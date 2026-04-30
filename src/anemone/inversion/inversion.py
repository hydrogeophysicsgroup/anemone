import torch
from anemone import rdtype


import scipy as sp
import numpy as np
from torch.linalg import solve

from nosj import nosj
from .models import BaseModel
from anemone.forwards import Forward

@nosj
class BaseInversion:
    data_observed                    : torch.Tensor
    data_error                       : torch.Tensor
    starting_model                   : BaseModel
    forward                          : Forward
    jac_perturbation_fraction        : float = 0.02
    data_misfit_acceptance_threshold : float = 0.1
    data_misfit_minimum_change       : float = 0.007
    damp_value                       : float = 5000.
    damp_reduction_factor            : float = 2**0.5
    iter_cap                         : int = 50
    final_model                      : BaseModel = None
    
    description                      : str = "Inversion Class"


    def __init__(self, data_observed, data_error, starting_model,  forward, **kwargs):
        """Create an inversion object, the necessary """
        self.data_observed                    = data_observed
        self.data_error                       = data_error
        self.starting_model                   = starting_model.copy()
        self.model                            = starting_model
        self.forward                          = forward
        self.nM                               = len(self.m)
        self.nD                               = len(data_observed)
        self.recip_error_squared              = 1./self.data_error**2
        self.log                              = kwargs.get('logger', print)
        self.device                           = self.model.vec.device
        self.pslices                          = self.model.pslice
        self.jac_perturbation_fraction        = kwargs.get('jac_perturbation_fraction', self.jac_perturbation_fraction)
        self.data_misfit_acceptance_threshold = kwargs.get('data_misfit_acceptance_threshold', self.data_misfit_acceptance_threshold)
        self.data_misfit_minimum_change       = kwargs.get('data_misfit_minimum_change', self.data_misfit_minimum_change)
        self.damp_value                       = kwargs.get('damp_value', self.damp_value)
        self.damp_reduction_factor            = kwargs.get('damp_reduction_factor', self.damp_reduction_factor)
        self.iter_cap                         = kwargs.get('iter_cap', self.iter_cap)
        self.final_model                      = kwargs.get('final_model', self.final_model)
        self._current_mvec = None


    def set_device(self, device):
        # self.device=device
        for var_name in vars(self):
            var = getattr(self, var_name)
            if torch.is_tensor(var):
                setattr(self, var_name, var.to(device))
            elif isinstance(var, list):
                for i, v in enumerate(var):
                    if torch.is_tensor(v):
                        var[i] = v.to(device)

    @property
    def m(self):
        return self.model.vec
    
    def data_misfit(self):
        return (self.data_observed-self.data_predicted)**2 * self.recip_error_squared

    def misfit(self):
        return torch.sqrt(self.data_misfit().mean())
    
    def jacobian(self, mvec):
        dm   = mvec*self.jac_perturbation_fraction
        dm[dm==0] = self.jac_perturbation_fraction
        m_pert    = torch.diag(dm) + mvec.clone()[:, None]
        d_data    = self.mforward(m_pert)
        Gt = (d_data - self.data_predicted) / dm[:, None]
        return Gt.T
    
    def _model_update(self, mvec):
        self.mvec = mvec
        self._current_mvec = self.mvec*1.
        self._dpred = self.mforward(self.mvec)
        self._G = self.jacobian(self.mvec)
        self._GtG = self._G.T @ self.Cdi @ self._G
        self._r_data = (self._G.T @ self.Cdi @ (self.data_observed - self._dpred))
        

    @property
    def data_predicted(self):
        if torch.any(self.mvec!=self._current_mvec) or self._dpred is None:
            self._model_update(self.mvec)
        return self._dpred
    
    @property
    def G(self):
        if self._current_mvec is None or not torch.all(self.mvec==self._current_mvec) or self._G is None:
            self._model_update(self.mvec)
        return self._G
    
    @property
    def GtG(self):
        if torch.any(self.mvec!=self._current_mvec):
            self._model_update(self.mvec)
        return self._GtG

    @property
    def r_data(self):
        if torch.any(self.mvec!=self._current_mvec) or self._r_data is None:
            self._model_update(self.mvec)
        return self._r_data

    @property
    def damp_matrix(self):
        damp = self.GtG.diagonal()/self.damp_value
        return damp*self.Imod

    def setup_matrices(self):
        self.mvec = self.m*1.
        self._current_mvec = self.mvec*1.1 if torch.all(self.mvec!=0) else torch.ones_like(self.mvec)# make sure the jacobian is calculated at least once
        self.Cdi = torch.diag(self.recip_error_squared).to(rdtype)
        self.Imod = torch.eye(len(self.mvec), device=self.mvec.device)
        self._G = None
        
        self.iteration   = 0
        self.misfits     = [np.inf]
        self.set_device(self.mvec.device)


    def mforward(self, mvec):
        mparam = self.model.modelize(mvec)
        if hasattr(self.forward, 'device'):
            mparam = [x.to(self.forward.device) for x in mparam]
        return self.forward(*mparam)

    @property
    def stop_inversion(self):
        if self.iteration<=1:
            return False
        if (abs(self.misfits[-2]-self.misfits[-1]) / self.misfits[-1]) < self.data_misfit_minimum_change and self.misfits[-1]>self.misfits[-2]:
            self.log(f'misfit change is less than {self.data_misfit_minimum_change}')
            return True
        if self.misfits[-1]<self.data_misfit_acceptance_threshold:
            self.log(f'misfit is below {self.data_misfit_acceptance_threshold}, stopping inversion')
            return True
        if self.iteration>=self.iter_cap:
            self.log('reached iteration cap..')
            return True
        return False

    def calc_dm(self):
        r_data = self.G.T @ self.Cdi @ (self.data_observed-self.data_predicted)
        Aown = self.GtG + self.damp_matrix
        return solve(Aown, r_data)
    
    def run_iteration(self):
        dM   = self.calc_dm()
        self.damp_value *= self.damp_reduction_factor
        # self.mvec = self.model.make_physical(self.mvec + dM)
        self.mvec += torch.nan_to_num(dM)
        self.iteration += 1
        self.misfits.append(self.misfit())
        self.log("{:03d}\tmisfit:\t{:2.2f}".format(self.iteration, self.misfits[-1]))
    
    def invert(self, **kwargs):
        self.setup_matrices()
        self.iter_cap = kwargs.pop('iter_cap', self.iter_cap)

        while not self.stop_inversion:
            self.run_iteration()
        self.final_model = self.model.update_values(self.mvec)
        return self.final_model

class Tikhonov(BaseInversion):
    vertical_smoothness = 0.2

    def setup_matrices(self):
        from math import log
        super().setup_matrices()
        self.HtH = self.model.H().T @ (self.model.H() * log(1.+self.vertical_smoothness)**-2)

    def calc_dm(self):
        r_data = self.G.T @ self.Cdi @ (self.data_observed-self.data_predicted)
        damp_mask = (self.GtG.diagonal() < 0.01*self.HtH.diagonal()).to(self.GtG.dtype)
        c_damp = damp_mask @ self.damp_matrix
        Aown = self.GtG + c_damp + self.HtH
        return solve(Aown, r_data)


def fastjacobian(cls):
    class FastJacobian(cls):
        def __post_init__(self):
            super().__post_init__()
            assert self.forward.kind=='res', "FastJacobian only works for resistivity forward models"
            assert self.model.free[-1]==False, "FastJacobian only works for models with fixed thicknesses"

        def jacobian(self, mvec):
            rho, thick = self.model.modelize(mvec)
            if isinstance(thick, np.ndarray): thick = torch.from_numpy(thick)
            G = self.forward(rho, thick, jac=True)
            if self.model.log_scale:
                G *= rho[:, None]
            if self.forward.log_scale:
                G /= self.data_predicted.exp()[None, :]
            return G.T
    return FastJacobian



@nosj
class Inversion(BaseInversion):
    log = print
    vertical_smoothness = 0.5
    one_over_error = None
    n_sites = None
    _GtG = None
    _r_data = None
    max_step = torch.log(torch.tensor(3.0))
    min_step = torch.log(torch.tensor(1.1))
    damp_exponent_start = 6
    diag_normalizer = 5e3
    damping_factor = 3.0
    step_down = torch.log(torch.tensor(2.0))
    step_up   = torch.log(torch.tensor(1.2))
    increase_count = 0
    data_misfits = np.inf

    def __init__(self, data_observed, data_error, starting_model,  forward, **kwargs):
        """Create an inversion object, the necessary """
        super().__init__(data_observed, data_error, starting_model, forward, **kwargs)
    
    def setup_matrices(self):
        if not isinstance(self.data_misfits, list): self.data_misfits = [self.data_misfits]

        super().setup_matrices()
        self.Rp = self.model.H()
        self.mvec = self.mvec.to(rdtype)
        self.C_rp = np.diag(np.ones(self.Rp.shape[0]) / np.log1p(self.vertical_smoothness)**2)
        self.C_rp = self.C_rp.astype(np.float32)
        self.HtH = self.Rp.T @ self.C_rp @ self.Rp
        self.CtC = self.HtH
        self.damp_exponent = torch.ones(len(self.pslices))*self.damp_exponent_start
        self._model_update(self.mvec)
        self.max_step = torch.ones(len(self.pslices))*self.max_step
        self.min_step = torch.ones(len(self.pslices))*self.min_step
        self.damp_increase = torch.ones(len(self.pslices)).bool()
        self.n_local=0

    def vert_misfit(self):
        r_mod = -self.Rp @ self.mvec
        m_misfit = r_mod @ self.C_rp @ r_mod
        return m_misfit.mean()

    def misfit(self):
        return torch.sqrt((self.data_misfit().sum()+self.vert_misfit())/(len(self.data_observed)+self.Rp.shape[0]))
    
    def damp_matrix(self):
        if self.damping_attempts==0:
            diag_GTG = torch.tensor(self.GtG.diagonal())
            self.damp = torch.empty_like(diag_GTG)
            max_GTG = [diag_GTG[pslice].max()/self.diag_normalizer for pslice in self.pslices]
            for i,pslice in enumerate(self.pslices):
                if self.damp_exponent[i]>0:
                    self.damp[pslice] =  max_GTG[i]*self.damping_factor**(self.damp_exponent[i])
                else:
                    self.damp[pslice] = 0
        else:
            for i,pslice in enumerate(self.pslices):
                if self.test_damping[i]:
                    if self.damp_increase[i]:
                        if self.damp_exponent[i]>0:
                            self.damp[pslice] =  self.damp[pslice]*self.damping_factor
                            self.damp_exponent[i] +=1
                        else:
                            diag_GTG = self.GtG.diagonal()
                            self.damp_exponent[i] +=1
                            self.damp[pslice] =  diag_GTG[pslice].max()/self.diag_normalizer*self.damping_factor**self.damp_exponent[i]
                    else:
                        if self.damp_exponent[i]>0:
                            self.damp[pslice] =  self.damp[pslice]/self.damping_factor
                            self.damp_exponent[i] -=1
                        else:
                            self.damp[pslice] = 0
        #self.damp = torch.clamp(self.damp, min=1.)  # Ensure damping is at least 1
        

        if sp.sparse.issparse(self.Imod):
            self.damp *= (self.GtG.diagonal()>0.05*self.CtC.diagonal()).astype(float)
            return sp.sparse.csr_matrix(self.Imod.multiply(self.damp.cpu().numpy()))# self.damp*self.Imod
        else:
            self.damp *= (self.GtG.diagonal()>0.05*self.CtC.diagonal()).to(float)
            return self.damp * self.Imod

    def find_damping(self):
        self.damping_attempts = 0
        self.test_damping = [True]*len(self.pslices)

        while any(self.test_damping):
            if self.damping_attempts > 20:
                self.test_damping = [False] * len(self.pslices)
                break

            

            dM = self.calc_dm()
            dM_abs = torch.abs(dM)

            for i, pslice in enumerate(self.pslices):
                max_dM = dM_abs[pslice].max()

                if max_dM > self.max_step[i]:
                    if self.damp_increase[i]:
                        continue
                    else:
                        self.damp_increase[i] = True
                        self.test_damping[i] = False
                        break  # force recalculation next while-iteration

                else:  # max_dM <= max_step[i]
                    if self.damp_increase[i] or self.damp_exponent[i] == 0:
                        self.test_damping[i] = False
            self.damping_attempts += 1
        return dM
        

    def calc_dm(self):
        Aown = self._GtG + self.CtC + self.damp_matrix()
        dm = sp.sparse.linalg.spsolve(Aown, self._r_data)

        return torch.nan_to_num(torch.tensor(dm, device=self.mvec.device, dtype=self.mvec.dtype))

    
    def run_iteration(self):
        import time
        time_old = time.time()
        self.iteration += 1
        self.damping_attempts = 0
        self.test_damping = [True]*len(self.pslices)


        dM = self.find_damping()
        self.mvec      = self.mvec+torch.nan_to_num(dM)
        self.new_misfit = self.misfit()
        self.n_local += 1
        self.data_predicted
        
        misfit_increased =  self.new_misfit > self.misfits[-1]
        misfit_change_small = (abs(self.misfits[-1] - self.new_misfit) / self.new_misfit < self.data_misfit_minimum_change)
        step_ratio_significant = torch.any(torch.abs(1 - self.max_step / self.min_step) >= 0.001)


        if (misfit_increased and not self.iteration < 1):
            for i, _ in enumerate(self.pslices):
                self.max_step[i] -= self.step_down
            self.iteration -= 1
            self.n_local = 0
            self.mvec = self.mvec-dM
            self.damp_exponent[:] += 2
        elif self.n_local > 2:
            for i, _ in enumerate(self.pslices):
                self.max_step[i] += self.step_up
            self.misfits.append(self.new_misfit)
            self.diag_normalizer *= 3
            self.n_local = 0
            self.misfits.append(self.new_misfit)
        elif misfit_change_small:
            if step_ratio_significant:
                for i, _ in enumerate(self.pslices):
                    self.max_step[i] -= self.step_down
                self.n_local = 0
                self.misfits.append(self.new_misfit)
                print('Reducing step size due to small misfit change')
        else:
            self.misfits.append(self.new_misfit)
        # Check if max_step is below min_step
        for i in range(len(self.max_step)):
            if self.max_step[i] < self.min_step[i]:
                self.max_step[i] = self.min_step[i]

        # Decrease damping before next iteration.
        for i in range(len(self.damp_exponent)):
            self.damp_exponent[i] = max(self.damp_exponent[i] - 1, 0)
        if self.misfits[-1] == self.new_misfit:
            time_new = time.time()
            if hasattr(self, "inv_list") and isinstance(self.inv_list, list):
                self.log("{:03d}\tData misfit:\t{:2.2f}\tVertical misfit:\t{:2.2f}\tHorizontal misfit:\t{:2.2f}\tTotal misfit:\t{:2.2f}\tStep size:\t{:2.2f}\tLast iteration time:\t{:2.2f}\n".format(
                    self.iteration,
                    torch.sqrt(self.data_misfit().sum() / len(self.data_observed)),
                    self.vert_misfit()/self.Rp.shape[0],
                    self.horz_misfit()/self.RL.shape[0],
                    self.misfits[-1],
                    self.max_step[0],
                    time_new - time_old
                ))
            else:
                self.log("{:03d}\tData misfit:\t{:2.2f}\tVertical misfit:\t{:2.2f}\tTotal misfit:\t{:2.2f}\tStep size:\t{:2.2f}\tLast iteration time:\t{:2.2f}\n".format(
                    self.iteration,
                    torch.sqrt(self.data_misfit().sum() / len(self.data_observed)),
                    self.vert_misfit()/self.Rp.shape[0],
                    self.misfits[-1],
                    self.max_step[0],
                    time_new - time_old
                ))
        else:
            print("{:03d}\tTotal misfit increased, smaller step\t".format(self.iteration))


    @property
    def stop_inversion(self):
        if super().stop_inversion:
            return True
        if self.iteration<=1:
            return False
        if (abs(self.misfits[-2]-self.misfits[-1]) / self.misfits[-1]) < self.data_misfit_minimum_change and self.misfits[-1]>self.misfits[-2]:
            self.log(f'misfit change is less than {self.data_misfit_minimum_change}')
            return True
        if torch.any(self.max_step == self.min_step):
            print('stop for reaching min step size')
            self.run_iteration()
            return True
        return False

    def invert(self, **kwargs):
        self.setup_matrices()
        self.misfits = [self.misfit()]
        self.iter_cap = kwargs.pop('iter_cap', self.iter_cap)

        while not self.stop_inversion:
            self.run_iteration()
        self.final_model = self.model.update_values(self.mvec)
        self.doi = self.calc_doi()
        # self.doi2 = self.calc_doi_fred()
        return self.final_model


    def calc_doi(self, mvec=None, threshold=0.8):
        if mvec is None:
            mvec = self.m
        G = self.G
        lin_data_err = torch.exp(self.data_error)
        z = torch.cumsum(self.model.values[1],dim=0)
        err_norm_G = torch.abs(G) / lin_data_err[:, None]
        sum_norm_G = torch.sum(err_norm_G, dim=0)
        cum_sens = torch.cumsum(sum_norm_G.flip(0), dim=0).flip(0)
        if torch.min(cum_sens) > threshold: idx = len(z)-1
        elif torch.searchsorted(cum_sens.flip(0), threshold, right=True) == 1: idx = len(z)-1
        else: idx = len(cum_sens) - torch.searchsorted(cum_sens.flip(0), threshold, right=True)
        x0, x1 = cum_sens[idx], cum_sens[idx - 1]
        y0, y1 = z[idx].log(), z[idx - 1].log()
        z_doi = torch.exp(y0 + (threshold - x0) * (y1 - y0) / (x1 - x0))
        return z_doi


fastJac = fastjacobian(Inversion)

class LCI(Inversion):
    #log = print
    one_over_error = None
    n_sites        = None
    _GtG           = None
    _r_data        = None
    horizontal_smoothness = 0.8
    data_misfits   = np.inf
    reuse_jac = True

    @property
    def m(self):
        return torch.concat([inv.m for inv in self.inv_list])

    def __init__(self, inv_list, x=None, y=None,log=print):
        self.log = log
        self.damping_attempts = 0
        self.inv_list = inv_list
        self.data_observed = torch.concat([inv.data_observed for inv in self.inv_list])
        self.data_error    = torch.concat([inv.data_error    for inv in self.inv_list])
        self.recip_error_squared = 1. / self.data_error**2
        self.iter_cap = 50
        self.misfits = [None]
        self.NDamp = 6  # Initial damping exponent
        self.x = x
        self.y = y
        self.n_model_updates=0

    def make_smoothness_matrix(self):
        n = len(self.mvec)
        m_per_inv = len(self.inv_list[0].m)
        RL2 = (sp.sparse.eye(n, format="csr")- sp.sparse.eye(n, k=m_per_inv, format="csr"))[:-m_per_inv]
        # RL = (sp.sparse.eye(n) - sp.sparse.eye(n, k=m_per_inv)).tocsr()[:-m_per_inv]
        return RL2

    def setup_matrices(self):
        if not isinstance(self.data_misfits, list): self.data_misfits = [self.data_misfits]
        self.mvec = self.m.to(torch.float)        
        self.device =self.mvec.device
        if not hasattr(self, 'step_down'): self.step_down = torch.log(torch.tensor(2.0))
        if not hasattr(self, 'step_up'):   self.step_up   = torch.log(torch.tensor(1.2))


        self.Cdi = self.recip_error_squared.to(torch.float)
        from math import log
        # Vertical constraints
        # self.Rp = (sp.sparse.eye(len(self.mvec),format="csr")-sp.sparse.eye(len(self.mvec), k=1,format="csr"))
        # print(self.Rp.shape)
        n_cols = len(self.mvec)
        n_rows = n_cols - len(self.inv_list)
        

        rows = []
        cols = []
        data = []

        for i in range(n_rows):
            shift = i // (len(self.inv_list[0].m)-1)
            j = i + shift

            if j + 1 < n_cols:
                rows.append(i)
                cols.append(j)
                data.append(1.0)
                rows.append(i)
                cols.append(j + 1)
                data.append(-1.0)

        self.Rp = sp.sparse.coo_matrix((data, (rows, cols)), shape=(n_rows, n_cols))
        self.Rp = self.Rp.tocsr()

        self.C_rp = sp.sparse.eye(self.Rp.shape[0])/log(1.+self.vertical_smoothness)**2
        HtH = self.Rp.T @ self.C_rp @ self.Rp

        # Horizontal constraints
        self.RL= self.make_smoothness_matrix()
        self.C_rl = sp.sparse.eye(self.RL.shape[0])/log(1.+self.horizontal_smoothness)**2
        LtL = self.RL.T @ self.C_rl @ self.RL
        
        self.CtC = (HtH + LtL)

        self.iteration     = 0
        self._model_update(self.mvec)

        all_pslices = [[] for _ in self.inv_list[0].model.pslice]
        for inv in self.inv_list:
            for i, pslice in enumerate(inv.model.pslice):
                pslice_bool = torch.zeros_like(inv.m, dtype=torch.bool)
                pslice_bool[pslice] = True
                all_pslices[i].append(pslice_bool)
        self.pslices = [torch.concat(slices).to(self.device) for slices in all_pslices]
        self.damp_exponent = torch.ones(len(self.pslices))*self.damp_exponent_start
        if hasattr(self.inv_list[0].model, 'max_step'):
            self.max_step = self.inv_list[0].model.max_step
        else:
            self.max_step = torch.ones(len(self.inv_list[0].pslices))*self.max_step_init

        if hasattr(self.inv_list[0].model, 'min_step'):
            self.min_step = self.inv_list[0].model.min_step
        else:
            self.min_step = torch.ones(len(self.inv_list[0].pslices))*self.min_step
        self.damp_increase = torch.ones(len(self.pslices)).bool()
        return #super().setup_matrices()

    def horz_misfit(self):
        r_mod = -self.RL * self.mvec
        m_misfit = r_mod @ self.C_rl @ r_mod
        return m_misfit.mean()

    def misfit(self):
        return torch.sqrt((self.data_misfit().sum()+self.vert_misfit()+self.horz_misfit())/(len(self.data_observed)+self.RL.shape[0]+self.Rp.shape[0]))

    def mforward(self, mvec):
        if self.n_sites is None: self.n_sites = len(self.inv_list)
        mvecs = mvec.reshape(self.n_sites, -1).T
        _, thickness = self.inv_list[0].model.mod
        rhos = mvecs.exp()
        if hasattr(self.inv_list[0].forward, 'device'):
            rhos = rhos.to(self.inv_list[0].forward.device)
            thickness = thickness.to(self.inv_list[0].forward.device)
        self._dpredgrid = self.inv_list[0].forward(rhos, thickness)
        return self._dpredgrid.flatten()
    
    def _model_update(self, mvec):
        self.n_model_updates += 1
        self.mvec = mvec
        self._current_mvec = self.mvec*1.

        self._dpred = self.mforward(self.mvec)
        self.jacobian(self.mvec)
        # self._r_data -= self.CtC @ self.cmvec 
    
    def calc_dm(self):
        Aown = sp.sparse.csr_matrix(self._GtG) + self.CtC + self.damp_matrix()
        dm = sp.sparse.linalg.spsolve(Aown, self._r_data)

        return torch.nan_to_num(torch.tensor(dm, device=self.mvec.device, dtype=self.mvec.dtype))


    def jacobian(self, mvec):
        if self.one_over_error is None:
            self.n_sites = len(self.inv_list)
            self.one_over_error = torch.sqrt(self.recip_error_squared).reshape(self.n_sites, -1)[:, None, :]
            self._dobsgrid = self.data_observed.reshape(self.n_sites, -1)
            self._derrgrid = self.data_error.reshape(self.n_sites, -1)**2
            _,thick = self.inv_list[0].model.mod
            self.Ggrid = torch.empty((self.n_sites, len(thick)+1, self._dobsgrid.shape[-1]), dtype=rdtype, device=self.device)
        mvecs = mvec.reshape(self.n_sites, -1).T
        _,thick = self.inv_list[0].model.mod
        rhos = mvecs.exp()
        
        if hasattr(self.inv_list[0].forward, 'device'):
            rhos = rhos.to(self.inv_list[0].forward.device)
            thick = thick.to(self.inv_list[0].forward.device)

        if self.reuse_jac:
            mvec0 = mvecs[:, 0]
            Gsingle = self.inv_list[0].forward(rhos[:, 0], thick, jac=True)
            self.Ggrid[0] = Gsingle
            for i_site in range(1, self.n_sites):
                cvec = mvecs[:, i_site]
                rel_diff = torch.abs(cvec - mvec0) / torch.abs(mvec0)
                if torch.all(rel_diff < 1e-1):
                    self.Ggrid[i_site] = Gsingle
                else:
                    mvec0 = cvec
                    self.Ggrid[i_site] = self.inv_list[i_site].forward(rhos[:, i_site], thick, jac=True)
        else:
            batch=50
            if self.n_sites<batch:
                self.Ggrid = self.inv_list[0].forward(rhos, thick, jac=True)
            else:
                i=0
                while (i+2)*batch<self.n_sites:
                    self.Ggrid[i*batch:(i+1)*batch] = self.inv_list[0].forward(rhos[:, i*batch:(i+1)*batch], thick, jac=True)
                    i += 1
                self.Ggrid[i*batch:] = self.inv_list[0].forward(rhos[:, i*batch:], thick, jac=True)


        self.Ggrid /= self._dpredgrid.exp()[:, None, :]
        self.Ggrid /= rhos.cpu().T[:,:,None]
        r_data = torch.einsum('ijk,ijk->ij', self.Ggrid, ((self._dobsgrid-self._dpredgrid)/self._derrgrid)[:, None, :])
        self.Ggrid *= self.one_over_error

        GtG = self.Ggrid @ self.Ggrid.swapaxes(-1,-2)
        GtG_np = GtG.cpu().numpy()  # shape: (n_sites, br, bc)
        n_blocks, br, bc = GtG_np.shape
        

        if self._GtG is None:
            indices = np.arange(n_blocks, dtype=np.int32)      # one diag block per row-block
            indptr  = np.arange(n_blocks + 1, dtype=np.int32)  # cumulative count

            self._GtG = sp.sparse.bsr_matrix(
                (GtG_np.copy(), indices, indptr),
                shape=(n_blocks * br, n_blocks * bc),
            )
            self.blocksize = (br, bc)
            eye_block = np.eye(br, bc, dtype=GtG_np.dtype)
            eye_blocks = np.repeat(eye_block[None, :, :], n_blocks, axis=0)
            self.Imod = sp.sparse.bsr_matrix(
                (eye_blocks, indices, indptr),
                shape=self._GtG.shape,
            )
        else:
            self._GtG.data[...] = GtG_np*1.  # no flattening
        
        self._r_data = r_data.cpu().numpy().flatten()
        self.cmvec = self.mvec.cpu().numpy()*1.

    
    def invert(self, iter_cap=10, **kwargs):
        self.setup_matrices()
        self.misfits = [self.misfit()]
        self.iter_cap = iter_cap
        self.n_local=0
        while not self.stop_inversion:
            self.run_iteration()
            
        # Split mvec back to each inversion
        mres_list = []
        doi_list = []
        mvecs = self.mvec.reshape(len(self.inv_list), -1)
        self.Ggrid /= self.one_over_error
        for i_site, (inv, mvec) in enumerate(zip(self.inv_list, mvecs)):
            mres = inv.model.copy()
            inv.mvec = mvec
            inv._current_mvec=mvec
            inv._G = torch.nan_to_num(self.Ggrid[i_site]).T

            mres.values = mres.modelize(mvec)
            mres_list.append(mres.copy())
            doi_value = inv.calc_doi(mvec=mvec)
            doi_list.append(doi_value)
        self.doi = doi_list
        return mres_list
    
    @property
    def stop_inversion(self):
        if super().stop_inversion:
            return True
        return False



class SCI(LCI):
    def make_smoothness_matrix(self):
        from scipy.spatial import Delaunay

        assert len(self.x) == len(self.y), "x and y must have the same length"
        assert len(self.x) == len(self.inv_list), "x and y must match the number of sites"

        points = np.vstack((self.x, self.y)).T
        n_points = len(points)

        # --- Delaunay connectivity ---
        tri = Delaunay(points)

        connected_pairs = set()
        for simplex in tri.simplices:
            simplex = np.sort(simplex)
            connected_pairs.add((simplex[0], simplex[1]))
            connected_pairs.add((simplex[0], simplex[2]))
            connected_pairs.add((simplex[1], simplex[2]))

        connected_pairs = list(connected_pairs)
        n_pairs = len(connected_pairs)

        # --- Edge lengths ---
        all_dist = np.zeros(n_pairs)
        for i, (i1, i2) in enumerate(connected_pairs):
            all_dist[i] = np.linalg.norm(points[i1] - points[i2])

        # --- Minimum neighbor distance per node ---
        min_dist = np.full(n_points, all_dist.max())
        for d, (i1, i2) in zip(all_dist, connected_pairs):
            min_dist[i1] = min(min_dist[i1], d)
            min_dist[i2] = min(min_dist[i2], d)

        constraint_strength = min_dist.mean() / all_dist

        # --- Model layout ---
        n_layers = self.inv_list[0].model.n_layers
        n_constraints = n_pairs * n_layers
        n_params = n_points * n_layers

        # --- Sparse assembly ---
        rows = []
        cols = []
        data = []

        for pair_idx, ((p1, p2), w) in enumerate(zip(connected_pairs, constraint_strength)):
            for layer in range(n_layers):
                row = pair_idx * n_layers + layer

                col1 = p1 * n_layers + layer
                col2 = p2 * n_layers + layer

                rows.extend([row, row])
                cols.extend([col1, col2])
                data.extend([w, -w])

        c_mat = sp.sparse.coo_matrix(
            (data, (rows, cols)),
            shape=(n_constraints, n_params)
        ).tocsr()

        return c_mat