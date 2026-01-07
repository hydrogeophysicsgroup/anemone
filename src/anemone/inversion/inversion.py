import torch
from anemone import rdtype


import scipy as sp
import numpy as np
from torch.linalg import solve

class BaseInversion:
    """ Bare-bones damped least squares inversion class. """
    jac_perturbation_fraction        = 0.02
    data_misfit_acceptance_threshold = 0.1
    data_misfit_minimum_change       = 0.007
    damp_value                       = 5000.
    damp_reduction_factor            = np.sqrt(2)
    iter_cap                         = 50
    _dpred                           = None

    def __init__(self, data_observed, data_error, starting_model,  forward_operator, logger=print):
        """Create an inversion object, the necessary """
        self.data_observed       = data_observed
        self.data_error          = data_error
        self.model               = starting_model
        self.forward             = forward_operator
        self.result              = None
        self.nM                  = len(self.m)
        self.nD                  = len(data_observed)
        self.recip_error_squared = 1./self.data_error**2
        self.log                 = logger
        self.device = self.model.vec.device
        self.pslices = self.model.pslice

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
        self.mvec = mvec
        dm   = mvec*self.jac_perturbation_fraction
        dm[dm==0] = self.jac_perturbation_fraction
        m_pert    = torch.diag(dm) + mvec[:, None]
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
        if not torch.all(self.mvec==self._current_mvec) or self._G is None:
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
        self._current_mvec = self.mvec*1.1 if torch.all(self.mvec!=0) else torch.ones_like(self.mvec)# make sure the jacobian is calculated at least once
        self._G = None
        
        self.iteration   = 0
        self.misfits     = [np.inf]
        self.misfits.append(self.misfit())
        self.set_device(self.mvec.device)


    def mforward(self, mvec):
        return self.forward(*self.model.modelize(mvec))

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
        #if self.misfits[-1]>self.misfits[-2] and self.misfits[-3]==min(self.misfits):
        #    self.log('misfit is increasing, stopping inversion')
        #    return True
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
        #self.mvec = self.model.make_physical(self.mvec + dM)
        self.mvec += torch.nan_to_num(dM)
        self.iteration += 1
        self.misfits.append(self.misfit())
        self.log("{:03d}\tmisfit:\t{:2.2f}".format(self.iteration, self.misfits[-1]))
    
    def invert(self, **kwargs):
        self.setup_matrices()
        self.iter_cap = kwargs.pop('iter_cap', self.iter_cap)

        while not self.stop_inversion:
            self.run_iteration()

        mres = self.model.copy()
        mres.values = mres.modelize(self.mvec)[:self.model.n_ptypes]
        return mres

class Tikhonov(BaseInversion):
    vertical_smoothness = 1e2

    def setup_matrices(self):
        super().setup_matrices()
        self.HtH = self.model.H().T @ self.model.H() * self.vertical_smoothness

    def calc_dm(self):
        r_data = self.G.T @ self.Cdi @ (self.data_observed-self.data_predicted)
        Aown = self.GtG + self.damp_matrix + self.HtH
        return solve(Aown, r_data)

    def invert(self, iter_cap=10, **kwargs):
        return super().invert(iter_cap=iter_cap, **kwargs)


class BaseInversion(BaseInversion):
    def mforward(self, mvec):
        return super().mforward(mvec).detach().cpu().numpy()

class Tikhonov(Tikhonov):
    def mforward(self, mvec):
        return super().mforward(mvec).detach().cpu().numpy()

def fastjacobian(cls):
    class FastJacobian(cls):
        def jacobian(self, mvec):
            rho, thick = self.model.modelize(mvec)
            # rho = torch.from_numpy(rho)
            thick = torch.from_numpy(thick)# if thick is not None else None
            G = self.forward(rho, thick, jac=True)#.detach().cpu().numpy()
            G *= rho[:, None]
            G /= self.data_predicted.exp()
            return G.T
        def copy(self):
            from copy import deepcopy
            return deepcopy(self) 
        def calc_dm(self):
            dm = super().calc_dm()
            return dm
    return FastJacobian

class Inversion(Tikhonov):
    vertical_smoothness = 0.4
    one_over_error = None
    n_sites = None
    _GtG = None
    _r_data = None
    target_step = 0.5
    dampf = 1./5000.
    increase_count = 0
    data_misfits = np.inf
    actual_step = None
    
    def setup_matrices(self):
        if not isinstance(self.target_step, list): self.target_step = [self.target_step]
        if not isinstance(self.dampf, list): self.dampf = [self.dampf]
        if not isinstance(self.actual_step, list): self.actual_step = []
        if not isinstance(self.data_misfits, list): self.data_misfits = [self.data_misfits]

        super().setup_matrices()
        self.Rp = self.model.H()
        self.C_rp = np.diag(np.ones(self.Rp.shape[0]) / np.log1p(self.vertical_smoothness)**2)
        self.HtH = self.Rp.T @ self.C_rp @ self.Rp
        self.HtH = self.HtH.to(self.mvec.dtype)
        self.CtC = self.HtH

    def calc_dm(self):
        cdamp = self.dampf[-1]*1.
        gtgmax = self._GtG.diagonal().max()
        cdamp *= gtgmax
        Aown = self._GtG + self.CtC
        cImod = self.Imod.clone()
        dm = solve(Aown+cImod*cdamp, self._r_data)
        dm_rep = torch.max(torch.abs(dm))
        
        attempts=1
        measure = [dm_rep]
        
        while attempts<25:
            if dm_rep<self.target_step[-1]*0.9:
                cdamp /= (1.2)**attempts
            elif dm_rep>self.target_step[-1]*1.1:
                cdamp *= (1.2)**attempts
            else:
                break
            dm = solve(Aown+cImod*cdamp, self._r_data)
            dm_rep = torch.max(torch.abs(dm))
            measure.append(dm_rep)
            attempts+=1
        self.actual_step.append(dm_rep)
        self.dampf.append(cdamp/gtgmax)
        return torch.nan_to_num(torch.tensor(dm, device=self.mvec.device, dtype=self.mvec.dtype))

    
    def run_iteration(self):
        dM   = self.calc_dm()


        self.mvec += dM
        self.iteration += 1
        new_misfit = self.misfit()
        
        dMisfit = (new_misfit-self.misfits[-1])/self.misfits[-1]
        d_misfit = self.data_misfit().mean().item()
    
        dDatMisfit = (d_misfit - self.data_misfits[-1])/self.data_misfits[-1]

        if dMisfit>0. or dDatMisfit>0:
            self.dampf.append(self.dampf[-1]*3.)
            self.mvec -= dM
            self.target_step[-1] *= 0.5
            self.increase_count +=1
            self.actual_step.pop(-1)

        else:
            self.misfits.append(new_misfit)
            if self.misfits[-1]<1.2:
                self.target_step.append(0.2)
            else:
                self.target_step.append(self.target_step[-1]+0.1)
            self.increase_count = 0
            self.data_misfits.append(d_misfit)

        self.log("{:03d}\tmisfit:\t{:2.2f}".format(self.iteration, self.misfits[-1]))
        print('data misfit: ', d_misfit)
    
    @property
    def stop_inversion(self):
        if super().stop_inversion:
            return True
        if self.increase_count>2:
            print('stop for repeated misfit increase')
            return True
        if self.target_step[-1]<0.01 and self.iteration>5:
            print('stop for too small target step')
            return True
        return False
    
fastJac = fastjacobian(Inversion)

from .dolphinv import LCI as dolphinv_LCI, Inversion as dInversion
class LCI(dolphinv_LCI):
    one_over_error = None
    n_sites        = None
    _GtG           = None
    _r_data        = None
    target_step    = 0.5
    dampf          = 1./5000.
    increase_count = 0
    data_misfits   = np.inf
    actual_step    = None

    def setup_matrices(self):
        if not isinstance(self.target_step, list): self.target_step = [self.target_step]
        if not isinstance(self.dampf, list): self.dampf = [self.dampf]
        if not isinstance(self.actual_step, list): self.actual_step = []
        if not isinstance(self.data_misfits, list): self.data_misfits = [self.data_misfits]        
        return super().setup_matrices()

    def mforward(self, mvec):
        if self.n_sites is None: self.n_sites = len(self.inv_list)
        mvecs = mvec.reshape(self.n_sites, -1).T
        rhos = mvecs.exp()
        self._dpredgrid = self.inv_list[0].forward(rhos, self.inv_list[0].model.thickness)
        return self._dpredgrid.flatten()
    
    def _model_update(self, mvec):
        self.n_model_updates += 1
        self.mvec = mvec
        self._current_mvec = self.mvec*1.

        self._dpred = self.mforward(self.mvec)
        self.jacobian(self.mvec)
        self._r_data -= self.CtC @ self.cmvec 

    def jacobian(self, mvec):
        if self.one_over_error is None:
            self.n_sites = len(self.inv_list)
            self.one_over_error = torch.sqrt(self.recip_error_squared).reshape(self.n_sites, -1)[:, None, :]
            self._dobsgrid = self.data_observed.reshape(self.n_sites, -1)
            self._derrgrid = self.data_error.reshape(self.n_sites, -1)**2
        mvecs = mvec.reshape(self.n_sites, -1).T
        rhos = mvecs.exp()
        thick = self.inv_list[0].model.thickness
        Ggrid = torch.empty((self.n_sites, len(thick)+1, self._dobsgrid.shape[-1]), dtype=rdtype, device=self.device)
        if self.n_sites<100:
            Ggrid = self.inv_list[0].forward(rhos, thick, jac=True)
        else:
            batch=50
            i=0
            while (i+2)*batch<self.n_sites:
                Ggrid[i*batch:(i+1)*batch] = self.inv_list[0].forward(rhos[:, i*batch:(i+1)*batch], thick, jac=True)
                i += 1
            Ggrid[i*batch:] = self.inv_list[0].forward(rhos[:, i*batch:], thick, jac=True)


        Ggrid /= self._dpredgrid.exp()[:, None, :]
        Ggrid *= rhos.T[:,:,None]
        r_data = torch.einsum('ijk,ijk->ij', Ggrid, ((self._dobsgrid-self._dpredgrid)/self._derrgrid)[:, None, :])
        Ggrid *= self.one_over_error

        GtG = Ggrid @ Ggrid.swapaxes(-1,-2)
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
            self.CtC = sp.sparse.bsr_matrix(self.CtC.cpu().numpy())
            
        else:
            self._GtG.data[...] = GtG_np*1.  # no flattening
        
        self._r_data = r_data.cpu().numpy().flatten()
        self.cmvec = self.mvec.cpu().numpy()*1.

    def calc_dm(self):
        
        cdamp = self.dampf[-1]*1.
      #  if self.iteration==0:
        gtgmax = self._GtG.diagonal().max()
        cdamp *= gtgmax
        # cdamp = max(cdamp, 1.)
        # if self.iteration>2:
        #     Aown = self._GtG*self.data_misfits[-1] + self.CtC*self.misfits[-1]
        # else:
        Aown = self._GtG + self.CtC
        cImod = self.Imod.copy()
        # cImod.data[:] = 1.
        # cImod.data[self._GtG.data<1e-6] = 0.
        # dm = sp.sparse.linalg.spsolve(Aown+cImod*cdamp, self._r_data)[0]
        dm = sp.sparse.linalg.cg(Aown+cImod*cdamp, self._r_data, maxiter=500)[0]
        dm_rep = np.max(np.abs(dm))
        
        attempts=1
        measure = [dm_rep]
        
        while attempts<25:
            if dm_rep<self.target_step[-1]*0.75:
                cdamp /= (1.2)**attempts
            elif dm_rep>self.target_step[-1]*1.5:
                cdamp *= (1.2)**attempts
            else:
                break
            # dm = sp.sparse.linalg.spsolve(Aown+cImod*cdamp, self._r_data)[0]
            dm = sp.sparse.linalg.cg(Aown+cImod*cdamp, self._r_data, maxiter=500)[0]
            dm_rep = np.max(np.abs(dm))
            measure.append(dm_rep)
            attempts+=1
        self.actual_step.append(dm_rep)
        print(attempts, ' attempts ', measure)
        self.dampf.append(cdamp/gtgmax)
        return torch.nan_to_num(torch.tensor(dm, device=self.mvec.device, dtype=self.mvec.dtype))

    
    def run_iteration(self):
        dM   = self.calc_dm()


        self.mvec += dM
        self.iteration += 1
        new_misfit = self.misfit()
        
        dMisfit = (new_misfit-self.misfits[-1])/self.misfits[-1]
        d_misfit = self.data_misfit().mean().item()
        
        print('dMisfit: ', dMisfit)
        dDatMisfit = (d_misfit - self.data_misfits[-1])/self.data_misfits[-1]

        if dMisfit>0. or dDatMisfit>0:
            self.dampf.append(self.dampf[-1]*3.)
            self.mvec -= dM
            self.target_step[-1] *= 0.5
            self.increase_count +=1
            self.actual_step.pop(-1)
        # elif dMisfit>0.1:
        #     self.dampf.append(self.dampf[-1]*1.1)
        #     self.target_step.append(max(self.target_step[-1]/2., self.target_step[-1]*-0.1))
        #     self.misfits.append(new_misfit)
        #     self.data_misfits.append(d_misfit)
        #     self.increase_count += 1
        else:
            self.misfits.append(new_misfit)
            if self.misfits[-1]<1.2:
                self.target_step.append(0.2)
            else:
                self.target_step.append(self.target_step[-1]+0.1)
            self.increase_count = 0
            self.data_misfits.append(d_misfit)

        self.log("{:03d}\tmisfit:\t{:2.2f}".format(self.iteration, self.misfits[-1]))
        print('data misfit: ', d_misfit)
    
    @property
    def stop_inversion(self):
        if super().stop_inversion:
            return True
        if self.increase_count>2:
            print('stop for repeated misfit increase')
            return True
        if self.target_step[-1]<0.01 and self.iteration>5:
            print('stop for too small target step')
            return True
        return False



class SCI(LCI):
    def make_smoothness_matrix(self):
        from scipy.spatial import Delaunay

        assert len(self.x) == len(self.y), "x and y must have the same length"
        assert len(self.x) == len(self.inv_list), "x and y must match the number of sites"

        points = np.vstack((self.x, self.y)).T
        tri = Delaunay(points)
        connected_pairs = []
        for simplex in tri.simplices:
            simplex.sort()
            p1 = int(simplex[0]), int(simplex[1])
            p2 = int(simplex[0]), int(simplex[2])
            p3 = int(simplex[1]), int(simplex[2])
            if p1 not in connected_pairs:
                connected_pairs.append(p1)
            if p2 not in connected_pairs:
                connected_pairs.append(p2)
            if p3 not in connected_pairs:
                connected_pairs.append(p3)

        all_dist = np.zeros(len(connected_pairs))


        for i,pair in enumerate(connected_pairs):
            p1 = points[pair[0]]
            p2 = points[pair[1]]
            all_dist[i] = np.linalg.norm(p1 - p2)

        min_dist = np.ones(len(points))*all_dist.max()

        for i,pair in enumerate(connected_pairs):    
            min_dist[pair[0]] = min(min_dist[pair[0]], all_dist[i])

        constraint_strength = min_dist.mean() / all_dist

        n_layers = self.inv_list[0].model.n_layers
        c_mat = np.zeros((len(connected_pairs), len(points)*n_layers))
        for i, pair in enumerate(connected_pairs):
            c_mat[i, ::n_layers][pair[0]] =  constraint_strength[i]
            c_mat[i, ::n_layers][pair[1]] = -constraint_strength[i]

        c_mat = np.repeat(c_mat, n_layers, axis=0)

        for i in range(n_layers):
            c_mat[i::n_layers, :] = np.roll(c_mat[i::n_layers, :], i, axis=1)
        return torch.from_numpy(c_mat)
    