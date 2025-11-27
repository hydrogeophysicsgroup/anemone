from .inversion import BaseInversion
import torch
import numpy as np
from anemone import rdtype
class Inversion(BaseInversion):
    vertical_smoothness = 0.1
    max_step_init                    = torch.log(torch.tensor(3.0))
    min_step                         = torch.log(torch.tensor(1.1))

    damping_factor                   = 3.
    damp_initmax                     = 1e6  
    damp_exponent_start              = 6
    diag_normalizer                  = 5e3
    step_down                        = torch.log(torch.tensor(2.0))
    step_up                          = torch.log(torch.tensor(1.2))
    Rp                               = None
    C_rp                             = None

    @property
    def stop_inversion(self):
        if self.iteration<=1:
            return False
        if (abs(self.misfits[-2]-self.misfits[-1]) / self.misfits[-1]) < self.data_misfit_minimum_change and self.misfits[-1]>self.misfits[-2]:
            self.log(f'misfit change is less than {self.data_misfit_minimum_change}')
            return True
        # if self.misfits[-1]<self.data_misfit_acceptance_threshold:
        #     self.log(f'misfit is below {self.data_misfit_acceptance_threshold}, stopping inversion')
        #     return True
        if torch.any(torch.abs(1 - self.max_step / self.min_step) <= 0.001):
            self.log(f'Maxstep is equal to minstep with no decrease in misfit, stopping inversion')
            return True
        if self.iteration>=self.iter_cap:
            self.log('reached iteration cap..')
            return True
        return False

    @property
    def damp_matrix(self):
        # Damping scheme following Aarhusinv. Used for both Inversion and LCI.
        # Creates a damping matrix to add to Aown.
        if self.damping_attempts==0:
            diag_GTG = self.GtG.diagonal()
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
        self.damp = torch.clamp(self.damp, min=1.)  # Ensure damping is at least 1
        return self.damp*self.Imod

    def vert_misfit(self):
        # Calculate the vertical constraints misfit.
        r_mod = - self.Rp @ self.mvec.to(torch.float)
        r = r_mod.permute(*torch.arange(r_mod.ndim - 1, -1, -1)) @ self.C_rp @ r_mod
        return r
    
    def misfit(self):
        return 10. / (1.+self.iteration)
#        return torch.sqrt((self.data_misfit().sum()+self.vert_misfit().sum())/(self.nD+self.nM-1))

    def setup_matrices(self):
        self.mvec = self.m.to(torch.float)
        self.device = self.mvec.device
        self.pslices = self.model.pslice
        self.Rp   = self.model.H().to(torch.float).to(self.device)
        self.C_rp = torch.eye(len(self.Rp), dtype=self.Rp.dtype, device=self.device)/(torch.log1p(torch.tensor(self.vertical_smoothness, dtype=torch.float32, device=self.device)) ** 2)
        self.HtH = self.Rp.T @ self.C_rp @ self.Rp
        super().setup_matrices()

        
        self.diag_normalizer = self.diag_normalizer
        self.Cdi = torch.diag(self.recip_error_squared)
        self.Imod = torch.eye(self.nM, device=self.mvec.device)
        self._current_mvec = self.mvec*1.1 if torch.all(self.mvec!=0) else torch.ones_like(self.mvec)# make sure the jacobian is calculated at least once
        self._G = None
        self.damp_exponent = torch.ones(len(self.pslices))*self.damp_exponent_start
        self.max_step = (torch.ones(len(self.pslices), device=self.device)*self.max_step_init)
        self.min_step = (torch.ones(len(self.pslices), device=self.device)*self.min_step)
        self.set_device(self.mvec.device)
        self.damp_increase = torch.ones(len(self.pslices), device=self.device)

        self.set_device(self.mvec.device)
        self.set_device(rdtype)

    def run_iteration(self):
        self.n_local=0
        step_down = False
        while not self.stop_inversion:
            self.damping_attempts = 0
            self.test_damping = [True]*len(self.pslices)

            if step_down:
                for i, _ in enumerate(self.pslices):
                    self.max_step[i] -= self.step_down
                    
            dM   = self.calc_dm()
            dM_abs = torch.abs(dM)
            for i,pslice in enumerate(self.pslices):
                if dM_abs[pslice].max() < self.max_step[i]:
                    self.damp_increase[i] = False
                else:
                    self.damp_increase[i] = True
            while any(self.test_damping):
                if self.damping_attempts > 20:
                    self.test_damping = [False]*len(self.pslices)
                self.damping_attempts += 1
                dM   = self.calc_dm()
                dM_abs = torch.abs(dM)
                for i,pslice in enumerate(self.pslices):
                    max_dM = dM_abs[pslice].max()
                    if max_dM > self.max_step[i]:
                        if self.damp_increase[i]:
                            continue
                        else:
                            self.damp_increase[i] = True
                            dM = self.calc_dm().to(self.device)  # Recalculate only once
                            self.test_damping[i] = False
                    elif max_dM < self.max_step[i]:
                        if self.damp_increase[i]:
                            self.test_damping[i] = False
                        else:
                            if self.damp_exponent[i] == 0:
                                self.test_damping[i] = False
                            continue
                    
                   
            print('damping attempts:', self.damping_attempts)
            self.mvec      = (self.mvec+torch.nan_to_num(dM)).to(torch.float)
            self.iteration += 1
            self.n_local += 1
            # self.data_predicted
            # Check if misfit has increased or if the change is small. new_misfit is only appended if the misfit is not increasing and the change is significant.
            # n_local is used to determine if the misfit has been stable for a few iterations and will allow for an increase in the step size.
            new_misfit = self.misfit()
            misfit_increased =  new_misfit > self.misfits[-1]
            misfit_change_small = (abs(self.misfits[-1] - new_misfit) / new_misfit < self.data_misfit_minimum_change)
            # print('max step device: ', self.max_step.device)
            # print('min step device:, ', self.min_step)
            step_ratio_significant = torch.any(torch.abs(1 - self.max_step / self.min_step) >= 0.001)
            
            if (misfit_increased and not self.iteration < 2) or (misfit_change_small and step_ratio_significant):
                if step_ratio_significant:
                    step_down = True
                self.iteration -= 1
                self.n_local = 0
                self.mvec = self.mvec-dM
                self.damp_exponent[:] += 2
            elif self.n_local > 2:
                for i, _ in enumerate(self.pslices):
                    self.max_step[i] += self.step_up
                self.misfits.append(new_misfit)
                self.diag_normalizer *= 3
                self.n_local = 0
            else:
                self.misfits.append(new_misfit)

            # Check if max_step is below min_step
            for i in range(len(self.max_step)):
                if self.max_step[i] < self.min_step[i]:
                    self.max_step[i] = self.min_step[i]
            # Decrease damping before next iteration.
            for i in range(len(self.damp_exponent)):
                self.damp_exponent[i] = max(self.damp_exponent[i] - 1, 0)
            if self.misfits[-1] == new_misfit:
                self.log("{:03d}\tData misfit:\t{:2.2f}\tTotal misfit:\t{:2.2f}".format(self.iteration, torch.sqrt(self.data_misfit().sum()/len(self.data_observed)), self.misfits[-1]))
            else:
                self.log("{:03d}\tTotal misfit increased, smaller step\t".format(self.iteration))

    def invert(self, iter_cap=10, **kwargs):
        self.setup_matrices()
        self.n_local = 0
        self.iter_cap = iter_cap
        
        while not self.stop_inversion:
            self.run_iteration()

        self.log('\n'+str(self.iteration))
        mres = self.model.copy()
        mres.values = mres.modelize(self.mvec)
        return mres    

    def calc_dm(self):
        self.Cdi = torch.tensor(self.Cdi,dtype=torch.float32)
        r_data = self.G.T @ self.Cdi @ (self.data_observed-self.data_predicted)
        r_data += self.Rp.T @ self.C_rp @ ( -1.*self.Rp @ torch.tensor(self.mvec,dtype=torch.float32) ) 
        Aown = self.GtG + self.damp_matrix + self.HtH
        r_data  = self.G.T @ self.Cdi @ (self.data_observed-self.data_predicted)
        r_data += self.Rp.T @ self.C_rp @ ( -1.*self.Rp @ self.mvec ) 
        return solve(Aown, r_data)
    

class LCI(Inversion):
    log = print
    horizontal_smoothness = 0.3
    vertical_smoothness = 0.3
    diag_normalizer = 5e3
    damping_factor = 3.0
    smoothness_vert = 0.1
    smoothness_lat  = 0.1
    max_step = 3.
    diag_normalizer = 5e3
    damping_factor = 3.

    def __init__(self, inv_list, x=None, y=None):
        self.inv_list = inv_list
        self.data_observed = torch.concat([inv.data_observed for inv in self.inv_list])
        self.data_error    = torch.concat([inv.data_error    for inv in self.inv_list])
        self.recip_error_squared = 1. / self.data_error**2
        self.iter_cap = 50
        self.misfits = [np.inf]
        self.NDamp = 6  # Initial damping exponent
        self.x = x
        self.y = y
        self.n_model_updates=0

    def horiz_misfit(self):
        # Calculate the horizontal constraints misfit.
        r_mod = - self.RL.to(torch.float) @ self.mvec.to(torch.float)
        r = r_mod.permute(*torch.arange(r_mod.ndim - 1, -1, -1)) @ self.C_rl @ r_mod
        return r

    def misfit(self):
        return ((self.r_data.T @ self.r_data) / self.misfit_norm)**0.5
        
    #    return torch.sqrt((self.data_misfit().mean()))
    #     # return torch.sqrt((self.data_misfit().sum()+self.vert_misfit()+self.horiz_misfit())/(len(self.data_observed)+len(self.C_rp)+len(self.C_rl)))
    
    


    @property
    def m(self):
        return torch.concat([inv.m for inv in self.inv_list])

    def mforward(self, mvec):
        mvecs = mvec.reshape(len(self.inv_list), -1)
        datas = [inv.mforward(mv) for inv, mv in zip(self.inv_list, mvecs)]
        return torch.concat(datas)

    def jacobian(self, mvec):
        mvecs = mvec.reshape(len(self.inv_list), -1)
        jacs = [inv.jacobian(mv) for inv, mv in zip(self.inv_list, mvecs)]
        return torch.block_diag(*jacs)

    def make_smoothness_matrix(self):
        n = len(self.mvec)
        m_per_inv = len(self.inv_list[0].m)
        RL = torch.eye(n) - torch.roll(torch.eye(n), m_per_inv, dims=1)
        RL = RL[:-m_per_inv]
        return RL.to(torch.float)

    def setup_matrices(self):
        self.mvec          = self.m.to(torch.float)
        self.device=self.mvec.device
        self.Cdi           = torch.diag(self.recip_error_squared)
        self.Imod          = torch.eye(len(self.mvec))
        self._G            = None
        self._r_data       = None
        self.iteration     = 0

        # # Damping parameters
        # self.damp_exponent = torch.ones(len(self.inv_list[0].pslices))*self.damp_exponent_start
        # if hasattr(self.inv_list[0].model, 'max_step'):
        #     self.max_step = self.inv_list[0].model.max_step
        # else:
        #     self.max_step = torch.ones(len(self.inv_list[0].pslices), device=self.device)*self.max_step_init
        # if hasattr(self.inv_list[0].model, 'min_step'):
        #     self.min_step = self.inv_list[0].model.min_step
        # else:
        #     self.min_step = torch.ones(len(self.inv_list[0].pslices), device=self.device)*self.min_step
        # self.damp_increase = torch.ones(len(self.inv_list[0].pslices))
        
        # Vertical constraints
        self.Rp = torch.block_diag(*[inv.model.H() for inv in self.inv_list]).to(torch.float) # make it similar for C_rp
        self.C_rp = torch.eye(len(self.Rp))/(torch.log1p(torch.tensor(self.vertical_smoothness, dtype=torch.float32)) ** 2)
        self.HtH = self.Rp.T @ self.C_rp @ self.Rp

        # Horizontal constraints
        self.RL = self.make_smoothness_matrix()#.to(torch.float)
        self.C_rl = torch.eye(len(self.RL), dtype=self.RL.dtype)/(torch.log1p(torch.tensor(self.horizontal_smoothness, dtype=torch.float32)) ** 2)
        self.LtL = self.RL.T @ self.C_rl @ self.RL

        # since these don't change, they can be added.
        self.CtC = (self.HtH + self.LtL)
        self.misfit_norm = (len(self.data_observed)+len(self.C_rp)+len(self.C_rl))
        for inv in self.inv_list:
            inv.setup_matrices()

        self._model_update(self.mvec)


        # Create pslices for damping
        all_pslices = [[] for _ in self.inv_list[0].model.pslice]
        for inv in self.inv_list:
            for i, pslice in enumerate(inv.model.pslice):
                pslice_bool = torch.zeros_like(inv.mvec, dtype=torch.bool)
                pslice_bool[pslice] = True
                all_pslices[i].append(pslice_bool)
        self.pslices = [torch.concat(slices).to(self.device) for slices in all_pslices]

        self.misfits = [np.inf]
        self.misfits.append(self.misfit())
        

    def _model_update(self, mvec):
        self.n_model_updates += 1
        super()._model_update(mvec)
        self._GtG += self.CtC
        self._r_data -= self.CtC @ self.mvec

    def calc_dm(self):
        r_data = self.r_data
        Aown = self.GtG + self.damp_matrix
        return solve(Aown, r_data).to(torch.float)

    def invert(self, iter_cap=10, **kwargs):
        self.setup_matrices()
        self.iter_cap = iter_cap
        while not self.stop_inversion:
            self.run_iteration()
        # Split mvec back to each inversion
        mres_list = []
        mvecs = self.mvec.reshape(len(self.inv_list), -1)
        for inv, mvec in zip(self.inv_list, mvecs):
            mres = inv.model.copy()
            mres.values = mres.modelize(mvec)[:inv.model.n_ptypes]
            inv.model = mres
            mres_list.append(mres)
        return mres_list