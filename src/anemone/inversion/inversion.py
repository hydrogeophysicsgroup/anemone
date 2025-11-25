from dolphinv import *
import torch
from anemone import constants as c, rdtype, cdtype
import scipy as sp


class Resistivity(BaseModel):
    def __init__(self, values, **kwargs):
        res_bnds = kwargs.pop('res_bnds', (1e-1, 2e4))
        thk_bnds = kwargs.pop('thk_bnds', (1, 100))
        
        if not isinstance(values, list):
            values = [values]
        upper_bounds = [torch.ones_like(values[0])*res_bnds[1]]
        lower_bounds = [torch.ones_like(values[0])*res_bnds[0]]
        names = ['resistivity']
        if len(values) == 2:
            upper_bounds.append(torch.ones_like(values[1])*thk_bnds[1])
            lower_bounds.append(torch.ones_like(values[1])*thk_bnds[0])
            names.append('thickness')
        kwargs['names'] = names
        kwargs.setdefault('lower_bounds', lower_bounds)
        kwargs.setdefault('upper_bounds', upper_bounds)
        super().__init__(values, **kwargs)


class MPA(BaseModel):
    def __init__(self, values, **kwargs):
        res_bnds = kwargs.pop('res_bnds', (1,    1e3))
        phi_bnds = kwargs.pop('phi_bnds', (1e-5, 1e-1))
        tau_bnds = kwargs.pop('tau_bnds', (1e-3, 1e2))
        c_bnds   = kwargs.pop('c_bnds',   (1e-3, 1e4))
        thk_bnds = kwargs.pop('thk_bnds', (1,    100))

        upper_bounds = [torch.ones_like(val)*bnds[1] for val,bnds in zip(values, [res_bnds, phi_bnds, tau_bnds, c_bnds])]
        lower_bounds = [torch.ones_like(val)*bnds[0] for val,bnds in zip(values, [res_bnds, phi_bnds, tau_bnds, c_bnds])]
        if not isinstance(values, list):
            values = [values]
        names = ['resistivity', 'phi', 'tau', 'c']
        if len(values) == 5:
            upper_bounds.append(torch.ones_like(values[-1])*thk_bnds[1])
            lower_bounds.append(torch.ones_like(values[-1])*thk_bnds[0])
            names.append('thickness')
        kwargs['names'] = names
        kwargs.setdefault('lower_bounds', lower_bounds)
        kwargs.setdefault('upper_bounds', upper_bounds)
        super().__init__(values, **kwargs)

class ColeCole(BaseModel):
    def __init__(self, values, **kwargs):
        res_bnds = kwargs.pop('res_bnds', (1,    1e3))
        m_bnds   = kwargs.pop('m_bnds',   (1e-5, 1e-1))
        tau_bnds = kwargs.pop('tau_bnds', (1e-3, 1e2))
        c_bnds   = kwargs.pop('c_bnds',   (1e-3, 1e4))
        thk_bnds = kwargs.pop('thk_bnds', (1,    100))

        upper_bounds = [torch.ones_like(val)*bnds[1] for val,bnds in zip(values, [res_bnds, m_bnds, tau_bnds, c_bnds])]
        lower_bounds = [torch.ones_like(val)*bnds[0] for val,bnds in zip(values, [res_bnds, m_bnds, tau_bnds, c_bnds])]
        if not isinstance(values, list):
            values = [values]
        names = ['resistivity', 'm', 'tau', 'c']
        if len(values) == 5:
            upper_bounds.append(torch.ones_like(values[-1])*thk_bnds[1])
            lower_bounds.append(torch.ones_like(values[-1])*thk_bnds[0])
            names.append('thickness')
        kwargs['names'] = names
        kwargs.setdefault('lower_bounds', lower_bounds)
        kwargs.setdefault('upper_bounds', upper_bounds)
        super().__init__(values, **kwargs)    

LogResistivity = logify(Resistivity)
LogMPA = logify(MPA)
LogColeCole = logify(ColeCole)

def fastjacobian(cls):
    class FastJacobian(cls):
        def jacobian(self, mvec):
            rho, thick = self.model.modelize(mvec)
            G = self.forward(rho, thick, jac=True)
            G *= rho[:, None]
            G /= self.data_predicted.exp()
            return G.T
        def copy(self):
            from copy import deepcopy
            return deepcopy(self)               
    return FastJacobian

fastJac = fastjacobian(Tikhonov)


class specialLCI2(LCI):
    one_over_error = None
    n_sites = None
    _GtG = None
    _r_data = None
    target_step = [0.5]
    dampf = [1./5000.]
    increase_count = 0
    data_misfits = [np.inf]
    actual_step = []

    def mforward(self, mvec):
        if self.n_sites is None: self.n_sites = len(self.inv_list)
        mvecs = mvec.reshape(self.n_sites, -1).T
        rhos = mvecs.exp()
        self._dpredgrid = self.inv_list[0].forward(rhos, self.inv_list[0].model.thickness).T
        # print('shape of dpred grid: ', self._dpredgrid.shape)
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
        batch=50
        i=0
        while (i+2)*batch<self.n_sites:
            Ggrid[i*batch:(i+1)*batch] = self.inv_list[0].forward(rhos[:, i*batch:(i+1)*batch], thick, jac=True)
            i += 1
        Ggrid[i*batch:] = self.inv_list[0].forward(rhos[:, i*batch:], thick, jac=True)


        Ggrid /= self._dpredgrid.exp()[:, None, :]
        Ggrid *= rhos.T[:,:,None]
        r_data = torch.einsum('ijk,ijk->ij', Ggrid, ((self._dobsgrid - self._dpredgrid)/self._derrgrid)[:, None, :])
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
        cdamp = max(cdamp, 1.)
        if self.iteration>2:
            Aown = self._GtG*self.data_misfits[-1] + self.CtC*self.misfits[-1]
        else:
            Aown = self._GtG + self.CtC
        cImod = self.Imod.copy()
        # cImod.data[:] = 1.
        # cImod.data[self._GtG.data<1e-6] = 0.
        # dm = sp.sparse.linalg.spsolve(Aown+cImod*cdamp, self._r_data)[0]
        dm = sp.sparse.linalg.cg(Aown+cImod*cdamp, self._r_data, maxiter=500)[0]
        dm_rep = np.median(np.abs(dm))
        
        attempts=1
        measure = [dm_rep]
        
        while attempts<25 and self.iteration<5:
            if dm_rep<self.target_step[-1]*0.5:
                cdamp /= (1.1)**attempts
            elif dm_rep>self.target_step[-1]*2:
                cdamp *= (1.1)**attempts
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

    @property
    def damp_matrix(self):
        return self.dampf[-1] *self.GtG.diagonal().max() * self.Imod

    
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
            self.target_step[-1] *= 0.7
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
        if self.increase_count>=2:
            return True
        if self.target_step[-1]<0.01:
            print('stop for too small target step')
            return True
        return False