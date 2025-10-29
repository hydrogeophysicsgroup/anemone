# %%
from anemone.forwards import *
import matplotlib.pyplot as plt
import numpy as np
import torch
import scipy as sp

# %%
from physics import rte_raiche1999 as rte
from physics import rte_halfspace, drte_dsigma

# %%
def make_H2t(t2t, geomz, omega, lmb, tolerance=1e-6, sigma_ref=0.01):
    # if t_indices is None:
    t_indices = np.arange(t2t.shape[0])

    t1 = (t2t[0, :, None]*geomz.real[None, :]).numpy()
    full_mask = np.zeros_like(t1, dtype=bool)

    rte_ref = rte_halfspace(torch.tensor(sigma_ref), omega[:,None], lmb[None,:]).squeeze().cpu().numpy()

    for i in t_indices:
        t1 = (t2t[i, :, None]*geomz.real[None, :]).numpy()*rte_ref
        t1_shape = t1.shape
        t1sort = np.abs(t1).flatten()
        # indices = np.arange(len(t1sort))

        sort_idx = np.argsort(t1sort)
        t1sort = t1sort[sort_idx]
        t1cumsum = np.cumsum(t1sort)
        t1cumsum /= t1cumsum[-1]
        idx = np.searchsorted(t1cumsum, tolerance)#-1

        mask = np.zeros_like(t1sort, dtype=bool)
        mask[idx:] = True


        inv_sort = np.argsort(sort_idx)
        mask = mask[inv_sort].reshape(t1_shape)
        full_mask += mask
    
    omega_i = np.arange(omega.squeeze().shape[0])
    lamda_i = np.arange(lmb.squeeze().shape[0])
    omega_i, lamda_i = np.meshgrid(omega_i, lamda_i, indexing='ij')

    omega_i = omega_i[full_mask]
    lamda_i = lamda_i[full_mask]
    # print(omega_i.min(), omega_i.max(), lamda_i.min(), lamda_i.max())
    print('n omegas: ', omega_i.max()-omega_i.min())
    print('n lambdas: ', lamda_i.max()-lamda_i.min())

    # full_mask[:,:] = False
    # full_mask[int(omega_i.min()):int(omega_i.max()), int(lamda_i.min()):int(lamda_i.max())] = True

    H2t = np.zeros(t2t.shape[:1]+(full_mask.sum(),), dtype=complex)

    for i in range(t2t.shape[0]):
        t1 = (t2t[i, :, None]*geomz.real[None, :]).numpy()
        H2t[i] = t1[full_mask].flatten()

    H2tweight = np.abs(H2t).sum(0)#.numpy()

    idx_H2t = np.argsort(H2tweight)[::-1]
    H2t = H2t[:, idx_H2t]

    Omega,Lambda = np.meshgrid(omega.squeeze().cpu().numpy(), lmb.squeeze().cpu().numpy(), indexing='ij')
    Omega = Omega[full_mask]
    Lambda = Lambda[full_mask]
    return torch.tensor(H2t), torch.tensor(Omega[idx_H2t])+0j, torch.tensor(Lambda[idx_H2t])+0j

class ForwardFlow(Forward):
    geomz = None
    t2t   = None

    def __init__(self, source, receiver, times, waveform=None, filterfunc=None, t_gatewidth=None, tolerance=1e-7):
        device = times.device

        super().__init__(source, receiver, times.cpu(), waveform, filterfunc, t_gatewidth, tolerance, precalc=False)
        self.t2t = None
        self.setup()
        self.device=device
        self.H2t = self.H2t.to(self.device).to(torch.complex64)
        self.omega = self.omega.to(self.device).to(torch.complex64)
        self.lmb = self.lmb.to(self.device).to(torch.complex64)
        self.H2tjac = self.H2t.to(self.device).to(torch.complex64)


    def setup(self):
        self.rho   = 40*torch.ones(2)[:, None, None] + 0j
        self.thick = 20*torch.ones(1)[:, None, None] + 0j

        def frq_integrand(f):
            self.omega = c.twopi*f[None,:,None]
            def wvlngth_integrand(wave_length):
                """ Hz from horizontal electric dipole. """
                self.lmb = wave_length[None, None, :]          
                #self.tmp_u0 = u0(self.omega, self.lmb)
                self.sfield_rte = torch.exp(-self.lmb*self.sfield_dz)*self.lmb / c.fourpi
                intg = rte(self.sigma, self.thick, self.omega, self.lmb)*self.sfield_rte
                return intg.squeeze()
            hkl1 = Hankel(wvlngth_integrand, 1, device=self.device)
            def geomz_r(r):
                return self.source.geomz(self.receiver, r).to(torch.double).to(self.device)
            Hr, self.geomz, self.lmb = hkl1.preintegrate(geomz_r, self.rmin, self.rmax)
            
            self.geomz *= torch.exp(-self.lmb*self.sfield_dz)*self.lmb / c.fourpi

            return Hr.squeeze()
        
        if self.t2t is None:
            self.cbft=invFourier(frq_integrand, device=self.device)
            def t2t_func(t_in):
                if self.t_gatewidth is not None:
                    waveform_matrix = wv_to_t(t_in, t_in, waveform=self.waveform)
                    gate_matrix = gating_matrix(t_in, self.t_st, self.t_end, n_eval=1000)
                    return gate_matrix @ waveform_matrix
                else:
                    return wv_to_t(self.times, t_in, waveform=self.waveform)

            
            Ht, self.t2t, self.omega1 = self.cbft.preintegrate(t2t_func, self.tmin, self.tmax*100.)
            scalar = self.times**(5/2)
            self.scalar = scalar/scalar.min()
            # self.t2t *= self.scalar[:, None]

            if self.filterfunc is not None:
                self.t2t *= self.filterfunc.omega_filter(self.omega1.squeeze())
            if self.waveform is not None:
                self.t2t /= -1j#*self.omega1.squeeze()
            else:
                self.t2t *= self.omega1.squeeze()

        self.H2t, self.omega, self.lmb = make_H2t(self.t2t, self.geomz, self.omega1, self.lmb, tolerance=self.tolerance)

        self.H2t       = self.H2t.to(self.device).to(torch.complex64)
        self.H2t      /= torch.sqrt(self.omega)
        self.H2t      /= c.sqrt_twopi
        self.omega    *= c.sqrt_twopi
        self.H2tjac    = self.H2t[:, None, :].to(self.device)                                   #[:, :3*len(self.omega)//6]
        # self.omegajac  = self.omega                                  #[:3*len(self.omega)//6]
        # self.lmbjac    = self.lmb                                    #[:3*len(self.omega)//6]
    
    def forward(self):
        with torch.no_grad():
            cRTE = rte(self.sigma, self.thick, self.omega, self.lmb).squeeze()
            return torch.einsum('ik,...k->i...', self.H2t, cRTE).real

    def jacobian(self):
        dRTE = drte_dsigma(self.sigma, self.thick, self.omega, self.lmb).squeeze()
        
        if len(dRTE.shape)>2:
            Gr = torch.einsum('ijk,lk->jil', dRTE, self.H2tjac).real
            # print(Gr.shape)
            # print(self.rho.squeeze().shape)
            Gr /= (self.rho.real.squeeze().T**2)[:,:,None]
        else:
            Gr = torch.einsum('ik,...k->i...', dRTE.real, self.H2t.real) - torch.einsum('ik,jk->ij', dRTE.imag, self.H2t.imag)
            Gr /= (self.rho.real.squeeze()**2)[:,None]
        return Gr
    
    def __call__(self, rho, thickness, jac=False):
        with torch.no_grad():
            if not isinstance(rho, torch.Tensor):       rho       = torch.tensor(rho)
            if not isinstance(thickness, torch.Tensor): thickness = torch.tensor(thickness)        
            self.rho = ((rho+0j)[..., None, None]).to(torch.complex64)#.to(self.device)
            if rho.ndim > thickness.ndim:
                thickness = thickness[:, None, ...]
            self.thick = ((thickness+0j)[..., None, None]).to(torch.complex64)
            if jac:
                G = self.jacobian()
                return G*c.mu0
            else:            
                return self.forward()*c.mu0     

    def copy(self):
        from copy import deepcopy
        return deepcopy(self)   



from anemone.experiment import compute_relstd
from anemone.inversion import *

class specialLCI(LCI):
    one_over_error = None
    n_sites = None
    damp_value = 2500.

    def mforward(self, mvec):
        if self.n_sites is None: self.n_sites = len(self.inv_list)
        mvecs = mvec.reshape(self.n_sites, -1).T
        rhos = mvecs.exp()
        self._dpredgrid = self.inv_list[0].forward(rhos, self.inv_list[0].model.thickness).T
        # print('shape of dpred grid: ', self._dpredgrid.shape)
        return self._dpredgrid.flatten()
    
    def jacobian(self, mvec):
        if self.one_over_error is None:
            self.n_sites = len(self.inv_list)
            self.one_over_error = torch.sqrt(self.recip_error_squared).reshape(self.n_sites, -1)[:, None, :]
            self._dobsgrid = self.data_observed.reshape(self.n_sites, -1)
            self._derrgrid = self.data_error.reshape(self.n_sites, -1)**2
        mvecs = mvec.reshape(self.n_sites, -1).T
        rhos = mvecs.exp()
        Ggrid = self.inv_list[0].forward(rhos, self.inv_list[0].model.thickness, jac=True)

        Ggrid /= self._dpredgrid.exp()[:, None, :]
        Ggrid *= rhos.T[:,:,None]
        r_data = torch.einsum('ijk,ijk->ij', Ggrid, ((self._dobsgrid - self._dpredgrid)/self._derrgrid)[:, None, :])
        Ggrid *= self.one_over_error

        GtG = Ggrid @ Ggrid.swapaxes(-1,-2)
        self._GtG = torch.block_diag(*[GtG[i] for i in range(self.n_sites)])
        self._r_data = r_data.flatten()
        self.cmvec = self.mvec*1.
    
    def _model_update(self, mvec):
        self.n_model_updates += 1
        self.mvec = mvec
        self._current_mvec = self.mvec*1.

        self._dpred = self.mforward(self.mvec)
        self.jacobian(self.mvec)
        self._r_data -= self.CtC @ self.cmvec        
    
    # @property
    # def damp_matrix(self):
    #     damp = self.GtG.diagonal().max()/self.damp_value
    #     return damp*self.Imod

    
    # def run_iteration(self):
    #     dM   = self.calc_dm()
    #     self.damp_value *= self.damp_reduction_factor
    #     #self.mvec = self.model.make_physical(self.mvec + dM)
    #     self.mvec += dM
    #     self.iteration += 1
    #     self.misfits.append(self.misfit())
    #     self.log("{:03d}\tmisfit:\t{:2.2f}".format(self.iteration, self.misfits[-1]))

class specialLCI2(LCI):
    one_over_error = None
    n_sites = None
    _GtG = None
    _r_data = None
    target_step = [0.05]
    damp = []
    increase_count = 0
    data_misfits = []

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
        Ggrid = torch.empty((self.n_sites, len(thick)+1, self._dobsgrid.shape[-1]), dtype=torch.float32, device=self.device)
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
            self.damp.append(self._GtG.diagonal().max()/2500.)
        else:
            self._GtG.data[...] = GtG_np  # no flattening

        self._r_data = r_data.cpu().numpy().flatten()
        self.cmvec = self.mvec.cpu().numpy()*1.

    def calc_dm(self):
        cdamp = self.damp[-1]

        Aown = self._GtG + self.CtC
        cImod = self.Imod.copy()
        cImod.data[self._GtG.data==0.] = 0.
        dm = sp.sparse.linalg.cg(Aown+cImod*cdamp, self._r_data)[0]
        dm_rep = np.max(np.abs(dm))
        attempts=1
        while np.abs(dm_rep-self.target_step[-1])/dm_rep>1. and attempts<20:
            cdamp *= np.abs(dm_rep-self.target_step[-1])/dm_rep
            dm = sp.sparse.linalg.cg(Aown+cImod*cdamp, self._r_data)[0]
            dm_rep = np.abs(dm).max()
            attempts+=1
        self.damp.append(max(cdamp, 1e-5))
        print('attempts to reach target step: ', attempts)
        return torch.nan_to_num(torch.tensor(dm, device=self.mvec.device, dtype=self.mvec.dtype))

    @property
    def damp_matrix(self):
        # damp = self.GtG.diagonal().max()/self.damp_value
        return self.damp[-1] * self.Imod

    
    def run_iteration(self):
        dM   = self.calc_dm()
        # self.damp_value *= self.damp_reduction_factor
        #self.mvec = self.model.make_physical(self.mvec + dM)
        self.mvec += dM
        self.iteration += 1
        new_misfit = self.misfit()
        
        dMisfit = (new_misfit-self.misfits[-1])/self.misfits[-1]
        d_misfit = self.data_misfit().mean().item()
        self.data_misfits.append(d_misfit)
        print('dMisfit: ', dMisfit)
        if dMisfit>0:
            self.damp[-1] = self.damp[-2]*10.
            self.mvec -= dM
            self.target_step.append(self.target_step[-1]*0.5)
            self.increase_count +=1
            
        else:
            self.misfits.append(new_misfit)
            self.target_step.append(max(0.02, d_misfit/10))
            self.increase_count = 0

        self.log("{:03d}\tmisfit:\t{:2.2f}".format(self.iteration, self.misfits[-1]))
        print('data misfit: ', d_misfit)
    
    @property
    def stop_inversion(self):
        if super().stop_inversion:
            return True
        if self.increase_count>=2:
            return True
        return False    # @property
    # def damp_matrix(self):
    #     # Damping scheme following Aarhusinv. Used for both Inversion and LCI.
    #     # Creates a damping matrix to add to Aown.
    #     if self.damping_attempts==0:
    #         diag_GTG = torch.from_numpy(self.GtG.diagonal()).to(self.device).to(torch.double)
    #         self.damp = torch.empty_like(diag_GTG)
    #         max_GTG = [diag_GTG[pslice].max()/self.diag_normalizer for pslice in self.pslices]
    #         for i,pslice in enumerate(self.pslices):
    #             if self.damp_exponent[i]>0:
    #                 self.damp[pslice] =  max_GTG[i]*self.damping_factor**(self.damp_exponent[i])
    #             else:
    #                 self.damp[pslice] = 0
    #     else:
    #         for i,pslice in enumerate(self.pslices):
    #             if self.test_damping[i]:
    #                 if self.damp_increase[i]:
    #                     if self.damp_exponent[i]>0:
    #                         self.damp[pslice] =  self.damp[pslice]*self.damping_factor
    #                         self.damp_exponent[i] +=1
    #                     else:
    #                         diag_GTG = torch.from_numpy(self.GtG.diagonal()).to(self.device).to(torch.double)
    #                         self.damp_exponent[i] +=1
    #                         self.damp[pslice] =  diag_GTG[pslice].max()/self.diag_normalizer*self.damping_factor**self.damp_exponent[i]
    #                 else:
    #                     if self.damp_exponent[i]>0:
    #                         self.damp[pslice] =  self.damp[pslice]/self.damping_factor
    #                         self.damp_exponent[i] -=1
    #                     else:
    #                         self.damp[pslice] = 0
    #     self.damp = torch.clamp(self.damp, min=1.)  # Ensure damping is at least 1
    #     return self.damp.cpu().numpy()*self.Imod
        
 
gpu = torch.device('cuda')
print('number of threads: ', torch.get_num_threads())
logForward = logify_forward(ForwardFlow)
t = torch.logspace(-5.5, -3, 31, device=gpu, dtype=torch.float)
fwr = logForward.example(times=t, tolerance=1e-5)
fastJac = fastjacobian(Inversion)

z = torch.cumsum(torch.logspace(0, np.log10(50.), 20), 0)
z /= z.max()

z *= 150

rho_true = torch.tensor([75., 10., 60.], device=gpu, dtype=torch.double)
thk_true = torch.tensor([50, 30], device=gpu, dtype=torch.double)


res_start = LogResistivity.smooth(torch.ones(len(z)+1)*50., z, device=gpu)
res_start.set_device(gpu)
thick1 = torch.linspace(0, 20, 250).to(torch.float)
inv_list0 = []
inv_list1 = []
m_trues=[]

for i,dthick in enumerate(thick1):
    this_fwr = fwr.copy()
    thk = thk_true.clone()
    thk[0] += dthick
    crho = rho_true.clone()
    crho[1] += abs(i-len(thick1)//2)*2.
    crho[2] -= abs((i-5)*70/len(thick1))
    crho[2] = max(crho[2], 20.)
    m_true = LogResistivity.few_layers(torch.clamp(crho, 5), thk)
    m_trues.append(m_true)
    
    dbdt = this_fwr(crho, thk).exp()
    d_err = compute_relstd(fwr.times, dbdt.cpu(), current=25.).to(gpu)
    d_obs = torch.abs(dbdt * (1. + d_err*torch.randn_like(dbdt))).log().to(gpu)
    tmp_inv = fastJac(d_obs, (1. + d_err).log(), res_start.copy(), this_fwr)

    # tmp_inv.setup_matrices()
    inv_list0.append(tmp_inv.copy())    

    tmp_inv = fastJac(d_obs, (1. + d_err).log(), res_start.copy(), this_fwr)

    inv_list1.append(tmp_inv.copy())

print(f"Created {len(inv_list0)} inversions")
from time import time as timer
bt = timer()
lci = specialLCI2(inv_list0)
lci.set_device(gpu)
res_mods = lci.invert(iter_cap=25)
# print('lci._GtG.shape: ', lci._GtG.shape)
# print('lci.CtC.shape: ', lci.CtC.shape)

print(f'time for normal inversion: {timer()-bt:2.2f} s')

# plt.figure()
# plt.matshow(lci.CtC.cpu().numpy())
# plt.show()

# bt = timer()
# lci = specialLCI(inv_list1)
# lci.set_device(gpu)
# res_mods = lci.invert(iter_cap=10)
# print(f'time for special inversion: {timer()-bt:2.2f} s')
# # lci.smoothness_lat

# # lci.smoothness_vert = 1e3
# # lci.smoothness_horiz = 1e2
# from time import time as timer
# import cProfile
# import pstats
# profiler = cProfile.Profile()
# profiler.enable()

# res_mods = lci.invert(iter_cap=10)

# print(f'NUMBER OF ITERATIONS: {lci.iteration}')
# print(f'NUMBER OF MODEL UPDATES: {lci.n_model_updates}')

# profiler.disable()
# stats = pstats.Stats(profiler)
# stats.sort_stats('tottime').print_stats(20)  # Print top 20 by cumulative time

if __name__ == '__main__':
    import matplotlib.pyplot as plt
    from plog import Log, Section, PieceWiseLine

    fig,ax = plt.subplots(2,1)
    result = []
    actual = []
    _,thick = res_mods[0].mod
    z = torch.cumsum(thick, 0)
    for i, (mtrue, res_mod) in enumerate(zip(m_trues, res_mods)):
        result.append(Log.standard(res_mod.values[0].cpu().log10().numpy(), z.cpu().numpy(), 'rho', xy=(i,0)))
        cres,thick = mtrue.mod
        actual.append(Log.standard(cres.cpu().log10().numpy(), np.cumsum(thick.cpu().numpy()), 'rho', xy=(i,0)))
    result = Section(result, proj_pnts=[(-1, 0), (i/2, 0), (i+1, 0)])
    actual = Section(actual, proj_pnts=[(-1, 0), (i/2, 0), (i+1, 0)])
    result.plot(axs=ax[0], cmap='turbo',  vmin=0, vmax=2)
    actual.plot(axs=ax[1], cmap='turbo',  vmin=0, vmax=2)


    plt.figure()
    plt.semilogy(lci.misfits, label='misfits')
    plt.semilogy(lci.damp, label='damp_fs')
    plt.semilogy(lci.target_step, label='target_step')
    plt.semilogy(lci.data_misfits, label='data_misfits')
    plt.legend()    
    plt.show()

    # %%



