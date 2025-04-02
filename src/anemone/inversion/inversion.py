import torch
from torch.linalg import solve, inv
from scipy.linalg import block_diag
import numpy as np

class EarthModel:
    def __init__(self, property_values, property_names, device=None, **kwargs):
        self.values       = property_values
        self.names        = property_names
        self.lower_bounds = kwargs.get('lower_bounds', [-1.*torch.ones_like(val)*torch.inf for val in self.values])
        self.upper_bounds = kwargs.get('upper_bounds', [ torch.ones_like(val)*torch.inf for val in self.values])
        self.dim          = len(self.values[0].shape)
        self.pslice       = []
        self.pshape       = []

        counter=0
        for val in self.values:
            self.pslice.append(slice(counter, counter+len(val.flatten())))
            self.pshape.append(val.shape)
            counter += len(val.flatten())
        self.lbnds = self.vectorize(self.lower_bounds)
        self.ubnds = self.vectorize(self.upper_bounds)
        if device is not None:
            for var_name in vars(self):
                var = getattr(self, var_name)
                if torch.is_tensor(var):
                    setattr(self, var_name, var.to(device))
                elif isinstance(var, list):
                    for i, v in enumerate(var):
                        if torch.is_tensor(v):
                            var[i] = v.to(device)
        self.n_params = len(self.vec)

    @property
    def mod(self):
        """ the reason for return modelize of the vector representation is to accomodate models with fixed parameters
            the many layer smooth models for example have fixed thicknesses, so modelize function will return the thickness as well.
            if self.values is used, the thickness will not be included.
        """
        return self.modelize(self.vec)
    
    @property
    def vec(self):
        return self.vectorize(self.values)
    
    @staticmethod
    def vectorize(list_of_arrays):
        return torch.concatenate([v for v in list_of_arrays])

    def modelize(self, mvec):
        assert len(mvec) == self.n_params, 'incorrect number of parameters in update vector'
        i_m = 0
        values = []
        for pslice,pshape in zip(self.pslice, self.pshape):
            values.append(mvec[pslice].reshape(pshape+(-1,)).squeeze())
        return values

    def make_physical(self, mvec):
        new_mvec = np.clip(mvec, self.lbnds, self.ubnds)
        return new_mvec

    def H(self, **kwargs):
        if    self.dim==1: return block_diag(*[self.H1d(val) for val in self.values])
        elif  self.dim==2: return block_diag(*[self.H2d(mat) for mat in self.values])
        else: raise NotImplementedError()

    def H1d(self, values):
        H = (np.eye(len(values))-np.eye(len(values), k=-1))[1:]
        # H = (np.eye(len(values))-2*np.eye(len(values), k=1)+np.eye(len(values), k=2))[2:-2]
        return H

    def H2d(self, matrix):
        Hv = block_diag(*[self.H1d(matrix[:, i]) for i in range(matrix.shape[1])])
        Hh = np.eye(len(matrix.flatten())) - np.eye(len(matrix.flatten()), k=len(matrix[0]))
        norm = Hh.sum(1)
        Hh = Hh[norm==0]
        return np.vstack((Hv, 0.1*Hh))
    
    def update_model(self, m):
        self.model.update_values(m)
        self.model.check_physical()
        return self

    def update_values(self, vec):
        assert len(vec) == len(self.vec), 'incorrect number of parameters in update vector'
        i_m = 0
        for i_p in range(len(self.values)):
            shp = self.values[i_p].shape
            self.values[i_p] = vec[i_m:i_m+len(self.values[i_p].flatten())].reshape(shp)
            i_m += len(self.values[i_p].flatten())
        return self

    def check_physical(self):
        for i in range(len(self.values)):
            self.values[i] = np.clip(self.values[i], self.lower_bounds[i], self.upper_bounds[i])*1.
        return self
    
    def copy(self):
        from copy import deepcopy
        return deepcopy(self)
    
    def plot(self, axs=None, **kwargs):
        if 'thickness' in self.names:
            thk = self.values[self.names.index('thickness')]
        else:
            thk = self.thick
        if axs is None:
            from matplotlib.pyplot import subplots
            fig,axs = subplots(1,4, sharey=True)
        for v,ax,pname in zip(self.values, axs, self.names):
            z1 = np.cumsum(thk.numpy())
            z2 = np.zeros((len(z1)+1,2))
            z2[1:,0]=z1
            z2[:-1,1]=z1
            z2[-1,1] = z1.max()*1.5
            v2 = np.repeat(v.numpy(), 2).T
            
            ax.plot(v2.flatten(), z2.flatten(), **kwargs)
            ax.set_title(pname)
        axs[0].set_ylim(z2.max(), z2.min())
        axs[0].set_ylabel('depth [m]')
        return axs    


class FewLayerModel(EarthModel):
    def __init__(self, res, thick, device=None):
        lbnds = [torch.ones_like(res)*5e-5, torch.ones_like(thick)*0.1]
        ubnds = [torch.ones_like(res)*1e1,  torch.ones_like(thick)*5e2]
        super().__init__([res, thick], ['resistivity', 'thick'], lower_bounds=lbnds, upper_bounds=ubnds, device=device)
        self.n_layers = len(res)
        if device is not None:
            for attr_name in self.__dict__:
                attr_value = getattr(self, attr_name)
                if torch.is_tensor(attr_value):
                    setattr(self, attr_name, attr_value.to(device))
        self.device=device

    @classmethod
    def starting_model(cls, n_layers, rho=40., **kwargs):
        res = rho*torch.ones(n_layers)
        thick = torch.logspace(0, 0.75, n_layers-1)*5        
        return cls(res, thick, **kwargs)
    
    @staticmethod
    def vectorize(list_of_arrays):
        return torch.log10(torch.concatenate(list_of_arrays))

    def modelize(self, mvec):
        return 10**mvec[:self.n_layers], 10**mvec[self.n_layers:]
    
    
    def make_physical(self,mvec):
        new_mvec = torch.clip(mvec, self.lbnds, self.ubnds)
        return new_mvec
    
    def plot(self, ax=None, **kwargs):
        if ax is None:
            from matplotlib.pyplot import subplots
            fig,ax = subplots()
        z1 = np.cumsum(self.values[1].numpy())
        z2 = np.zeros((len(z1)+1,2))
        z2[1:,0]=z1
        z2[:-1,1]=z1
        z2[-1,1] = z1.max()*1.5
        v = np.repeat(self.values[0].numpy(), 2).T
        
        ax.plot(v.flatten(), z2.flatten(), **kwargs)
        return ax


class SmoothModel(EarthModel):
    def __init__(self, res, thick, device=None):
        lbnds = [torch.ones_like(res)*5e-5]
        ubnds = [torch.ones_like(res)*1e1]
        super().__init__([res], ['resistivity'], lower_bounds=lbnds, upper_bounds=ubnds, device=device)
        self.n_layers = len(res)
        self.thick = thick
        if device is not None:
            for attr_name in self.__dict__:
                attr_value = getattr(self, attr_name)
                if torch.is_tensor(attr_value):
                    setattr(self, attr_name, attr_value.to(device))
        self.device=device

    
    @classmethod
    def starting_model(cls, n_layers, depth_bottom=150, rho=40, **kwargs):
        res = rho*torch.ones(n_layers)
        thick = torch.logspace(0, 1, n_layers-1)
        thick *= depth_bottom/thick.sum()
        return cls(res, thick, **kwargs)
    
    @staticmethod
    def vectorize(list_of_arrays):
        return torch.log10(list_of_arrays[0])

    def modelize(self, mvec):
        return 10**mvec, self.thick
    
    def make_physical(self,mvec):
        new_mvec = torch.clip(mvec, self.lbnds, self.ubnds)
        return new_mvec

    def plot(self, ax=None, **kwargs):
        if ax is None:
            from matplotlib.pyplot import subplots
            fig,ax = subplots()
        v,thk = self.modelize(self.vec)
        z1 = np.cumsum(thk.numpy())
        z2 = np.zeros((len(z1)+1,2))
        z2[1:,0]=z1
        z2[:-1,1]=z1
        z2[-1,1] = z1.max()*1.5
        v = np.repeat(self.values[0].numpy(), 2).T
        
        ax.plot(v.flatten(), z2.flatten(), **kwargs)
        return ax
    
    
class SmoothModelColeCole(EarthModel):
    def __init__(self, rho, tau, m, C, thick, device=None):
        lbnds = [torch.ones_like(rho)*1, torch.ones_like(tau)*1e-3, torch.ones_like(m)*1e-3, torch.ones_like(C)*1e-3]
        ubnds = [torch.ones_like(rho)*2e3, torch.ones_like(tau)*1e3, torch.ones_like(m)*1e3, torch.ones_like(C)*1e3]
        super().__init__([rho, tau, m, C], ['resistivity', 'tau', 'chargeability', 'c'], lower_bounds=lbnds, upper_bounds=ubnds, device=device)
        self.n_layers = len(rho)
        self.thick = thick
        if device is not None:
            for attr_name in self.__dict__:
                attr_value = getattr(self, attr_name)
                if torch.is_tensor(attr_value):
                    setattr(self, attr_name, attr_value.to(device))
        self.device=device

    
    @classmethod
    def starting_model(cls, n_layers, depth_bottom=150, rho=40, tau=3e-4, m=0.2, C=0.6, **kwargs):
        res = rho*torch.ones(n_layers)
        tau = tau*torch.ones(n_layers)
        m   = m*torch.ones(n_layers)
        C   = C*torch.ones(n_layers)

        thick = torch.logspace(-1, 1, n_layers-1)
        thick *= depth_bottom/thick.sum()

        return cls(res, tau, m, C, thick, **kwargs)
    
    
    @staticmethod
    def vectorize(list_of_arrays):
        return torch.log10(torch.concat(list_of_arrays))

    def modelize(self, mvec):
        rho = 10**mvec[:self.n_layers]
        tau = 10**mvec[self.n_layers:2*self.n_layers]
        m   = 10**mvec[2*self.n_layers:3*self.n_layers]
        c   = 10**mvec[3*self.n_layers:]
        return rho, tau, m, c, self.thick
    
    def make_physical(self,mvec):
        new_mvec = torch.clip(mvec, self.lbnds, self.ubnds)
        return new_mvec

    def plot(self, axs=None, **kwargs):
        thk = self.thick
        if axs is None:
            from matplotlib.pyplot import subplots
            fig,axs = subplots(1,4, sharey=True)
        for v,ax,pname in zip(self.values, axs, self.names):
            z1 = np.cumsum(thk.numpy())
            z2 = np.zeros((len(z1)+1,2))
            z2[1:,0]=z1
            z2[:-1,1]=z1
            z2[-1,1] = z1.max()*1.5
            v2 = np.repeat(v.numpy(), 2).T
            
            ax.plot(v2.flatten(), z2.flatten(), **kwargs)
            ax.set_title(pname)
        axs[0].set_ylim(z2.max(), z2.min())
        axs[0].set_ylabel('depth [m]')
        return axs
    
class FewLayerColeCole(EarthModel):
    def __init__(self, rho, tau, m, C, thick, device=None):
        if not isinstance(rho, torch.Tensor):   rho   = torch.tensor(rho)
        if not isinstance(tau, torch.Tensor):   tau   = torch.tensor(tau)
        if not isinstance(m, torch.Tensor):     m     = torch.tensor(m)
        if not isinstance(C, torch.Tensor):     C     = torch.tensor(C)
        if not isinstance(thick, torch.Tensor): thick = torch.tensor(thick)

        lbnds = [torch.ones_like(rho)*1, torch.ones_like(tau)*1e-3, torch.ones_like(m)*1e-3, torch.ones_like(C)*1e-3, torch.ones_like(thick)]
        ubnds = [torch.ones_like(rho)*2e3, torch.ones_like(tau)*1e3, torch.ones_like(m)*1e3, torch.ones_like(C)*1e3, torch.ones_like(thick)*1e2]
        super().__init__([rho, tau, m, C, thick], ['resistivity', 'tau', 'chargeability', 'c', 'thickness'], lower_bounds=lbnds, upper_bounds=ubnds, device=device)
        self.n_layers = len(rho)
        if device is not None:
            for attr_name in self.__dict__:
                attr_value = getattr(self, attr_name)
                if torch.is_tensor(attr_value):
                    setattr(self, attr_name, attr_value.to(device))
        self.device=device

    @classmethod
    def starting_model(cls, n_layers, depth_bottom=150, rho=40, tau=3e-4, m=0.2, C=0.6, **kwargs):
        res = rho*torch.ones(n_layers)
        tau = tau*torch.ones(n_layers)
        m = m*torch.ones(n_layers)
        C = C*torch.ones(n_layers)

        thick = torch.logspace(-1, 1, n_layers-1)
        thick *= depth_bottom/thick.sum()

        return cls(res, tau, m, C, thick, **kwargs)
    
    
    @staticmethod
    def vectorize(list_of_arrays):
        return torch.log10(torch.concat(list_of_arrays))

    def modelize(self, mvec):
        rho   = 10**mvec[:self.n_layers]
        tau   = 10**mvec[self.n_layers:2*self.n_layers]
        m     = 10**mvec[2*self.n_layers:3*self.n_layers]
        c     = 10**mvec[3*self.n_layers:4*self.n_layers]
        thick = 10**mvec[4*self.n_layers:]
        return rho, tau, m, c, thick

class SmoothModels(EarthModel):
    def __init__(self, res, thick, device=None):
        lbnds = [torch.ones_like(res)]
        ubnds = [torch.ones_like(res)*1e3]
        super().__init__([res], ['resistivity'], lower_bounds=lbnds, upper_bounds=ubnds, device=device)
        self.n_layers,self.n_models = res.shape
        self.thick = thick
        if device is not None:
            for attr_name in self.__dict__:
                attr_value = getattr(self, attr_name)
                if torch.is_tensor(attr_value):
                    setattr(self, attr_name, attr_value.to(device))
        self.device=device

    
    @classmethod
    def starting_model(cls, n_layers, n_models, rho=40., depth_bottom=150.,  **kwargs):
        res = rho*torch.ones((n_layers, n_models))
        thick = torch.logspace(-1, 1, n_layers-1)
        thick *= depth_bottom/thick.sum()
        return cls(res, thick, **kwargs)
    
    @staticmethod
    def vectorize(list_of_arrays):
        return torch.log10(list_of_arrays[0].T.flatten())

    def modelize(self, mvec):
        return 10**mvec.reshape((-1,self.n_layers)).T, self.thick
    
    def make_physical(self,mvec):
        new_mvec = torch.clip(mvec, self.lbnds, self.ubnds)
        return new_mvec

    def plot(self, ax=None, **kwargs):
        import matplotlib.pyplot as plt
        
        if ax is None:
            from matplotlib.pyplot import subplots
            fig,ax = subplots()
        v,thk = self.modelize(self.vec)
        z1 = np.insert(np.cumsum(thk.cpu().numpy()), 0,0)
        xdist = np.arange(v.shape[1])
        pcm=ax.pcolor(xdist, z1, np.log10(v.cpu().numpy()), **kwargs)
        plt.colorbar(pcm)
        ax.set_ylim(z1.max(), z1.min())
        return ax

def passthrough(d):
    return d

class Inversion:
    jac_perturbation_fraction        = 0.02
    data_misfit_acceptance_threshold = 0.5
    data_misfit_minimum_change       = 0.01
    diag_normalizer = 5.e3
    max_step = 0.5
    damping_factor=3.
    max_parallel_models=200

    def __init__(self, data_observed, data_error, starting_model,  forward_operator, logger=print, data_transform=passthrough, device=None):
        """Create an inversion object, the necessary """
        self.data_transform   = data_transform
        self.data_observed    = data_transform(data_observed)
        self.data_error       = torch.log10(1.+data_error/data_observed)
        self.model            = starting_model
        self.forward = forward_operator
        self.result        = None
        self.nM = len(self.m)
        self.nD = len(data_observed)
        self.recip_error_squared = 1./self.data_error**2
        self.log=logger
        self.device = self.forward.device

        for attr_name in self.__dict__:
            attr_value = getattr(self, attr_name)
            if torch.is_tensor(attr_value):
                setattr(self, attr_name, attr_value.to(self.device).to(torch.double))

    @property
    def m(self):
        return self.model.vec
    
    @property
    def data_predicted(self):
        return self.mforward(self.m)

    def misfit(self, *args):
        if len(args)==1: data_predicted = args[0]
        else: data_predicted = self.data_predicted
        misfit = (self.data_observed-data_predicted)**2 * self.recip_error_squared
        return torch.sqrt(torch.mean(misfit))
    
    def jacobian(self, mvec):
        dm   = mvec*self.jac_perturbation_fraction
        dm[dm==0] = self.jac_perturbation_fraction
        
        m_pert    = torch.diag(dm) + mvec[:, None]
        c_data    = self.mforward(mvec)
        d_data    = self.mforward(m_pert)#.T)
        Gt = (d_data - c_data) / dm[:, None]
        return Gt.T

    def Cm(self, mvec):
        G = self.jacobian(mvec)
        return inv( G.T @ np.diag(self.recip_error_squared) @ G )

    def mforward(self, mvec):
        return self.data_transform(self.forward(*self.model.modelize(mvec)))

    @property
    def stop_inversion(self):
        if self.iteration<=1:
            return False
        if (abs(self.misfits[-2]-self.misfits[-1]) / self.misfits[-1]) < self.data_misfit_minimum_change and self.misfits[-1]<1:
            self.log(f'misfit change is less than {self.data_misfit_minimum_change}')
            return True
        if self.misfits[-1]<self.data_misfit_acceptance_threshold:
            return True
        if self.misfits[-1]>self.misfits[-2] and self.misfits[-3]==min(self.misfits):
            return True
        if self.iteration>=self.iter_cap:
            self.log('reached iteration cap..')
            return True
        return False
    
    def C_est(self, mvec):
        from torch.linalg import inv
        G = self.jacobian(mvec)
        C = inv(G.T @ torch.diag(self.recip_error_squared) @ G)
        return C
    
    def sensitivity_factor(self, mvec):
        C = self.C_est(mvec)
        diag = torch.sqrt(torch.diag(C))
        return torch.exp(diag)

    def plot_sensitivity_analysis(self, ax=None, x_offset=0, cbar=False):
        stdf, model = self.sensitivity_factor(self.model.vec), self.model
        import matplotlib.colors as colors
        cmap = colors.ListedColormap(['red', 'darkorange', 'gold', 'deepskyblue', 'blue'])
        norm = colors.BoundaryNorm(np.array([1., 1.2, 1.5, 2., 3., 4.]), cmap.N)    
        if ax is None:
            fig,ax = plt.subplots()

        dy = np.arange(len(model.vec)+1)
        tick_pos = (dy[:-1]+dy[1:])/2
        tick_lab = [f'$\\rho_{i+1}$' for i in range(len(model.values[0]))]
        tick_lab +=[f'thk$_{i+1}$' for i in range(len(model.values[1]))]
        dx = np.outer(np.ones_like(dy), np.array([-0.5, 0.5]))+x_offset
        dy = np.outer(dy, np.ones(2))
        val = stdf[:, None].cpu().numpy()
        val[np.isnan(val)] = 5.
        val[val>4] = 5
        pcm = ax.pcolormesh(dx, dy, val, cmap=cmap, norm=norm, edgecolor='k', linewidth=0.5)
        ax.set_yticks(tick_pos)
        ax.set_yticklabels(tick_lab)
        if cbar:
            import matplotlib.pyplot as plt
            plt.colorbar(pcm, ax=ax, label='sensitivity factor')
        return ax
    

    def invert(self, smoothness=0., iter_cap=10, **kwargs):
        from torch.linalg import inv, solve
        mvec = self.m*1.


        self.iteration   = 0
        self.iter_cap    = iter_cap
        self.misfits     = [np.inf]
        
        NDamp = 1

        data_predicted = self.mforward(mvec)
        self.misfits.append(self.misfit(data_predicted))
        Cdi = torch.diag(self.recip_error_squared)

        H = torch.from_numpy(self.model.H()).to(mvec.device)
        HtH = (H.T @ H)*smoothness
        dM = torch.zeros_like(mvec)

        G = self.jacobian(mvec)
        GtG = G.T @ Cdi @ G
        max_diag = torch.max(torch.diag(GtG))/self.diag_normalizer

        while not self.stop_inversion:
            if self.misfits[-1]<self.misfits[-2]:
                G = self.jacobian(mvec)
                GtG = G.T @ Cdi @ G
                max_diag = torch.max(torch.diag(GtG))/self.diag_normalizer
                
            else:
                mvec -= dM*0.9
                NDamp += 2

            damping_attempts = 0
            test_damping = True
                

            while test_damping:
                if NDamp>0:
                    damp = max_diag*self.damping_factor**(NDamp-1)
                else:
                    damp = 0.
                r_data = G.T @ Cdi @ (self.data_observed-data_predicted)
                Aown = GtG + torch.eye(len(mvec), device=mvec.device)*damp + HtH
                dM   = solve(Aown, r_data)

                dM_abs = torch.abs(dM)
                
                if torch.any(dM_abs>self.max_step):
                    NDamp += 1
                    damping_attempts+=1
                elif damping_attempts>20:
                    test_damping = False
                else:
                    test_damping = False
                
                
            
            if damping_attempts==0 and self.iteration>1:
                # self.log('increased max step')
                self.max_step += 0.5
            elif damping_attempts==20:
                # self.log('decreased max step')
                self.max_step -= 0.1
        
            mvec                += torch.nan_to_num(dM)
            self.iteration      += 1
            data_predicted     = self.mforward(mvec)
            self.misfits.append(self.misfit(data_predicted))
            NDamp += -1
            self.log("{:03d}\tmisfit:\t{:2.2f}".format(self.iteration, self.misfits[-1]))
                                
        if self.iteration==iter_cap: self.log('reached iteration cap..')
        print('\n',self.iteration)
        mres = self.model.copy()
        mres.values = mres.modelize(mvec)
        return mres

    
class LCI(Inversion):
    def jacobian(self, mvec):
        n_mods = self.model.n_models
        n_layers = self.model.n_layers
        c_data = self.mforward(mvec)
        n_data = int(len(c_data)/n_mods)
        Gt = torch.zeros(mvec.shape+c_data.shape, device=c_data.device, dtype=torch.double)

        dm   = mvec*self.jac_perturbation_fraction
        dm[dm==0] = self.jac_perturbation_fraction
        
        dm_mat = torch.zeros(mvec.shape+(n_layers,), device=c_data.device)
        for i in range(n_mods):
            dm_mat[i*n_layers:(i+1)*n_layers] = torch.diag(dm[i*n_layers:(i+1)*n_layers])
        
        m_pert    = (dm_mat + mvec[:, None]).T

        d_data    = self.mforward(m_pert)
        # else:
        #     d_data  = torch.hstack([self.mforward(m_pert[:, i*n_layers:(i+1)*n_layers]) for i in range(n_mods)])
        # print('shape of d_data: ', d_data.shape)

        

        for i in range(n_mods):
            Gt[i*n_layers:(i+1)*n_layers, i*n_data:(i+1)*n_data] = d_data[:, i*n_data:(i+1)*n_data]-c_data[i*n_data:(i+1)*n_data]
        Gt /= dm[:, None]

        return Gt.T    