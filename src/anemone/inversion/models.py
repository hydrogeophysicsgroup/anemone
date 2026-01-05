import torch
from scipy.linalg import block_diag
import numpy as np

class BaseModel:
    def __init__(self, layer_properties, **kwargs):
        if not isinstance(layer_properties, list): layer_properties = [layer_properties]
        self.values       = layer_properties
        self.names        = kwargs.get('names', ['physical_property_{}'.format(i) for i in range(len(self.values))])
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
        if kwargs.get('device') is not None:
            self.set_device(kwargs['device'])
            
        self.n_params = len(self.vec)
        self.n_ptypes = len(self.values)

    def set_device(self, device):
        self.device=device
        for var_name in vars(self):
            var = getattr(self, var_name)
            if torch.is_tensor(var):
                setattr(self, var_name, var.to(device))
            elif isinstance(var, list):
                for i, v in enumerate(var):
                    if torch.is_tensor(v):
                        var[i] = v.to(device)


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
        new_mvec = torch.clamp(mvec, self.lbnds, self.ubnds)
        return new_mvec

    def H(self, **kwargs):
        Hs = block_diag(*[self.H1d(val) for val in self.values])
        if 'thickness' in self.names:
            Hs = Hs[:-self.n_layers+2]
        return torch.from_numpy(Hs)

    def H1d(self, values):
        H = (np.eye(len(values))-np.eye(len(values), k=1))[:-1]
        return H
    
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
        n_params = len(self.values)
        if 'thickness' in self.names:
            thk = self.values[self.names.index('thickness')].cpu()
            n_params += -1
        else:
            thk = self.thickness.cpu()
        if axs is None:
            from matplotlib.pyplot import subplots
            fig,axs = subplots(1,n_params, sharey=True)
            if not isinstance(axs, np.ndarray):axs = [axs]
        for v,ax,pname in zip(self.values, axs, self.names):
            if pname == 'thickness':
                continue
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
    
    @classmethod
    def smooth(cls, property_values, z_bounds, **kwargs):
        class SmoothModel(cls):
            n_layers = len(z_bounds)+1
            z = z_bounds
            thickness = torch.diff(z_bounds, prepend=torch.zeros(1, dtype=z_bounds.dtype))
            def __init__(self, property_values, **kwargs):
                super().__init__(property_values, **kwargs)
                self.thickness = self.thickness#.to(self.device)
            
            def modelize(self, mvec):
                return super().modelize(mvec)+[self.thickness]
        
        return SmoothModel(property_values, **kwargs)
    
    @classmethod
    def few_layers(cls, property_values, thickness, **kwargs):
        class FewLayersModel(cls):
            n_layers = len(thickness)+1
            def __init__(self, property_values, **kwargs):
                super().__init__(property_values, **kwargs)
                if 'thickness' not in self.names:
                    self.names.append('thickness')
                    self.values.append(thickness)

        if not isinstance(property_values, list): property_values = [property_values]
        property_values += [thickness]
        return FewLayersModel(property_values, **kwargs)

def logify(cls):
    class LogModel(cls):
        def vectorize(self, list_of_arrays):
            return torch.log(super().vectorize(list_of_arrays))
        
        def modelize(self, mvec):
            return super().modelize(mvec.exp())
    return LogModel


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
