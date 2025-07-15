from dolphinv import *
import torch

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

LogResistivity = logify(Resistivity)
LogMPA = logify(MPA)

def fastjacobian(cls):
    class FastJacobian(cls):
        def jacobian(self, mvec):
            rho, thick = self.model.modelize(mvec)
            G = self.forward(rho, thick, jac=True)
            G *= rho[:, None]
            G /= 10**self.data_predicted
            return G.T
    return FastJacobian