import torch

class constants:
    pi = torch.pi
    twopi = 2*torch.pi
    fourpi = 4*torch.pi
    sqrt_twopi = (2*torch.pi)**(1./2.)
    sqrt_pi = torch.pi**(1./2.)
    mu0 = 4*torch.pi*1e-7
    epsilon0 = 8.854187817e-12
    log2 = 0.6931471805599453

default_dtype = 'double'
if default_dtype=='float':
    torch.set_default_dtype(torch.float32)
elif default_dtype=='double':
    torch.set_default_dtype(torch.float64)

def rdtype():
    return torch.get_default_dtype()

def cdtype():
    cfloat = rdtype()
    if   cfloat==torch.float32: return torch.complex64
    elif cfloat==torch.float64: return torch.complex128
    raise NotImplementedError(f"no complex type associated with current default float {cfloat}")
