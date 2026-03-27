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

default_dtype = 'float'
if default_dtype=='float':
    rdtype = torch.float32
    cdtype = torch.complex64
elif default_dtype=='double':
    rdtype = torch.float64
    cdtype = torch.complex128
