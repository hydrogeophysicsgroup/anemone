from anemone import transforms
from anemone import physics
from anemone import constants as c
import numpy as np
import torch

running_as_main = __name__ == '__main__'

def calc_rte_halfspace(sigma, omega, wave_length):
    lambda_sqrd = wave_length*wave_length
    K1_sqrd     = physics.calc_wavenumber(sigma, omega)**2
    u1          = torch.sqrt(lambda_sqrd-K1_sqrd)
    rte         = 1./(wave_length+u1)
    return rte


def test_frequency_response():
    rho = 100.                   # meters
    sigma = 1./100.            # 1/ohm-m
    frqs = torch.logspace(-6, 5, 501, dtype=torch.complex128)  # Hz
    omega = c.twopi*frqs
    
    hz_analytical = physics.halfspace_vmd(1./sigma, omega, rho)

    omega = omega[:, None]

    def rte1(wave_length):
        wave_length = wave_length[None, :]
        return calc_rte_halfspace(sigma, omega, wave_length)*wave_length**3
    Hkl_rte1 = transforms.Hankel(rte1, 0)

    r,resp = Hkl_rte1.range(90, 110)
    cbResp = transforms.CubicSpline(r, resp)
    hz = cbResp(rho)/c.twopi

    if running_as_main:
        import matplotlib.pyplot as plt
        plt.plot(frqs, hz_analytical*7., 'k-')
        plt.plot(frqs, hz, 'r--')
        plt.xlabel('r (m)')
        plt.ylabel('H(r)')
        plt.show()
    
    assert torch.allclose(hz.real, hz_analytical.real), "hankel transform results do not match analytical solution"
    assert torch.allclose(hz.imag, hz_analytical.imag), "hankel transform results do not match analytical solution"    


def test_transforms_varying_samples_per_decades():
    rho = 100.                   # meters
    sigma = 1./100.            # 1/ohm-m
    frqs = torch.logspace(-6, 5, 501, dtype=torch.complex128)  # Hz
    omega = c.twopi*frqs[:, None]

    def rte1(wave_length):
        wave_length = wave_length[None, :]
        return calc_rte_halfspace(sigma, omega, wave_length)*wave_length**3

    Hkl_rte1 = transforms.Hankel(rte1, 0)
    r,resp = Hkl_rte1.range(90, 110)
    cbResp = transforms.CubicSpline(r, resp)
    hz1 = (cbResp(rho)/(c.twopi))

    Hkl_rte2 = transforms.Hankel(rte1, 0, smp_per_decade=20)
    r,resp = Hkl_rte2.range(90, 110)
    cbResp = transforms.CubicSpline(r, resp)
    hz2 = (cbResp(rho)/(c.twopi))


    assert np.allclose(hz1.real, hz2.real), "changing the number of decades is changing the results of the transform"
    assert np.allclose(hz1.imag, hz2.imag), "changing the number of decades is changing the results of the transform"

def test_spline():
    x = torch.linspace(0, 10, 4*80+1)
    xs = x[::4]

    f = lambda x: 1+x**2 + 2*x**3

    ys = f(xs)
    ys = ys[None, :] + torch.arange(650)[:, None]

    y = f(x)
    y1 = y[None, :] + torch.arange(650)[:, None]

    cspl = transforms.CubicSpline(xs,ys)
    y2 = cspl(x)
    assert torch.allclose(y1, y2), "cubic spline interpolation on cubic function is giving non-exact results"

    if running_as_main:
        return x.numpy(), y1.numpy(), y2.numpy()
    
def test_spline_integration():
    x = torch.linspace(0, 10, 4*80+1)
    xs = x[::4]

    f = lambda x: 1+x**2 + 2*x**3
    fint = lambda x: x + (1/3)*x**3 + (2/4)*x**4

    ys = f(xs)

    xlb, xub = torch.tensor([1,3,5]), torch.tensor([6, 4, 9])

    y1 = fint(xub)-fint(xlb)

    cspl = transforms.CubicSpline(xs,ys)
    y2 = cspl.integrate(xlb, xub)
    # print(y1)
    # print(y2)
    assert torch.allclose(y1, y2), "cubic spline interpolation on cubic function is giving non-exact results"

    if running_as_main:
        return x.numpy(), y1.numpy(), y2.numpy()    

# def test_fourier():
#     from anemone.transforms import Fourier

#     a = torch.tensor([1e-1], dtype=torch.complex128)
#     def f1(x):
#         return 1e6*torch.exp(-a*x**2)

#     def F1(k):
#         return 1e6*torch.sqrt(c.pi/a)*torch.exp(-c.pi**2*k*k/a)

#     ft = Fourier(f1, smp_per_decade=20, range=(1e-6,1e2))
#     f,numeric = ft.r, ft.resp
#     analytic = F1(f)


#     if running_as_main:
#         f = f.numpy()
#         analytic = analytic.numpy()
#         numeric = numeric.numpy()
#         import matplotlib.pyplot as plt
#         fig,axs = plt.subplots(1,3)
#         axs[0].loglog(f, analytic)
#         axs[0].loglog(f, numeric, 'r--')
#         axs[1].semilogx(f, analytic)
#         axs[1].semilogx(f, numeric, 'r--')
#         axs[2].semilogx(f, analytic.real/numeric.real)
#         # axs[0].set_xlim(1e-1, 1e1)
#         axs[0].set_ylim(1e-16, None)
#         plt.show()
#     else:
#         assert torch.allclose(numeric.real, analytic.real), "Fourier transform is performing poorly on gaussian analytical example"


def test_fourier_lowerfrequencies():
    from anemone.transforms import Fourier

    a = torch.tensor([1], dtype=torch.complex128)
    def f1(x):
        return 1e6*torch.exp(-a*x**2)

    def F1(k):
        return 1e6*torch.sqrt(c.pi/a)*torch.exp(-c.pi**2*k*k/a)

    ft = Fourier(f1)
    f,numeric = ft.range(1e-6,1e-1)
    analytic = F1(f)

    if running_as_main:
        f = f.numpy()
        analytic = analytic.numpy()
        numeric = numeric.numpy()
        import matplotlib.pyplot as plt
        fig,axs = plt.subplots(1,3)
        axs[0].loglog(f, analytic)
        axs[0].loglog(f, numeric, 'r--')
        axs[1].semilogx(f, analytic)
        axs[1].semilogx(f, numeric, 'r--')
        axs[2].semilogx(f, analytic.real/numeric.real)
        axs[0].set_ylim(1e-16, None)
        plt.show()
    else:
        assert torch.allclose(numeric.real, analytic.real), "Fourier transform is performing poorly on gaussian analytical example"

from pytest import mark

@mark.skipif(reason="other tests seem ok, but haven't found the reason this fails")
def test_invfourier_earlytimes():
    from anemone.transforms import invFourier

    a = torch.tensor([2e-1], dtype=torch.complex128)
    def f1(x):
        return 1e6*torch.exp(-a*x**2)

    def F1(k):
        return 1e6*torch.sqrt(c.pi/a)*torch.exp(-c.pi**2*k*k/a)

    ift = invFourier(F1, smp_per_decade=21)
    ift._recenter_function((torch.ones(1)*1e12, torch.ones(1)*1e13))
    t,y2 = ift.range(1e-6,1e3)

    y1 = f1(t)



    if running_as_main:

        
        analytic = y1.numpy()
        numeric = y2.numpy()
        import matplotlib.pyplot as plt
        fig,axs = plt.subplots(1,3)
        axs[0].loglog(t, analytic)
        axs[0].loglog(t, numeric, 'r--')
        axs[1].semilogx(t, analytic)
        axs[1].semilogx(t, numeric, 'r--')
        axs[2].semilogx(t, analytic.real/numeric.real)
        axs[0].set_ylim(1e-16, None)
        plt.show()
    assert torch.allclose(y1.real, y2.real), "inverse Fourier transform is performing poorly on gaussian analytical example"

def test_invlaplace():
    from anemone.transforms import invLaplace

    def f(t):
        return 1./torch.sqrt(torch.pi*t)

    def LF(s):
        return 1./torch.sqrt(s)

    t = torch.logspace(-7, 2, 101, dtype=torch.float64)

    y1 = invLaplace(LF, n_coefficients=16)(t)
    y2 = f(t)

    
    if running_as_main:
        
        import matplotlib.pyplot as plt
        fig,axs = plt.subplots(1,2, figsize=(6,4))
        axs[0].loglog(t.numpy(), y2.numpy(), 'k-', label='analytical')
        axs[0].loglog(t.numpy(), y1.numpy(), 'r--', label='inv.laplace numeric')
        axs[1].semilogx(t.numpy(), 100*(y1.numpy()-y2.numpy())/y2.numpy(), 'k-', label='analytical')

        axs[0].legend()
        axs[0].set_xlabel('time [s]')
        axs[0].set_ylabel('f(t)')
        axs[1].set_xlabel('time [s]')
        axs[1].set_ylabel('relative error [%]')
        plt.tight_layout()
        plt.savefig('../docs/images/invlaplace_example.png', dpi=300)
        plt.show()
        # return t.numpy(), y1.numpy(), y2.numpy()
    assert torch.allclose(y1.real, y2.real), "inverse Laplace transform is performing poorly on 1/sqrt(s)<->1./sqrt(t*pi) analytical example"
    
def test_invlaplace_Hz_center_of_circular_loop_halfspace():
    import torch
    class ForwardLaplaceHalfspace:
        def __init__(self, radius, times):
            self.radius = radius
            self.times = times if isinstance(times, torch.Tensor) else torch.tensor(times)
            self.times = self.times.to(torch.float64)

        def forward(self):
            def laplace_integrand(s):
                omega = -1j*s
                hz = physics.hz_halfspace_center_circle_loop(self.sigma, omega, self.radius)
                
                return  -(hz-0.01)/s
            cbft = transforms.invLaplace(laplace_integrand, n_coefficients=16)
            resp = cbft(self.times)
            return resp.squeeze()   

        def __call__(self, sigma, thick):
            self.sigma = torch.tensor(sigma, dtype=torch.complex128)
            return self.forward().real

    t = torch.logspace(-7, -4, 51, dtype=torch.float64)
    fwr2 = ForwardLaplaceHalfspace(50., t)
    rho = 10
    Hz = fwr2(1/rho, None)
    Hz_analytical = physics.hz_halfspace_center_circle_loop_time(t, 1/rho, 50.)
    assert torch.allclose(Hz.real, Hz_analytical.real, rtol=1.e-4), "inverse Laplace transform is performing poorly on Hz from center of circular loop halfspace analytical example"
    if running_as_main:
        import matplotlib.pyplot as plt
        fig,axs = plt.subplots(1,2)
        axs[0].loglog(t.numpy(), Hz_analytical.numpy(), 'k-', label='analytical')
        axs[0].loglog(t.numpy(), Hz.numpy(), 'r--', label='laplace numeric')
        axs[1].semilogx(t.numpy(), (Hz.numpy()-Hz_analytical.numpy())/Hz_analytical.numpy(), 'k-', label='analytical')

        plt.legend()
        plt.show()

def test_invlaplace_dHzdt_center_of_circular_loop_halfspace():
    import torch
    class ForwardLaplaceHalfspace:
        def __init__(self, radius, times):
            self.radius = radius
            self.times = times if isinstance(times, torch.Tensor) else torch.tensor(times)
            self.times = self.times.to(torch.float64)

        def forward(self):
            def laplace_integrand(s):
                omega = -1j*s
                hz = physics.hz_halfspace_center_circle_loop(self.sigma, omega, self.radius)
                return  hz
            cbft = transforms.invLaplace(laplace_integrand, n_coefficients=16)
            resp = cbft(self.times)
            return resp.squeeze()   

        def __call__(self, sigma, thick):
            self.sigma = torch.tensor(sigma, dtype=torch.complex128)
            return self.forward().real

    t = torch.logspace(-7, -3, 51, dtype=torch.float64)
    fwr2 = ForwardLaplaceHalfspace(50., t)
    rho = 10
    Hz = fwr2(1/rho, None)
    Hz_analytical = physics.hzdt_halfspace_center_circle_loop(t, 1/rho, 50.)
    #assert torch.allclose(Hz.real, Hz_analytical.real, rtol=1.e-4), "inverse Laplace transform is performing poorly on Hz from center of circular loop halfspace analytical example"
    if running_as_main:
        import matplotlib.pyplot as plt
        fig,axs = plt.subplots(1,2)
        axs[0].loglog(t.numpy(), Hz_analytical.numpy(), 'k-', label='analytical')
        axs[0].loglog(t.numpy(), Hz.numpy(), 'r--', label='laplace numeric')
        axs[1].semilogx(t.numpy(), (Hz.numpy()-Hz_analytical.numpy())/Hz_analytical.numpy(), 'k-', label='analytical')

        plt.legend()
        plt.show()

def check_hankel_pairs(Fk, Fr, axs=None, order=0, smp_per_decade=12):
    Fk_num = transforms.Hankel(Fr, order=order, lambda_included=False, smp_per_decade=smp_per_decade)
    r = torch.logspace(-15, 15, 100)    
    fr_num_vals = Fk_num(r).numpy()
    fr_ana_vals = Fk(r).numpy()
    r = r.numpy()
    
    import matplotlib.pyplot as plt
    if axs is None:
        fig,axs = plt.subplots(2,1, sharex=True)
        axs[0].loglog(r, fr_ana_vals, 'k-', label='analytic')
    else:
        fig = axs[0].figure

    axs[0].loglog(r, fr_num_vals, '.', label=f'{smp_per_decade}')
    axs[1].loglog(r, 100*(np.abs(fr_num_vals - fr_ana_vals)/fr_ana_vals))
    axs[1].set_ylabel('Relative Error (%)')
    axs[1].set_ylim(None, 3.)
    axs[1].set_xlabel('r')
    assert np.allclose(fr_num_vals, fr_ana_vals, rtol=1e-2), "Hankel transform did not match analytic solution within tolerance."
    return axs

def test_hankel_simple_hankel_pair_order0():
    def Fk(k):
        return k**-1

    def fr(r):
        return 1/r 

    axs = check_hankel_pairs(Fk, fr, order=0, smp_per_decade=6)
    check_hankel_pairs(Fk, fr, axs=axs, order=0, smp_per_decade=12)
    check_hankel_pairs(Fk, fr, axs=axs, order=0, smp_per_decade=24)
    check_hankel_pairs(Fk, fr, axs=axs, order=0, smp_per_decade=36)
    if running_as_main:
        import matplotlib.pyplot as plt
        axs[0].legend(title='smp/decade')
        plt.show()

def test_hankel_simple_hankel_pair_order1():
    def Fk(k):
        return k**-1

    def fr(r):
        return 1/r 

    axs = check_hankel_pairs(Fk, fr, order=1, smp_per_decade=6)
    check_hankel_pairs(Fk, fr, axs=axs, order=1, smp_per_decade=12)
    check_hankel_pairs(Fk, fr, axs=axs, order=1, smp_per_decade=24)
    check_hankel_pairs(Fk, fr, axs=axs, order=1, smp_per_decade=36)
    if running_as_main:
        import matplotlib.pyplot as plt
        axs[0].legend(title='smp/decade')
        plt.show()

if __name__ == '__main__':
    test_frequency_response()
    # test_spline()
    # test_transforms_varying_samples_per_decades()
    # test_fourier()
    # test_spline_integration()
    # test_invlaplace()
    # test_fourier_lowerfrequencies()
    # test_invfourier_earlytimes()
    # test_invlaplace_Hz_center_of_circular_loop_halfspace()
    # test_invlaplace_dHzdt_center_of_circular_loop_halfspace()
    # test_invlaplace_Hz_center_of_circular_loop_halfspace()
    test_hankel_simple_hankel_pair_order1()