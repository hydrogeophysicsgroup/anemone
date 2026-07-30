import numpy as np
import torch
from anemone import constants as c
from nosj import nosj


S1 = lambda x: 1 - (9/4)*x**2 * ( 1 - 2./9*x*(1 + 5/2*x*(1 - 2/5 * x)))
S2 = lambda x: -1/2*x*(1 - 11/6*x*(1 - 2/11*x*(1 + 5/2*x*(1-2/5*x))))

def lamontagne(x):
    phi   = x*0.
    xm2m1 = (-2 <= x)*(x < -1)
    xm1m0 = (-1 <= x)*(x < 0)
    x0p1  = (0 <= x)*(x <= 1)
    x1p2  = (1 < x)*(x <= 2)
    
    phi[xm2m1] = S2(-x[xm2m1]-1)
    phi[xm1m0] = S1(-x[xm1m0])
    phi[x0p1] = S1(x[x0p1])
    phi[x1p2] = S2(x[x1p2]-1)
    
    return phi

@nosj
class Receiver: 
    x       : np.ndarray
    y       : np.ndarray
    z       : np.ndarray
    grid    : bool = True

    def __init__(self, x=0., y=0., z=0., grid=False) -> None:
        self.x = np.array(x) if not isinstance(x, np.ndarray) else x
        self.y = np.array(y) if not isinstance(y, np.ndarray) else y
        self.z = np.array(z) if not isinstance(z, np.ndarray) else z
        self.grid = grid
        
        if self.grid:
            self._grid()
        else:
            all_points = [coord.size for coord in (self.x, self.y, self.z)]
            npoints = max(all_points)
            self.X = self.x if self.x.size==npoints else np.ones(npoints)*self.x
            self.Y = self.y if self.y.size==npoints else np.ones(npoints)*self.y
            self.Z = self.z if self.z.size==npoints else np.ones(npoints)*self.z


    def _grid(self):
        X,Y,Z = np.meshgrid(self.x, self.y, self.z, indexing='ij')
        self.shp = X.shape+(3,)
        self.X = X.squeeze()
        self.Y = Y.squeeze()
        self.Z = Z.squeeze()
        return self
    
    @property
    def xyz(self):
        return np.squeeze(np.moveaxis(np.stack((
                            self.X.flatten(), 
                            self.Y.flatten(), 
                            self.Z.flatten())), 0, -1))



    def __len__(self):
        return self.X.size

@nosj
class Loop:
    x : np.ndarray
    y : np.ndarray
    z : np.ndarray

    def __init__(self, x, y, z, **kwargs) -> None:
        self.x = np.array(x) if not isinstance(x, np.ndarray) else x
        self.y = np.array(y) if not isinstance(y, np.ndarray) else y
        self.z = np.array(z) if not isinstance(z, np.ndarray) else z
        
        if not np.all(self.z == self.z[0]): raise NotImplemented('current implementation only supports horizontal transmitter loops; all z values must be equal.')
        r2 = self.x[0]**2 + self.y[0]**2
        if np.allclose(self.x**2 + self.y**2, r2) and len(self.x)>=20:
            self.area = np.pi*r2
            self.radius = np.sqrt(r2)

    @classmethod
    def circle(cls, radius, n=40, **kwargs):
        r = radius
        
        t = np.linspace(0, 2*np.pi, n)
        x = r*np.cos(t)
        y = r*np.sin(t)
        z = np.ones_like(x)*kwargs.pop('z', 0.)
        lp = cls(x, y, z, **kwargs)
        lp.area = np.pi*radius**2
        lp.radius = r
        return lp

    @classmethod
    def square(cls, side, **kwargs):
        half = side / 2.
        x = np.array([-half,  half, half, -half, -half])
        y = np.array([-half, -half, half,  half, -half])
        z = np.ones_like(x)*kwargs.pop('z', 0.)
        lp = cls(x, y, z)
        lp.area = side*side
        return lp

    @classmethod
    def fig8(cls, side, **kwargs):
        half = side / 2.
        x = np.array([-side, 0, 0, side, side, 0, 0, -side, -side])
        y = np.array([-half, -half, half, half, -half, -half, half, half, -half])
        z = np.ones_like(x)*kwargs.pop('z', 0.)
        return cls(x, y, z)
    
    @classmethod
    def differential(cls, length, width, seperation=0.4, height=0.2):
        hl = length/2
        hw =  width/2
        x = [-hl, hl, hl, -hl, -hl]
        y = [-hw, -hw, hw, hw, -hw ]
        z = [height]*5
        bottom_coil = cls(x,y,z)
        z = [height+seperation]*5
        top_coil = cls(x[::-1], y[::-1], z)
        return bottom_coil, top_coil


    @property
    def xyz(self):
        return np.moveaxis(np.stack((self.x,self.y,self.z)), 0, -1)
    
    def min_horizontal_distance(self, rx):
        overall_min = np.inf
        for i in range(len(self)-1):
            v1x,v1y = rx.x-self.x[i], rx.y-self.y[i]
            v2x,v2y = self.x[i+1]-self.x[i], self.y[i+1]-self.y[i]
            v2r = v2x**2 + v2y**2
            line_dist = v1x*v2x + v1y*v2y
            
            # conditional to check to ensure rx points lies between the two end points.
            if ((0 <= line_dist)*(line_dist<=v2r)).any():
                numerator = np.abs(v1x*v2y-v2x*v1y)
                cmin = np.min(numerator/np.sqrt(v2r))
                if cmin < overall_min: overall_min = cmin*1
        return overall_min
    
    def max_horizontal_distance(self, rx):
        overall_max = 0
        for i in range(len(self)):
            cdist2 = np.max((self.x[i]-rx.x)**2 + (self.y[i]-rx.y)**2)
            if cdist2 > overall_max: overall_max = cdist2*1
        return np.sqrt(overall_max)

    def __len__(self):
        return self.x.size
    
    def geomz(self, rx, R):
        if hasattr(self, 'radius') and rx.xyz.size==3 and rx.x==0. and rx.y==0.:
            return self._geomz_circle(rx, R)
        logR = np.log(R)
        dR = logR[1]-logR[0]

        rx_xyz = rx.xyz
        n_rec  = len(rx)

        geomz      = np.zeros((n_rec, )+R.shape)

        R      = R [None, None, :]
        
        lp_xyz = self.xyz[:, None] - rx_xyz[None, :]
        
        for i in range(len(self)-1):
        
            Rz,iRz = getRz(lp_xyz[i], lp_xyz[i+1])
            p1 = np.einsum('...ij,...j->...i', Rz, lp_xyz[i])
            p2 = np.einsum('...ij,...j->...i', Rz, lp_xyz[i+1])
            x1,y1 = p1[..., 0], p1[..., 1]
            x2,y2 = p2[..., 0], p2[..., 1]
            ix = int(100*np.abs(x2-x1).max())
            tmpx = np.linspace(-1, 1, ix)
            x_samp = tmpx
            x_samp += 1.
            length = np.abs(x2-x1)/2.


            x_ = (x1[:,None] + length[:,None]*x_samp[None, :])
            dx = (x_[..., 1] - x_[..., 0])[:,None]
            
            x1 = x1[:, None]
            x2 = x2[:, None]
            y1 = y1[:, None, None]
            x_ = x_[:, :, None]
            dx = dx[:, None]


            cr = np.sqrt(y1**2 + x_**2)
            phi = lamontagne((np.log(cr)-logR)/dR)
            cs =  -y1 * phi * dx / cr
            geomz += cs.sum(axis=1, keepdims=False)
        if hasattr(rx, 'weighting') and rx.weighting is not None:
            geomz = np.einsum('i,ij...->j...', rx.weighting, geomz)

        return geomz

    def _geomz_circle(self, rx, R):
        logR = np.log(R)
        dR = logR[1]-logR[0]
        phi = lamontagne((np.log(self.radius)-logR)/dR)
        geomz = 2*np.pi*phi*self.radius
        return geomz

    def geomz_circle2(self, rx, R):
        """Compute geomz for a circular loop with off-centre receivers.

        Uses per-R clipped Gauss–Legendre quadrature in θ-space, splitting at
        the four phi polynomial-piece boundaries.  This replaces a uniform
        1 000-point θ grid with 4 sub-intervals × 3 GL points = 12 evaluations
        per (receiver, R) value while keeping relative error below ~5 × 10⁻⁴.

        The method exploits the fact that phi has compact support in log-r space:
        for each filter abscissa R_k, only the arc of the loop where
        |log(r(θ)/R_k)| < 2·dR contributes.  Within that arc the integrand is
        split at the four phi polynomial-piece boundaries (u = −1, 0, 1) so
        that the 3-point GL rule integrates the dominant polynomial part exactly.

        Derivation
        ----------
        r(θ) = √(radius² + d² − 2·radius·d·cos θ)  is monotone on [0, π],
        so each r value maps to a unique θ = arccos((radius²+d²−r²)/(2·radius·d)).
        The integrand   f(θ)·phi(u(θ))   is smooth within each phi piece, so GL
        converges exponentially.  A factor of 2 accounts for the symmetric arc
        at −θ (i.e. the full circle integral ∫₀²π = 2·∫₀ᵖ).

        Requires: minimum receiver-to-wire distance ≥ 1 cm.
        """
        from numpy.polynomial.legendre import leggauss
        import torch

        rdtype = R.dtype
        logR   = torch.log(R).cpu()
        dR     = float(logR[1] - logR[0])
        R_cpu  = R.cpu()

        radius = (self.radius if isinstance(self.radius, torch.Tensor)
                  else torch.tensor(self.radius, dtype=rdtype))

        d     = torch.sqrt(rx.X**2 + rx.Y**2).flatten().cpu()  # (n_rec,)
        n_rec = len(d)
        n_R   = len(R_cpu)

        # 3-point GL is exact for polynomials of degree ≤ 5 (phi is piecewise deg-5)
        gp_np, gw_np = leggauss(3)
        gp = torch.tensor(gp_np, dtype=rdtype)
        gw = torch.tensor(gw_np, dtype=rdtype)

        geomz = torch.zeros(n_rec, n_R, dtype=rdtype)

        # ── Centered receivers (d ≈ 0): r = radius everywhere on the loop ──
        centered = d < 1e-10
        if centered.any():
            phi_c = lamontagne((torch.log(radius) - logR) / dR)
            geomz[centered] = 2 * torch.pi * radius * phi_c

        oc_mask = ~centered
        if not oc_mask.any():
            return geomz.to(R.device)

        d_oc  = d[oc_mask]
        r_min = torch.abs(radius - d_oc)   # (n_oc,) min r to loop
        r_max = radius + d_oc              # (n_oc,) max r to loop

        # Active r window for each (receiver, R_k), clipped to [r_min, r_max]:
        r_lo = torch.maximum(r_min[:,None], R_cpu[None,:] * float(np.exp(-2*dR)))  # (n_oc, n_R)
        r_hi = torch.minimum(r_max[:,None], R_cpu[None,:] * float(np.exp( 2*dR)))  # (n_oc, n_R)

        def r_to_theta(r_val):
            # Invert r(θ)=√(radius²+d²−2·radius·d·cosθ):
            cos_t = (radius**2 + d_oc[:,None]**2 - r_val**2) / (2 * radius * d_oc[:,None])
            return torch.acos(cos_t.clamp(-1., 1.))  # θ ∈ [0, π]

        geomz_oc = torch.zeros(int(oc_mask.sum()), n_R, dtype=rdtype)

        for ub_lo, ub_hi in [(-2., -1.), (-1., 0.), (0., 1.), (1., 2.)]:
            r_lo_k = torch.maximum(r_lo, R_cpu[None,:] * float(np.exp(ub_lo * dR)))
            r_hi_k = torch.minimum(r_hi, R_cpu[None,:] * float(np.exp(ub_hi * dR)))
            valid  = r_lo_k < r_hi_k
            if not valid.any():
                continue

            # r increases monotonically with θ on [0,π], so r_lo→θ_lo, r_hi→θ_hi:
            theta_lo   = r_to_theta(r_lo_k)           # (n_oc, n_R)
            theta_hi   = r_to_theta(r_hi_k)           # (n_oc, n_R)
            theta_half = (theta_hi - theta_lo) / 2
            theta_mid  = (theta_lo + theta_hi) / 2

            theta_gl = theta_mid[:,:,None] + theta_half[:,:,None] * gp[None,None,:]  # (n_oc, n_R, 3)
            cos_gl   = torch.cos(theta_gl)
            r_gl     = torch.sqrt(radius**2 + d_oc[:,None,None]**2
                                  - 2*radius * d_oc[:,None,None] * cos_gl)

            f_gl   = radius * (radius - d_oc[:,None,None] * cos_gl) / r_gl
            u_gl   = (torch.log(r_gl) - logR[None,:,None]) / dR
            phi_gl = lamontagne(u_gl)

            # ×2: the loop has symmetric arcs at +θ and −θ contributing equally
            contrib   = 2 * (f_gl * phi_gl * theta_half[:,:,None] * gw[None,None,:]).sum(-1)
            geomz_oc += torch.where(valid, contrib, torch.zeros_like(contrib))

        geomz[oc_mask] = geomz_oc

        if hasattr(rx, 'weighting') and rx.weighting is not None:
            geomz = np.einsum('i,ij->j', rx.weighting, geomz)

        return geomz.to(R.device)

    def geomz_new(self, rx, R):
        print('using new geomz')
        if hasattr(self, 'radius') and rx.xyz.size == 3 and rx.x == 0. and rx.y == 0.:
            return self._geomz_circle(rx, R)

        from numpy.polynomial.legendre import leggauss

        logR = np.log(R)
        dR   = float(logR[1] - logR[0])
        n_R  = len(R)

        rx_xyz = rx.xyz
        n_rec  = len(rx)

        geomz_out = np.zeros((n_rec, n_R))

        lp_xyz = self.xyz[:, None] - rx_xyz[None, :]  # (n_pts, n_rec, 3)

        # 3-pt GL is exact for degree ≤ 5 polynomials; phi is piecewise degree-5.
        # Splitting at each phi piece boundary means the polynomial part is
        # integrated exactly and only the smooth ln(cosh) composition incurs error.
        gp, gw = leggauss(3)  # (3,)
        u_breaks = np.array([-2., -1., 0., 1., 2.])

        for i in range(len(self) - 1):
            Rz, _ = getRz(lp_xyz[i], lp_xyz[i+1])
            p1    = np.einsum('...ij,...j->...i', Rz, lp_xyz[i])    # (n_rec, 3)
            p2    = np.einsum('...ij,...j->...i', Rz, lp_xyz[i+1])  # (n_rec, 3)
            x1, y1 = p1[..., 0].flatten(), p1[..., 1].flatten()     # (n_rec,)
            x2     = p2[..., 0].flatten()                            # (n_rec,)

            # Skip receivers closer than 1 cm to the wire (y1 is perp. distance)
            safe = np.abs(y1) > 1e-3
            if not safe.any():
                continue

            y1_s    = y1[safe];  abs_y1_s = np.abs(y1_s)  # (n_safe,)
            x1_s    = x1[safe];  x2_s     = x2[safe]

            # Change of variable  x = y1·sinh(α)
            #   r = |y1|·cosh(α),   dx = y1·cosh(α) dα
            #   ∫ −y1·phi/r dx  =  ∫ −y1²/|y1|·phi dα  =  ∫ −|y1|·phi dα
            a1 = np.arcsinh(x1_s / y1_s)  # (n_safe,)
            a2 = np.arcsinh(x2_s / y1_s)  # (n_safe,)

            # beta[j,k] = (log|y1_j| − logR_k) / dR
            beta = (np.log(abs_y1_s)[:, None] - logR[None, :]) / dR  # (n_safe, n_R)

            # Alpha breakpoints for phi piece boundaries, per (receiver, R).
            # alpha_break corresponds to u_break via cosh(alpha) = exp((u−beta)·dR).
            # Set to 0 (lower bound of positive-alpha domain) when breakpoint < 0.
            with np.errstate(over='ignore'):
                cosh_b = np.exp(np.minimum(
                    (u_breaks[None, None, :] - beta[:, :, None]) * dR, 500.
                ))  # (n_safe, n_R, 5)
            alpha_breaks = np.where(
                cosh_b >= 1., np.arccosh(np.maximum(cosh_b, 1.)), 0.
            )  # (n_safe, n_R, 5)

            def F_pos(a_pos):
                """∫_0^{a_pos} −|y1|·phi(beta + ln(cosh α)/dR) dα → (n_safe, n_R)"""
                out = np.zeros((int(safe.sum()), n_R))
                cap = a_pos[:, None]  # (n_safe, 1)
                for k in range(4):
                    lo = np.minimum(alpha_breaks[:, :, k  ], cap)
                    hi = np.minimum(alpha_breaks[:, :, k+1], cap)
                    if not (lo < hi).any():
                        continue
                    ah = np.where(lo < hi, (hi - lo) / 2., 0.)  # (n_safe, n_R)
                    am = np.where(lo < hi, (hi + lo) / 2., 0.)  # (n_safe, n_R)
                    alpha_gl = am[:, :, None] + ah[:, :, None] * gp[None, None, :]
                    r_gl     = abs_y1_s[:, None, None] * np.cosh(alpha_gl)
                    u_gl     = (np.log(r_gl) - logR[None, :, None]) / dR
                    # ah=0 for inactive cells → zero contribution without explicit mask
                    out += (-abs_y1_s[:, None]
                            * (lamontagne(u_gl) * ah[:, :, None] * gw[None, None, :]).sum(-1))
                return out

            # Fundamental theorem: ∫_{a1}^{a2} = sgn(a2)·F_pos(|a2|) − sgn(a1)·F_pos(|a1|)
            contrib = (  np.sign(a2)[:, None] * F_pos(np.abs(a2))
                       - np.sign(a1)[:, None] * F_pos(np.abs(a1)))
            geomz_out[safe] += contrib
            
        if hasattr(rx, 'weighting') and rx.weighting is not None:
            geomz_out = np.einsum('i,ij...->j...', rx.weighting, geomz_out)
        return geomz_out


def perp_distance_to_line(x,y, line_points):
	((x1,y1), (x2,y2)) = line_points
	numerator = np.abs((x2-x1)*(y1-y) - (x1-x)*(y2-y1))
	denominator = np.sqrt((x2-x1)**2 + (y2-y1)**2)
	return numerator / denominator


def box_filter(r, r_min, r_par_max):
    r_max = np.sqrt(r_min**2 + r_par_max**2)
    y = np.ones_like(r)
    y[r<r_min]=0
    y[r>r_max]=0
    # if np.all(y==0):
    #      ir = np.searchsorted(r, r_min)
    #      y[ir] = 0.5
    #      y[ir+1] = 0.5
    y *= np.sign(r_par_max)
    return y

def norm(tensor, axis=-1, keepdims=False):
    return np.sqrt(np.sum(tensor**2, axis=axis, keepdims=keepdims))


def getRz(P,Q):
    QP = Q-P
    mQP = norm(QP)
    cos_theta = QP[..., 0] / mQP
    sin_theta = QP[..., 1] / mQP

    Rz = np.zeros(QP.shape+(3,))

    Rz[..., 0,0] = cos_theta
    Rz[..., 1,1] = cos_theta
    Rz[..., 0,1] = sin_theta
    Rz[..., 1,0] = -sin_theta
    Rz[..., 2,2] = 1.
    
    return Rz, np.swapaxes(Rz, -1, -2)


if __name__ == '__main__':
    import cProfile
    import pstats
    import matplotlib.pyplot as plt

    
    tx = Loop.square(40.)
    rx = Receiver(x=np.linspace(-1.,1.,51), y=np.linspace(-1.,1.,51), z=0., grid=True)
    rx.weighting = np.ones_like(rx.X.flatten())/rx.X.size

    r = np.logspace(-1, 2, 201)

    profiler = cProfile.Profile()
    profiler.enable()

    geomz = tx.geomz_new(Receiver(), r)

    profiler.disable()
    stats = pstats.Stats(profiler)
    stats.sort_stats('cumtime').print_stats(20)  # Print top 20 by cumulative time

    profiler = cProfile.Profile()
    profiler.enable()

    geomz2 = tx.geomz_new(rx, r)

    profiler.disable()
    stats = pstats.Stats(profiler)
    stats.sort_stats('cumtime').print_stats(20)  # Print top 20 by cumulative time


    plt.figure()
    plt.loglog(r, geomz.T, 'k-')
    plt.loglog(r, geomz2.T, '--')
    plt.show()