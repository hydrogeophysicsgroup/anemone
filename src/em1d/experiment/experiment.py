import numpy as np
from dataclasses import dataclass


class Receiver:
    def __init__(self, x=0., y=0., z=0., grid=True) -> None:
        self.x = np.array(x)
        self.y = np.array(y)
        self.z = np.array(z)
        
        if grid:
            self.grid()
        else:
            npoints = max(len(self.x), len(self.y),len(self.z))
            if len(self.x)==1: self.x = np.ones(npoints)*self.x
            if len(self.y)==1: self.y = np.ones(npoints)*self.y
            if len(self.z)==1: self.z = np.ones(npoints)*self.z

    def grid(self):
        X,Y,Z = np.meshgrid(self.x, self.y, self.z, indexing='ij')
        self.x = np.squeeze(X)
        self.y = np.squeeze(Y)
        self.z = np.squeeze(Z)
        return self
    
    @property
    def xyz(self):
        return np.squeeze(np.moveaxis(np.stack((self.x, self.y, self.z)), 0, -1))

    def __len__(self):
        return np.size(self.x)

@dataclass
class Loop:
    def __init__(self, x, y, z):
        self.x = np.array(x)
        self.y = np.array(y)
        self.z = np.array(z)
        if not np.all(self.z == self.z[0]): raise NotImplemented('current implementation only supports horizontal transmitter loops; all z values must be equal.')



    @classmethod
    def square(cls, side, **kwargs):
        half = side / 2.
        x = np.array([-half,  half, half, -half, -half])
        y = np.array([-half, -half, half,  half, -half])
        z = np.zeros_like(x)
        return cls(x, y, z)

    @classmethod
    def fig8(cls, side, **kwargs):
        half = side / 2.
        x = np.array([-side, 0, 0, side, side, 0, 0, -side, -side])
        y = np.array([-half, -half, half, half, -half, -half, half, half, -half])
        z = np.zeros_like(x)
        return cls(x, y, z)

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
            if ((0 < line_dist)*(line_dist<v2r)).any():
                numerator = np.abs(v1x*v2y-v2x*v1y)
                cmin = np.min(numerator/np.sqrt(v2r))
                if cmin < overall_min: overall_min = cmin*1
        return overall_min
    
    def max_horizontal_distance(self, rx):
        overall_max = 0
        for i in range(len(self)):
            cdist2 = np.max((self.x[i]-rx.x)**2 + (self.y[i]-rx.y)**2)
            if cdist2 > overall_max: overall_min = cdist2*1
        return np.sqrt(cdist2)

    def __len__(self):
        return np.size(self.x)
    
    def s_of_r(self, rx, R):
        # dR = np.gradient(R)
        
        # s = np.zeros_like(R)

        # lp_xyz = self.xyz - xyz[None, ...]

        # for i in range(len(self)-1):
        #     Rz,iRz = getRz(lp_xyz[i], lp_xyz[i+1])
        #     p1 = np.einsum('...i,...i->...', Rz, lp_xyz[i])
        #     p2 = np.einsum('...i,...i->...', Rz, lp_xyz[i+1])

        #     r_perp = p1[1]
        #     interval = box_filter(R, np.abs(r_perp), p2[0])-box_filter(R, np.abs(r_perp), p1[0])

        #     ds = np.nan_to_num(interval*R*dR/np.sqrt(R**2 - r_perp**2))
        #     ds *= np.abs(p2[0]-p1[0])/np.abs(np.nansum(ds))
        #     ds *= r_perp
        #     s +=  ds
        # return np.nan_to_num(s)
        dR_alpha = np.sqrt(R[1]/R[0])
        R_edge1 = (R/dR_alpha)**2
        R_edge2 = (R*dR_alpha)**2

        
        s = np.zeros_like(R)

        lp_xyz = self.xyz - rx.xyz[None, :]
        for i in range(len(self)-1):
            Rz,iRz = getRz(lp_xyz[i], lp_xyz[i+1])
            p1 = np.einsum('...i,...i->...', Rz, lp_xyz[i])
            p2 = np.einsum('...i,...i->...', Rz, lp_xyz[i+1])
            y2 = p1[1]**2

            x1,x2 = min(np.abs(p1[0]), np.abs(p2[0])), max(np.abs(p1[0]), np.abs(p2[0]))
            r1 = x1**2 + y2
            r2 = x2**2 + y2

            ds = np.nan_to_num(np.sqrt(R_edge2 - y2)) - np.nan_to_num(np.sqrt(R_edge1 - y2))

            ds1,ds2 = ds*1., ds*1.
            
            i_s = np.searchsorted(R_edge1, r1)-1
            ds1[i_s] = np.nan_to_num(np.sqrt(r1 -y2)) - np.nan_to_num(np.sqrt(R_edge1[i_s] -y2))
            
            ds1[i_s+1:] = 0

            i_s = np.searchsorted(R_edge1, r2)-1
            ds2[i_s] = np.nan_to_num(np.sqrt(r2 -y2)) - np.nan_to_num(np.sqrt(R_edge1[i_s] -y2))
            ds2[i_s+1:] = 0
            
            sign = np.sign(p1[0])*np.sign(p2[0])
            # if np.sign(p1[0])==np.sign(p2[0]): multiplier = 1
            # else: multiplier = 2
            
            ds = np.nan_to_num(ds2) - np.nan_to_num(sign*ds1)
            ds *= np.abs(p2[0]-p1[0])/np.nansum(ds)
            
            s += p1[1]*ds/R

        return s        
    
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

def getRz(P,Q):
    QP = Q-P
    mQP = np.linalg.norm(QP, axis=-1)
    cos_theta = QP[..., 0] / mQP
    sin_theta = QP[..., 1] / mQP

    Rz = np.zeros(QP.shape+(3,))

    Rz[..., 0,0] = cos_theta
    Rz[..., 1,1] = cos_theta
    Rz[..., 0,1] = sin_theta
    Rz[..., 1,0] = -sin_theta
    Rz[..., 2,2] = 1.
    
    return Rz, Rz.swapaxes(-1, -2)