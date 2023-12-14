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
            if np.any(0 < line_dist < v2r):
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