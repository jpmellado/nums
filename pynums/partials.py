# Finite difference approximations to partial derivatives of a function f(x,y) 
# discretized into f[j,i] assuming a homogeneous grid spacing
# i runs in x, and j in y, so that elements in horizontal lines are contiguous in memory

# This is just the appropriate wrapper for the schemes in the fdm library

import numpy as np

class partial:
    def __init__(self, scheme, step_x=1.0, step_y=1.0):
        self.scheme = scheme
        self.step_x = step_x
        self.step_y = step_y

    def dx(self,f):
        d = np.empty_like(f)

        ny = np.shape(f)[0]
        for j in range(ny):
            d[j,:] = self.scheme(f[j,:])
            
        return d/self.step_x

    def dy(self,f):
        d = np.empty_like(f)

        nx = np.shape(f)[1]
        for i in range(nx):
            d[:,i] = self.scheme(f[:,i])
            
        return d/self.step_y

    # def dxx(self,f):
    #     d = self.dx(f)
    #     return d/self.step_x

    # def dyy(self,f):
    #     d = self.dy(f)
    #     return d/self.step_y
