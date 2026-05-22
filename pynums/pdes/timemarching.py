import numpy as np

# Define time-marching schemes to advance one single step

if __name__ != "__main__":
    from __main__ import h, diffusivity, velocity

def diffusionNumber(dt):
    if diffusivity != None:
        return dt * diffusivity / h**2.0
    else:
        return None


def advectionNumber(dt):
    if velocity != None:
        return dt * velocity / h
    else:
        return None
    
class Explicit:
    def __init__(self, delta_t = None, dnum = None, cnum = None):
        # We use one over delta_t to deal with cases of diffusivity=0, velocity=0
        if dnum != None and diffusivity != None: # Variable time step from diffusion stability constraint
            one_ov_dtd = diffusivity / h**2.0 / dnum
        else:
            one_ov_dtd = 0.0

        if cnum != None and velocity != None: # Variable time step from advection stability constraint
            one_ov_dtc = velocity / h / cnum
        else:
            one_ov_dtc = 0.0

        one_ov_dt = max(one_ov_dtc, one_ov_dtd)
        if one_ov_dt > 0.0:
            self.dt = 1.0/one_ov_dt
        else:
            self.dt = delta_t
        
class Euler(Explicit):
    dnumMax = 2./4.
    cnumMax = 0.

    def __init__(self, delta_t = None, dnum = None, cnum = None):
        super().__init__(delta_t, dnum, cnum)  # Call parent constructor

    def advanceStep(self, rhs, u, t):
    # rhs: function that calculates the rhs of the evolution equation
    # u: vector with state variables
    # t: time
        f = u +self.dt *rhs(u,t)   
        return f, t +self.dt

class RungeKutta3(Explicit):
    dnumMax = 2.57/4.
    cnumMax = 1.73

    def __init__(self, delta_t = None, dnum = None, cnum = None):
        super().__init__(delta_t, dnum, cnum)  # Call parent constructor

    def advanceStep(self, rhs, u, t):
    # rhs: function that calculates the rhs of the evolution equation
    # u: vector with state variables
    # t: time
        k1 = rhs( u,                     t          )
        k2 = rhs( u +self.dt *0.5 *k1,        t +0.5 *self.dt )
        k3 = rhs( u +self.dt *( 2. *k2 -k1 ), t +     self.dt )
        f = u +self.dt *( k1 +4. *k2 +k3 ) /6.
        return f, t +self.dt
   
