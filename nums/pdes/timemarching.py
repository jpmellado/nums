import numpy as np

# Define time-marching schemes

if __name__ != "__main__":
    from __main__ import dt


def Euler(rhs,u,t):
# rhs: function that calculates the rhs of the evolution equation
# u: vector with state variables
# t: time
    f = u +dt *rhs(u,t)   
    return f, t +dt

def RungeKutta3(rhs,u,t):   
# rhs: function that calculates the rhs of the evolution equation
# u: vector with state variables
# t: time
    k1 = rhs( u,                     t          )
    k2 = rhs( u +dt *0.5 *k1,        t +0.5 *dt )
    k3 = rhs( u +dt *( 2. *k2 -k1 ), t +     dt )
    f = u +dt *( k1 +4. *k2 +k3 ) /6.
    return f, t +dt
   
