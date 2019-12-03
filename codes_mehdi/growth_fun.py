from scipy import special
import math

# define growth factor G(z)
def growth_factor(z, Omega_m):
    a = 1.0/(1.0+z)
    v = (1.0+z)*(Omega_m/(1.0-Omega_m))**(1.0/3.0)
    phi = math.acos((v+1.0-3.0**0.5)/(v+1.0+3.0**0.5))
    m = (math.sin(75.0/180.0* math.pi))**2.0
    part1c = 3.0**0.25 * (1.0+ v**3.0)**0.5
    # first elliptic integral
    F_elliptic = special.ellipkinc(phi, m)
    # second elliptic integral
    Se_elliptic = special.ellipeinc(phi, m)
    part1 = part1c * ( Se_elliptic - 1.0/(3.0+3.0**0.5)*F_elliptic)
    part2 = (1.0 - (3.0**0.5 + 1.0)*v*v)/(v+1.0+3.0**0.5)
    d_1 = 5.0/3.0*v*(part1 + part2)
    # if a goes to 0, use d_11, when z=1100, d_1 is close to d_11
    #    d_11 = 1.0 - 2.0/11.0/v**3.0 + 16.0/187.0/v**6.0
    return a*d_1
