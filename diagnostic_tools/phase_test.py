import numpy as np
from scipy import optimize as opt

def phase(timestep, phase_std=np.sqrt(2), ctime=.4):
    return phase_std * np.sqrt(timestep/ctime) * np.random.randn((len(timestep)))

def test_std(pstd):
    a = phase(np.full(1000000, .4), phase_std=pstd)
    Imax = (np.cos(a) ** 2).sum()
    Imin = (np.cos(a + np.pi/2) ** 2).sum()

    return (Imax - Imin) / (Imax + Imin)

def do_fit():
    def residual(p):
        pstd = p[0]
        return 1/np.e - test_std(pstd)
    return opt.least_squares(residual, (np.pi/4))

