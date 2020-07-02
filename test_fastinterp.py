#
import numba
import numpy as np


def fast_interp1d(instant, t, ary):

    # first, our optimized interpolator
    # make a formula for finding the array index for a specific time
    slope = (len(t) - 1) / (t[-1] - t[0])
    intercept = - t[0] * slope
    # import ipdb; ipdb.set_trace()

    # find the floating point index to the array
    idx = slope * instant + intercept 
    x1 = int(np.floor(idx))
    x2 = int(np.ceil(idx))

    return ary.T[x1] + (ary.T[x2] - ary.T[x1]) * (idx - x1)


