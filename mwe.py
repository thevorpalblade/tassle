import numpy as np
import numba


@numba.njit(parallel=True)
def a():
    f = np.array([1 for i in range(1000)])
    return np.std(f), np.mean(f)


@numba.njit
def b():
    return a()


@numba.njit
def spin(n):
    i = 0
    for a in range(n):
        i += 2 * a**2
        i = i % 10
    return i


@numba.njit(parallel=True)
def ja():
    a = 0
#     x = np.array([spin(1e6)
#                   for i in range(int(1e6))])
    n = int(1e5)
    x = np.zeros(n)
    for i in range(n):
        x[i] = spin(n)
    return np.std(x), np.mean(x)


@numba.njit(parallel=True)
def jb():
    a = 0
    x = np.array([spin(1e6)
        for i in range(int(1e6))])
    return np.std(x), np.mean(x)

@numba.njit
def jc():
    return ja()
