from functools import partial
from itertools import accumulate
import numpy as np


def do_rw(weight_fun, n=1000000, w=0.1, sigma=1):
   deltas = sigma * np.random.randn(n)

   weight = partial(weight_fun, w)
   x = np.array(list(accumulate(deltas, lambda x0, xi: x0 + xi - weight(x0) )))
   return x


def linear_weight(w, x):
    return x * w



def thingy(w=.01):
    bins = np.linspace(-2, 2, 100)
    y_gauss = np.zeros(99)
    y_linear = np.zeros(99)
    y_quad = np.zeros(99)
    for i in range(100):
        x2 = do_rw(linear_weight, w=w, n=10000, sigma=.1)
        y_linear = y_linear + np.histogram(x2, bins)[0]

    return y_gauss, y_linear, y_quad

