from itertools import accumulate

import numba
import numpy as np
from matplotlib import pyplot as plt


def do_rw(n=1000000, w=0.1, sigma=1):
    deltas = sigma * np.random.randn(n)

    x = np.array(list(accumulate(deltas, lambda x0, xi: x0 + xi - w * x0)))
    return x


@numba.njit(fastmath=True)
def do_numba_rw(n=1000, w=0.99, sigma=1.0, init_sigma=7):
    val = np.random.randn() * init_sigma
    for i in range(n):
        val = val * w + np.random.randn() * sigma
    return val


@numba.njit(fastmath=True)
def find_final_std(w, sigma, tolerance=0.1, n=1000, nrounds=1000):
    # get two initial data points
    vals = np.array([do_numba_rw(n, w=w, sigma=sigma) for i in range(nrounds)])
    old_std = np.std(vals)
    # on the second one, use the std dev we found the first time to jumpstart
    # things
    vals = np.array([
        do_numba_rw(n, w=w, sigma=sigma, init_sigma=old_std)
        for i in range(nrounds)
    ])
    new_std = np.std(vals)
    # declare a couple variables. monotonic keeps track of how long the increase
    # in std is monotonic, stds is an array to accumulate the stds once they b
    monotonic = True
    stds = []
    # make sure we don't trigger the first time
    stderr = tolerance + 1

    # the loop has two phases, the initial monotonic phase, and the phase where
    # we are actually recording stderrs.
    while monotonic or (stderr > tolerance):

        print("quack")
        print("new std: ", new_std)
        # if we are in the first phase, we will seed the next point with the
        # last std
        if not monotonic:
            avg = new_std
        # check for monotonicity and switch modes if appropriate
        if old_std > new_std:
            monotonic = False
            # when we switch modes, populate the std array with two initial
            # values
            stds.append(old_std)
            stds.append(new_std)

        old_std = new_std
        # while running, run another nrounds random walks and compute the
        # standard deviation
        vals = np.array([
            do_numba_rw(n, w=w, sigma=sigma, init_sigma=old_std)
            for i in range(nrounds)
        ])
        new_std = np.std(vals)
        # in the second mode, compute standard errors of the standard deviations
        # as well as the mean.
        if not monotonic:
            stds.append(new_std)
            stds_array = np.array(stds)
            stderr = np.std(stds_array) / np.sqrt(len(stds_array) - 1)
            print("stderr: ", stderr)

            avg = np.mean(stds_array)

    return (avg, np.std(stds_array))


@numba.njit(fastmath=True)
def find_coh_length(w, sigma, measured_sigma=-1., tolerance=.1, nrounds=1000):
    # if not passed a measured_sigma (final sigma of RR), find it:
    measured_sigma_std = 0
    if measured_sigma == -1.:
        measured_sigma, measured_sigma_std = find_final_std(w, sigma)
    # now do a bunch of random walks and find the coherence length for each one
    coherence_lengths = []
    # initialize stderr to a large value so we enter the loop the first time
    stderr = tolerance + 1
    while stderr > tolerance:
        # initialize a random walk
        init_val = np.random.randn() * measured_sigma
        val = init_val
        # keeps track of how far the random walk goes
        counter = 0
        # continue the random walk until we have wandered sigma/e (the threshold
        # for "coherence" for a unweighted random walk. Assuming this is a
        # resonable estimate for a weighted random walk may be bad, but life is
        # short.
        while np.abs(init_val - val) < measured_sigma / np.e:
            val = val * w + np.random.randn() * sigma
            counter += 1
        coherence_lengths.append(counter)
        if len(coherence_lengths) > 5:
            stderr = np.std(np.array(coherence_lengths)) / np.sqrt(
                len(coherence_lengths))
            print("coherence_length: ", counter)
            print("stderr: ", stderr)
    avg = np.mean(np.array(coherence_lengths))
    std_dev = np.std(np.array(coherence_lengths))
    return measured_sigma, measured_sigma_std, avg, std_dev


def thingy(w=0.99, n=10000, sigma=1, count=100, init_sigma=7):
    last_points = []
    for i in range(count):
        x = do_numba_rw(w=w, n=n, sigma=sigma, init_sigma=init_sigma)
        last_points.append(x)
    plt.hist(last_points, 30)
    print(np.std(last_points))

    return last_points


def do_bounded_rw(n=10000):
    val = np.random.randn()
    vals = np.zeros(n)
    vals[0] = val

    for i in range(1, n):
        val = np.random.randn() + val / np.sqrt(i)
        vals[i] = val
    return vals
