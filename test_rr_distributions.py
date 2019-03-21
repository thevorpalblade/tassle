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
def find_final_std(w, sigma, tolerance=0.01, n=1000, nrounds=1000):
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
    percent_err = tolerance + 1

    # the loop has two phases, the initial monotonic phase, and the phase where
    # we are actually recording stderrs.
    while monotonic or (percent_err > tolerance):

        # print("new std: ", new_std)
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
            # print("stderr: ", stderr)
            avg = np.mean(stds_array)
            percent_err = stderr / avg

    return (avg, np.std(stds_array))


@numba.njit(fastmath=True)
def find_coh_length(w, sigma, measured_sigma=-1., tolerance=.01, nrounds=1000):
    # if not passed a measured_sigma (final sigma of RR), find it:
    measured_sigma_std = 0
    if measured_sigma == -1.:
        measured_sigma, measured_sigma_std = find_final_std(w, sigma)
    # now do a bunch of random walks and find the coherence length for each one
    coherence_lengths = []
    # initialize stderr to a large value so we enter the loop the first time
    percent_err = tolerance + 1
    while percent_err > tolerance:
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
            coh_len_array = np.array(coherence_lengths)
            stderr = np.std(coh_len_array) / np.sqrt(len(coherence_lengths))
            percent_err = stderr / np.mean(coh_len_array)
            # print("coherence_length: ", counter)
            # print("stderr: ", stderr)
            # print("percent error: ", percent_err)
    avg = np.mean(np.array(coherence_lengths))
    std_dev = np.std(np.array(coherence_lengths))
    return measured_sigma, measured_sigma_std, avg, std_dev

@numba.njit(parallel=True)
def search_coh_lengths():
    # first, define the range of w and sigma to search
    ws = np.linspace(-3, -1, 10)
    ws = 1 - np.power(10, ws)
    sigmas = np.linspace(.001, 10, 10)

    results = np.zeros((len(ws), len(sigmas), 4))
    for i in numba.prange(len(ws)):
        for j in numba.prange(len(sigmas)):
            print("w: ", ws[i])
            print("sigma: ", sigmas[j])
            results[i][j] = find_coh_length(ws[i], sigmas[j])

    return ws, sigmas, results


def plot_properties(ws, sigmas, a):
    plt.figure(1)
    for i in range(len(ws)):
        plt.errorbar(ws, a[:, i, 0], a[:, i, 1], label="sigma="+str(sigmas[i]))
    plt.legend()
    plt.xlabel("W")
    plt.ylabel("Final Sigma")

    plt.figure(2)
    for i in range(len(ws)):
        plt.errorbar(ws, a[:, i, 2], a[:, i, 3], label="sigma="+str(sigmas[i]))
    plt.legend()
    plt.xlabel("W")
    plt.ylabel("Coherence Length")

    plt.figure(3)
    for i in range(len(sigmas)):
        plt.errorbar(sigmas, a[i, :,  0], a[i, :,  1], label="w="+str(ws[i]))

    plt.legend()
    plt.xlabel("Sigma")
    plt.ylabel("Final Sigma")
