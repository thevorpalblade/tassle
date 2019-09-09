from itertools import accumulate

import numba
import numpy as np
from matplotlib import pyplot as plt
from scipy.optimize import curve_fit
from scipy import stats


def do_rw(n=1000000, w=0.1, sigma=1):
    deltas = sigma * np.random.randn(n)

    x = np.array(list(accumulate(deltas, lambda x0, xi: x0 + xi - w * x0)))
    return x


@numba.njit(fastmath=True)
def binned_mode(x, bins=100):
    his = np.histogram(np.abs(x), bins)
    max_idx = np.argmax(his[0])
    return np.mean(np.array([his[1][max_idx], his[1][max_idx + 1]]))


@numba.njit(fastmath=True)
def do_numba_rw(n=1000, w=0.99, sigma=1.0, init_sigma=0):
    for i in range(n):
        if i == 0:
            val = (np.random.randn() + np.random.randn() * 1j) * init_sigma
        else:
            val = val * w + (np.random.randn() + np.random.randn() * 1j) * sigma
    return val


@numba.njit(fastmath=True, parallel=True)
def do_rw_ensemble(n=1000, w=0.99, sigma=1.0, init_sigma=0, nrounds=1000):
    rvals = np.zeros(nrounds)
    ivals = np.zeros(nrounds)
    for i in numba.prange(nrounds):
        val = do_numba_rw(n, w, sigma, init_sigma)
        rvals[i] = np.real(val)
        ivals[i] = np.imag(val)
    return ivals * 1j + rvals


@numba.njit(fastmath=True)
def do_numba_full_rw(n=1000, w=0.99, sigma=1.0, init_sigma=0):
    vals = np.zeros(n)
    for i in range(n):
        if i == 0:
            val = (np.random.randn() + np.random.randn() * 1j) * init_sigma
        else:
            val = val * w + (np.random.randn() + np.random.randn() * 1j) * sigma
            vals[i] = np.abs(val)
    return vals


@numba.njit(fastmath=True, parallel=True)
def do_avg_full_rw(n=1000, w=0.99, sigma=1.0, init_sigma=0, nrounds=1000):
    # initialize a empty array
    rw = np.zeros(n)
    for i in numba.prange(nrounds):
        # do a bunch of random walks and average them all
        rw += do_numba_full_rw(n, w, sigma)
    rw = rw / nrounds
    return rw


# a fit function
def exp_rec(x, A, tau):
    return A * (1 - np.exp(-((x)/tau)))**(1/2)


def find_recovery_time(w=0.99, sigma=1.0, iters=1000, n_init=1000):
    rw = do_avg_full_rw(n_init, w, sigma, nrounds=iters)

    x = np.linspace(0, n_init, n_init)
    # get the exponential time constant from the recovery curve
    try:
        tau = curve_fit(exp_rec, x, rw)[0][-1]
    except RuntimeError:
        # probably means that the curve isn't smooth enough, let's average more
        print("caught exception!")
        return find_recovery_time(w, sigma, iters=iters*10, n_init=n_init)

    if tau * 3 > n_init:
        print("tau is ", tau, " and n_init is ", n_init, ". recursing")
        return find_recovery_time(w, sigma, iters, n_init=int(tau*5))
    else:
        #plt.plot(x, rw)
        print("done")
        return tau

@numba.jit
def find_final_sigma(w, sigma,  n=1000, nrounds=1000):
    # first make sure we have the full recovery arc
    tau = find_recovery_time(w, sigma)

    rws = do_rw_ensemble(n=tau*4, w=w, sigma=sigma, nrounds=nrounds)
    #rwsi, rwsr = do_rw_ensemble(n=tau*4, w=w, sigma=sigma, nrounds=nrounds)
    # standard deviation of just the real or imaginary part is the same as the
    # median of the absolute value (the shape parameter of the rayleigh dist)
    rwsvals = np.concatenate((rws.real, rws.imag))
    sigma = np.std(rwsvals)
    stderr = sigma / np.sqrt(len(rwsvals))

    return sigma, stderr

@numba.jit(fastmath=True)
def find_coh_length(w, sigma, measured_sigma=-1., tolerance=.01, nrounds=1000):
    # if not passed a measured_sigma (final sigma of RR), find it:
    measured_sigma_std = 0
    if measured_sigma == -1.:
        measured_sigma, measured_sigma_std = find_final_sigma(w, sigma)
    # now do a bunch of random walks and find the coherence length for each one
    coherence_lengths = []
    # initialize stderr to a large value so we enter the loop the first time
    percent_err = tolerance + 1
    while percent_err > tolerance:
        # initialize a random walk
        init_val = measured_sigma * (np.random.randn() + np.random.randn() * 1j)
        val = init_val
        # keeps track of how far the random walk goes
        counter = 0
        # continue the random walk until we have wandered sigma/e (the threshold
        # for "coherence" for a unweighted random walk. Assuming this is a
        # resonable estimate for a weighted random walk may be bad, but life is
        # short.
        while np.abs(init_val - val) < measured_sigma / np.e:
            val = val * w + (np.random.randn() + np.random.randn() * 1j) * sigma
            counter += 1
        coherence_lengths.append(counter)
        if len(coherence_lengths) > 5:
            coh_len_array = np.array(coherence_lengths)
            stderr = np.std(coh_len_array) / np.sqrt(len(coherence_lengths))
            percent_err = stderr / np.mean(coh_len_array)
            # print("coherence_length: ", counter)
            # print("stderr: ", stderr)
            # print("percent error: ", percent_err)
    co_len_ary = np.array(coherence_lengths)
    avg = np.mean(co_len_ary)
    std_dev = np.std(co_len_ary)
    return measured_sigma, measured_sigma_std, avg, std_dev / np.sqrt(len(co_len_ary))


@numba.jit(parallel=True)
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
        plt.errorbar(
            ws, a[:, i, 0], a[:, i, 1],
            label="sigma=" + str(sigmas[i]),
            fmt=".")
    plt.legend()
    plt.xlabel("W")
    plt.ylabel("Final Sigma")

    plt.figure(2)
    for i in range(len(ws)):
        plt.errorbar(
            ws, a[:, i, 2], a[:, i, 3], 
            fmt=".",
            label="sigma=" + str(sigmas[i]))
    plt.legend()
    plt.xlabel("W")
    plt.ylabel("Coherence Length")

    plt.figure(3)
    for i in range(len(sigmas)):
        plt.errorbar(sigmas, a[i, :, 0], a[i, :, 1],
                     fmt=".",
                     label="w=" + str(ws[i]))

    plt.legend()
    plt.xlabel("Sigma")
    plt.ylabel("Final Sigma")


def fitting(ws, sigmas, results):
    # relationship between the weights, w, and the measured sigmas:
    def final_sigma_fit(w, amplitude):
        return amplitude / (1 - w)**0.5

    # fit all the meas_sigma vs ws curves to the above form, and extract
    # the proportionality constant in each case
    amps = [
        curve_fit(final_sigma_fit, ws, results[:, i, 0])[0][0]
        for i in range(len(ws))
    ]
    # plot this to check
    plt.figure(1)
    curves = [plt.plot(ws, final_sigma_fit(ws, i)) for i in amps]

    # the result is linear in the initial sigmas:
    magic_const = np.polyfit(sigmas, amps, 1)
    print("the magic constant is: ", magic_const)
    plt.figure(4)
    plt.plot(sigmas, np.poly1d(magic_const)(sigmas))
    plt.plot(sigmas, amps, ">")
    plt.xlabel("sigmas")
    plt.ylabel("C1")

    # the magic constant C1 is 0.7004203389745852, so the final relationship is:
    # final_sigma = 0.7004203389745852 * init_sigma / (1 - w)**0.5
    plt.figure(3)
    for i in ws:
        plt.plot(sigmas, 0.7004203389745852 * sigmas / (1 - i)**0.5)

    # OK, so now lets compute the relationship between w, sigma, and
    # coherence time
    # Based on the plots, it seems like there's no sigma dependence.
    #
    def coht_fit(w, C2, y0):
        return y0 + C2 / (1 - w)
    p = [curve_fit(coht_fit, ws, results[:, i, 2])[0] for i in range(len(ws))]
    p = np.array(p)
    plt.figure(2)
    curves = [plt.plot(ws, coht_fit(ws, *i)) for i in p]
    print(np.mean(p, axis=0))

    # we get C2 = 0.03801156, and y0 = 1.29206731, so
    # coh_time = 1.29206731 + 0.03801156 / (1 - w)
    # although I remain sceptical of our definition of coherence time here.
    
    # lets invert!
    # (1 - w)t = y0(1-w) + C2
    # (1-w)(t-y0) = C2
    # w-1 = -C2/(t - y0)
    ### w = 1 - C2/(t - y0)

    # sigma_f = C1 * sigma / sqrt(1-w)
    # sigma = sigma_f * sqrt(1-w) / C1
    #### sigma = sigma_f * np.sqrt(C2/(t - y0)) / C1
