import tassle
import time
import numpy as np
from scipy.signal import welch
from multiprocessing import Pool

def main(n=100, processes=4, days=0.5):
    # initialize results array
    print("initializing with practice run")
    t_start = time.time()
    f, fft, psd = job(days)
    psd_f, psd_m = psd

    p = Pool(processes)
    for i in range(n):
        t = time.time()
        print("run number ", n)
        results = p.map(job, days * np.ones(processes))
        for j in results:
            fft += j[1]
            psd_m += j[2][1]
        print("finished in ", time.time() - t, " seconds")
    psd_m = psd_m / (n * processes)
    fft = fft / (n * processes)
    print("finished all in ", t_start - time.time(), "seconds")

    return [f, fft, psd_f, psd_m]


def job(days):
    a = tassle.Axion()
    t, r = a.do_sim(days)
    fft = np.fft.rfft(r[0])
    f = np.fft.rfftfreq(len(r[0]), 1 / (5 * a.frequency))
    psd = welch(r[2], t[1] - t[0], nperseg=2**25)
    return f, fft, psd
