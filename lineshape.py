import tassle
import time
import multiprocessing
import numpy as np
from scipy.signal import welch
from dask.distributed import as_completed
from dask.distributed import Client
from dask_jobqueue import SLURMCluster


def main(n=100, processes=64, days=1):
    # Set up cluster
    cluster = SLURMCluster(queue='regular',
                           project='tassle',
                           cores=processes,
                           memory="1 TB"
                           )
    cluster.adapt(maximum_jobs=processes - 1)

    client = Client(cluster)


    # initialize results array
    print("initializing with practice run")
    t_start = time.time()
    f, fft, psd = job(days)
    psd_f, psd_m = psd

    # if processes is None:
    #     # account for hyperthreading, we are NOT IO bound
    #     processes = multiprocessing.cpu_count() // 2
    t = time.time()
    result_num = 0
    futures = client.map(job, days * np.ones(n))    
    for future, result in as_completed(futures, with_results=True):
        result_num += 1
        fft += result[1]
        psd_m += result[2][1]
        print("finished job ", result_num, " at ", time.time() - t, " seconds")
    # when all results are gathered and summed, do the averaging
    psd_m = psd_m / n
    fft = fft / n
    print("finished all in ", time.time() - t_start, "seconds")

    np.savez_compressed("lineshape_results.npz", f=f, fft=fft,
                        psd_f=psd_f, psd_m=psd_m)


def job(days):
    a = tassle.Axion()
    t, r = a.do_sim(days)
    fft = np.fft.rfft(r[0])
    f = np.fft.rfftfreq(len(r[0]), 1 / (5 * a.frequency))
    psd = welch(r[2], t[1] - t[0], nperseg=2**26)
    return f, fft, psd


if __name__ == '__main__':
    main()
