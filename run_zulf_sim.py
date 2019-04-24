# Run a simulation in a CASPEr ZULF scenario

import numpy as np

from axion_generator import Axion


def run_sim(mass, start, stop, sampling_rate):
    """Run sim for single mass.
    """
    axion = Axion(mass=mass)
    return axion.do_fast_axion_sim(start,
                                   stop,
                                   sampling_rate)


def main():
    """Run axion simulations for different masses in the CASPEr-ZULF
       frequency range and setup. Output sims to npz files.
    """
    start = 1554994269  # unix timestamp, fixed for reproducability
    stop = start + 850 * 61  # number of acqs * time between acqs
    sampling_rate = 512.  # Hz

    # Nyquist freq needs to be larger than frequency of J-peaks
    nyquist = sampling_rate / 2 + 1
    assert nyquist > 250

    # Test single mass for now
    mass = 2e-15
    result = run_sim(mass, start, stop, sampling_rate)

    sim_name = 'sim_mass_{:g}_rate_{:g}.npz'.format(mass, sampling_rate)
    np.savez(sim_name, times=result[0], amplitudes=result[1])
    print('saved: {}'.format(sim_name))


if __name__ == '__main__':
    main()
