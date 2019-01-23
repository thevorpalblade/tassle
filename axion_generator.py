# This module is the axion generator for the CASPEr-wind MC.

import itertools
import pickle
import time
import timeit
from itertools import accumulate

import numba
import numpy as np
from matplotlib import pyplot as plt
from numba import njit
from scipy import constants as const
from scipy.interpolate import interp1d


class Axion:
    def __init__(
            self,
            mass=1e-12,
            coupling=1,
            velocities_file="axion_wind.pkl",
            sampling_rate=800,
            sampling_time=60 * 60 * 24 * 10,
    ):
        # mass in eV
        self.mass = mass
        # our signal is multiplied by the coupling
        self.coupling = float(coupling)
        # in seconds, 1/sampling rate (the dwell time, for NMR fans)
        # self.step_size = step_size
        # the axion mass, in eV, converted to a frequency in Hz
        self.frequency = mass / const.value("Planck constant in eV s")
        # our coherence time (1/ the real linewidth)
        # self.coherence_time = 1 / (linewidth * self.frequency)
        self.coherence_time = 40e-6 * 100e-6 / self.mass
        # placeholder value, in km
        # TODO: this is just a placeholder value
        # self.coh_length = 10 * self.coherence_time
        self.coh_length = 6.2 * 100e-6 / self.mass

        # width of 1d velocity distribution in km/sec
        # TODO real value here
        self.v_std = 200
        self.phase_value = 0
        # the standard deviation of the distribution the phase change is drawn
        # from, when normalized by timestep
        self.phase_std = 1 / np.sqrt(2)
        # velocity distribution width to draw from for velocity random walk
        # (may be different from the width of the maxwell velocity
        # distribution) TODO: check this
        self.vel_rr_std = self.v_std / np.sqrt(2)

        # the random axion velocity
        self.v0 = np.array([
            self.v_std * np.random.randn(),
            self.v_std * np.random.randn(),
            self.v_std * np.random.randn(),
        ])

        # load the average axion wind data
        with open(velocities_file, "rb") as wind_file:
            t, xhat, yhat, zhat, v_wind = pickle.load(wind_file)
        # times at which v_wind was computed, in unix time (Seconds since
        # 1 Jan 1970 UTC)
        self.t_raw = t.value
        # unit vectors in the CASPEr frame, unitless
        self.xhat = interp1d(self.t_raw, np.float64(xhat.xyz.value))
        self.yhat = interp1d(self.t_raw, np.float64(yhat.xyz.value))
        self.zhat = interp1d(self.t_raw, np.float64(zhat.xyz.value))
        # velocity of the DM at CASPEr, on average, in km/sec
        self.v_wind = interp1d(self.t_raw, np.float64(v_wind.d_xyz.value))

    def gen_phases(self, root_time_fractions, n):
        """
        computes the phase evolution of the axion according to it's coherence
        time. The array of all the phase values for n timesteps is returned.
        n should be greater than one. Timestep is assumed to be in seconds.
        """

        # efficiently calculate a bunch of phase deltas
        phase_deltas = self.phase_std * root_time_fractions * np.random.randn(
            n)

        # for convenience, add the old phase value to the first phase delta
        phase_deltas[0] = phase_deltas[0] + self.phase_value
        # now we can compute all the phases along the way efficiently
        phases = np.cumsum(phase_deltas)

        return phases

    def gen_vels(self, time_fractions, n, w=0.001):
        """
        Generate the velocities random walk! :D
        Need to choose a good value for w, which specifies the width of the
        valley the random walk occurs in.

        """
        # compute the velocity steps
        vdelt = np.array(
            self.vel_rr_std * np.column_stack([np.sqrt(time_fractions)] * 3) *
            np.random.randn(n, 3),
            dtype=np.float64,
        )

        # weighted_sum = lambda v0, v: v0 * (1 - w) + v
        # uf_weighted_sum = np.frompyfunc(weighted_sum, 2, 1)
        # do a modified random walk, which penalizes deviations from the mean
        for i, vdi in enumerate(vdelt):
            if i == 0:
                vdelt[0][0] = self.vx
                vdelt[0][1] = self.vy
                vdelt[0][2] = self.vz
            else:
                vdelt[i] += vdelt[i - 1] * (1 - w)
        return vdelt.T

        # vs = uf_weighted_sum.accumulate(vdelt, axis=0, dtype=np.object).astype(np.float64)
        # vs = np.array(list(accumulate(vdelt, lambda v0, v: v0 * (1 - w) + v)))
        # return vs.T

    def get_pure_axion(self, v_wind, vels, phases, t):
        """
        returns x and y for a pure (no noise) axion at given timesteps, for a
        given number of samples (n).
        Returns x and y for the axion intesity.
        """
        v = v_wind + vels
        wind = np.sum(
            v * self.xhat(t), axis=0) + np.sum(
                v * self.yhat(t), axis=0)
        axion = self.coupling * wind * np.sin(self.frequency * t + phases)
        return axion

    def do_axion_sim(self, start_t, end_t, sampling_rate):
        """
        """
        # sanity check
        assert start_t >= self.t_raw[0]
        assert end_t <= self.t_raw[-1]

        # Define the time-step, number of samples, and time points.
        # Start and stop time taken from the DM Halo velocity data
        sampling_time = end_t - start_t

        n = int(sampling_time * sampling_rate)

        print("Generating time points")
        t = np.linspace(start_t, end_t, n)

        print("Calculating axion wind magnitudes")
        strt = time.time()
        v_wind = self.v_wind(t)
        v_wind_mag = np.sqrt(np.sum(v_wind**2, axis=0))
        stp = time.time()
        print(stp - strt)
        print("Calculating effective coherence times")
        # effective coherence times based on DM halo velocities
        coherence_times = 1 / (
            1 / self.coherence_time + v_wind_mag / self.coh_length)
        # compute the time fractions, relevant for deciding the width of the
        # random distribution we pull from for the phase/velocity deltas
        root_time_fractions = np.sqrt(1 / (sampling_rate * coherence_times))

        print("Calculating phases")
        strt = time.time()
        phases = self.gen_phases(root_time_fractions, n)
        stp = time.time()
        print(stp - strt)

        print("Calculating Velocities")
        strt = time.time()
        vels = gen_vels(self.vel_rr_std, root_time_fractions, self.v0, n)
        stp = time.time()
        print(stp - strt)
        print("Calculating axion field values")

        strt = time.time()
        result = self.get_pure_axion(v_wind, vels, phases, t)
        # result = get_pure_axion(vels, v_wind, phases, xhat, yhat,
        #                        self.coupling, self.frequency, t)
        stp = time.time()
        print(stp - strt)
        return t, result

    def plot_psd(self, t, axion, sampling_rate):
        f = np.fft.rfftfreq(len(t), 1 / sampling_rate)
        p0 = abs(np.fft.rfft(axion))**2
        plt.plot(f, p0)
        return f, p0


@njit(cache=True)
def gen_vels(vel_rr_std, root_time_fractions, v0, n, w=0.001):
    """
    Generate the velocities random walk! :D
    Need to choose a good value for w, which specifies the width of the
    valley the random walk occurs in.

    """
    # compute the velocity steps
    vdelt = (vel_rr_std * np.column_stack(
        (root_time_fractions, root_time_fractions, root_time_fractions)) *
             np.random.randn(n, 3))

    # do a modified random walk, which penalizes deviations from the mean
    for i in range(len(vdelt)):
        if i == 0:
            vdelt[0] = v0
        else:
            vdelt[i] += vdelt[i - 1] * (1 - w)
    return vdelt.T


@njit(cache=True)
def get_pure_axion(vels, v_wind, phases, xhat, yhat, coupling, frequency, t):
    """
    returns x and y for a pure (no noise) axion at given timesteps, for a
    given number of samples (n).
    Returns x and y for the axion intesity.
    """
    axion = np.empty_like(t, dtype=np.float64)
    for i in range(len(t)):
        v = v_wind[:, i] + vels[:, i]
        wind = np.sum(v * (xhat[:, i] + yhat[:, i]))
        axion[i] = coupling * wind * np.sin(frequency * t[i] + phases[i])

    return axion


def main(days=1 / 24):
    a = Axion()
    start = a.t_raw[0]
    end = start + 60 * 60 * 24 * days
    r = a.do_axion_sim(start, end, a.frequency * 2.5)
    return r


if __name__ == "__main__":
    main()