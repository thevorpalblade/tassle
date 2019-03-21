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
            velocities_file="axion_wind_sparse.pkl",
            sampling_rate=800,
            sampling_time=60 * 60 * 24 * 10,
            phase0=None
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
        self.coh_time = 40e-6 * 100e-6 / self.mass
        # placeholder value, in km
        # TODO: this is just a placeholder value
        # self.coh_length = 10 * self.coherence_time
        self.coh_length = 6.2 * 100e-6 / self.mass

        # width of 1d velocity distribution in km/sec
        # TODO real value here
        self.v_std = 200
        if phase0 is None:
            self.phase0 = 2 * np.pi * np.random.random()
        # the standard deviation of the distribution the phase change is drawn
        # from, when normalized by timestep
        self.phase_rr_std = 1 / np.sqrt(2)
        # velocity distribution width to draw from for velocity random walk
        # (may be different from the width of the maxwell velocity
        # distribution) TODO: check this
        self.vel_rr_std = self.v_std / np.sqrt(2)
        self.a_rr_std = 1.
        self.a0 = 1

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
        phase_deltas = self.phase_rr_std * root_time_fractions * np.random.randn(
            n)

        # for convenience, add the old phase value to the first phase delta
        phase_deltas[0] = phase_deltas[0] + self.phase0
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
        import ipdb; ipdb.set_trace()
        return vdelt.T

        # vs = uf_weighted_sum.accumulate(vdelt, axis=0, dtype=np.object).astype(np.float64)
        # vs = np.array(list(accumulate(vdelt, lambda v0, v: v0 * (1 - w) + v)))
        # return vs.T

    def get_pure_axion(self, v_wind, vels, phases, t, debug=False):
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
        if debug:
            plt.subplot(211)
            plt.plot(t, phases)
            plt.subplot(212)
            axion = np.sin(self.frequency * t + phases)
            axion_nophase = np.sin(self.frequency * t + self.phase_value)
            plt.plot(t, (axion + axion_nophase) ** 2)
            plt.show()
        return axion

    def do_fast_axion_sim(self,  start_t, end_t, sampling_rate):
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

        print("Calculating interpolated quantities")
        strt = time.time()
        v_wind = self.v_wind(t)
        zhat = self.zhat(t)

        stp = time.time()
        print(stp - strt)

        print("doing heavy lifting")

        strt = time.time()
        r = heavy_lifting(self.vel_rr_std, self.v0, self.a_rr_std,
                          self.a0, self.phase_rr_std, self.phase0,
                          n, v_wind, zhat, t, sampling_rate,
                          self.frequency, self.coupling,
                          self.coh_time, self.coh_length, w=0.001)
        stp = time.time()
        print(stp - strt)
        return t, r

    def do_axion_sim(self, start_t, end_t, sampling_rate, debug=False):
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
        coh_times = 1 / (
            1 / self.coh_time + v_wind_mag / self.coh_length)
        # compute the time fractions, relevant for deciding the width of the
        # random distribution we pull from for the phase/velocity deltas
        root_time_fractions = np.sqrt(1 / (sampling_rate * coh_times))

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
        if debug:
            result = self.get_pure_axion(v_wind, vels, phases, t, True)
        else:
            xhat = self.xhat(t)
            yhat = self.yhat(t)
            result = get_pure_axion(vels, v_wind, phases, xhat, yhat,
                                    self.coupling, self.frequency, t)
        stp = time.time()
        print(stp - strt)
        return t, result

    def plot_psd(self, t, axion, sampling_rate):
        f = np.fft.rfftfreq(len(t), 1 / sampling_rate)
        p0 = abs(np.fft.rfft(axion))**2
        plt.plot(f, p0)
        return f, p0


@njit(cache=True, fastmath=True)
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


#@njit(cache=True, fastmath=True)
def heavy_lifting(vel_rr_std, v0, a_rr_std, a0, phase_rr_std, phase0,
                  n, v_wind, z_hat, t, sampling_rate,
                  frequency, coupling,
                  coh_time, coh_length, w=0.001):
    """
    This inner loop combines several tasks into one optimized loop, so that we
    only have to run one iteration.
    """
    # the axion array
    axion = np.zeros(n)
    # variables to hold the last phase, velocity, and amplitude (the things
    # being random-walked
    phase = phase0
    vel = v0
    amp = a0
    v_wind_mag = np.sqrt((v_wind.T[0]).dot(v_wind.T[0]))
    # the time fraction gives the width of the distribution to draw from,
    # taking into account the velocity at this point and the sampling rate.
    # (you have to take the square root to get the real width though)
    time_fraction = (1/coh_time + v_wind_mag / coh_length) / sampling_rate
    # The dad correction factor: as the random walk progresses, we want to limit
    # it's eventual size to a steady state. So keep track of the cumulative draw
    # width and divide it back out. 
    total_width = time_fraction
    # calculate the first axion point
    a = v_wind.T[0] + vel
    b = z_hat.T[0]
    wind = np.sqrt(a.dot(a) * b.dot(b) - (a.dot(b)) ** 2)
    axion[0] = wind * coupling * amp * np.sin(frequency * t[0] + phase)

    # for debug purposes, record the phasres, vels, and amplitudes
    phases = np.zeros(n)
    amps = np.zeros(n)
    vels = np.zeros((n, 3))

    phases[0] = phase
    amps[0] = amp
    vels[0] = vel

    # do a modified random walk, which penalizes deviations from the mean
    for i in range(1, n):
        # compute the time fraction (based on the effective coherence time)
        v_wind_mag = np.sqrt((v_wind.T[i]).dot(v_wind.T[i]))
        time_fraction = (1/coh_time + v_wind_mag / coh_length) / sampling_rate
        total_width += time_fraction
        root_time_fraction = np.sqrt(time_fraction)
        root_total_width = np.sqrt(total_width)
        # the velocity random walk
        vel = (vel + (np.random.randn(3)
                      * root_time_fraction
                      * np.array([1, 1, 1]))) / root_total_width
        # the amplitude random walk
        amp = (amp + a_rr_std * np.random.randn() * root_time_fraction) / root_total_width
        # the phase random walk
        phase += phase_rr_std * root_time_fraction * np.random.randn()

        # an optimized form for the magnitude cross product
        a = v_wind.T[i] + vel
        b = z_hat.T[i]
        wind = np.sqrt(a.dot(a) * b.dot(b) - (a.dot(b)) ** 2)

        axion[i] = wind * coupling * amp * np.sin(frequency * t[i] + phase)

        # store these values for debugging
        phases[i] = phase
        amps[i] = amp
        vels[i] = vel

    return axion, phases, amps, vels


@njit(cache=True, parallel=True)
def get_pure_axion(vels, v_wind, phases, xhat, yhat, coupling, frequency, t):
    """
    returns x and y for a pure (no noise) axion at given timesteps, for a
    given number of samples (n).
    Returns x and y for the axion intesity.
    """
    axion = np.empty_like(t, dtype=np.float64)
    for i in numba.prange(len(t)):
        v = v_wind.T[i] + vels.T[i]
        wind = np.sum(v * (xhat.T[i] + yhat.T[i]))
        axion[i] = coupling * wind * np.sin(frequency * t[i] + phases[i])

    return axion


def main(days=1, debug=False):
    a = Axion()
    start = a.t_raw[0]
    end = start + 60 * 60 * 24 * days
    r = a.do_fast_axion_sim(start, end, a.frequency * 2.5)
    #r = a.do_axion_sim(start, end, a.frequency * 2.5)
    return r


if __name__ == "__main__":
    main()
