# This module is the axion generator for the CASPEr-wind MC.

import pickle
import time

import numba
import numpy as np
from matplotlib import pyplot as plt
from numba import njit
from scipy import constants as const
from scipy.interpolate import interp1d


class Axion:
    def __init__(self,
                 mass=1e-12,
                 coupling=1,
                 velocities_file="axion_wind_sparse.pkl",
                 phase0=None):
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
        else:
            self.phase0 = phase0
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

    def change_mass(self, mass):
        """set a new axion mass"""
        self.mass = mass
        self.frequency = mass / const.value("Planck constant in eV s")
        self.coh_time = 40e-6 * 100e-6 / self.mass
        self.coh_length = 6.2 * 100e-6 / self.mass

    def change_freq(self, freq):
        """set a new axion frequency"""
        self.frequency = freq
        self.mass = self.frequency * const.value("Planck constant in eV s")
        self.coh_time = 40e-6 * 100e-6 / self.mass
        self.coh_length = 6.2 * 100e-6 / self.mass

    def change_coh(self, coh_ratio=1e6):
        """
        Set the coherence length to something else.
        WARNING: Likel to give unphysical results!
        """
        self.coh_time = coh_ratio / self.frequency
        self.coh_length = self.coh_time * (6.2 / 40e-6)

    def do_fast_axion_sim(self, start_t, end_t, sampling_rate,
                          rayleigh_amp=True, debug=False,
                          compute_wind=True):
        """
        """
        # sanity check
        assert start_t >= self.t_raw[0]
        assert end_t <= self.t_raw[-1]

        # Define the time-step, number of samples, and time points.
        # Start and stop time taken from the DM Halo velocity data
        sampling_time = end_t - start_t

        n = int(sampling_time * sampling_rate)

        if debug:
            print("Generating time points")
        t = np.linspace(start_t, end_t, n)

        if debug:
            print("Calculating interpolated quantities")
        strt = time.time()
        v_wind = self.v_wind(t)
        zhat = self.zhat(t)

        stp = time.time()
        if debug:
            print(stp - strt)
            print("doing heavy lifting")

        strt = time.time()
        r = heavy_lifting(
            self.vel_rr_std,
            self.v0,
            self.a_rr_std,
            self.a0,
            self.phase_rr_std,
            self.phase0,
            n,
            v_wind,
            zhat,
            t,
            sampling_rate,
            self.frequency,
            self.coupling,
            self.coh_time,
            self.coh_length,
            w=0.001,
            rayleigh_amp=rayleigh_amp,
            compute_wind=compute_wind,
            debug=debug,
        )
        stp = time.time()
        if debug:
            print(stp - strt)
        return t, r

    def do_sim(self, days=.01, debug=False, rayleigh_amp=True, compute_wind=True):
        """A convenience function for doing simulations from the beginning
        of the axion wind data for a number of days"""
        start = self.t_raw[0]
        end = start + 60 * 60 * 24 * days
        r = self.do_fast_axion_sim(start, end, self.frequency * 2.5,
                                   debug=debug,  rayleigh_amp=rayleigh_amp,
                                   compute_wind=compute_wind)
        return r


@njit(cache=True, fastmath=True)
def heavy_lifting(vel_rr_std,
                  v0,
                  a_rr_std,
                  a0,
                  phase_rr_std,
                  phase0,
                  n,
                  v_wind,
                  z_hat,
                  t,
                  sampling_rate,
                  frequency,
                  coupling,
                  coh_time,
                  coh_length,
                  w=0.001,
                  rayleigh_amp=True,
                  compute_wind=True,
                  debug=True):
    """
    This inner loop combines several tasks into one optimized loop, so that we
    only have to run one iteration.
    """
    # the axion array
    axion = np.zeros(n)
    if debug:
        phases = np.zeros(n)
        vels = np.zeros((n, 3))
        amps = np.zeros(n)
        winds = np.zeros(n)
        print("wind ", compute_wind)
        print("Rayleigh ", rayleigh_amp)
    # variables to hold the last phase, velocity, and amplitude (the things
    # being random-walked
    phase = phase0
    vel = v0
    # if we are calcuating the wind, do the first point
    v_wind_mag = np.sqrt((v_wind.T[0]).dot(v_wind.T[0]))
    a = v_wind.T[0] + vel
    b = z_hat.T[0]
    wind = np.sqrt(a.dot(a) * b.dot(b) - (a.dot(b))**2)
    if not compute_wind:
        # if not computing the wind, the wind strength is the speed of light
        # here given in km/sec
        wind = 3e5

    amp = a0
    # the time fraction gives the width of the distribution to draw from,
    # taking into account the velocity at this point and the sampling rate.
    # (you have to take the square root to get the real width though)
    time_fraction = (1 / coh_time + v_wind_mag / coh_length) / sampling_rate
    # calculate the first axion point
    axion[0] = wind * coupling * np.abs(amp) * np.sin(frequency * t[0] + phase)

    # do a modified random walk, which penalizes deviations from the mean
    for i in range(1, n):
        # the axion wind speed
        if compute_wind:
            v_wind_mag = np.sqrt((v_wind.T[i]).dot(v_wind.T[i]))

        # compute the time fraction (based on the effective coherence time)
        effective_coh_time = 1 / (1 / coh_time + v_wind_mag / coh_length)
        time_fraction = 1 / (effective_coh_time * sampling_rate)
        #  time_fraction = (
        #      1 / coh_time + v_wind_mag / coh_length) / sampling_rate
        root_time_fraction = np.sqrt(time_fraction)
        if compute_wind:
            # calculate the weight and sigma for the velocity weighted random walk
            # from the
            # standard deviation and coherence time of the velocity

            w, sigma = get_rr_properties(effective_coh_time, vel_rr_std,
                                        "velocity")
            vel = (vel * w + np.random.randn(3) * sigma *
                   #root_time_fraction *
                   np.array([1, 1, 1]))
            # an optimized form for the magnitude cross product
            a = v_wind.T[i] + vel
            b = z_hat.T[i]
            wind = np.sqrt(a.dot(a) * b.dot(b) - (a.dot(b))**2)


        # the amplitude random walk is a random-walk in the complex plane,
        # we do similar calcuations to get it's properties

        w, sigma = get_rr_properties(effective_coh_time, a_rr_std, "amplitude")
        if rayleigh_amp:
            amp = (amp * w + (np.random.randn() + np.random.randn() * 1j) *
                   #root_time_fraction *
                   sigma)
        # the phase random walk
        phase += phase_rr_std * root_time_fraction * np.random.randn()


        axion[i] = wind * coupling * np.abs(amp) * np.sin(frequency * t[i] +
                                                          phase)
        if debug:
            winds[i] = wind
            amps[i] = np.abs(amp)
            phases[i] = phase
            vels[i] = vel

    return axion, phases, vels, amps, winds

@numba.njit
def get_rr_properties(coh_t, std, rr_type):
    if rr_type == "velocity":
        C1 = 0.71105544
        C2 = 0.07758531
        y0 = 2.3798211
    elif rr_type == "amplitude":
        C1 = 0.7004203389745852
        C2 = 0.03801156
        y0 = 1.29206731
    else:
        print(rr_type)
        raise ValueError("rr_type must be either 'velocity' or 'amplitude'")

    w = 1 - C2 / (coh_t - y0)
    sigma = std * np.sqrt(1 - w) / C1
#    print(w, sigma)
    return w, sigma


def main(days=.01, debug=False):
    a = Axion()
    start = a.t_raw[0]
    end = start + 60 * 60 * 24 * days
    r = a.do_fast_axion_sim(start, end, a.frequency * 2.5)
    #r = a.do_axion_sim(start, end, a.frequency * 2.5)
    return r


if __name__ == "__main__":
    main()
