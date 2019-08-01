#!/usr/bin/python
# This module is the axion generator for the axion event generator MC.
# Written by Matthew Lawson in 2019 at Stockholm University
# mmlawson@ucdavis.edu
import os
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
                 velocities_file=None,
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
        # coherence length, in km
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

        # load the average axion wind data. Its in a npz archive
        # This is the default file
        if velocities_file is None:
            path = os.path.dirname(__file__) + "/axion_wind_sparse.npz"
        else:
            path = velocities_file
        archive = np.load(path)
        # sort the array names to make absolutely sure there is no funny
        # business
        archive.files.sort()
        # unpack the arrays
        t, v_wind, xhat, yhat, zhat = [archive[i] for i in archive.files]
        # times at which v_wind was computed, in unix time (Seconds since
        # 1 Jan 1970 UTC)
        self.t_raw = t
        # unit vectors in the experiment frame, unitless
        self.xhat = interp1d(self.t_raw, xhat)
        self.yhat = interp1d(self.t_raw, yhat)
        self.zhat = interp1d(self.t_raw, zhat)
        # velocity of the DM at experiment, on average, in km/sec
        self.v_wind = interp1d(self.t_raw, v_wind)

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

    def do_fast_axion_sim(self,
                          start_t,
                          end_t,
                          sampling_rate,
                          sensitive_axes=0,
                          axion_wind=True,
                          random_amp=True,
                          random_vel=True,
                          random_phase=True,
                          debug=False):
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
        if sensitive_axes == 0 or sensitive_axes == 3:
            xhat = self.xhat(t)
            yhat = self.yhat(t)
        else:
            xhat = np.array([[]])
            yhat = np.array([[]])

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
            xhat,
            yhat,
            zhat,
            t,
            sampling_rate,
            self.frequency,
            self.coupling,
            self.coh_time,
            self.coh_length,
            sensitive_axes=sensitive_axes,
            axion_wind=axion_wind,
            random_amp=random_amp,
            random_vel=random_vel,
            random_phase=random_phase,
            debug=debug,
        )
        stp = time.time()
        if debug:
            print(stp - strt)
        return t, r

    def do_sim(self,
               days=.01,
               debug=False,
               rayleigh_amp=True,
               compute_wind=True):
        """A convenience function for doing simulations from the beginning
        of the axion wind data for a number of days"""
        start = self.t_raw[0]
        end = start + 60 * 60 * 24 * days
        r = self.do_fast_axion_sim(start,
                                   end,
                                   self.frequency * 2.5,
                                   debug=debug,
                                   rayleigh_amp=rayleigh_amp,
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
                  x_hat,
                  y_hat,
                  z_hat,
                  t,
                  sampling_rate,
                  frequency,
                  coupling,
                  coh_time,
                  coh_length,
                  sensitive_axes=0,
                  axion_wind=True,
                  random_amp=True,
                  random_vel=True,
                  random_phase=True,
                  debug=True):
    """
    This inner loop combines several tasks into one optimized loop, so that we
    only have to run one iteration.
    """
    if debug:
        phases = np.zeros(n)
        vels = np.zeros((n, 3))
        amps = np.zeros(n)
        winds = np.zeros(n)
    # if sensitive_axes == 0 we want to keep the full vector output of the axion
    # wind simulation.
    if sensitive_axes == 0:
        axion_y = np.zeros(n)
        axion_z = np.zeros(n)
    # otherwise a simple scalar will do. We will use axion_x in either the
    # scalar case (as the only asnwer) or the vector case (as the x component)
    axion_x = np.zeros(n)
    # variables to hold the last phase, velocity, and amplitude (the things
    # being random-walked
    phase = phase0
    vel = v0
    # if we are calcuating the wind, do the first point
    if axion_wind:
        total_wind = v_wind.T[0] + vel
        if sensitive_axes == 0:
            wind_vect = np.array([
                x_hat.T[0].dot(total_wind), y_hat.T[0].dot(total_wind),
                z_hat.T[0].dot(total_wind)
            ])
        elif sensitive_axes == 1:
            wind = z_hat.T[0].dot(total_wind)
        elif sensitive_axes == 2:
            # an optimized form for the magnitude cross product
            v = total_wind
            z = z_hat.T[0]
            wind = np.sqrt(z.dot(z) * v.dot(v) - (v.dot(z))**2)
        elif sensitive_axes == 3:
            wind = np.linalg.norm(total_wind)
    else:
        # if not computing the wind, the wind strength is the speed of light
        # here given in km/sec
        wind = 3e5
        random_vel = False

    if (not random_vel) and axion_wind:
        # if we are an axion wind experiment but are ignoring the stocastic
        # component (for computational efficiency or debugging) just set the
        # random component to zero for all time
        vel = np.array([0., 0., 0.])

    amp = a0
    # in the case where we are resolving the full 3d axion velocity, we have
    # to compute each velocity component seperately to keep the numba typing
    # system happy
    if sensitive_axes == 0:
        axion_no_wind = coupling * np.abs(amp) * np.sin(frequency * t[0] +
                                                        phase)
        axion_x[0] = wind_vect[0] * axion_no_wind
        axion_y[0] = wind_vect[1] * axion_no_wind
        axion_z[0] = wind_vect[2] * axion_no_wind

    else:
        axion_x[0] = wind * coupling * np.abs(amp) * np.sin(frequency * t[0] +
                                                            phase)

    # do a modified random walk, which penalizes deviations from the mean
    for i in range(1, n):
        # the axion wind speed, for computing the effective coherence time
        if axion_wind:
            v_wind_mag = np.sqrt((v_wind.T[i]).dot(v_wind.T[i]))
        # calculate the effective coherence time, the coherence time when taking
        # into account velocity through the halo
        effective_coh_time = 1 / (1 / coh_time + v_wind_mag / coh_length)
        if axion_wind:
            # calculate the weight and sigma for the velocity weighted random
            # walk from the
            # standard deviation and coherence time of the velocity
            if random_vel:
                w, sigma = get_rr_properties(effective_coh_time, vel_rr_std,
                                             "velocity")
                vel = (vel * w +
                       np.random.randn(3) * sigma * np.array([1, 1, 1]))
            # get the component of the wind along the sensitive axis/axes of the
            # experiment
            total_wind = v_wind.T[i] + vel
            if sensitive_axes == 0:
                wind_vect = np.array([
                    x_hat.T[i].dot(total_wind), y_hat.T[i].dot(total_wind),
                    z_hat.T[i].dot(total_wind)
                ])
            elif sensitive_axes == 1:
                wind = z_hat.T[i].dot(total_wind)
            elif sensitive_axes == 2:
                # an optimized form for the magnitude cross product
                v = total_wind
                z = z_hat.T[i]
                wind = np.sqrt(z.dot(z) * v.dot(v) - (v.dot(z))**2)
            elif sensitive_axes == 3:
                wind = np.linalg.norm(total_wind)
        # the amplitude random walk is a random-walk in the complex plane,
        # we do similar calcuations to get it's properties

        if random_amp:
            w, sigma = get_rr_properties(effective_coh_time, a_rr_std,
                                         "amplitude")
            amp = (amp * w +
                   (np.random.randn() + np.random.randn() * 1j) * sigma)
        # the phase random walk
        if random_phase:
            # compute the time fraction (based on the effective coherence time)
            time_fraction = 1 / (effective_coh_time * sampling_rate)
            root_time_fraction = np.sqrt(time_fraction)
            phase += phase_rr_std * root_time_fraction * np.random.randn()

        # in the case where we are resolving the full 3d axion velocity, we have
        # to compute each velocity component seperately to keep the numba typing
        # system happy
        if sensitive_axes == 0:
            axion_no_wind = coupling * np.abs(amp) * np.sin(frequency * t[i] +
                                                            phase)
            axion_x[i] = wind_vect[0] * axion_no_wind
            axion_y[i] = wind_vect[1] * axion_no_wind
            axion_z[i] = wind_vect[2] * axion_no_wind

        else:
            axion_x[i] = wind * coupling * np.abs(amp) * np.sin(frequency *
                                                                t[i] + phase)
        if debug:
            winds[i] = wind
            amps[i] = np.abs(amp)
            phases[i] = phase
            vels[i] = vel

    return axion_x, axion_y, axion_z, phases, vels, amps, winds


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
    return r


if __name__ == "__main__":
    main()
