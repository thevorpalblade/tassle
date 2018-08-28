import numpy as np
import scipy as sp
from scipy import constants as const
import itertools

from matplotlib import pyplot as plt


class Axion:
    def __init__(self, mass=1e-12, coupling=1, linewidth=1e-6):
        # mass in eV
        self.mass = mass
        # our signal is multiplied by the coupling
        self.coupling = coupling
        # as a fraction, so a linewidth of 1e-6 is a 1Hz linewidth at 1MHz
        self.linewidth = linewidth
        # in seconds, 1/sampling rate (the dwell time, for NMR fans)
        # self.step_size = step_size
        # the axion mass, in eV, converted to a frequency in Hz
        self.frequency = mass / const.value("Planck constant in eV s")
        # our coherence time (1/ the real linewidth)
        self.coherence_time = 1 / (linewidth * self.frequency)

        self.phase_value = 0
        # the standard deviation of the distribution the phase change is drawn
        # from, when normalized by
        self.phase_std = 1 / np.sqrt(2)


    def get_next_phase(self, timestep, n=1):
        """
        computes the phase evolution of the axion according to it's coherence
        time. If n=1 (the default) takes one timestep, and returns the new phase.
        If n >= 2, return the array of all the phase values along the way
        as well. Etiher way, self.phase_value is updated. n should be an
        integer greater than 0.
        """

        # compute the time fraction, relevant for deciding the width of the
        # random distribution we pull from for the phase deltas
        time_fraction = timestep/self.coherence_time
        # efficiently calculate a bunch of phase deltas
        phase_deltas = (self.phase_std * np.sqrt(time_fraction)
                        * np.random.randn(n))

        # for convenience, add the old phase value to the first phase delta
        phase_deltas[0] = phase_deltas[0] + self.phase_value
        # now we can compute all the phases along the way efficiently
        phases = np.cumsum(phase_deltas)
        # and the new phase is just the sum of all the phase deltas
        # which is also the last element of the cumulative sum
        self.phase_value = phases[-1]

        if n == 1:
            return self.phase_value
        else:
            return phases

    def get_pure_axion(self, timestep=None, n=None):
        """
        returns x and y for a pure (no noise) axion at given timesteps, for a
        given number of samples (n).
        Returns x and y for the axion intesity.
        """
        if timestep is None:
            timestep = 0.1/self.frequency
        if n is None:
            n = int(self.coherence_time * 2 / timestep)

        x = np.linspace(0, n*timestep, n)
        phases = self.get_next_phase(timestep, len(x))
        axion = np.sin(self.frequency * x + phases)
        return (x, axion)

