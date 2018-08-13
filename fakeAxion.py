import numpy as np
import scipy as sp
from scipy import constants as const
import itertools

from matplotlib import pyplot as plt


class Axion:
    def __init__(self, mass=1e-8, coupling=1, linewidth=1e-6, step_size=1e-6):
        # mass in eV
        self.mass = mass
        # our signal is multiplied by the coupling
        self.coupling = coupling
        # as a fraction, so a linewidth of 1e-6 is a 1Hz linewidth at 1MHz
        self.linewidth = linewidth
        # in seconds, 1/sampling rate (the dwell time, for NMR fans)
        self.step_size = step_size
        # the axion mass, in eV, converted to a frequency in Hz
        self.frequency = mass / const.value("Planck constant in eV s")
        # our coherence time (1/ the real linewidth)
        self.coherence_time = 1 / (linewidth * self.frequency)

        self.phase_value = 0
        # the standard deviation of the distribution the phase change is drawn
        # from
        self.phase_std = 2 * np.pi * 0.68 * (step_size / self.coherence_time)

    def get_next_phase(self):
        phase = self.phase_value + self.phase_std * np.random.randn()
        print(phase)
        self.phase_value = phase

    def phase_test(self, points=1e5):
        x = np.arange(0, 1e-4, self.step_size)
        y1 = np.sin(self.frequency * x)
        phases = itertools.accumulate(self.phase_std
                                      * np.random.randn(len(x)))
        y2 = np.sin(self.frequency * x + np.array(list(phases)))
        plt.plot(x, y1)
        plt.plot(x, y2)
        return (x, y1, y2)

