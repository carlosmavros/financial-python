"""This is a class for simulating Brownian motion.

More precisely, this class simulates geometric Brownian motion.
"""
import logging
import sys
import numpy as np
import matplotlib.pyplot as plt
from numpy.random import standard_normal as z
from math import sqrt


logging.basicConfig(stream=sys.stdout, level=logging.DEBUG)


class GeomBrownian:
    """Models a geometric Brownian path."""

    def __init__(self, r: float, sigma: float,
                 T: float, S0: float):
        """Initialize brownian motion object.

        Args:
        r (float)     : the risk free rate.
        sigma (float) : the volatility.
        T (float)     : maturity.
        S0 (float)    : initial value
        """
        self.r = r
        self.sigma = sigma
        self.T = T
        self.S0 = S0

    def simulate_paths(self, nsteps: int, npaths: int):
        """Simulate paths using an Euler discretization scheme.

        Args:
        nsteps (int) : number of steps.
        npaths (int) : number of paths to be simulated.
        """
        dt = self.T/nsteps
        S = np.zeros((nsteps+1, npaths))
        S[0] = self.S0

        for t in range(1, nsteps + 1):
            S[t] = S[t-1] * np.exp((self.r - 0.5 * self.sigma ** 2)
                                   * dt + self.sigma * sqrt(dt)
                                   * z(npaths))
        self.__simulated_paths = S

    def get_paths(self):
        """Return the simulated paths."""
        return(self.__simulated_paths)

    def plot_paths(self, n=10):
        """Plot the first n simulate paths.

        Args:
        n (int) : number of paths to plot
        """
        try:
            # try plotting the paths.
            plt.figure(figsize=(12, 8))
            plt.plot(self.__simulated_paths[:, :n], lw=1)
            plt.xlabel('time')
            plt.ylabel('S')
            plt.show()
        except IndexError as error:
            logging.log_exception(error)
        except Exception as exception:
            logging.log_exception(exception, False)
