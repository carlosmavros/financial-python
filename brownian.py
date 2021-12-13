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
        t = np.zeros((nsteps+1, 1))

        for i in range(1, nsteps + 1):
            S[i] = S[i-1] * np.exp((self.r - 0.5 * self.sigma ** 2)
                                   * dt + self.sigma * sqrt(dt)
                                   * z(npaths))
            t[i] = i*dt

        self.__simulated_paths = (t, S)

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
            plt.plot(self.__simulated_paths[0],
                     self.__simulated_paths[1][:, :n], lw=1)
            plt.xlabel('time')
            plt.ylabel('S')
            plt.show()
        except IndexError as error:
            logging.exception(error)
        except Exception as exception:
            logging.exception(exception, False)

    def test():
        """Simple test."""
        S0 = 100.0
        r = 0.05
        sigma = 0.25
        T = 2  # horizon year in fractions
        nsteps = 1000
        npaths = 20
        gb = GeomBrownian(r, sigma, T, S0)
        gb.simulate_paths(nsteps, npaths)
        gb.plot_paths()
