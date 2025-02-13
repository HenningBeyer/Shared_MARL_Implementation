import abc
from typing import NamedTuple

import chex
import jax
import jax.numpy as jnp

""" Lasers are simply the agents that move that laser.
   The real laser physics might happen with the flow field and the particle"""

class LaserGroupUpdateFn(abc.ABC):
    """ Base class for custom Particles.

        Note: particles are like swimmers without actions, particles might be added into the mix with swimmers.

        Can be extended in the future, for example with non-spherical particles with any kind of properties.
    """

    def __init__(self):
        self.k_b = 1.380649e-23 # J/K boltzman constant

    def __call__(self, state : State):
        """ Give the environment state to the function to update it """

class NoLaserUpdateGroupFn(abc.ABC):
    """ The class chosen, if there is no laser group.
        Gives great flexibility as it allows to use and to not use any laser without recoding the entire environment.
    """

    def __init__(self):
        self.k_b = 1.380649e-23 # J/K boltzman constant

    def __call__(self, state : State, v_x, v_y):
        """ Give the environment state to the function to update it """
        state.particle_state = NoParticleState()

class SimplifiedLaser2D(LaserGroupUpdateFn):

    def step(self, ...)

class RealisticLaser2D(LaserGroupUpdateFn):
    pass
    # TODO: RealisticLaser: Required???
        # would require a good physics model of how the lasers would heat the fluid; will then also need *an exact fluid simulation* with turbulent flow
        # What is the heat emitted by the laser to the fluid?
        # How does the heat dissipate around the fluid over time?
        # how to numerically solve the navier-stokes equations and model the turbulences?
        # ... this can be relevant for the setup in 2D/3D, but would additional research.

        # How is the laser to be defined as an agent??