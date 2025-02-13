import abc
from typing import NamedTuple

import chex
import jax
import jax.numpy as jnp
from jax.random import uniform

from physics_environments.envs.SwimmerEnvs.env_types import State
from physics_environments.envs.SwimmerEnvs.physics.potential import PotentialForceFunc

""" Note Particles are exactly swimmers without action or controllability.
    They might be used for other swimmers to navigate around, or to by other swimmers, lasers, and control flows to arange or sort.

    Particles only follow the langevin equations for now; could be extended later if needed for different kinds of environments.
    Particles follow WCA or no WCA
"""

class ParticleGroupGetUpdateFn(abc.ABC):
    """ Base class for custom Particles.

        Note: particles are like swimmers without actions, particles might be added into the mix with swimmers.

        Can be extended in the future, for example with non-spherical particles with any kind of properties.
    """

    def __init__(self, potential_force_fn : PotentialForceFunc):
        self.k_b                = 1.380649e-23 # J/K boltzman constant
        self.potential_force_fn = potential_force_fn

    @abc.abstractmethod
    def __call__(self, state : State):
        """ Get the new particle state based on the current environment state. """

class NoParticleGroupGetUpdateFn(ParticleGroupGetUpdateFn):
    """ The class chosen, if there is no particle group.
        Gives flexibility as it allows to use and to not use any particle without recoding the entire environment.
    """

    def __init__(self, potential_force_fn : PotentialForceFunc):
        self.k_b = 1.380649e-23 # J/K boltzman constant

    def __call__(self, state : State):
        """ Get the new and updated particle state based on the current environment state. """
        state.particle_state = NoParticleState()


class TranslationalSphereParticle2DGetUpdateFn(ParticleGroupGetUpdateFn):
    """ Spheric Particles implemented together with the Langevin termostat.
        This is a simple sphere particle without torque or rotation and only direct x and y movement.

    """

    def __init__(self,
                 dt                 : chex.Numeric,
                 gamma_t            : chex.Numeric,
                 xi_t               : chex.Numeric,
                 T                  : chex.Numeric,
                 radii              : chex.Array, # all particle radii, often all radii are the same
                 potential_force_fn : PotentialForceFunc
                 ):
        super().__init__(potential_force_fn = potential_force_fn) # initializes k_b and potential_force_fn
        self.dt      = dt
        self.gamma_t = gamma_t # translational friction; converted to jax just in case
        self.xi_t    = xi_t
        self.T       = T # T is constant
        self.radii   = radii


    def __call__(self, state : State, F_act_x = 0.0, F_act_y = 0.0):
        """ Get the new particle state based on the current environment state and potentially action forces. """
        x = particle_state.particles.x

        F_act_x      = F_act_x
        F_act_y      = F_act_y
        F_pot_x, F_pot_y = self.potential_force_fn(state = state)

        F_net_x = F_act_x + F_flow_x + F_part_net_x
        F_net_y = F_act_y + F_flow_y + F_part_net_y

        #M_net_i = M_act_i + M_flow_i + M_part_i_net # M_net_i: M_net per particle
        dr_x_dt = 1/self.gamma_t*F_net_x + jnp.sqrt(2*self.k_b*self.T)*self.xi_t
        dr_y_dt = 1/self.gamma_t*F_net_y + jnp.sqrt(2*self.k_b*self.T)*self.xi_t

        x_new = x + dr_x_dt*self.dt
        y_new = y + dr_y_dt*self.dt

        return  TranslationalSphereParticle2DState(x = x_new,
                                                   y = y_new)


class SelfPropelledSphereParticle2DGetUpdateFn(ParticleGroupGetUpdateFn):
    """ Spheric Particles implemented together with the Langevin termostat.

    """

    def __init__(self,
                 dt      : chex.Numeric,
                 gamma_t : chex.Numeric,
                 xi_t    : chex.Numeric,
                 T       : chex.Numeric,
                 radii   : chex.Array # all particle radii, often all radii are the same
                 ):
        super().__init__() # initializes k_b
        self.dt      = dt
        self.gamma_t = gamma_t # translational friction; converted to jax just in case
        self.xi_t    = xi_t
        self.T       = T # T is constant
        self.radii   = radii

    def __call__(self, state : State):
        """ Get the new particle state based on the current environment state. """
        #F_act      = ...
        F_part_net = ... # net force other particles induce to the i-th particle
        F_flow     = ...

        # M_act      = ...
        # M_part_net = 0.0 # net force other particles induce to the i-th particle; assumed to be 0.0
        # M_flow     = 0.0 # assumed to be 0.0

        F_net = F_act + F_flow + F_part_net # F_net_i: F_net per particle
        M_net = M_act + M_flow + M_part_net # M_net_i: M_net per particle
        drdt = 1/self.gamma_t*F_net + jnp.sqrt(2*self.k_b*self.T)*self.xi_t


epsilon_vect = 4*jnp.pi*${env.physics_consts.dyn_viscosity}*r