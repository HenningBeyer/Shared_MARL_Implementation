import abc
from typing import NamedTuple

import chex
import jax
import jax.numpy as jnp
from jax.random import uniform

from physics_environments.envs.rl_pendulum.env_types import State
from physics_environments.envs.SwimmerEnvs.physics.physics_types import SimpleTarget2DState


class TargetGroupUpdateFn(abc.ABC):
    """ Base class for custom Targets.
        Can be extended in the future, for example as TargetAreas that can hold 10 particles,
            have a square shape, attract paritcles, can attract or eject particles, or particles can become loose over time, etc.
    """

    def __init__(self):
        pass

    @abc.abstractmethod
    def update_target_state(self, state : State):
        """ Give the environment state to the function to update it """

class SimpleTarget2DUpdateFn(TargetGroupUpdateFn):
    """ A simple target for the particle environment.
            - This target is a simple point target that a particle or swimmer should hold, a particle can get ejected by another particle / swimmer,
                or the swimmer could leave to make space for another swimmer
            - If one particle has entered the target acceptance area is_occupied is set to True for that step

        All data is handled in types.py to separate data from the computation functions:

        class SimpleTarget2DState(NamedTuple):
            x           : chex.Array # an array holding all x positions of all targets
            y           : chex.Array # an array holding all y positions of all targets
            is_occupied : chex.Array # an array holding all bools of all targets
    """

    def __init__(self, acceptance_radius : float):
        assert acceptance_radius > 0.0
        self.acceptance_radius = acceptance_radius

    def single_target_occupy(self,
                             x_swimmers  : chex.Array,  # shape (n_swimmers)
                             y_swimmers  : chex.Array,  # shape (n_swimmers)
                             target_x    : chex.Array,  # shape (1)
                             target_y    : chex.Array,  # shape (1)
                             o_target    : chex.Array,  # shape (1)
                             ) -> chex.Array:
        """ This function is vmapped for multiple targets in parallel. """

        r       = jnp.sqrt((target_x - x_swimmers)**2      + (target_y - y_swimmers)**2)
        nearest_particle_idx = jnp.argmin(r)
        did_occupy = (r <= self.acceptance_radius)
        did_occupy = did_occupy[nearest_particle_idx]#*(~o_target) # take the closest particle from did_occupy
        return did_occupy

    def return_target_state(self, state : State):
        """ Check all swimmers/particles if they got into the acceptance radius, and accept the closest.
            returns
        """
        x_swimmers = state.swimmer_state.x
        y_swimmers = state.swimmer_state.y
        target_x   = state.target_state.x
        target_y   = state.target_state.y
        o_target   = state.target_state.is_occupied

        all_did_occupies = jax.vmap(self.single_target_occupy, in_axes=(None,None,0,0,0))(x_swimmers = x_swimmers, # shape (n_swimmers)
                                                                                          y_swimmers = y_swimmers, # shape (n_swimmers)
                                                                                          target_x   = target_x,   # shape (m_targets)
                                                                                          target_y   = target_y,   # shape (m_targets)
                                                                                          o_target   = o_target)   # shape (m_targets)

        new_target_state = SimpleTarget2DState(x           = target_x,
                                               y           = target_y,
                                               is_occupied = all_did_occupies)

        return new_target_state