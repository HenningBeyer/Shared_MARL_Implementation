import abc

import chex
import jax
import jax.numpy as jnp
from jax.random import uniform

from physics_environments.envs.SwimmerEnvs.env_types import State, EnvConstants

class Flow(abc.ABC):
    """ Base class for custom Flow functions / models. Classes should just return flow functions.
        Most classes are just for example.
    """

    def __init__(self, env_constants : EnvConstants):
        self.xv, self.yv = self.flow_meshgrid = env_constants.flow_meshgrid

    @abc.abstractmethod
    def __call__(self, state : State):
        """ Create a flow field, and apply it to each particle, and store it in the state"""

class NoField2D(Flow):
    """ Create a flow field, and apply it to each particle.
            - No flow means no change at all, just return state
    """

    def __init__(self, env_constants : EnvConstants):
        self.xv, self.yv = env_constants.flow_meshgrid

    def __call__(self, state : State) -> chex.Array:
        flow_field = jnp.stack([jnp.zeros_like(self.xv), jnp.zeros_like(self.yv)], axis=2) # shape (x_cells, y_cells, 2)
        return flow_field

class SpiralField2D(Flow):
    """ Create a flow field that flows to the center, and apply it to each particle.
        - This is more an example class for later, one can implement custom flow functions with it.
    """

    def __init__(self,
                 env_constants : EnvConstants,
                 rotate_factor : float,
                 inward_factor : float
                 ):
        self.xv, self.yv = env_constants.flow_meshgrid
        self.x_center = env_constants.env_size[0]*0.5
        self.y_center = env_constants.env_size[1]*0.5

        self.A = rotate_factor
        self.B = inward_factor

    def __call__(self, state : State) -> State:
        r_x = self.xv - self.x_center
        r_y = self.yv - self.y_center

        flow_field = jnp.stack([-self.A*r_x + self.B*r_y,
                                -self.A*r_y - self.B*r_x],
                                axis=2)

        return flow_field # shape (x_cells, y_cells, 2)

class SimplifiedMultiLaserField2D(Flow):
    """ Creates a sum of local inward or outward flows per laser agent instance.
        This field is only used if laser agents are present.
    """

    def __init__(self,
                 env_constants  : EnvConstants,
                 r_max          : float,
                 outward_factor : float
                 ):
        self.xv, self.yv    = env_constants.flow_meshgrid
        self.r_max          = r_max # radius at which flow approaches 0
        self.outward_factor = outward_factor

    def single_laser_field(self, laser_pos_x, laser_pos_y):
        r_x = self.xv - laser_pos_x
        r_y = self.yv - laser_pos_y

        r_mag    = jnp.sqrt(r_x**2 + r_y**2)
        r_x_unit = r_x/(r_mag + 1e-8)
        r_y_unit = r_y/(r_mag + 1e-8)

        v   = self.outward_factor*jnp.clip(1 - r_mag**3/self.r_max, min=0)
        v_x = v*r_x_unit
        v_y = v*r_y_unit


        flow_field = jnp.stack([v_x, v_y],
                                axis=2)

        return flow_field # shape (x_cells, y_cells, 2)

    def __call__(self, state : State) -> State:
        vec_laser_x_pos = state.laser_state.x # shape (n_lasers)
        vec_laser_y_pos = state.laser_state.y # shape (n_lasers)


        all_laser_fields = jax.vmap(self.single_laser_field, in_axes=(0,0))(vec_laser_x_pos, vec_laser_y_pos) # shape (n_lasers, x_cells, y_cells, 2)
        flow_field       = all_laser_fields.sum(axis=0) # shape (x_cells, y_cells, 2); sums up individual laser flowfield contributions

        return flow_field # shape (x_cells, y_cells, 2)

class RealisticMultiLaserField2D:
    pass
    # TODO: RealisticMultiLaserField2D: # Required??
        # would require a good physics model of how the lasers would heat the fluid; will then also need *an exact fluid simulation* with turbulent flow
        # What is the heat emitted by the laser to the fluid?
        # How does the heat dissipate around the fluid over time?
        # how to numerically solve the navier-stokes equations and model the turbulences?
        # ... this can be relevant for the setup in 2D/3D, but would additional research.