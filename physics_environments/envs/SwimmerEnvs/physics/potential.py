import abc
from typing import NamedTuple

import chex
import jax
import jax.numpy as jnp
from jax.random import uniform

from physics_environments.envs.SwimmerEnvs.env_types import State, EnvPhysicsConstants

class PotentialForceFunc(abs.ABC):
    """ Potential functions mostly for spherical particles, microswimmers and colloids """

    def __init__(self, env_physics_consts : EnvPhysicsConstants):
        self.k_b           = env_physics_consts.k_b
        self.dyn_viscosity = env_physics_consts.dyn_viscosity # often around 0.800e-3 mPa*s; dynamic fluid viscosity η; varies strongly per fluid, temperature, and pressure; assumes them constant

    @abc.abstractmethod
    def __call__(self, state : State):
        """ Calculate and return the individual force components per particle """

class NoPotentialForceFunc(PotentialForceFunc):
    """ Used, if no inter-particle interaction a wished for simplicity/testing. """

    def __init__(self, env_physics_consts : EnvPhysicsConstants):
        pass

    def __call__(self, state : State):
        """ Leave each individual force components per particle as 0.0"""
        x = state.particle_states.x
        F_x = F_y = jnp.zeros_like(x)
        return (F_x, F_y)

class WCAPotentialForceFunc(PotentialForceFunc):
    """ Avoids particles and colloids oveerlapping and stacking with each other.
        The WCA potential is only the purely repulsive potential of the complete LJ potential, or the LJ without attractive forces.
        (Collisions are still not modelled, they are more compute intensive, and hard to do, but WCA mostly avoids that)
        (WCA closely resembles the interactions between colloid, but might not it)

        Without some repulsive force, most RL tasks we have in mind may won't work.

        Sources: - https://www.researchgate.net/publication/257218769_Augmented_van_der_Waals_equation_of_state_for_the_Lennard-Jones_fluid
                 - https://dasher.wustl.edu/chem430/readings/md-intro-2.pdf
                 - https://jax.readthedocs.io/en/latest/notebooks/autodiff_cookbook.html
    """

    def __init__(self, env_physics_consts : EnvPhysicsConstants):
        super().__init__(env_physics_consts = env_physics_consts)
        self.WCA_epsilon = self.dyn_viscosity*self.k_b

    def __call__(self, state : State):
        """ Calculate and return the individual force components per particle """
        # epsilon is the depth of the potential well
        x = state.particle_states.x
        y = state.particle_states.y
        r = state.particle_states.r  # shape (n_particles); particle_radii

        # sigma_matrix can be a constant = 2*r if the particle radius is constant, but implementing the general formula here for later.
        sigma_matrix = r[:, None] + r[None, :] # Shape (n_particles, n_particles) # broadcast to a pairwise matrix, and add both radii
        r_cutoff_matrix = jnp.pow(2, 1/6)*sigma_matrix # shape (n_particles, n_particles)

        coordinates       = jnp.stack((x,y), axis=1)                               # Shape (n_particles, 2)              # jnp.stack((x, y, z), axis=1) for 3D
        r_xyz_tensor      = coordinates[:, None, :] - coordinates[None, :, :]      # Shape (n_particles, n_particles, 2) # broadcast to a pairwise matrix, and take differences
        r_matrix          = jnp.linalg.norm(r_xyz_tensor, axis=-1, keepdims=False) # Shape (n_particles, n_particles)
        r_xyz_unit_tensor = r_xyz_tensor / r_matrix                                # Shape (n_particles, n_particles, 2)

        def potential_func(r_matrix, sigma_matrix, r_cutoff_matrix):
            """ The WCA potential function to be differentiated for the force components.
                Function is not used, but stays here to show the methodology.
            """

            # Avoid division by zero on the diagonal
            r_matrix = jnp.where(r_matrix == 0, jnp.inf, r_matrix)

            r6 = jnp.pow((sigma_matrix / r_matrix), 6)

            LJ_potential  = 4 * self.WCA_epsilon * (jnp.pow(r6, 2) - r6 + 0.25)
            WCA_potential = jnp.where(r_matrix <= r_cutoff_matrix, LJ_potential, jnp.array(0.0))

            return jnp.sum(WCA_potential) / 2 # gets the entire potential; halved to avoid double counting

        def potential_grad_func(r_matrix, sigma_matrix, r_cutoff_matrix):
            """ potential_grad_func = jax.grad(potential_grad_func, in_axes=(0,None,None))
                    The potential_func was symbolically differentiated by the distances r_ij for force calculation
                    - doing it here symbolically for performance and precision.
            """
            r_matrix = jnp.where(r_matrix == 0, jnp.inf, r_matrix)

            r6 = jnp.pow((sigma_matrix / r_matrix), 6)

            LJ_potential_Derivative = 24 * self.WCA_epsilon / r_matrix * (2 * jnp.pow(r6, 2) - r6) # differentiated w.r.t. r_matrix
            WCA_potential_Force     = jnp.where(r_matrix <= r_cutoff_matrix, LJ_potential_Derivative, jnp.array(0.0))

            return WCA_potential_Force

        F_mag    = -potential_grad_func(r_matrix, sigma_matrix, r_cutoff_matrix) # Shape (n_particles) # partial derivatives by r; jax allows automatic differentiation.
        F_x = jnp.sum(F_mag * r_xyz_unit_tensor[:, :, 0], axis=-1)          # Shape (n_particles); x forces per particle
        F_y = jnp.sum(F_mag * r_xyz_unit_tensor[:, :, 1], axis=-1)          # Shape (n_particles)

        return (F_x, F_y)