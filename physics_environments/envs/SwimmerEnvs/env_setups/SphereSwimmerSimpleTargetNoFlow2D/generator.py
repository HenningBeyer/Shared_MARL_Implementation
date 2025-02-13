import abc

import chex
import jax
import jax.numpy as jnp
from jax.random import uniform

# from physics_environments.envs.rl_pendulum.types import Constants, State
# #from typing import override


# class Generator(abc.ABC):
#     """ Base class for generators """

#     def __init__(self, constants : Constants):
#         self.constants = constants

#     @abc.abstractmethod # Abstract methods mean that the inheriting class should implement the function
#     def generate_initial_episode_state(self, key: chex.PRNGKey) -> State:
#         return

#     def __call__(self, key: chex.PRNGKey) -> State:
#         """ Generate the first state of an episode for an environment.

#             Do some common calculation and then call self.generate_initial_episode_state to
#             initialize an initial environment state with a custom procedure defined in the inheriting class.
#         """
#         key, sample_key = jax.random.split(key)                # create a new random key each reset
#         state = self.generate_initial_episode_state(key = key) # call the abstract method specific to each inheriting class's implementation

#         return state


# class DefaultHangingGenerator(Generator):
#     """ The pendulum will be initialized in a standard hanging position.
#         All velocities and accelerations are also set to 0 """

#     def generate_initial_episode_state(self, key: chex.PRNGKey) -> State:
#         thetas   = dthetas = ddthetas = jnp.zeros(self.constants.n, dtype=jnp.float32)
#         thetas   = thetas + jnp.pi
#         a_x      = jnp.array(0, jnp.float32)
#         state = State(t            = jnp.array(0, jnp.float32),
#                       step_count   = jnp.array(0, jnp.int32),
#                       key          = key,
#                       solver_state = None, # initialized in env.py
#                       y_solver     = jnp.concatenate([thetas, dthetas, jnp.array([a_x])]),
#                       thetas       = thetas,
#                       dthetas      = dthetas,
#                       ddthetas     = ddthetas,
#                       s_x          = jnp.array(0, jnp.float32),
#                       v_x          = jnp.array(0, jnp.float32),
#                       a_x          = a_x)

#         return state

# class DefaultStandingGenerator(Generator):
#     """ The pendulum will be initialized in a standard standing position with a small random offset.
#         All velocities and accelerations are also set to 0 """

#     def generate_initial_episode_state(self, key: chex.PRNGKey) -> State:

#         n        = self.constants.n
#         thetas   = uniform(key, shape=(n,), minval=-0.01, maxval=0.01)
#         dthetas  = uniform(key, shape=(n,), minval=-0.01, maxval=0.01)
#         ddthetas = uniform(key, shape=(n,), minval=-0.01, maxval=0.01)
#         a_x      = jnp.array(0, jnp.float32)

#         state = State(t            = jnp.array(0, jnp.float32),
#                       step_count   = jnp.array(0, jnp.int32),
#                       key          = key,
#                       solver_state = None, # initialized in env.py
#                       y_solver     = jnp.concatenate([thetas, dthetas, jnp.array([a_x])]),
#                       thetas       = thetas,
#                       dthetas      = dthetas,
#                       ddthetas     = ddthetas,
#                       s_x          = jnp.array(0, jnp.float32),
#                       v_x          = jnp.array(0, jnp.float32),
#                       a_x          = a_x)

#         return state




# class UniformGenerator(Generator):
#     """ The pendulum will be initialized in a totally random state.
#         This can help the robustness of the agent. """

#     def generate_initial_episode_state(self, key: chex.PRNGKey) -> chex.Array:
#         """ Returns the state of a randomly initialized episode. """

#         n                     = self.constants.n
#         max_s_x_offset        = 0.50*(self.constants.track_width - self.constants.cart_width)/2
#         max_cart_acceleration = self.constants.max_cart_acceleration

#         thetas   = uniform(key, shape=(n,), minval=-1*jnp.pi, maxval=1*jnp.pi)
#         dthetas  = uniform(key, shape=(n,), minval=-2.0, maxval=2.0)
#         ddthetas = uniform(key, shape=(n,), minval=-5.0, maxval=5.0)
#         s_x      = uniform(key, shape=(1), minval=-max_s_x_offset, maxval=max_s_x_offset)[0] # [0] changes shape (1,) to shape (1)
#         v_x      = uniform(key, shape=(1), minval=-0.5, maxval=0.5)[0]
#         a_x      = uniform(key, shape=(1), minval=-max_cart_acceleration, maxval=max_cart_acceleration)[0]

#         state = State(t            = jnp.array(0, jnp.float32),
#                       step_count   = jnp.array(0, jnp.int32),
#                       key          = key,
#                       solver_state = None, # initialized in env.py
#                       y_solver     = jnp.concatenate([thetas, dthetas, jnp.array([a_x])]),
#                       thetas       = thetas,
#                       dthetas      = dthetas,
#                       ddthetas     = ddthetas,
#                       s_x          = s_x,
#                       v_x          = v_x,
#                       a_x          = a_x)

#         return state



# class BenchmarkGenerator(Generator):
#     """ Get a specific benchmark setting for testing specific initial conditions, much like as a benchmark.
#         generate_initial_episode_state will now receive a benchmark_id parameter to receive specific benchmarks.
#     """

#     #@override
#     def __call__(self, key: chex.PRNGKey, benchmark_id : int) -> State:
#         """ Generate the first state of an episode for an environment.

#             Do some common calculation and then call self.generate_initial_episode_state to
#             initialize an initial environment state with a custom procedure defined in the inheriting class.
#         """
#         key, sample_key = jax.random.split(key)                                    # create a new random key each reset
#         state = self.generate_initial_episode_state(key = key, benchmark_id = benchmark_id) # call the abstract method specific to each inheriting class's implementation

#         return state

#     def generate_initial_episode_state(self, key: chex.PRNGKey, benchmark_id : int) -> State:
#         if benchmark_id == 0:
#             """ Hanging configuration """
#             print(""" Hanging configuration """)
#             thetas = dthetas = ddthetas = jnp.zeros(self.constants.n, dtype=jnp.float32)
#             thetas = thetas + jnp.pi
#             a_x    = jnp.array(0, jnp.float32)
#             state = State(t            = jnp.array(0, jnp.float32),
#                           step_count   = jnp.array(0, jnp.int32),
#                           key          = key,
#                           solver_state = None, # initialized in env.py
#                           y_solver     = jnp.concatenate([thetas, dthetas, jnp.array([a_x])]),
#                           thetas       = thetas,
#                           dthetas      = dthetas,
#                           ddthetas     = ddthetas,
#                           s_x          = jnp.array(0, jnp.float32),
#                           v_x          = jnp.array(0, jnp.float32),
#                           a_x          = a_x)

#         elif benchmark_id == 1:
#             """ Standing configuration """
#             print(""" Standing configuration """)
#             thetas = dthetas = ddthetas = jnp.zeros(self.constants.n, dtype=jnp.float32)
#             a_x    = jnp.array(0, jnp.float32)

#             state = State(t            = jnp.array(0, jnp.float32),
#                           step_count   = jnp.array(0, jnp.int32),
#                           key          = key,
#                           solver_state = None, # initialized in env.py
#                           y_solver     = jnp.concatenate([thetas, dthetas, jnp.array([a_x])]),
#                           thetas       = thetas,
#                           dthetas      = dthetas,
#                           ddthetas     = ddthetas,
#                           s_x          = jnp.array(0, jnp.float32),
#                           v_x          = jnp.array(0, jnp.float32),
#                           a_x          = a_x)

#         elif benchmark_id == 2:
#             """ Standing sub-equilibrium configuration """
#             print(""" Standing sub-equilibrium configuration """)
#             # thetas = dthetas = ddthetas = jnp.zeros(self.constants.n, dtype=jnp.float32)
#             n        = self.constants.n
#             thetas   = uniform(key, shape=(n,), minval=-0.01, maxval=0.01)
#             dthetas  = uniform(key, shape=(n,), minval=-0.01, maxval=0.01)
#             ddthetas = uniform(key, shape=(n,), minval=-0.01, maxval=0.01)
#             thetas = thetas.at[1::2].add(jnp.pi) # making every second rod point down; this is a results in a folded, but standing pendulum
#             a_x    = jnp.array(0, jnp.float32)

#             state = State(t            = jnp.array(0, jnp.float32),
#                           step_count   = jnp.array(0, jnp.int32),
#                           key          = key,
#                           solver_state = None, # initialized in env.py
#                           y_solver     = jnp.concatenate([thetas, dthetas, jnp.array([a_x])]),
#                           thetas       = thetas,
#                           dthetas      = dthetas,
#                           ddthetas     = ddthetas,
#                           s_x          = jnp.array(0, jnp.float32),
#                           v_x          = jnp.array(0, jnp.float32),
#                           a_x          = a_x)

#         elif benchmark_id == 3:
#             """ Tilted Cube configuration """
#             print(""" Tilted Cube configuration """)
#             dthetas = ddthetas = jnp.zeros(self.constants.n, dtype=jnp.float32)

#             start  =  jnp.pi/4
#             step   = -jnp.pi/2
#             thetas = start + jnp.arange(self.constants['n']) * step # starting at pi/4, let each element be smaller by pi/2, forming a tilted coude of rods.
#             a_x    = jnp.array(0, jnp.float32)

#             state = State(t            = jnp.array(0, jnp.float32),
#                           step_count   = jnp.array(0, jnp.int32),
#                           key          = key,
#                           solver_state = None, # initialized in env.py
#                           y_solver     = jnp.concatenate([thetas, dthetas, jnp.array([a_x])]),
#                           thetas       = thetas,
#                           dthetas      = dthetas,
#                           ddthetas     = ddthetas,
#                           s_x          = jnp.array(0, jnp.float32),
#                           v_x          = jnp.array(0, jnp.float32),
#                           a_x          = a_x)

#         elif benchmark_id == 4:
#             """ Pull-up configuration
#                 - each rod hangs down and experiences an extreme angular velocity and acceleration of 10 a.u. on each rod.
#                 - also, the sign for each rod changes, resulting in an upward pull.
#                 - The agent will likely fail this task, unless it is trained robustly
#             """
#             print(
#             """ Pull-up configuration
#                 - each rod hangs down and experiences an extreme angular velocity and acceleration of 10 a.u. on each rod.
#                 - also, the sign for each rod changes, resulting in an upward pull.
#                 - The agent will likely fail this task, unless it is trained robustly """)
#             thetas   = dthetas = ddthetas = jnp.zeros(self.constants.n, dtype=jnp.float32)
#             thetas   = thetas + jnp.pi
#             dthetas  = dthetas + 10.0
#             ddthetas = ddthetas + 10.0
#             dthetas  = dthetas.at[1::2].mul(-1)
#             ddthetas = ddthetas.at[1::2].mul(-1)
#             a_x      = jnp.array(0, jnp.float32)

#             state = State(t            = jnp.array(0, jnp.float32),
#                           step_count   = jnp.array(0, jnp.int32),
#                           key          = key,
#                           solver_state = None, # initialized in env.py
#                           y_solver     = jnp.concatenate([thetas, dthetas, jnp.array([a_x])]),
#                           thetas       = thetas,
#                           dthetas      = dthetas,
#                           ddthetas     = ddthetas,
#                           s_x          = jnp.array(0, jnp.float32),
#                           v_x          = jnp.array(0, jnp.float32),
#                           a_x          = a_x)

#         return state


