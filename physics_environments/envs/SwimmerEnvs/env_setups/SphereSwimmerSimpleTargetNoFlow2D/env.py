
from typing import Literal, Tuple
from functools import cached_property

import chex
import jax
import jax.numpy as jnp
from jumanji import specs
from jumanji.env import Environment
from jumanji.types import TimeStep, restart, termination, transition, truncation

from physics_environments.baselines.baseline_types import \
    ALL_BASELINES, CTCE_baselines, CTDE_baselines, DTDE_baselines,\
    reward_per_agent_baselines, reward_per_group_baselines, reward_per_agent_or_group_baselines

from physics_environments.envs.SwimmerEnvs.env_setups.env_types import \
    State, EnvConstants, EnvPhysicsConstants, Observation, SwimmerEnvironment2DParams


from physics_environments.envs.SwimmerEnvs.env_setups.SphereSwimmerSimpleTargetNoFlow2D.reward \
    import CooperativeAgentGroupRewardFn

from physics_environments.envs.SwimmerEnvs.env_setups.SphereSwimmerSimpleTargetNoFlow2D.observation \
    import FullGlobalObservationFn, ReducedFullGlobalObservationFn, LocalNNearestObjectsObservationFn

class SphereSwimmerEnvironment2D(Environment[State, specs.MultiDiscreteArray, Observation]):
    """
    ** SphereSwimmerEnvironment2D **
        A simple and flexible 2D MARL environment for microswimmers.

        Given, that there are a limited number of targets, and that the agents can push each other,
        this environment should a playground for different kinds of MARL scenarios (independent, cooperative, or even competitive)
        based on the given reward given.
        There are also various observability settings to test each PPO baseline on in terms of scalability,
        sample-efficiency, training stability, run-time speed and performance.

    Environment Information:
    - environment metric / goal:
        - The goal in the environment is to have each particle occupy each available target as fast as possible,
          (or to have each swimmer having one target occupied).
        - So the metric is the TotalTimeToOccupy and the MeanIndividualTimeToOccupy
    - reward:
        - One type of reward can be chosen: CooperativeAgentGroupRewardFn
            - Other available rewards, as IndependentAgentRewardFn, CollaborativeAgentGroupRewardFn should be defined in a different environment setup.
        - Note: All the rewards might be better to fulfill the environment task; they guide the MAS to a specific behavior, but this behavior can be hard to define safely.
    - action:
        - These are defined with the swimmer
    - state:
        - t, step_num, key, swimmer_state, target_state, flow_field_state
    - observation:
        - FullGlobalObservationFn, ReducedFullGlobalObservationFn, LocalNNearestObjectsObservationFn
    - episode termination:
        - upon fullfilling the task
        - and upon raching the episode time limit
    - episode truncation:
        - no episode truncation as there no forbidden states/actions

    Notes:
        - The environment is originally designed to have constant atmospheric pressure, constant room temperature, and water as a fluid.
        - This environment is only designed to hold SphereSwimmers as JanusSwimmers need slightly different observation information.
        - This environment is assumed to have fully homogeneous swimmers occupying a variable amount of available target positions.
        - Only one type of particle type and swimmer type is simulated in the environment at a time.
        - All particles in the environment are steerable swimmers.
            - There are no non-agent particles in this environment that need to be pushed away, or need to be sorted.
    """

    def __init__(self,
                 params : SwimmerEnvironment2DParams):

        ##  Checking parameter choices  ##
        assert (len(params.env.env_size) == 2), (f"The environment is in 2D. but params.env.env_size implied {len(params.env.env_size)} dimensions.")
        assert (params.env.env_size[0]  == params.env.env_size[1]), ("The shape of the environment has to be square for this implementation, for simplicity.")
        assert (params.env.n_swimmers  == params.env.n_targets), ("This environment should only focus on tasks with one target per agent. ")

        assert (params.env.baseline_name in ALL_BASELINES), (f"One has to use a supported baseline from {ALL_BASELINES} so that the environment is configured accordingly! But {params.env.baseline_name} was selected.")

        def get_flow_meshgrid2d(n_cells, box_len):
            xs = jnp.linspace(0.5*box_len/n_cells, box_len - 0.5*box_len/n_cells,
                              num=n_cells, endpoint=True) # the flow should be in the middle of each grid cell
            ys = xs
            xv, yv = jnp.meshgrid(xs, ys)  # linear x,y coordinates
            return (xv, yv)

        self.env_constants = EnvConstants(t_simulation_episode = params.env.t_simulation_episode,
                                          act_dt               = params.env.act_dt,
                                          sim_dt               = params.env.sim_dt,
                                          particle_radius      = params.env.particle_radius,
                                          env_size             = params.env.env_size,
                                          n_swimmers           = params.env.n_swimmers,
                                          n_targets            = params.env.n_targets,
                                          n_flow_grid_cells    = params.env.n_flow_grid_cells,
                                          flow_mesh            = get_flow_meshgrid2d(n_cells = params.env.n_flow_grid_cells,
                                                                                     box_len = params.env.env_size[0]),
                                          env_physics_consts   = EnvPhysicsConstants() # not intended to change these constants as of now; using default values
                                          )

        ##  Defining environment functions  ##
        # Note, these rewards are intended to benefit the task


        if self.params.reward_func == "CooperativeAgentGroupRewardFn":
            print('Note: Using CooperativeAgentGroupRewardFn which is compatible for any baseline.')
            self.reward_func = CooperativeAgentGroupRewardFn(env_constants = self.env_constants) # returns shape (1)


        else:
            raise NotImplementedError(f"self.params.reward_func {self.params.reward_func} was not implemented in env.py or reward.py. " + \
                                      f"Or {self.params.reward_func} was not a valid choice for SwimmerEnvironment2D")

        # Observations per Agent

        if self.params.observation_func == "FullGlobalObservationFn":
            centralized_fullglobal = params.baseline.name in CTCE_baselines
            self.observation_func = FullGlobalObservationFn(centralized=centralized_fullglobal)
        elif self.params.observation_func == "ReducedFullGlobalObservationFn":
            self.observation_func = ReducedFullGlobalObservationFn(centralized=False)
        elif self.params.observation_func == "LocalNNearestObjectsObservationFn":
            self.observation_func = LocalNNearestObjectsObservationFn(centralized=False)
        else:
            raise NotImplementedError(f"self.params.observation_func {self.params.observation_func} was not implemented in env.py or observation.py. " + \
                                      f"Or {self.params.observation_func} was not a valid choice for SwimmerEnvironment2D")

        # # Particle-Particle Interactions
        if self.params.particle_interaction_func == "NoInteraction":
            self.particle_interaction_func = NoInteraction()
        elif self.params.particle_interaction_func == "WCAInteraction":
            self.particle_interaction_func = WCAInteraction()
        else:
            raise NotImplementedError(f"self.params.particle_interaction_func {self.params.particle_interaction_func} was not implemented in env.py or particle_interaction.py. " + \
                                      f"Or {self.params.particle_interaction_func} was not a valid choice for SwimmerEnvironment2D")

        # Thermostat of Particles
        # if self.params.thermostat_type == "Langevin":
        #     self.thermostat_type = Langevin()
        # else:
        #     raise NotImplementedError(f"self.params.thermostat_type {self.params.thermostat_type} was not implemented in env.py or thermostat.py. " + \
        #                               f"Or {self.params.thermostat_type} was not a valid choice for SwimmerEnvironment2D")


        # Swimmer Type / Agent Type
        # if self.params.swimmer_type == "DiscreteSphereSwimmer2D":
        #     self.get_updated_swimmer_state_func = DiscreteSphereSwimmer2D()
        # elif self.params.swimmer_type == "ContinuousSphereSwimmer2D":
        #     self.get_updated_swimmer_state_func = ContinuousSphereSwimmer2D()
        # else:
        #     raise NotImplementedError(f"self.params.swimmer_type {self.params.swimmer_type} was not implemented in env.py or swimmer.py. " + \
        #                               f"Or {self.params.swimmer_type} was not a valid choice for SwimmerEnvironment2D")

        # Episode Generator Type
        # if self.params.episode_generator_func == "CustomSceneGenerator":
        #     self.episode_generator_func   = CustomSceneGenerator()
        #     self.benchmark_generator_func = self.episode_generator_func
        # elif self.params.episode_generator_func == "UniformCustomSceneGenerator":
        #     self.episode_generator_func   = UniformCustomSceneGenerator()
        #     self.benchmark_generator_func = self.episode_generator_func
        # elif self.params.episode_generator_func == "RandomEpisodeGenerator":
        #     self.episode_generator_func   = RandomEpisodeGenerator()
        #     self.benchmark_generator_func = self.episode_generator_func
        # else:
        #     raise NotImplementedError(f"self.params.swimmer_type {self.params.episode_generator_func} was not implemented in env.py or swimmer.py. " + \
        #                               f"Or {self.params.episode_generator_func} was not a valid choice for SwimmerEnvironment2D")


        # Particles
        # --> No Particles used; just swimmers used

        # Flow Field
        # --> No flow field used

        # Laser Agent
        # --> No Laser Agent used

        # Target Type
        if self.params.flow_field_func == "SimpleTarget2D":
            self.target_update_func = SimpleTarget2D(acceptance_radius = self.params.target.acceptance_radius)
        else:
            raise NotImplementedError(f"self.params.flow_field_func {self.params.flow_field_func} was not implemented in env.py or flow.py. " + \
                                      f"Or {self.params.flow_field_func} was not a valid choice for SwimmerEnvironment2D")


        ##  Defining constants ##

        # It is relevant to define each constant as a JAX data type as a good practice.
            # This may avoids unexpected speed bottle-necks for GPU code and errors later that you don't find well via debugging.



        self.terminate_done_fn   = TimeTerminateDoneFn(t_episode = self.constants.t_episode)
        self.truncate_done_fn    = NoTruncateDoneFn(bump_x = self.constants.bump_x)

        # self.viewer              = PendulumViewer(physics_env = physics_env,
        #                                           constants   = self.constants)
        # super().__init__()


    def reset(self, key: chex.PRNGKey) -> Tuple[State, TimeStep[Observation]]:
        """ Resets the environment. """
        # state              = self.training_generator(key)
        # # Initialize solver
        # solver_state       = self.math_objects.solver.init(terms=self.math_objects.ode_term, y0=state.y_solver, t0=state.t, t1=state.t+self.constants.dt, args=self.constants.ode_constants)
        # state.solver_state = solver_state

        # observation = self._state_to_observation(state=state)
        # timestep    = restart(observation=observation) # returns class TimeStep set with initial observation and reward
        # return state, timestep

    def reset_to_benchmark(self, key : chex.PRNGKey, benchmark_id : int) -> Tuple[State, TimeStep[Observation]]:
        """ Resets the environment for a Benchmark. """
        state              = self.benchmark_generator(key, benchmark_id)

        # Initialize solver
        solver_state       = self.math_objects.solver.init(terms=self.math_objects.ode_term, y0=state.y_solver, t0=state.t, t1=state.t+self.constants.dt, args=self.constants.ode_constants)
        state.solver_state = solver_state
        state.key          = key

        observation = self._state_to_observation(state=state)
        timestep    = restart(observation=observation) # returns class TimeStep set with initial observation and reward
        return state, timestep

    def reset_to_random_benchmark(self, key : chex.PRNGKey, benchmark_id : int) -> Tuple[State, TimeStep[Observation]]:
        """ Resets the environment for a Benchmark. """
        state              = self.random_benchmark_generator(key, benchmark_id)

        # Initialize solver
        solver_state       = self.math_objects.solver.init(terms=self.math_objects.ode_term, y0=state.y_solver, t0=state.t, t1=state.t+self.constants.dt, args=self.constants.ode_constants)
        state.solver_state = solver_state
        state.key          = key

        observation = self._state_to_observation(state=state)
        timestep    = restart(observation=observation) # returns class TimeStep set with initial observation and reward
        return state, timestep



    def step(self, state: State, action: chex.Array) -> Tuple[State, TimeStep[Observation]]:
        """ Run one timestep of the environment's dynamics. action is expected to have shape (1,). """



        t_next            = state.t+self.constants.dt
        step_count        = state.step_count + 1

        # The update order is important here!
        # future implementations might adapt to this given order.
        # (not used) next_laser_state      = self.target_update_func(state = state)             # only moves the laser / updates it based on everything: past lasers / past particles / past swimmers /...
        # (not used) next_flow_field_state = self.target_update_func(state = state)             # only updates the flow field based on everything: lasers / past particles / past swimmers /...
        # (not used) next_particle_state   = self.get_particle_swimmer_state_func(state = state)             # only updates the particles based on everything: lasers / past particles / past swimmers / ...
        next_swimmer_state    = self.get_updated_swimmer_state_func(state = state) # only updates the swimmers based on everything: lasers / particles / past swimmers /...
        next_target_state     = self.target_update_func(state = state)             # only updates the target states based on everything: lasers / particles / swimmers / ...

        reward          = self.reward_function(state)
        #reward = jnp.nan_to_num(reward, nan=0.0)



        next_state = State(t                = t_next,
                           step_count       = step_count,
                           key              = state.key,
                           target_state     = next_target_state,
                           laser_state      = next_laser_state,
                           particle_state   = next_particle_state,
                           swimmer_state    = next_swimmer_state,
                           flow_field_state = next_flow_field_state)

        next_observation = self._state_to_observation(state=next_state)

        terminate = self.terminate_done_fn(state = next_state)  # terminate is the naturally intended episode ending, or when the goal was reached by the agent
        truncate = self.truncate_done_fn(state = state)         # truncate is an early episde ending through invalid actions, failed goals, etc.
        next_timestep = jax.lax.switch(
            terminate + 2 * truncate,
            [lambda rew, obs, shape: transition( reward=rew, observation=obs, shape=shape),  # 0: !terminate and !truncate
             lambda rew, obs, shape: termination(reward=rew, observation=obs, shape=shape),  # 1:  terminate and !truncate
             lambda rew, obs, shape: truncation( reward=rew, observation=obs, shape=shape),  # 2: !terminate and  truncate
             lambda rew, obs, shape: termination(reward=rew, observation=obs, shape=shape)], # 3:  terminate and  truncate
            reward,
            next_observation,
            (), # shape parameter of rewards; needed for MARL envs
        )
        return next_state, next_timestep

    def _state_to_observation(self, state: State) -> Observation:
        """ Takes a state from env.step and converts it to an observation, i.e. the agent input.

            It is possible to feature engineering
        """

        # base_features               = jnp.array([s_x, v_x, a_x, d_corner])

        #agent_inputs_global = jnp.nan_to_num(agent_inputs_global, nan=0.0) # very important, there are very rarely nan for an unknown reason

        return Observation(agent_inputs_global = agent_inputs_global)


    def render(self, state: State) -> None:
        raise NotImplementedError("Refer to all the visualization methods via RL_Cart_Pendulum_Environment.viewer!")

    def close(self) -> None:
        pass # no extra cleanup necessary for closing this environment

    @cached_property
    def observation_spec(self) -> specs.Spec[Observation]:

        # keeping a mostly permissive definition for flexibility:
        agent_inputs_global = specs.BoundedArray(
            shape=(4 + 6*self.constants.n + 6*self.constants.n, ), # refer to _state_to_observation to get the shape
            dtype=jnp.float32,
            minimum=-jnp.inf,
            maximum=jnp.inf,
            name="agent_inputs_global",
        )

        return specs.Spec(
            Observation,
            "ObservationSpec",
            agent_inputs_global=agent_inputs_global
        )

    @cached_property
    def action_spec(self) -> specs.Array:
        return specs.BoundedArray(
                shape=(1,),
                name="cart acceleration",
                minimum=-self.constants.max_cart_acceleration,
                maximum=self.constants.max_cart_acceleration,
                dtype=jnp.float32,
            )

    def __repr__(self):
        return self.__doc__