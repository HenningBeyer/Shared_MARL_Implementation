import abc

import chex
import jax
import jax.numpy as jnp
from jax.random import uniform

from physics_environments.envs.SwimmerEnvs.env_types import State, EnvConstants

""" Mostly one observation function per environment setup is considered, to maintain simplicity.
        - Specific environments with specific research goals might be converted to an additional and separate environment setup, occasionally.
        - Otherwise it might also be required to make his own obervation function in this file.

    The applicability of each observation function varies per setup.

    For this setup observing actions of other agents was not considered already, for simplicity. (optional TODO)
    (Optianal TODO: vision cones or realistic perception types per agent could be implemented still)
"""

class ObservationFn(abc.ABC):
    """ Base class for custom observation functions. Classes should just return observation functions. """

    def __init__(self, centralized : bool):
        """
        centralized : bool:
            - wheter the observation is centralized or decentralized for each agent/actor.
            - should be centralized only for CPPO or any C- baseline.
            - else can be decentralized for IPPO and MAPPO actors or any MARL baseline in general.

            Note: for simplicity the actors will just centralize all their actor observations, or pass the exact observation from the actor.

        """
        self.centralized = centralized

    @abc.abstractmethod
    def __call__(self, state : State):
        """ Return the observation for the actor/actors """

class FullGlobalObservationFn(ObservationFn):
    """ The agent gets all possible information.
        *Viable for centralized baselines such as CPPO*
        *Viable for any decentralized baseline such as IPPO and MAPPO*

        Note: Here the swimmer is the agent and observer.

        All possible relative features between agents, targets, and particles are included.

        # Absolute Features:
            - all agent / swimmer features are observed (x, y, Δx, Δy) for each agent
                - actions from agents are not observed for simplicity and coding flexibility
                - absolute coordinates are provided to the decentralized agent as local information now == global information
            - all target features are observed (x, y, is_occupied) for each agent

        # Relative Features:
            - all n-n agent-agent features are observed (x_rel, y_rel, Δx_rel, Δy_rel, r_rel, Δr_rel) for each actor/agent, respectively
            - all n-m agent-target features are observed (x_rel, y_rel, Δx_rel, Δy_rel, r_rel, Δr_rel) for each actor/agent, respectively
                - These features could be n-n and n-m, but this would be too much information and bad scalability.
                - feature count is (4*n_swimmers) + (3*m_targets) + (n_swimmers*[n_swimmers - 1]*6) + (n_swimmers*m_targets*6) *per agent if decentralized* or *for one agent if centralized*
    """

    def __init__(self, centralized : bool):
        self.centralized = centralized
        if self.centralized is True:
            self.call_func = self.centralized_call_func
        elif self.centralized is False:
            self.call_func = self.decentralized_call_func

    def __call__(self, state : State) -> chex.Array:
        observation = self.call_func(state = state)
        return observation

    def centralized_call_func(self, state : State):
        """ - For baselines such as CPPO: do not include duplicate agent/swimmer, and target features.
            - Calculate Relative features more efficiently
        """
        x_swimmer          = state.swimmer_state.x # shape (n_swimmers,)
        y_swimmer          = state.swimmer_state.y
        x_past_swimmer     = state.swimmer_state.x_past
        y_past_swimmer     = state.swimmer_state.y_past

        x_target           = state.target_state.x
        y_target           = state.target_state.y

        is_occupied_target = state.target_state.is_occupied

        ### swimmer features ###
        swimmer_obs = jnp.concatenate((x_swimmer, y_swimmer, x_swimmer - x_past_swimmer, y_swimmer - y_past_swimmer), axis=-1)

        ### target features ###
        target_obs  = jnp.concatenate((x_target, y_target, is_occupied_target), axis=-1)

        ### n-n swimmer-swimmer features ###
        x_rel         = x_swimmer[:, None]      - x_swimmer[None, :] # shape (n_swimmers, n_swimmers)
        y_rel         = y_swimmer[:, None]      - y_swimmer[None, :]

        x_rel_past    = x_past_swimmer[:, None] - x_past_swimmer[None, :]
        y_rel_past    = y_past_swimmer[:, None] - y_past_swimmer[None, :]

        mask_swimmers = ~jnp.eye(x_swimmer.shape[-1], dtype=bool)      # Make identity matrix with False on diagonal, True elsewhere

        x_rel         = x_rel[mask_swimmers]  # shape (n_swimmers*(n_swimmers-1))
        y_rel         = y_rel[mask_swimmers]
        x_rel_past    = x_rel_past[mask_swimmers]
        y_rel_past    = y_rel_past[mask_swimmers]
        r_rel         = jnp.sqrt((x_rel)**2 + (y_rel)**2)
        r_rel_past    = jnp.sqrt((x_rel_past)**2 + (y_rel_past)**2)
        delta_x_rel   = x_rel - x_rel_past
        delta_y_rel   = y_rel - y_rel_past
        delta_r_rel   = r_rel - r_rel_past

        rel_swimmer_swimmer_obs = jnp.concatenate((x_rel, y_rel, delta_x_rel, delta_y_rel, r_rel, delta_r_rel), axis=-1)

        ### n-m swimmer-target features ###
        x_rel         = x_swimmer[:, None]      - x_target[None, :] # shape (n_swimmers, n_targes)
        y_rel         = y_swimmer[:, None]      - y_target[None, :]

        x_rel_past    = x_past_swimmer[:, None] - x_past_swimmer[None, :]
        y_rel_past    = y_past_swimmer[:, None] - y_past_swimmer[None, :]

        # mask_swimmers         = jnp.triu(N=x_swimmer.shape[-1], k=0, dtype=bool)

        arr_shape = x_rel.shape
        x_rel         = x_rel.reshape(arr_shape[0] * arr_shape[1])  # shape (n_swimmers*m_targets)
        y_rel         = y_rel.reshape(arr_shape[0] * arr_shape[1])
        x_rel_past    = x_rel_past.reshape(arr_shape[0] * arr_shape[1])
        y_rel_past    = y_rel_past.reshape(arr_shape[0] * arr_shape[1])
        r_rel         = jnp.sqrt((x_rel)**2 + (y_rel)**2)
        r_rel_past    = jnp.sqrt((x_rel_past)**2 + (y_rel_past)**2)
        delta_x_rel   = x_rel - x_rel_past
        delta_y_rel   = y_rel - y_rel_past
        delta_r_rel   = r_rel - r_rel_past

        rel_swimmer_target_obs = jnp.concatenate((x_rel, y_rel, delta_x_rel, delta_y_rel, r_rel, delta_r_rel), axis=-1)

        centralized_obs = jnp.concatenate((swimmer_obs, target_obs, rel_swimmer_swimmer_obs, rel_swimmer_target_obs), axis=-1) # shape (n_features)
        return centralized_obs # shape (n_features)

    def decentralized_call_func(self, state : State):
        centralized_obs   = self.centralized_call_func(state = state)
        n_agents          = state.swimmer_state.x.shape[-1]
        decentralized_obs = jnp.tile(centralized_obs, (n_agents, 1)), # replicate the array shape (n_features) to each agent; decentralize the centralized array
        return decentralized_obs # shape (n_agents, n_features)

class ReducedFullGlobalObservationFn(ObservationFn):
    """ The agent gets only information relative to its own view, while a few absolute coordinate features are removed.

        Note: This is only a reduced form of FullGlobalObservationFn to allow more scalable observations for decentralized agents.
            --> centralized agents should just use FullGlobalObservationFn instead of ReducedGlobalObservationFn as there is too little difference

        *Not intended for centralized baselines such as CPPO*
        *Viable for any decentralized baseline such as IPPO and MAPPO*

        # Absolute Features:
            - only essential and personal agent / swimmer features are observed (x, y, Δx, Δy) for each agent
                - the agent only sees its own personal information; an not that of other swimmers
                - actions from agents are not observed for simplicity and coding flexibility
                - absolute coordinates are not visible to the decentralized agent
            - only essential target features are observed (is_occupied) for each agent

        # Relative Features:
            - only essential 1-n agent-agent features are observed (x_rel, y_rel, Δx_rel, Δy_rel, r_rel, Δr_rel) for each actor/agent, respectively
            - only essential 1-m agent-target features are observed (x_rel, y_rel, Δx_rel, Δy_rel, r_rel, Δr_rel) for each actor/agent, respectively
                - These features could be n-n and n-m, but this would be too much information and bad scalability.
                - feature count is (4) + (3*m_targets) + ((n_swimmers-1)*6) + (m_targets*6) *per agent if decentralized*
    """

    def __init__(self, centralized : bool):
        self.centralized = centralized
        assert (centralized is False), ('This ReducedGlobalObservationFn is not practical for centralized baselines such as CPPO. ' +
                                        'Consider using FullGlobalObservationFn for any centralized baseline instead.')
        self.call_func = self.decentralized_call_func

    def __call__(self, state : State) -> chex.Array:
        observation = self.call_func(state = state)
        return observation

    def centralized_call_func(self, state : State) -> chex.Array:
        raise NotImplementedError('It makes no sense to have a centralized ReducedGlobalObservationFn, '+\
                                  'when FullGlobalObservationFn gives essentially the same information for centralized=True.')

    def decentralized_call_func(self, state : State) -> chex.Array:

        x_swimmer          = state.swimmer_state.x
        y_swimmer          = state.swimmer_state.y
        x_past_swimmer     = state.swimmer_state.x_past
        y_past_swimmer     = state.swimmer_state.y_past

        x_target           = state.target_state.x
        y_target           = state.target_state.y

        is_occupied_target = state.target_state.is_occupied

        ### swimmer features ###
        x_rel = x_swimmer - x_past_swimmer # shape (N_swimmers)
        y_rel = y_swimmer - y_past_swimmer

        swimmer_obs = jnp.concatenate((x_rel[:, None], y_rel[:, None]), axis=-1) # shape (n_features, N_swimmers) (separate observed features per swimmer)

        ### target features ###

        target_obs = jnp.concatenate((is_occupied_target[:, None]), axis=-1) # shape (n_features, N_swimmers)

        ### n 1-n swimmer-swimmer features ###
        x_rel         = x_swimmer[:, None]      - x_swimmer[None, :] # assumes shape (n_swimmers, n_swimmers)
        y_rel         = y_swimmer[:, None]      - y_swimmer[None, :]

        x_rel_past    = x_past_swimmer[:, None] - x_past_swimmer[None, :]
        y_rel_past    = y_past_swimmer[:, None] - y_past_swimmer[None, :]

        mask_swimmers = ~jnp.eye(x_swimmer.shape[-1], dtype=bool)     # Make identity matrix with False on diagonal, True elsewhere
        n_swimmers_   = x_rel.shape[0]

        x_rel         = x_rel[mask_swimmers].reshape(n_swimmers_, -1)    # shape (n_swimmers, n_swimmers-1); just remove the main diagonal element here for each agent
        y_rel         = y_rel[mask_swimmers].reshape(n_swimmers_, -1)
        x_rel_past    = x_rel_past[mask_swimmers].reshape(n_swimmers_, -1)
        y_rel_past    = y_rel_past[mask_swimmers].reshape(n_swimmers_, -1)
        r_rel         = jnp.sqrt((x_rel)**2 + (y_rel)**2)
        r_rel_past    = jnp.sqrt((x_rel_past)**2 + (y_rel_past)**2)
        delta_x_rel   = x_rel - x_rel_past
        delta_y_rel   = y_rel - y_rel_past
        delta_r_rel   = r_rel - r_rel_past

        rel_swimmer_swimmer_obs = jnp.concatenate((x_rel, y_rel, delta_x_rel, delta_y_rel, r_rel, delta_r_rel), axis=-1)

        ### n 1-m swimmer-target features ###
        x_rel         = x_swimmer[:, None]      - x_target[None, :] # shape (n_swimmers, n_targets) --> already is separate feature per swimmer/agent
        y_rel         = y_swimmer[:, None]      - y_target[None, :]

        x_rel_past    = x_past_swimmer[:, None] - x_past_swimmer[None, :]
        y_rel_past    = y_past_swimmer[:, None] - y_past_swimmer[None, :]

        r_rel         = jnp.sqrt((x_rel)**2 + (y_rel)**2)
        r_rel_past    = jnp.sqrt((x_rel_past)**2 + (y_rel_past)**2)

        rel_swimmer_target_obs = jnp.concatenate((x_rel, y_rel, x_rel - x_rel_past, y_rel - y_rel_past, r_rel, r_rel - r_rel_past), axis=-1)

        observation = jnp.concatenate((swimmer_obs, target_obs, rel_swimmer_swimmer_obs, rel_swimmer_target_obs), axis=-1)

        return observation


class LocalNNearestObjectsObservationFn(ObservationFn):
    """ The agent observes only the information of the n nearest targets and agents/swimmers.
        This function could give a more limted view of the environment, such that the swimmers might have to explore the surroundings.

        *Not intended for centralized baselines such as CPPO*
        *Viable for any decentralized baseline such as IPPO and MAPPO*

        Note: Here the swimmer is the agent and observer.
        Note: This observation type might be relevant for some GNN baselines later.


        Only the observations of the n nearest objects (respectively for other agents/swimmers, targets, and particles) are included.

        # Absolute Features:
            - n nearest agent / swimmer features are observed (Δx, Δy) for each agent
                - actions from agents are not observed for simplicity and coding flexibility
                - absolute coordinates are not visible to the decentralized agent
            - n nearest agent target features are observed for each agent; the agent might be limited to see only occupied targets.

        # Relative Features:
            - n nearest agent 1-m agent-agent features are observed (x_rel, y_rel, Δx_rel, Δy_rel, r_rel, Δr_rel) for each actor/agent, respectively
            - n nearest agent 1-l agent-target features are observed (x_rel, y_rel, Δx_rel, Δy_rel, r_rel, Δr_rel) for each actor/agent, respectively

        The observations are always ordered from closest to farthest to maintain input consinstency.
    """

    def __init__(self, n : int, centralized : bool = False):
        self.centralized = centralized
        self.n = n
        if self.centralized is True:
            self.call_func = self.centralized_call_func
            raise NotImplementedError('This might not be needed yet for a centralized benchmark, so leaving it out.')
        elif self.centralized is False:
            self.call_func = self.decentralized_call_func

    def __call__(self, state : State) -> chex.Array:
        observation =  self.call_func(state = state)
        return observation

    def centralized_call_func(self, state : State):
        raise NotImplementedError('This might not be needed yet for a centralized benchmark, so leaving it out.')
        # decentralized_observation = decentralized_call_func(state = state)
        # centralized_observation   = decentralized_observation.flatten()
        # return centralized_observation

    def decentralized_call_func(self, state : State) -> chex.Array:
        x_swimmer          = state.swimmer_state.x
        y_swimmer          = state.swimmer_state.y
        x_past_swimmer     = state.swimmer_state.x_past
        y_past_swimmer     = state.swimmer_state.y_past

        x_target           = state.target_state.x
        y_target           = state.target_state.y

        is_occupied_target = state.target_state.is_occupied

        ### nearest targets, agents/swimmers, particles ###

        x_rel_swimmer_target  = x_swimmer[:, None]      - x_target[None, :] # shape (n_swimmers, n_targets) --> already is separate feature per swimmer/agent
        y_rel_swimmer_target  = y_swimmer[:, None]      - y_target[None, :]
        x_rel_swimmer_swimmer = x_swimmer[:, None]      - x_swimmer[None, :] # shape (n_swimmers, n_swimmers) --> already is separate feature per swimmer/agent
        y_rel_swimmer_swimmer = y_swimmer[:, None]      - y_swimmer[None, :]

        r_rel_swimmer_target  = jnp.sqrt((x_rel_swimmer_target)**2 + (y_rel_swimmer_target)**2)
        r_rel_swimmer_swimmer = jnp.sqrt((x_rel_swimmer_swimmer)**2 + (y_rel_swimmer_swimmer)**2)

        n_nearest_target_idx  = jnp.argsort(r_rel_swimmer_target)[:, :self.n]     # shape (n_swimmers, self.n)
        n_nearest_swimmer_idx = jnp.argsort(r_rel_swimmer_swimmer)[:, 1:self.n+1] # shape (n_swimmers, self.n) # n other swimmers (excluding itsself)

        x_rel_past_swimmer_swimmer = x_past_swimmer[:, None] - x_past_swimmer[None, :] # shape (n_swimmers, n_swimmers) --> already is separate feature per swimmer/agent
        y_rel_past_swimmer_swimmer = y_past_swimmer[:, None] - y_past_swimmer[None, :]
        r_rel_past_swimmer_swimmer = jnp.sqrt((x_rel_past_swimmer_swimmer)**2 + (y_rel_past_swimmer_swimmer)**2)

        x_rel_past_swimmer_target  = x_past_swimmer[:, None] - x_target[None, :] # shape (n_swimmers, n_swimmers) --> already is separate feature per swimmer/agent
        y_rel_past_swimmer_target = y_past_swimmer[:, None] - y_target[None, :]
        r_rel_past_swimmer_target = jnp.sqrt((x_rel_past_swimmer_target)**2 + (y_rel_past_swimmer_target)**2)


        ### utility function ###
        def select_nnearest_from_array(nnearest_idx : chex.Array, data_array : chex.Array) -> chex.Array:
            dim0_idx = jnp.arange(data_array.shape[0])
            return data_array[dim0_idx[:, None], n_nearest_target_idx]

        ### swimmer features ###
        delta_x_swimmer    = x_swimmer - x_past_swimmer # shape (n_swimmers)
        delta_y_swimmer    = y_swimmer - y_past_swimmer # shape (n_swimmers)

        swimmer_obs = jnp.concatenate((delta_x_swimmer[:, None], delta_y_swimmer[:, None]), axis=-1)
            # delta_x_swimmer[:, None] --> shape (n_swimmers, 1) (each swimmer does only observe itsself here)

        # ### target features ###
        is_occupied_target_ = jnp.array([is_occupied_target]).repeat(axis=0, repeats=x_swimmer.shape[0]) # repeat shape (1, n_targets) --> (n_swimmers, n_targets)
        is_occupied_target_ = select_nnearest_from_array(n_nearest_swimmer_idx, is_occupied_target_) # shape (n_swimmers, self.n)

        target_obs = jnp.concatenate((is_occupied_target_), axis=-1) # shape (n_swimmers, self.n + 0)

        # ### n 1-n swimmer-swimmer features ###

        # shape (n_swimmers, n_swimmers) --> (n_swimmers, self.n)
        input_arrays = jnp.array([x_rel_swimmer_swimmer,
                                  y_rel_swimmer_swimmer,
                                  x_rel_past_swimmer_swimmer,
                                  y_rel_past_swimmer_swimmer,
                                  r_rel_swimmer_swimmer,
                                  r_rel_past_swimmer_swimmer])

        x_rel_swimmer_swimmer_,\
        y_rel_swimmer_swimmer_,\
        x_rel_past_swimmer_swimmer_,\
        y_rel_past_swimmer_swimmer_,\
        r_rel_swimmer_swimmer_,\
        r_rel_past_swimmer_swimmer_ = jax.vmap(select_nnearest_from_array, in_axes=(None, 0))(n_nearest_swimmer_idx, input_arrays)

        delta_x_rel_swimmer_swimmer_ = x_rel_swimmer_swimmer_ - x_rel_past_swimmer_swimmer_
        delta_y_rel_swimmer_swimmer_ = y_rel_swimmer_swimmer_ - y_rel_past_swimmer_swimmer_
        delta_r_rel_swimmer_swimmer_ = r_rel_swimmer_swimmer_ - r_rel_past_swimmer_swimmer_

        rel_swimmer_swimmer_obs = jnp.concatenate((x_rel_swimmer_swimmer_,
                                                   y_rel_swimmer_swimmer_,
                                                   delta_x_rel_swimmer_swimmer_,
                                                   delta_y_rel_swimmer_swimmer_,
                                                   r_rel_swimmer_swimmer_,
                                                   delta_r_rel_swimmer_swimmer_),
                                                  axis=-1)

        # ### n 1-m swimmer-target features ###
        input_arrays = jnp.array([x_rel_swimmer_target,
                                  y_rel_swimmer_target,
                                  x_rel_past_swimmer_target,
                                  y_rel_past_swimmer_target,
                                  r_rel_swimmer_target,
                                  r_rel_past_swimmer_target])


        # r_rel         = jnp.sqrt((x_rel)**2 + (y_rel)**2)
        # r_rel_past    = jnp.sqrt((x_rel_past)**2 + (y_rel_past)**2)
        x_rel_swimmer_target_,\
        y_rel_swimmer_target_,\
        x_rel_past_swimmer_target_,\
        y_rel_past_swimmer_target_,\
        r_rel_swimmer_target_,\
        r_rel_past_swimmer_target_ = jax.vmap(select_nnearest_from_array, in_axes=(None, 0))(n_nearest_target_idx, input_arrays)

        delta_x_rel_swimmer_target_ = x_rel_swimmer_target_ - x_rel_past_swimmer_target_
        delta_y_rel_swimmer_target_ = y_rel_swimmer_target_ - y_rel_past_swimmer_target_
        delta_r_rel_swimmer_target_ = r_rel_swimmer_target_ - r_rel_past_swimmer_target_

        rel_swimmer_target_obs = jnp.concatenate((x_rel_swimmer_target_,
                                                  y_rel_swimmer_target_,
                                                  delta_x_rel_swimmer_target_,
                                                  delta_y_rel_swimmer_target_,
                                                  r_rel_swimmer_target_,
                                                  delta_r_rel_swimmer_target_),
                                                 axis=-1)

        observation = jnp.concatenate((swimmer_obs, target_obs, rel_swimmer_swimmer_obs, rel_swimmer_target_obs), axis=-1) # shape (n_swimmers, n_obs*n_features)
        return observation