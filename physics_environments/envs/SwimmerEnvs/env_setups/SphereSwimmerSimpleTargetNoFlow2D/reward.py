"""

Good read: https://agents.inf.ed.ac.uk/blog/multiagent-learning-environments/

We defined 3 different rewards:
    - an independent reward for each agent: each agent follows its own goal
        - there might be slight competition and a lack of cooperation
        - also allows for independent task
    - a collaborative reward for each agent: each agent receives the collective team reward + receives its independent reward.
        - there might be a good mix of independent behavoir while beeing cooperative
    - a cooperative reward for each agent one agent group: each agent receives the collective team reward.
        - agents might lack independent behavior.

In total, one is able define these rewards for any environment, depending on the needs:
    - CompetitiveAgentRewardFn, CompetitiveGroupRewardFn, MixedAgentRewardFn,
      IndependentAgentRewardFn, CooperativeAgentGroupRewardFn, CollaborativeAgentGroupRewardFn

Future development could allow further experiments with more complicated multi-agent settings:
    - There might be more than one agent groups with different rewards between two agent groups.
    - There might be more complex environments with more complex tasks that are more intertwined
        - This could require multiple types of specific MixedAgentReward functions

- a competitive reward does not make much sense in this exact environment setting.
"""
import abc

import chex
import jax.numpy as jnp
import jax

from physics_environments.envs.rl_pendulum.types import State, EnvConstants

class RewardFn(abc.ABC):
    @abc.abstractmethod
    def __call__(self, state: State) -> chex.Array:
        """ Call method for computing the reward given current state and selected action. """

class CompetitiveAgentRewardFn(RewardFn):
    """ Only compatible with baselines that assign their reward per agent!

        Keep this as placeholder and blue-print for other potential environments, where a competitive reward could make sense.

        The competition should be amongst each agent for CompetitiveAgentRewardFn.
        The loss reward of one agent is the loss of another agent either directly, or for fighting over limited reward sources.
    """
    pass

class CompetitiveGroupRewardFn(RewardFn):
    """ Fully compatible with every baseline, as the shared reward can be assigned per agent and agent group.

        Keep this as placeholder and blue-print for other potential environments, where a competitive reward could make sense.

        The competition should be amongst each agent group for CompetitiveGroupRewardFn.
            - This means the task also should be cooperative within each agent group!
        The loss reward of one agent group is the loss of another agent group either directly, or for fighting over limited reward sources.
    """
    pass

class MixedAgentRewardFn(RewardFn):
    """ Only compatible with baselines that assign their reward per agent!

        Keep this as placeholder and blue-print for other potential environments, where a more mixed reward could make sense.
            - The reward and the tasks tend to be more complicated and exhbit more mixed behavior (all collaborative, independent and competitive behavior)
            - One is able to define multiple types of mixed reward functions
                - One should still specify the expected behavior, such as MixedCollaborativeCompetitive to differentiate them better.

        Fall back to MixedAgentRewardFn, when all these other reward functions are to explicit:
            - CompetitiveAgentRewardFn, CompetitiveGroupRewardFn, IndependentAgentRewardFn,
              CooperativeAgentGroupRewardFn, CollaborativeAgentGroupRewardFn

        The loss reward of one agent group is the loss of another agent group either directly, or for fighting over limited reward sources.
    """
    pass

class IndependentAgentRewardFn(RewardFn):
    """ Only compatible with baselines that assign their reward per agent!
        This reward is may cooperative task or is of interest for independent learning.

        Competition and collaboration should not exist as there is always a reward for following an unoccupied target.

        Individual Rewards:
            - A Reward (can be positive and negative) is given for reducing the distance to the nearest (unoccupied) target.
                - When a target gets occupied by another agent, no penalty is given, some extra reward will be given for following the new target.
                - No target left means 0 distance reward.
            - An extra reward is given / added as long as the agent is occupying the target.

        - There might be slighly competitive behavior to observe
        - However this reward function might prove sufficient as rewards are directly attributed to each agent without having any cooperative rewards.

        Always stays an independent learning reward per agent
            - the reward is duplicated for each agent.
            - used for any IL method such as IPPO. Could be used for MAPPO-Ind as an exception.
    """

    def __init__(self, env_constants : EnvConstants):
        self.arena_length = env_constants.env_size[0]
        self.episode_t    = env_constants.t_simulation_episode
        self.act_dt       = env_constants.act_dt

    def single_agent_reward(self,
                            x_targets      : chex.Array,  # shape (n_targets)
                            y_targets      : chex.Array,  # shape (n_targets)
                            o_targets      : chex.Array,  # shape (n_targets)
                            swimmer_x_all  : chex.Array,  # shape (n_swimmers)
                            swimmer_y_all  : chex.Array,  # shape (n_swimmers)
                            swimmer_x      : chex.Array,  # shape (1) # the respective swimmer to calculate the reward for
                            swimmer_y      : chex.Array,  # shape (1) # the respective swimmer to calculate the reward for
                            swimmer_x_past : chex.Array,  # shape (1) # the respective swimmer to calculate the reward for
                            swimmer_y_past : chex.Array   # shape (1) # the respective swimmer to calculate the reward for
                            ) -> chex.Array:
        """ This function is vmapped for multiple agents in parallel. """
        r            = jnp.sqrt((x_targets - swimmer_x)**2      + (y_targets - swimmer_y)**2)
        r_past       = jnp.sqrt((x_targets - swimmer_x_past)**2 + (y_targets - swimmer_y_past)**2)
        dist_reward  = (r_past - r)/self.arena_length # norm the sum of the maximum possible reward to approximate value of one


        nearest_free_target_idx = jnp.argmin(r + o_targets*jnp.inf, axis=-1) # will pick the index of nearest unoccupied targets if they exist; edge case: all targets are occupied: returns 0
        dist_reward = dist_reward[nearest_free_target_idx]
        dist_reward = dist_reward*~o_targets[nearest_free_target_idx] # filtering the case if all targets occupied; set to 0 or leave the value as is


        nearest_target_idx       = jnp.argmin(r, axis=-1)
        nearest_swimmer_dist     = jnp.min(jnp.sqrt((x_targets[nearest_target_idx] - swimmer_x_all)**2 + (y_targets[nearest_target_idx] - swimmer_y_all)**2), axis=-1)

        agent_is_occupier_reward = (r == nearest_swimmer_dist)*o_targets[nearest_target_idx]  # reward if the agent is the nearest to an occupied target
        agent_is_occupier_reward = agent_is_occupier_reward/(self.episode_t // self.act_dt)*10 # norm the sum of the maximum possible rewards to approximate value of 10

        return dist_reward + agent_is_occupier_reward # shape (1)

    def __call__(self, state: State) -> chex.Array:
        x_targets = state.target_state.x
        y_targets = state.target_state.y
        o_targets = state.target_state.is_occupied

        x_swimmers      = state.swimmer_state.x
        y_swimmers      = state.swimmer_state.y
        x_past_swimmers = state.swimmer_state.x_past
        y_past_swimmers = state.swimmer_state.y_past

        all_swimmer_rewards = jax.vmap(self.single_agent_reward,
                                       in_axes=(None,None,None,None,None,0,0,0,0))(x_targets       = x_targets,
                                                                                   y_targets       = y_targets,
                                                                                   o_targets       = o_targets,
                                                                                   swimmer_x_all   = x_swimmers,
                                                                                   swimmer_y_all   = y_swimmers,
                                                                                   x_swimmers      = x_swimmers,
                                                                                   y_swimmers      = y_swimmers,
                                                                                   x_past_swimmers = x_past_swimmers,
                                                                                   y_past_swimmers = y_past_swimmers) # shape (n_swimmers)

        return all_swimmer_rewards # shape (n_swimmers)

class CooperativeAgentGroupRewardFn(RewardFn):
    """ Fully compatible with every baseline, as the shared reward can be assigned per agent and agent group.

        Team Rewards:
            - Each agent gets a reward for the mean of all individual distance differences from all agents and their repective nearest targets.
                - its the mean of the individual rewards
            - Each agent gets a reward when one agent occupies an unoccupied target.
                - its also just the mean of the individual rewards
            - If each target is occupied, each agent gets a large reward, to promote solving the task perfectly.
            - Hypothesis: There might be lazy agents without individual reward contributions; especially, the more agents there are
                - However, there won't be competitive behavior as of no individual reward terms.

        --> fully reusing IndependentAgentRewardFn as its code does all the needed calculations.
    """

    def __init__(self, env_constants : EnvConstants, centralized : bool):
        self.episode_t    = env_constants.t_simulation_episode
        self.act_dt       = env_constants.act_dt
        self.centralized  = centralized

        self.Independent_Reward_Func = IndependentAgentRewardFn(env_constants=env_constants)

    def __call__(self, state: State) -> chex.Array:
        x_swimmers      = state.swimmer_state.x # shape (n_swimmers)
        independent_rewards = self.Independent_Reward_Func(state=state) # calling IndependentAgentRewardFn

        all_targets_occupied         = state.target_state.is_occupied.all() # shape (1) bool
        all_agents_do_occupy         = (jnp.sum(state.target_state.is_occupied, axis=-1) == x_swimmers.shape[-1]) # shape (1) bool
        task_fulfilled_reward        = (all_targets_occupied | all_agents_do_occupy)/(self.episode_t // self.act_dt)*10 # norm the sum of the maximum possible rewards to approximate value of 10

        cooperative_rewards = jnp.mean(independent_rewards, axis=-1) + task_fulfilled_reward # shape (1)

        cooperative_rewards = self.cooperative_rewards

        return cooperative_rewards

class CollaborativeAgentGroupRewardFn(RewardFn):
    """ Only compatible with baselines that assign their reward per agent!
            - This excludes baselines as CPPO, and MAPPO, while IPPO and MAPPO-Ind are allowed.

        Individual Rewards:
            - A Reward (positive+negative) is given for reducing the distance to the nearest (unoccupied) target.
            - Each agent gets a reward for occupying a target for itsself

        Team Rewards:
            - Each agent gets a slight reward for the mean of all individual distance differences from all agents and their repective nearest targets.
                - its just the mean of the individual distance rewards taken for the team
            - Each agent gets a slight reward when one agent occupied an unoccupied target.
            - Each agent gets a reward when every agent occupied a target, then the task is completed

        - Hypothesis: The intention of this collaborative reward is to provide better credit-assignment for cooperative tasks,
            that accounts directly for the contribution from each individual agent compared to CooperativeAgentGroupRewardFn.


        --> This function is (by occasion) able to reuse IndependentAgentRewardFn + CooperativeAgentGroupRewardFn for convenience.
    """

    def __init__(self, env_constants : EnvConstants):
        self.episode_t    = env_constants.t_simulation_episode
        self.act_dt       = env_constants.act_dt

        self.Independent_Reward_Func = IndependentAgentRewardFn(env_constants=env_constants)
        self.Cooperative_Reward_Func = CooperativeAgentGroupRewardFn(env_constants=env_constants)

    def __call__(self, state: State) -> chex.Array:
        independent_rewards   = self.CooperativeIndependentAgentRewardFn(state=state) # shape (n_swimmers)
        cooperative_rewards   = self.Cooperative_Reward_Func(state=state)             # shape (1)
        collaborative_rewards = independent_rewards + cooperative_rewards             # shape (n_swimmers)

        return collaborative_rewards
