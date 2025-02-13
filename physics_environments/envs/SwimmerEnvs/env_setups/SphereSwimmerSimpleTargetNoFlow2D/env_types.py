from typing import TYPE_CHECKING, NamedTuple
import chex

if TYPE_CHECKING:
    from dataclasses import dataclass
else:
    from chex import dataclass


""" Define the custom Environment types for the setup 'SphereSwimmerSimpleTargetNoFlow2D' """

@dataclass
class SwimmerEnvironment2DParams:
    """ The common parameters every flexible swimmer environment is definable with. """
    #
    t_simulation_episode        : chex.Numeric = 1.00      # in s;   # the length of the simulation episode  Note: an agent can learn within an episode (num_updates = t_simulation_episode // batch_size, or similar)
    act_dt                      : chex.Numeric = 0.01000           # in s;   # act_dt sets the interval the agent can act in the simulation; act_dt should divide act_dt to not run into issues.
    sim_dt                      : chex.Numeric = 0.00100           # in s;   # sim_dt should be small enough for realistic simulations; sim_dt should divide t_simulation_episode to not run into issues.

    particle_radius             : chex.Numeric = 1e-6              # in m; radius for all particles
    env_size                    : chex.Array   = [250e-6, 250e-6]  # in m, might be square always.

    n_swimmers                  : chex.Numeric = 5
    n_targets                   : chex.Numeric = 5
    thermostat_type             : str          = "Langevin"              # defines the particle model                  # Literal["Langevin"]                                                                                              # could be extended with ["Brownian", "NoseHoover", ...], if needed for other simulations
    particle_interaction_type   : str          = "WCAPotential"          # defines the potential between particles     # Literal["NoPotential", "WCAPotential"]                                                                           # could be extended with other types such as ["LJ", ...], if needed for other types of simulations
    training_generator_type     : str          = "UniformGenerator"      # Literal["UniformGenerator"]                 # --> could extend this with ["SceneGenerator", "UniformSceneGenerator", "UniformGenerator"]

    flow_field_type             : str          = "NoField"               # defines the flowfield in simulation         # Literal["NoField", "SpiralField2D"]                                                                              # could be extended with other custom flow fields, and flowfields controlled by actions, ...
    flow_update_func            : str          = "ConstantFlow"          # defines the fluid simulation method "ConstantFlow" is only applying a constant the current flow field induced by other objects
        # --> TODO for flow simulation, we might need some type of fluid simulation model over time.
        # --> we might need to define which kind of accuracy is needed, and which kind of fluid simulation model we need.

    swimmer_type                : str          = "DiscreteSphereSwimmer" # defines the controllable swimmer type       # Literal["DiscreteSphereSwimmer", "DiscreteJanusSwimmer", "ContinuousSphereSwimmer", "ContinuousJanusSwimmer"]
    laser_agent_type            : str          = "NoLaser"

# @dataclass
# Useful Reference: https://github.com/hgrecco/pint/blob/c350f02117ec03e2fb6b58a0bce9b997d0c2fa23/pint/constants_en.txt
class EnvPhysicsConstants(NamedTuple):
    k_b           : chex.Numeric = 1.380649e-23
    dyn_viscosity : chex.Numeric = 0.800e-3     # in mPa*s; approximated value; dynamic fluid viscosity η; varies strongly per fluid, temperature, and pressure; assumes all of them constant!
    T             : chex.Numeric = 300.0        # in K; assumed to be constant for now at all parts of the environment!

class EnvConstants(NamedTuple):
    t_simulation_episode : chex.Numeric
    act_dt               : chex.Numeric
    sim_dt               : chex.Numeric
    particle_radius      : chex.Numeric
    env_size             : chex.Numeric
    n_swimmers           : chex.Numeric
    n_targets            : chex.Numeric
    n_flow_grid_cells    : chex.Numeric
    mesh_grid            : chex.Array # the constant meshgrid of the env
    env_physics_consts   : EnvPhysicsConstants


class MetricState(NamedTuple):
    """ We create a metric state to store all possibly relevant episode metrics for evaluation. Made for the environment 'SphereSwimmerSimpleTargetNoFlow2D'
        It was chosen to calculate these metrics within the jax loop as of faster and more flexible access to the state metrics per step
    """
    did_solve     : chex.Numeric
    time_to_solve : chex.Numeric
    occupied_target_percentage : chex.Numeric
    occupied_target_count      : chex.Numeric
    total_return               : chex.Numeric


# @dataclass
class State(NamedTuple):
    """ All the necessary information to calculate agent observations and the next environment State wthin an environment step.
        Also contains some state variables needed for logging, terminating, etc. """
    t                : chex.Numeric # step time
    step_count       : chex.Numeric
    key              : chex.PRNGKey # Jax.random.key(seed=42)
    swimmer_state    : SwimmerState
    target_state     : TargetState
    flow_field_state : FlowFieldState
    metric_state     : MetricState

# class SwimmerObservationLocal(NamedTuple):
#     """ Choosing a simple observation type for all agents with one entry of arbitrary dimension.
#         Image data could make a new entry if needed - i.e. feature groups that needed to be handled differently by the agent.
#     """
#     individual_agent_observation  : chex.Array # --> shape (4 + 9n + 9n)

# --> needs to be planned based on the baselines/pipeline
