# swimmers are kept fully homogeneus as physical particle, and agent type, for simplicity.
  # heterogeneous aspects might have to be introduced later; but would allow specific experiments
  # This stage of prototype focuses on fully homogeneous agents all along.

# Some performance notes for JAX:
# (Consider them for the performance-critical main-loops)
"""
    Data types in JAX should be immutable, i.e they have to be jnp.arrays and pytrees only; chex.dataclass and NamedTuple are very relevant too
        - a chex.dataclass is slightly more flexible as it is immutable
        - a NamedTuple is faster than a chex.dataclass, but might be less flexible as it is immuatable, else they are very similar
            --> everything can be a named tuple for performance; implementations outside the jax loop can be chex.dataclass or NamedTuple
            - a NamedTuple is more tuple-like, a chex.dataclass is more dict-like
     having numpy arrays (CPU data) or basic data types (CPU data) or phython dicts (CPU data) harms compiled JAX loops on a GPU
        - bad cases of this unbearable compilation times and slower step times.
    You also do not have any python loop constants or globals in JAX.
    You don't use any string types in a JAX loop, JAX array, or JAX pytree; its for numerical computing
        - Everything with strings, logging, printing might be separated entirely from the jax loop!
            - If you still want to do logging, you may make an extra python for-loop as in https://github.com/instadeepai/Mava/blob/develop/mava/systems/ppo/anakin/ff_mappo.py#L467
            - If you need it to be faster, and can leave out some logging, it needs to be a pure jax program.
    Also, no python for loops, while loops, and if statements should be in a JAX loop!
        - use lax.scan, lax.fori_loop, lax.while_loop, lax.cond, ... in https://jax.readthedocs.io/en/latest/jax.lax.html#control-flow-operators
    Consider reading https://jax.readthedocs.io/en/latest/notebooks/autodiff_cookbook.html, whenever you need to calculate derivatives in JAX.
"""

# Note: Only the environment parameters are mutable dataclass, as parameters might be overwritten by hydra or common overrides.abs
#   Else, other data types should be immutable NamedTuples as a good practice

# Note that every function is decomposed into an initializable function as this sets default attributes for the jax loop,
#   without having al these messy globals floating around!

# Note JAX is running FP-32 precision by default for simulations.

# You should use ruff as a VSCode extension for easier developing and clean code!
# On windows you need to set up a WSL ubuntu to run JAX on GPU, You need to connect VScode to Ubuntu via a WSL extension in VSCode

# Design Choice:
# States: Should only contain the minimal information to calculate the environment step and all types of observations per step along side the environment constants.
#   - The agent nevery receives state information directly!
# Observations: Should contain the information only intended for an agent; it gets calculated from the state.

# Design Choice:
# States were defined as mutable chex.dataclass in Jumanji, however there is no reason for having a chex.dataclass over a NamedTuple
#   --> so chosing a NamedTuple for performance and immutability.

""" TODO: a reliable benchmarking for MARL is essential as in:
    - Benchmarl https://github.com/facebookresearch/BenchMARL/tree/main, https://jmlr.org/papers/volume25/23-1612/23-1612.pdf,
    - Mava https://github.com/instadeepai/Mava, https://arxiv.org/pdf/2107.01460
    - Rliable https://github.com/google-research/rliable, https://arxiv.org/pdf/2108.13264.pdf
    - MARLeval https://github.com/instadeepai/marl-eval, https://arxiv.org/pdf/2209.10485
        --> detailed ablation studies for different rewards, observation types, ... are a good idea
        --> many detailed and different benchmark plots are important as well, at best automated!

    - A rigorous documentation of each baseline should exist in either paper/code comments/framework documentation
        - This is inspired by model cards https://arxiv.org/pdf/1810.03993 that were designed for ML models
        - One might describe a MARL systems and RL agorithms with their details too!

    - Note that the evaluation procedure for real-world tasks may not be fully developed yet.

    --> marl-eval https://github.com/instadeepai/marl-eval is a good library mava and benchmarl are using for marl evaluation
    - it is highly suggested and planned to integrate their evaluation scheme to the working experiments
"""
# Optional Papers
# https://arxiv.org/pdf/2312.08463
# DeRL https://proceedings.mlr.press/v80/colas18a/colas18a.pdf
# α-rank https://arxiv.org/pdf/1903.01373
# ResponseGraphUCB: https://proceedings.neurips.cc/paper_files/paper/2019/file/510f2318f324cf07fce24c3a4b89c771-Paper.pdf
# Melting Pot Evaluation Protocol... https://proceedings.mlr.press/v139/leibo21a/leibo21a.pdf
    -
""" TODO: We definitely need hydra and yaml files to manage all those parameters. BenchMARL, Mava, and TorchRL do this too. """
""" TODO: it will be interesting to have GNNs, RNNs, and CNNs for these simulations --> They can provide a lot of performance for the agents. """
