from typing import NamedTuple
import chex

class ParticleState(NamedTuple):
    """ The dataclass used for typing only; it represents any Particle state listed below."""

class NoParticleState(NamedTuple):
    """ Used when no particles exist, Used by NoParticleGroupFn """

class TranslationalSphereParticle2DState(NamedTuple):
    x           : chex.Array # an array holding all x positions of all particles
    y           : chex.Array # an array holding all x positions of all particles

class SelfPropelledSphereParticle2DState(NamedTuple):
    x           : chex.Array # an array holding all x positions of all particles
    y           : chex.Array # an array holding all x positions of all particles

