from typing import NamedTuple
import chex

class SwimmerState(NamedTuple):
    """ The dataclass used for typing only; it represents any Particle state listed below."""

class NoSwimmerState(NamedTuple):
    """ The dataclass used for particles, if there do not exist any. Used by NoParticleGroupFn """

class TranslationalSphereSwimmer2DState(NamedTuple):
    x           : chex.Array # an array holding all x positions of all particles
    y           : chex.Array # an array holding all y positions of all particles
    F_act_x     : chex.Array
    F_act_y     : chex.Array

class SelfPropelledSphereSwimmer2DState(NamedTuple):
    x           : chex.Array # an array holding all x positions of all particles
    y           : chex.Array # an array holding all y positions of all particles
    M_torque    : chex.Array
    F_propel    : chex.Array
    angle       : chex.Array
