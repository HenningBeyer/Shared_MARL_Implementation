from typing import NamedTuple
import chex

class SimpleTarget2DState(NamedTuple):
    x           : chex.Array # an array holding all x positions of all targets
    y           : chex.Array # an array holding all y positions of all targets
    is_occupied : chex.Array # an array holding all bools of all targets
