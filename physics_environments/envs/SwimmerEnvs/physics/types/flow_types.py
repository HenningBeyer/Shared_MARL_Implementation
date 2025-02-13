from typing import NamedTuple
import chex

class FlowField2DState(NamedTuple):
    # Visible Information (perceivable by other agents)
    X           : chex.Array # an array holding all x positions of all swimmers
    Y           : chex.Array # an array holding all y positions of all swimmers
