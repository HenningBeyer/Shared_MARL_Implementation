from typing import NamedTuple
import chex

class SimpleLaser2DState(NamedTuple):
    x           : chex.Array # an array holding all x positions of all lasers
    y           : chex.Array # an array holding all y positions of all lasers
    v_x_act     : chex.Array # an array holding all v_x actions
    v_y_act     : chex.Array # an array holding all v_y actions


class RealisticLaser2DState(NamedTuple):
    pass # maybe to do later
    # x           : chex.Array # an array holding all x positions of all lasers
    # y           : chex.Array # an array holding all y positions of all lasers
    # v_x_act     : chex.Array # an array holding all v_x actions
    # v_y_act     : chex.Array # an array holding all v_y actions
