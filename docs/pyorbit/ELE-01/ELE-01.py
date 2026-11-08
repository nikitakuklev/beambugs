"""ELE-01: Linac Gaussian rotation-error setters give every node the rms angle instead of a random draw

`StraightRotationX_NodesModification`, `StraightRotationY_NodesModification` and
`StraightRotationZ_NodesModification.setGaussDistributedAngle(σ, cut_off_level)` draw a truncated Gaussian angle for
each linac node and then broadcast and assign σ itself, so every node is rotated by the same angle +σ. A random set of
pitch, yaw or roll errors becomes a systematic one. For 200 quadrupoles with σ = 1 mrad and a 3σ cut, all 200 angles
are 1.000 mrad, where a Gaussian truncated at 3σ has mean 0 (standard error 0.07 mrad), rms 0.9866 mrad and both
signs. The displacement generator of the same module assigns its draws correctly.

The script attaches each of the three rotation generators to linac `Quad` nodes. First it replaces `random.gauss` by a
stub that returns known values, so the angle each node must receive is known exactly; then it draws 200 angles with a
fixed seed and compares their mean, rms and signs with the truncated normal distribution. The control is the
coordinate-displacement generator of the same module, run the same two ways.
"""

import math
import os
import random

for _v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ.setdefault(_v, "1")

from orbit.py_linac.lattice import Quad
from orbit.py_linac.lattice_modifications import (
    CoordinateDisplacementNodesModification,
    StraightRotationX_NodesModification,
    StraightRotationY_NodesModification,
    StraightRotationZ_NodesModification,
)

ROTATIONS = (
    StraightRotationX_NodesModification,
    StraightRotationY_NodesModification,
    StraightRotationZ_NodesModification,
)
N, SIGMA, CUT = 200, 1.0e-3, 3.0
# rms of a normal distribution truncated at +-c sigma: sigma*sqrt(1 - 2 c phi(c) / (2 Phi(c) - 1)),
# phi the standard normal density and 2 Phi(c) - 1 = erf(c / sqrt 2); for c = 3 this is 0.98658 sigma.
RMS_TRUNC = SIGMA * math.sqrt(
    1 - 2 * CUT * math.exp(-CUT * CUT / 2) / math.sqrt(2 * math.pi) / math.erf(CUT / math.sqrt(2))
)
STUB = [0.1, -0.2, 0.3, -0.4, 0.5]  # stub draws in units of sigma, all inside the 3 sigma cut


def quads(n):
    out = []
    for i in range(n):
        q = Quad(f"q{i}")
        q.setLength(0.4)
        out.append(q)
    return out


def assign(cls, n, sigma):
    """Attach the generator to n quadrupoles, call its Gaussian setter and return the assigned values."""
    m = cls("m")
    m.addLatticeNodes(quads(n))
    if cls is CoordinateDisplacementNodesModification:
        m.setGaussDistributedDisplacementParameter("dx", sigma, CUT)
        return [c.getDisplacementParameters()[0] for c in m.getErrorControllers()]
    m.setGaussDistributedAngle(sigma, CUT)
    return [c.getRotationAngle() for c in m.getErrorControllers()]


def stubbed(cls):
    draws = iter(STUB)
    real = random.gauss

    def fake(mu=0.0, sigma=1.0):  # the setter calls random.gauss(0.0, sigma) once per node
        return next(draws) * sigma

    random.gauss = fake  # ty: ignore[invalid-assignment]
    try:
        return assign(cls, len(STUB), SIGMA)
    finally:
        random.gauss = real


def stats(values):
    mean = sum(values) / len(values)
    rms = math.sqrt(sum(v * v for v in values) / len(values))
    return mean, rms, sum(1 for v in values if v < 0), len(set(values))


reproduced = True
expected = [s * SIGMA for s in STUB]  # each node must receive its own draw
for cls in ROTATIONS + (CoordinateDisplacementNodesModification,):
    control = cls is CoordinateDisplacementNodesModification
    label = cls.__name__ + (" dx (control)" if control else "")
    got = stubbed(cls)
    print(
        f"[{label}, stub draws] assigned {[f'{v:.1e}' for v in got]}; expected the draws {[f'{v:.1e}' for v in expected]}"
    )
    random.seed(20261107)
    mean, rms, neg, distinct = stats(assign(cls, N, SIGMA))
    print(
        f"[{label}, {N} nodes, sigma = 1e-3, cut 3 sigma] mean {mean:.4e}, rms {rms:.4e}, negatives {neg}, distinct "
        f"{distinct}; expected mean 0 +- {RMS_TRUNC / math.sqrt(N):.1e} (standard error), rms {RMS_TRUNC:.4e}, "
        f"about {N // 2} negatives, {N} distinct"
    )
    matches = (
        all(abs(g - e) < 1e-18 for g, e in zip(got, expected)) and distinct == N and neg > N // 4
    )
    reproduced &= matches if control else not matches and all(g == SIGMA for g in got)
print("DEFECT REPRODUCED" if reproduced else "NOT REPRODUCED")
