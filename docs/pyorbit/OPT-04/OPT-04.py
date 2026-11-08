"""OPT-04: `getTunes` adds a full turn for every negative-length element

`MATRIX_Lattice.trackTwissData`, which `TEAPOT_MATRIX_Lattice.getTunes` and the ring phase-advance
tables use, wraps each element's phase advance into [0, 2π) so that advances between π and 2π stay
positive. This also turns the small negative phase advance of a negative-length element into almost
2π, so the integrated tune grows by one for every slice of such an element. Inserting a −0.3 m drift
followed by a +0.3 m drift, whose product is the identity, into a six-cell FODO ring with tune
1.4753 raises `getTunes` to 2.4753, while the one-turn matrix and its fractional tune 0.4753 are
unchanged. Lattices of positive-length elements are unaffected.

The script builds the ring with and without the drift pair and prints `getTunes` and the
fractional tunes of the one-turn matrix; the ring without the pair is the control. It checks that
the pair's transfer matrix is the identity and recomputes the integrated tune of the pair ring
from the measured element matrices, once with the principal value of each element's phase advance
and once with the method's wrap into [0, 2π).
"""

import os

for _v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ.setdefault(_v, "1")

import math

from orbit.core.bunch import Bunch
from orbit.matrix_lattice import BaseMATRIX
from orbit.teapot import (
    DriftTEAPOT,
    QuadTEAPOT,
    TEAPOT_Lattice,
    TEAPOT_MATRIX_Lattice,
)


def make_ring(pair):
    lat = TEAPOT_Lattice("fodo")
    for c in range(6):
        lat.addNode(QuadTEAPOT(f"qf{c}", 0.25, 2, 1.2))
        lat.addNode(DriftTEAPOT(f"d{c}a", 2.0))
        if pair and c == 0:
            lat.addNode(DriftTEAPOT("neg", -0.3))
            lat.addNode(DriftTEAPOT("pos", 0.3))
        lat.addNode(QuadTEAPOT(f"qd{c}", 0.5, 2, -1.2))
        lat.addNode(DriftTEAPOT(f"d{c}b", 2.0))
        lat.addNode(QuadTEAPOT(f"qe{c}", 0.25, 2, 1.2))
    lat.initialize()
    return lat


def proton_bunch():
    b = Bunch()
    b.mass(0.938272)
    b.charge(1)
    b.getSyncParticle().kinEnergy(1.0)  # GeV
    return b


def integrated_tune(ml, alpha, beta, wrap):
    """Sum of per-element phase advances atan2(M01, beta0*M00 - alpha0*M01) with Twiss transport;
    the principal value of atan2 is the signed phase advance of each element."""
    phi = 0.0
    for node in ml.getNodes():
        if not isinstance(node, BaseMATRIX):
            continue
        m = node.getMatrix()
        m00, m01, m10, m11 = m.get(0, 0), m.get(0, 1), m.get(1, 0), m.get(1, 1)
        dphi = math.atan2(m01, beta * m00 - alpha * m01)
        phi += dphi % (2 * math.pi) if wrap else dphi
        gamma = (1 + alpha * alpha) / beta
        beta, alpha = (
            m00 * m00 * beta - 2 * m00 * m01 * alpha + m01 * m01 * gamma,
            -m00 * m10 * beta + (m00 * m11 + m01 * m10) * alpha - m01 * m11 * gamma,
        )
    return phi / (2 * math.pi)


res = {}
for pair in (False, True):
    ml = TEAPOT_MATRIX_Lattice(make_ring(pair), proton_bunch())
    d = ml.getRingParametersDict()
    res[pair] = ml.getTunes()
    tag = "with identity pair" if pair else "control, no pair"
    print(
        f"[{tag}] getTunes {res[pair][0]:.6f} {res[pair][1]:.6f}; "
        f"one-turn fractional tunes {d['fractional tune x']:.6f} {d['fractional tune y']:.6f}"
    )
    if pair:
        a, b = d["alpha x"], d["beta x [m]"]
        print(
            f"  x tune from element matrices: principal-value phase advances "
            f"{integrated_tune(ml, a, b, False):.6f}, wrapped into [0, 2pi) "
            f"{integrated_tune(ml, a, b, True):.6f}"
        )

# The -0.3 m / +0.3 m drift pair transports every particle back to its start: identity matrix
pl = TEAPOT_Lattice("pair")
pl.addNode(DriftTEAPOT("neg", -0.3))
pl.addNode(DriftTEAPOT("pos", 0.3))
pl.initialize()
m = TEAPOT_MATRIX_Lattice(pl, proton_bunch()).getOneTurnMatrix()
print(f"pair transfer matrix: M01 = {m.get(0, 1):.3e}, M23 = {m.get(2, 3):.3e} (identity: 0)")
for k, plane in enumerate("xy"):
    print(
        f"tune_{plane} with pair: observed {res[True][k]:.6f}, expected {res[False][k]:.6f} "
        "(ring without the identity pair)"
    )
bad = any(abs(res[True][k] - res[False][k]) > 1e-6 for k in (0, 1))
print("DEFECT REPRODUCED" if bad else "NOT REPRODUCED")
