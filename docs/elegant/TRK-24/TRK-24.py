"""TRK-24: KQUAD and KSEXT edge multipole cutoffs select rows by the body file's order labels

When KQUAD or KSEXT magnets use separate body and edge multipole files, minimum and maximum
normal/skew order cutoffs incorrectly select edge rows using the body file's order labels.
Relabelling zero-amplitude body rows must leave every physical field unchanged, but changes the edge
kicks. In a 0.17 m KQUAD with K₁ = −0.7 m⁻², edge orders 2 and 4 and minimum normal and skew orders
of 3, reversing body labels from [2, 4] to [4, 2] changes horizontal slope x′ from −1.754×10⁻³ to
−1.637×10⁻³. The largest vertical-slope error is 5.084×10⁻⁴; independent edge impulses and body
integration establish the expected result.

The script tracks 12 electrons through a KQUAD or KSEXT with an edge file of orders 2 and 4 and a
body file of zero amplitudes, with a minimum or a maximum order of 3, and varies only the body
file's order labels. It compares each exit coordinate with an independent integration: the kept
edge rows applied at entrance and exit around a straight magnet. The control is the body file with
the edge file's own labels; every relabelling must give the same result, since zero rows add no
field.
"""

import sys
from math import factorial

import numpy as np
from _elegant import modes, particles, run, verdict
from _multipole import P0, RADIUS, beam, canonical, observable, straight_body, table_text

LENGTH = 0.17  # m
EDGE_ORDERS, EDGE_NORMAL, EDGE_SKEW = [2, 4], [0.08, -0.04], [0.03, 0.02]
# Body files with zero amplitude in every row; only the order labels differ.
BODY_LABELS = {
    "aligned [2,4]": [2, 4],
    "reversed [4,2]": [4, 2],
    "low [0,1]": [0, 1],
    "high [5,6]": [5, 6],
}
KINDS = {"KQUAD": (1, -0.7), "KSEXT": (2, -17.0)}  # element: (main order m, strength K_m)
CUTOFFS = {  # cutoff text: (keeps order n if)
    "MIN_NORMAL_ORDER=3,MIN_SKEW_ORDER=3": lambda n: n >= 3,
    "MAX_NORMAL_ORDER=3,MAX_SKEW_ORDER=3": lambda n: n <= 3,
}
ATOL = 1e-13  # observed agreement with the oracle is ~1e-15; wrong row sets differ by >1e-7
SPLIT = 1e-7
P_ATOL = 2e-13  # no radiation, so the momentum is conserved


def edge_oracle(initial, m, strength, keep):
    """Exit x, xp, y, yp for the edge rows of order in `keep`, from the independent integrator.

    A thin edge multipole of order n with fractions (a_n, b_n) of the main field at radius R
    gives the impulse (dqx, dqy) = (-Re C_e, Im C_e) with
    C_e = K_m L R^m/m! * sum_n (a_n - i b_n) ((x + iy)/R)^n, at entrance and at exit, with a
    straight magnet of the main field between.
    """

    def impulse(state):
        x, y, qx, qy = state
        c_e = np.zeros(len(x), dtype=complex)
        for n, a, b in zip(EDGE_ORDERS, EDGE_NORMAL, EDGE_SKEW, strict=True):
            if n in keep:
                c_e += (
                    strength
                    * LENGTH
                    * RADIUS**m
                    / factorial(m)
                    * (a - 1j * b)
                    * ((x + 1j * y) / RADIUS) ** n
                )
        return np.stack((x, y, qx - c_e.real, qy + c_e.imag))

    state = impulse(canonical(initial))
    state = straight_body(state, initial, m, strength, LENGTH, harmonics=())
    result = observable(impulse(state), initial)
    if not np.all(np.isfinite(result)):
        sys.exit("the independent edge-impulse oracle is not finite")
    return result


def track(mode, kind, cutoff, labels, initial):
    m, strength = KINDS[kind]
    lattice = (
        f"E: {kind},L={LENGTH},K{m}={strength},N_SLICES=96,INTEGRATION_ORDER=4,{cutoff},"
        'SYSTEMATIC_MULTIPOLES="body.sdds",EDGE_MULTIPOLES="edge.sdds",SYNCH_RAD=0,ISR=0\n'
        "BL: LINE=(E)\n"
    )
    commands = (
        f"&run_setup lattice=lattice.lte, use_beamline=BL, p_central={P0}, output=out.sdds, "
        "random_number_seed=753219 &end\n"
        "&run_control n_steps=1, n_passes=1 &end\n"
        "&sdds_beam input=beam.sdds &end\n"
        "&track soft_failure=0 &end\n"
    )
    files = {
        "lattice.lte": lattice,
        "run.ele": commands,
        "body.sdds": table_text(labels, [0.0, 0.0], [0.0, 0.0]),
        "edge.sdds": table_text(EDGE_ORDERS, EDGE_NORMAL, EDGE_SKEW),
    }
    out = particles(run(files, mode, {"beam.sdds": initial})["out.sdds"])
    if len(out["p"]) != len(initial["p"]):
        sys.exit(f"{mode} {kind}: particles were lost")
    if not np.allclose(out["p"], initial["p"], rtol=0, atol=P_ATOL, equal_nan=False):
        sys.exit(f"{mode} {kind}: the momentum is not conserved")
    return np.column_stack((out["x"], out["xp"], out["y"], out["yp"]))


def last(a):
    return "[" + ", ".join(repr(float(v)) for v in a[-1]) + "]"


def compare(label, observed, expected):
    """Print the last particle's x, xp, y, yp and the largest difference over all particles.

    Nonfinite values make the comparison invalid: the run terminates instead of returning a NaN
    that every later `> ATOL` test would accept.
    """
    if not (np.all(np.isfinite(observed)) and np.all(np.isfinite(expected))):
        sys.exit(f"{label}: a nonfinite value makes the comparison invalid")
    error = float(np.max(np.abs(observed - expected)))
    print(f"{label}: observed={last(observed)} expected={last(expected)} max|diff|={error:.4g}")
    return error


defect = False
initial = beam()
for mode in modes():
    for kind, (m, strength) in KINDS.items():
        for cutoff, passes in CUTOFFS.items():
            # Correct selection tests the edge file's own order labels.
            keep = {n for n in EDGE_ORDERS if passes(n)}
            expected = edge_oracle(initial, m, strength, keep)
            for name, labels in BODY_LABELS.items():
                observed = track(mode, kind, cutoff, labels, initial)
                error = compare(f"{mode} {kind} {cutoff} body labels {name}", observed, expected)
                # What the code does instead: row i of the edge file is kept when row i of the
                # body file passes the cutoff.
                by_body = {n for n, label in zip(EDGE_ORDERS, labels, strict=True) if passes(label)}
                explained = edge_oracle(initial, m, strength, by_body)
                if not np.allclose(observed, explained, rtol=0, atol=ATOL, equal_nan=False):
                    sys.exit(f"{mode} {kind} {name}: result matches neither row selection")
                if name.startswith("aligned"):
                    if error > ATOL:
                        sys.exit(f"{mode} {kind} {cutoff}: the aligned control disagrees")
                else:
                    defect |= error > SPLIT
print(f"comparison: atol={ATOL:g} for agreement, {SPLIT:g} for disagreement")
verdict(defect)
