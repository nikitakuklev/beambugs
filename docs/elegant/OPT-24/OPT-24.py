"""OPT-24: closed_orbit ignores start_from_centroid and start_from_dp_centroid

Elegant's `closed_orbit` command documents that `start_from_centroid` (default 1) starts the search
from the beam centroid and that `start_from_dp_centroid` uses the beam's momentum centroid, so that
the closed orbit of an off-momentum beam can be computed; the code sets both flags to zero before
testing them, so every closed orbit is computed on momentum. For a 2.4 GeV, 70 m ring of 20 FODO
cells and a one-particle beam at x = 0.3 mm, x′ = 0.1 mrad and δ = 0.001, the closed orbit found by
tracking at δ = 0.001 starts at x = 2.458 mm, x′ = 0.4474 mrad. With either flag set, Elegant
returns δ = 0 and x = 2.799 nm, the on-momentum orbit.

The script runs serial Elegant `closed_orbit` before tracking a one-particle beam with x = 0.3 mm,
x′ = 0.1 mrad and δ = 0.001 through a 2.4 GeV, 70 m ring of 20 FODO cells with CSBENDs, with the
flags (start_from_centroid, start_from_dp_centroid) set to (0, 0), (1, 0) and (0, 1), and reads
the orbit at the ring start. The expected orbits for δ = 0.001 and, as a control, δ = 0 come from
Newton iteration on single-turn tracking without `closed_orbit`; the (0, 0) run must give the
on-momentum orbit.
"""

import sys

import numpy as np
from _elegant import particles, run, verdict

P0 = 2400 / 0.51099906  # reference momentum, units of m_e c (Elegant's electron mass)
DELTA = 1e-3
CENTROID = {"x": 3e-4, "xp": 1e-4}
LATTICE = f"""F: QUAD,L=0.3,K1=0.9
G: QUAD,L=0.3,K1=-0.85
D: DRIF,L=0.65
E: DRIF,L=0.4
B: CSBEND,L=1.2,ANGLE={2 * np.pi / 20!r},N_SLICES=40,EXPAND_HAMILTONIAN=0
C: LINE=(F,D,B,E,G,D)
R: LINE=(20*C)
"""
STEP = 2e-6


def shoot(delta):
    """Periodic (x, xp) at the ring start for momentum offset delta, by Newton iteration."""
    orbit = np.zeros(2)
    probes = np.array([[0, 0], [STEP, 0], [-STEP, 0], [0, STEP], [0, -STEP]])
    commands = (
        "&run_setup lattice=ring.lte, use_beamline=R, p_central_mev=2400, output=out.sdds &end\n"
        "&run_control &end\n&sdds_beam input=beam.sdds &end\n&track &end\n"
    )
    for _ in range(10):
        start = orbit + probes
        beam = {
            "x": start[:, 0],
            "xp": start[:, 1],
            "p": np.full(len(probes), P0 * (1 + delta)),
            "particleID": np.arange(1, len(probes) + 1),
        }
        o = particles(
            run({"ring.lte": LATTICE, "run.ele": commands}, beams={"beam.sdds": beam})["out.sdds"]
        )
        end = np.c_[o["x"], o["xp"]]
        residual = end[0] - orbit
        if np.abs(residual).max() < 1e-14:
            return orbit
        jacobian = np.c_[(end[1] - end[2]) / (2 * STEP), (end[3] - end[4]) / (2 * STEP)] - np.eye(2)
        orbit = orbit - np.linalg.solve(jacobian, residual)
    sys.exit("Newton iteration for the closed orbit did not converge.")


def closed_orbit(centroid, dp_centroid):
    commands = (
        "&run_setup lattice=ring.lte, use_beamline=R, p_central_mev=2400 &end\n"
        "&twiss_output filename=twiss.sdds, matched=1 &end\n"
        f"&closed_orbit output=orbit.sdds, start_from_centroid={centroid},"
        f" start_from_dp_centroid={dp_centroid} &end\n"
        "&run_control &end\n"
        "&bunched_beam n_particles_per_bunch=1, bunch=bunch.sdds,"
        f" centroid[0]={CENTROID['x']!r}, centroid[1]={CENTROID['xp']!r}, centroid[5]={DELTA!r} &end\n"
        "&track &end\n"
    )
    result = run(
        {"ring.lte": LATTICE, "run.ele": commands},
        outputs=("orbit.sdds", "bunch.sdds", "twiss.sdds"),
    )
    orbit = result["orbit.sdds"]
    bunch = particles(result["bunch.sdds"])
    eta = float(result["twiss.sdds"][0, "etax"][0])
    start = [float(orbit[0, "x"][0]), float(orbit[0, "xp"][0])]
    beam = [float(bunch["x"][0]), float(bunch["xp"][0]), float(bunch["p"][0]) / P0 - 1]
    return start, float(orbit["delta"][0]), beam, eta


on = shoot(0.0)
off = shoot(DELTA)
print(
    f"tracked closed orbit at the ring start: delta=0 x, xp={on.tolist()}; delta={DELTA:g} x, xp={off.tolist()}"
)
defect = False
for centroid, dp_centroid in ((0, 0), (1, 0), (0, 1)):
    start, delta, beam, eta = closed_orbit(centroid, dp_centroid)
    label = f"start_from_centroid={centroid} start_from_dp_centroid={dp_centroid}"
    if not np.allclose(beam, [CENTROID["x"], CENTROID["xp"], DELTA], rtol=1e-9, atol=1e-15):
        sys.exit(f"{label}: the beam centroid is not the requested one: {beam}")
    expected = on if (centroid, dp_centroid) == (0, 0) else off
    expected_delta = 0.0 if (centroid, dp_centroid) == (0, 0) else DELTA
    print(
        f"{label}: x, xp, delta observed=[{start[0]:.10g}, {start[1]:.10g}, {delta:.10g}]"
        f" expected=[{expected[0]:.10g}, {expected[1]:.10g}, {expected_delta:.10g}]"
    )
    bad = (
        not np.allclose(start, expected, rtol=1e-6, atol=1e-11)
        or abs(delta - expected_delta) > 1e-12
    )
    if (centroid, dp_centroid) == (0, 0):
        if bad:
            sys.exit(
                "The on-momentum control disagrees with the tracked closed orbit: invalid comparison."
            )
        print(f"control: dispersion at the start {eta:.10g} m; eta*delta = {eta * DELTA:.10g} m")
    else:
        defect |= bad
verdict(defect)
