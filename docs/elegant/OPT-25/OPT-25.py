"""OPT-25: moments_output crashes on a one-element beamline

Elegant's `moments_output` crashes with a segmentation fault on a beamline of one element when it
writes moments along the beamline (the default `final_values_only=0`): writing the starting row
reads the Coulomb logarithm of the element after the first, which does not exist. The manual
supports propagating given starting moments through a non-periodic beamline, and a single 1.7 m
drift entered with Twiss βx = 3 m, αx = 0.4 and emittance εx = 2 nm is valid input. With
`final_values_only=1`, or with a zero-length marker appended, Elegant finishes and writes the
drift-transported beam-matrix elements σ11 = 5.515×10⁻⁹ m² and σ12 = 5.147×10⁻¹⁰ m.

The script runs serial Elegant `moments_output` with `matched=0` and given starting ellipses on a
beamline holding one 1.7 m drift, with the default `final_values_only=0`, and reports whether the
run ends with a segmentation fault. The controls are the same input with `final_values_only=1` and
the drift followed by a zero-length marker with the default output; both must finish, and their
final horizontal covariance must equal the drift transport D Σ Dᵀ of the starting ellipse.
"""

import sys

import numpy as np
from _elegant import RunFailure, run, verdict

L = 1.7
EMIT, BETA, ALPHA = 2e-9, 3.0, 0.4
LATTICE = f"D: DRIF,L={L}\nM: MARK\nONE: LINE=(D)\nTWO: LINE=(D,M)\n"
# Starting horizontal covariance from (emittance, beta, alpha), drift-transported:
# s11 = e (beta - 2 alpha L + gamma L^2), s12 = e (-alpha + gamma L).
GAMMA = (1 + ALPHA**2) / BETA
EXPECTED = [EMIT * (BETA - 2 * ALPHA * L + GAMMA * L**2), EMIT * (-ALPHA + GAMMA * L)]


def case(line, final_only):
    commands = (
        f"&run_setup lattice=lattice.lte, use_beamline={line}, p_central_mev=2400, default_order=2 &end\n"
        "&moments_output filename=moments.sdds, matched=0, equilibrium=0, radiation=0,"
        f" final_values_only={final_only}, emit_x={EMIT!r}, beta_x={BETA!r}, alpha_x={ALPHA!r},"
        " emit_y=1e-9, beta_y=2, emit_z=1e-6, beta_z=4 &end\n"
    )
    try:
        result = run({"lattice.lte": LATTICE, "run.ele": commands}, outputs=("moments.sdds",))
    except RunFailure as failure:
        crashed = "SIGSEGV" in failure.log
        return None, f"exit status {failure.status}{', Terminated by SIGSEGV' if crashed else ''}"
    moments = result["moments.sdds"]
    final = [float(moments[0, "s1"][-1]) ** 2, float(moments[0, "s12"][-1])]
    return final, f"finished, final s11, s12 = [{final[0]:.10g}, {final[1]:.10g}]"


defect = False
print(f"expected final s11, s12 = [{EXPECTED[0]:.10g}, {EXPECTED[1]:.10g}] (m^2, m)")
for line, final_only, control in (("ONE", 1, True), ("TWO", 0, True), ("ONE", 0, False)):
    final, note = case(line, final_only)
    label = f"{'control, ' if control else ''}beamline {line} final_values_only={final_only}"
    print(f"{label}: observed={note} expected=finished")
    if control and (final is None or not np.allclose(final, EXPECTED, rtol=1e-12)):
        sys.exit("A control run failed or disagrees with the drift transport: invalid comparison.")
    defect |= final is None
verdict(defect)
