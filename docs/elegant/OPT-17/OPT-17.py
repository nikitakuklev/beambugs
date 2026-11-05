"""OPT-17: CCBEND radiation integral I4 subtracts the sextupole-centering offset twice

With a sextupole component K2, Elegant's CCBEND I4 radiation integral in `twiss_output` evaluates
the gradient as K1 + K2(x − Δx), where x is already measured from the magnet axis that the default
optimization (OPTIMIZE_DX) shifts by Δx; for a negative bend the offset enters with the opposite
sign. A magnet and its mirror image (opposite angle and K2) therefore get different I4. In a 73.60
m, 3 GeV ring of 32 one-meter CCBENDs with K2 = 3 m⁻³ (Δx = 12.26 mm), Elegant gives a horizontal
damping partition of 1.035 and a natural emittance of 351.5 nm, and for the mirrored ring 0.5606 and
648.8 nm. The two I4 differ by 0.5850 m⁻¹, close to 4K2·Δx·I1 = 0.5859 m⁻¹.

The script runs serial Elegant `twiss_output` with radiation integrals on a CCBEND with K2 and on
its mirror image (negative angle and K2, mirrored entrance dispersion), then on a ring of such
CCBENDs and the mirrored ring. A mirrored magnet with mirrored dispersion has the same radiation
integrals; the other integrals, the tunes and the optimized offset printed with VERBOSE=1 are equal
in each pair and serve as controls. With OPTIMIZE_DX=0, which sets the offset to zero, the
single-magnet pair must agree again; the script also compares the pair difference with
4·K2·Δx·I1, the size the doubled subtraction predicts.
"""

import re
import sys

import numpy as np
from _elegant import run, verdict

P_MEV = 3000.0  # reference momentum, MeV/c
RTOL = 1e-10

# Single magnet: entered with fixed optics; mirroring flips the sign of eta and eta'.
SINGLE = {"L": 1.3, "ANGLE": 0.12, "K1": 0.05, "K2": 4.0}
START = "beta_x=3.2, alpha_x=0.3, beta_y=2.8, alpha_y=-0.1"
ETA0, ETAP0 = 0.4, 0.03

# Ring: 16 FODO cells with two 1 m CCBENDs each, 2*pi/32 per bend, K2 = 3 m^-3, at 3 GeV.
N_BENDS = 32
RING = {"L": 1.0, "ANGLE": 2 * np.pi / N_BENDS, "K1": 0.0, "K2": 3.0}


def bend(p, sign, optimize_dx):
    # Mirror image in x: the dipole field and the sextupole term change sign, K1 does not.
    return (
        f"B: CCBEND,L={p['L']!r},ANGLE={sign * p['ANGLE']!r},K1={p['K1']!r},"
        f"K2={sign * p['K2']!r},N_SLICES=256,OPTIMIZE_DX={optimize_dx},VERBOSE=1"
    )


def twiss(p, sign, optimize_dx, ring):
    """Twiss parameters and the optimized offset dx of the (first) CCBEND."""
    if ring:
        lattice = (
            "F: QUAD,L=0.3,K1=2.4\nG: QUAD,L=0.3,K1=-2.4\nD: DRIF,L=0.5\nE: DRIF,L=0.5\n"
            f"{bend(p, sign, optimize_dx)}\nC: LINE=(F,D,B,E,G,D,B,E)\n"
            f"BL: LINE=({N_BENDS // 2}*C)\n"
        )
        matched = "matched=1"
    else:
        lattice = f"{bend(p, sign, optimize_dx)}\nBL: LINE=(B)\n"
        matched = f"matched=0, {START}, eta_x={sign * ETA0!r}, etap_x={sign * ETAP0!r}"
    commands = (
        f"&run_setup lattice=lattice.lte, use_beamline=BL, p_central_mev={P_MEV!r} &end\n"
        f"&twiss_output filename=twiss.sdds, {matched}, radiation_integrals=1 &end\n"
    )
    result = run({"lattice.lte": lattice, "run.ele": commands}, outputs=("twiss.sdds",))
    found = re.findall(r"optimized: FSE=\S+, dx=(\S+),", result["log"])
    dx = float(found[0]) if found else 0.0
    table = result["twiss.sdds"]
    names = ["I1", "I2", "I3", "I4", "I5", "nux", "Jx", "ex0"]
    return {k: float(table[k][0]) for k in names}, dx


def fmt(values):
    return "[" + ", ".join(f"{v:.10g}" for v in values) + "]"


defect = False
controls = ["I1", "I2", "I3", "I5", "nux"]
for label, p, ring in (("single magnet", SINGLE, False), ("ring", RING, True)):
    pos, dx_pos = twiss(p, 1, 1, ring)
    neg, dx_neg = twiss(p, -1, 1, ring)
    print(f"{label} optimized offset dx (m): bend={dx_pos:.12g} mirror={dx_neg:.12g}")
    print(
        f"{label} control I1, I2, I3, I5, nux: bend={fmt([pos[k] for k in controls])}"
        f" mirror={fmt([neg[k] for k in controls])}"
    )
    same = all(np.isclose(pos[k], neg[k], rtol=RTOL, atol=1e-15) for k in controls)
    if not same or not np.isclose(dx_pos, dx_neg, rtol=RTOL) or dx_pos == 0:
        sys.exit(
            f"{label}: the mirror pair differs in geometry or linear optics: invalid comparison."
        )
    # The mirror image must have the same I4; the doubled subtraction gives the positive bend
    # K1 + K2 (x - dx) and the mirror K1 + K2 (x + dx), so the I4 pair differs by
    # 4 K2 dx sum(ds eta h cos(atan x')) ~ 4 K2 dx I1 (cos(atan x') is within 0.2% of 1 here).
    print(f"{label} I4 (1/m): bend={pos['I4']:.12g} mirror={neg['I4']:.12g}")
    print(f"{label} I4 mirror - bend: observed={neg['I4'] - pos['I4']:.10g} expected=0")
    print(f"{label} 4*K2*dx*I1 = {4 * p['K2'] * dx_pos * pos['I1']:.10g} 1/m")
    defect |= not np.isclose(pos["I4"], neg["I4"], rtol=RTOL, atol=1e-15)
    if ring:
        print(f"{label} Jx: bend={pos['Jx']:.10g} mirror={neg['Jx']:.10g}")
        print(f"{label} ex0 (m): bend={pos['ex0']:.10g} mirror={neg['ex0']:.10g}")
    else:
        # Control: OPTIMIZE_DX=0 leaves dx = 0, and the mirror pair agrees again.
        pos0, _ = twiss(p, 1, 0, ring)
        neg0, _ = twiss(p, -1, 0, ring)
        print(
            f"{label} control OPTIMIZE_DX=0 I4 (1/m): bend={pos0['I4']:.12g}"
            f" mirror={neg0['I4']:.12g}"
        )
        if not np.isclose(pos0["I4"], neg0["I4"], rtol=RTOL, atol=1e-15):
            sys.exit("The OPTIMIZE_DX=0 mirror pair differs: the offset is not the only cause.")
verdict(defect)
