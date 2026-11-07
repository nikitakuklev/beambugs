"""SCT-17: ibsEmittance stops the equilibrium emittance at ε₀/(1+κ) when τx > τy and reports convergence

ibsEmittance, run for equilibrium emittances at coupling ratio κ = εy/εx, never returns a
horizontal emittance below ε₀/(1+κ), with ε₀ the uncoupled natural emittance. When the
horizontal damping time τx exceeds the vertical τy (horizontal damping partition Jx < 1) and IBS
is weak, the radiation–IBS balance the program solves has its root below that bound, at
ε₀/(1+κτx/τy) without IBS; the program returns the bound and reports "Emittance converged". For
a 3 GeV, 288.0 m ring with Jx = 0.9634, κ = 0.1 and 1 nC, it returns 48.96 nm versus 48.79 nm,
0.3408% high; the transverse balance misses zero by 0.3408% of its excitation term. This
lower-bound obstruction is absent for weak IBS when τx ≤ τy.

Setup: the `ibsEmittance` program of the Elegant build on `PATH`; the script runs it on the ring's
Twiss output.

The script builds a 3 GeV periodic model of 12 FODO cells, with 55.2 m total length and sector
bends totaling π, writes its Twiss and radiation integrals with elegant, and runs ibsEmittance for the equilibrium at κ = 0.1, at 10⁻¹² nC (no IBS)
and at 1 nC. It compares the returned εx with the root of the program's own radiation balance,
ε₀/(1+κτx/τy), and evaluates that balance, IBS terms included, at the returned emittances. The
control is the same model with a defocusing gradient in the bends, which makes τx < τy; there the
program returns the root.
"""

import os
import subprocess
import sys
import tempfile
from pathlib import Path

import pysdds
from _elegant import TIMEOUT, RunFailure, verdict

KAPPA = 0.1  # coupling ratio epsilon_y/epsilon_x given to -coupling
CELLS = 12


def lattice(bend_k1):
    # FODO ring of 12 cells, 24 sector bends of 1 m (circumference 12 x 4.6 m = 55.2 m). With
    # bend_k1 = 0 the bends have no gradient and the damping partition is Jx = 1 - D < 1, so
    # tau_x > tau_y; a defocusing gradient in the bends raises Jx above 1 (tau_x < tau_y).
    return (
        "D: DRIF,L=0.5\n"
        "QF: QUAD,L=0.3,K1=1.2\n"
        "QD: QUAD,L=0.3,K1=-1.2\n"
        f"B: SBEN,L=1,ANGLE={3.141592653589793 / (2 * CELLS)!r},K1={bend_k1!r}\n"
        "CELL: LINE=(QF,D,B,D,QD,D,B,D)\n"
        f"RING: LINE=({CELLS}*CELL)\n"
    )


TWISS = (
    '&run_setup lattice="ring.lte",use_beamline=RING,p_central_mev=3000,default_order=2 &end\n'
    '&twiss_output filename="ring.twi",matched=1,radiation_integrals=1 &end\n'
    "&run_control &end\n"
    "&bunched_beam &end\n"
    "&track &end\n"
)


def equilibrium(bend_k1, charge_nc):
    """Parameters of ibsEmittance's equilibrium output for the ring (no -integrate, no -growthRatesOnly)."""
    if "RPN_DEFNS" not in os.environ:
        sys.exit("Set RPN_DEFNS to the defns.rpn file of SDDS (see the elegant installation).")
    env = dict(os.environ, OMP_NUM_THREADS="1", OPENBLAS_NUM_THREADS="1", MKL_NUM_THREADS="1")
    with tempfile.TemporaryDirectory(prefix="elegant-repro-") as tmp:
        work = Path(tmp)
        (work / "ring.lte").write_text(lattice(bend_k1))
        (work / "twiss.ele").write_text(TWISS)
        for argv in (
            ["elegant", "twiss.ele"],
            # -length is the rms bunch length in mm; no RF is needed for a fixed length.
            [
                "ibsEmittance",
                "ring.twi",
                "ring.ibs",
                f"-charge={charge_nc!r}",
                f"-coupling={KAPPA!r}",
                "-length=5",
            ],
        ):
            done = subprocess.run(
                argv,
                cwd=work,
                env=env,
                capture_output=True,
                text=True,
                timeout=TIMEOUT,
                check=False,
            )
            if done.returncode:
                raise RunFailure(argv[0], done.returncode, done.stdout + done.stderr)
        out = pysdds.read(work / "ring.ibs")
    return {p.name: p.data[0] for p in out.parameters}


def case(label, bend_k1, charge_nc):
    p = equilibrium(bend_k1, charge_nc)
    ex0, taux, tauy = float(p["emitx0"]), float(p["taux"]), float(p["tauy"])
    ex, ey = float(p["emitx"]), float(p["emity"])
    # Equilibrium balance solved by IBSequations (ibsEmittance.c:1085-1090): summing the x and y
    # action equations with epsilon_y = kappa epsilon_x,
    #   0 = -2 ex/taux - 2 ey/tauy + 2 ex0/taux + gx ex + gy ey.
    # Without IBS (gx = gy = 0) its root is ex = ex0/(1 + kappa taux/tauy), the value the program
    # itself reports as emitxInput (ibsEmittance.c:747) and the manual gives for sddsbrightness.
    floor = ex0 / (1 + KAPPA)  # the simplex lower limit, ibsEmittance.c:806,810
    # With IBS, the root of the same balance with the rates gx, gy the program returns; at these
    # charges gx/(2/taux) < 1e-7, so holding the rates at their returned values is exact to that order.
    gx, gy = float(p["xGrowthRate"]), float(p["yGrowthRate"])
    root = (2 * ex0 / taux) / (2 / taux + 2 * KAPPA / tauy - gx - KAPPA * gy)
    # Balance at the returned point, normalized by the excitation term 2 ex0/taux as the program's
    # objective is.
    residual = (-2 * ex / taux - 2 * ey / tauy + 2 * ex0 / taux + gx * ex + gy * ey) / (
        2 * ex0 / taux
    )
    if gx * taux / 2 > 1e-7:
        raise SystemExit(f"{label}: IBS is not weak; the fixed-rate root is not exact")
    print(f"{label}: taux/tauy={taux / tauy:.6f} gx={gx:.4g}/s 2/taux={2 / taux:.6g}/s")
    print(f"  emitx observed={ex:.10g} expected={root:.10g} (relative {ex / root - 1:+.4%}), "
          f"floor ex0/(1+kappa)={floor:.10g}")  # fmt: skip
    print(f"  balance residual={residual:+.4%}; emitxInput={float(p['emitxInput']):.10g}; "
          f"Convergence=\"{p['Convergence']}\"")  # fmt: skip
    return ex, root, floor, residual, str(p["Convergence"])


# Control: bends with K1 = -0.02 m^-2 give tau_x < tau_y; the root lies above the floor.
ex, root, floor, residual, _ = case("control, bend K1=-0.02, 1e-12 nC", -0.02, 1e-12)
if not root > floor or abs(ex / root - 1) > 1e-9 or abs(residual) > 1e-9:
    raise SystemExit("control: the equilibrium misses the radiation balance")

defect = True
for charge in (1e-12, 1.0):
    ex, root, floor, residual, status = case(f"sector bends, {charge:g} nC", 0.0, charge)
    clamped = abs(ex / floor - 1) < 1e-12 and ex / root - 1 > 1e-3
    defect &= clamped and abs(residual) > 1e-3 and status == "Emittance converged"
print(
    "comparison: control rtol=1e-9; floor equality rtol=1e-12; defect needs >0.1% excess and residual"
)
verdict(defect)
