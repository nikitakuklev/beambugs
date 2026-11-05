"""TRK-46: p_central_mev multiplies a custom species' reference momentum by its charge magnitude

Selecting a custom species with charge magnitude Q in units of the elementary charge and setting run_setup.p_central_mev selects a reference momentum Q times the requested value. The conversion divides by rest energy per charge instead of the rest energy in MeV required by relativistic kinematics. A particle with approximately the helium nucleus mass and Q = 2 requested at 200.0 MeV/c is tracked at 400.0 MeV/c, crossing a 1.000 m drift in 31.26 ns instead of 62.26 ns. Supplying the equivalent dimensionless p_central, or using a singly charged species, gives the correct momentum and arrival time.

The script tracks six identical on-axis particles through an exact drift and compares momentum and flight time with relativistic mass-shell kinematics. Singly charged input, both charge signs and direct dimensionless momentum provide controls, with the same mass constants used in the simulation and oracle. Each case starts a fresh native process with a fixed seed; --mpi also runs three-rank Pelegant. Elegant and SDDS must be on PATH with RPN_DEFNS set.
"""

import math

import numpy as np
from _elegant import modes, particles, run, verdict

C = 299792458.0
# SDDS constants.h values isolate charge normalization from historical constant rounding.
M_E = 9.1093897e-31
E = 1.60217733e-19
MASS_RATIO = 7294.29954
MOMENTUM_MEV = 200.0
LENGTH = 1.0
PARTICLES = 6  # More than the three MPI ranks; all particles are identical.
REST_MEV = MASS_RATIO * M_E * C * C / E / 1e6
# p/(mc) = pc/(mc^2), beta = p/sqrt(1+p^2), and t = L/(beta*c).
EXPECTED_P = MOMENTUM_MEV / REST_MEV
EXPECTED_T = LENGTH * math.hypot(1, EXPECTED_P) / (C * EXPECTED_P)
TOLERANCE = 3e-14


def case(mode: str, charge_ratio: int, use_mev: bool) -> tuple[float, float]:
    momentum = f"p_central_mev={MOMENTUM_MEV}" if use_mev else f"p_central={EXPECTED_P:.17g}"
    commands = f"""&change_particle name=custom,mass_ratio={MASS_RATIO},charge_ratio={charge_ratio} &end
&run_setup lattice=lattice.lte,use_beamline=BL,{momentum},output=out.sdds,random_number_seed=13579 &end
&run_control &end
&bunched_beam n_particles_per_bunch={PARTICLES},emit_x=0,emit_y=0,sigma_dp=0,sigma_s=0 &end
&track &end
"""
    result = run(
        {"run.ele": commands, "lattice.lte": f"D: EDRIFT,L={LENGTH}\nBL: LINE=(D)\n"},
        mode=mode,
    )
    out = particles(result["out.sdds"])
    if len(out["p"]) != PARTICLES:
        raise RuntimeError("A particle was lost")
    if not np.all(out["p"] == out["p"][0]) or not np.all(out["t"] == out["t"][0]):
        raise RuntimeError("Identical particles gave different momenta or flight times")
    return float(out["p"][0]), float(out["t"][0])


def main() -> None:
    defects = []
    for mode in modes():
        for charge, use_mev in [(-1, True), (-2, True), (-2, False), (2, True)]:
            p, t = case(mode, charge, use_mev)
            label = (
                f"{mode} charge_ratio={charge}, input={'p_central_mev' if use_mev else 'p_central'}"
            )
            print(
                f"{label}: observed p/(mc)={p:.15g} expected={EXPECTED_P:.15g} "
                f"(pc/(mc^2), mc^2={REST_MEV:.15g} MeV); "
                f"observed pc={p * REST_MEV:.12g} MeV expected={MOMENTUM_MEV:.12g} MeV; "
                f"observed t={t:.15g} s expected={EXPECTED_T:.15g} s (L/(beta*c))"
            )
            correct = math.isclose(p, EXPECTED_P, rel_tol=TOLERANCE) and math.isclose(
                t, EXPECTED_T, rel_tol=TOLERANCE
            )
            if abs(charge) == 1 or not use_mev:
                if not correct:
                    raise RuntimeError("A reference-momentum control failed")
                continue
            # Check the precise faulty conversion, not merely any discrepancy.
            wrong_p = abs(charge) * EXPECTED_P
            wrong_t = LENGTH * math.hypot(1, wrong_p) / (C * wrong_p)
            faulty = math.isclose(p, wrong_p, rel_tol=TOLERANCE) and math.isclose(
                t, wrong_t, rel_tol=TOLERANCE
            )
            if not correct and not faulty:
                raise RuntimeError(
                    "Output matches neither the physical nor the charge-scaled result"
                )
            defects.append(faulty)
    verdict(all(defects))


if __name__ == "__main__":
    main()
