---
title: "TRK-47: CSBEND Gaussian quantum diffusion scales as charge to the fourth power"
hide:
  - toc
---

# TRK-47 — CSBEND Gaussian quantum diffusion scales as charge to the fourth power

| | |
|---|---|
| Code | [Elegant](../index.md) · Tracking |
| Priority | P3 |
| Severity | Wrong physics |
| Evidence | Reproduced, independently rechecked |
| Last reproduced | 2026-11-05 on Elegant commit [`0355dd6`](https://github.com/rtsoliday/elegant/commit/0355dd6aca12), committed 2026-11-03 |
| Tags | <span class="bb-tag">partly-source-only</span> [?](../../methodology.md#tags) |
| Source | [`track_through_csbend`,
csbend.c:1151–1166](https://github.com/rtsoliday/elegant/blob/0355dd6aca12c861971fbf8f3ae78a0ded991734/src/csbend.c#L1151-L1166)<br>[`process_particle_command`,
elegant.c:3292–3308](https://github.com/rtsoliday/elegant/blob/0355dd6aca12c861971fbf8f3ae78a0ded991734/src/elegant.c#L3292-L3308)<br>[`integrate_csbend_ordn`,
csbend.c:1960–1966](https://github.com/rtsoliday/elegant/blob/0355dd6aca12c861971fbf8f3ae78a0ded991734/src/csbend.c#L1960-L1966)<br>[`determineRadiationMatrix`,
analyze.c:1773–1783](https://github.com/rtsoliday/elegant/blob/0355dd6aca12c861971fbf8f3ae78a0ded991734/src/analyze.c#L1773-L1783)<br>[`change_particle`,
elegant.tex:1497–1517](https://github.com/rtsoliday/elegant/blob/0355dd6aca12c861971fbf8f3ae78a0ded991734/doc/elegant.tex#L1497-L1517)<br>[`CSBEND` parameter table,
track_data.c:2008–2011](https://github.com/rtsoliday/elegant/blob/0355dd6aca12c861971fbf8f3ae78a0ded991734/src/track_data.c#L2008-L2011)<br>[CSBEND radiation description,
csbend.tex:107–113](https://github.com/rtsoliday/elegant/blob/0355dd6aca12c861971fbf8f3ae78a0ded991734/doc/csbend.tex#L107-L113)<br>[`moments_output.radiation`,
elegant.tex:5608–5612](https://github.com/rtsoliday/elegant/blob/0355dd6aca12c861971fbf8f3ae78a0ded991734/doc/elegant.tex#L5608-L5612) |

For a multiply charged custom species, moments_output overestimates the CSBEND quantum variance σδ², where δ is relative momentum deviation, by Q², where Q is charge magnitude in units of the elementary charge. Its Gaussian diffusion coefficient gives variance proportional to Q⁴ instead of the Q² law obtained from the synchrotron photon spectrum. For a particle with approximately the lead nucleus mass, Q = 82 and momentum 573.8 TeV/c, a 14.30 m bend of radius 2804 m gives σδ² = 6.713×10⁻²³ instead of 9.984×10⁻²⁷, a factor 6724. Singly charged controls agree with the spectrum integral.

Run `python TRK-47.py` with [_elegant.py](../helpers/_elegant.py) in the same directory.

??? example "Reproducer: TRK-47.py"

    [Download TRK-47.py](TRK-47.py)

    ```python
    """TRK-47: CSBEND Gaussian quantum diffusion scales as charge to the fourth power

    For a multiply charged custom species, moments_output overestimates the CSBEND quantum variance σδ², where δ is relative momentum deviation, by Q², where Q is charge magnitude in units of the elementary charge. Its Gaussian diffusion coefficient gives variance proportional to Q⁴ instead of the Q² law obtained from the synchrotron photon spectrum. For a particle with approximately the lead nucleus mass, Q = 82 and momentum 573.8 TeV/c, a 14.30 m bend of radius 2804 m gives σδ² = 6.713×10⁻²³ instead of 9.984×10⁻²⁷, a factor 6724. Singly charged controls agree with the spectrum integral.

    The script computes deterministic moments_output covariance growth from zero initial covariance through a bend with the exact Hamiltonian. It compares the photon-spectrum integral with electron-mass and heavy unit-charge controls and heavy particles with charge magnitudes 2, 10 and 82, including both signs at 82. Dimensionless p_central isolates diffusion from momentum-input conversion, and no stochastic tracking variance is measured. Each case runs serial Elegant in a fresh process with a fixed seed;
    """

    import math

    import numpy as np
    from _elegant import run, verdict

    # SDDS constants.h values isolate the charge law from historical constant rounding.
    C = 299792458.0
    E = 1.60217733e-19
    M_E = 9.1093897e-31
    HBAR = 6.6260755e-34 / (2 * math.pi)
    EPS0 = 1e7 / (4 * math.pi * C * C)
    P0 = 2963.0
    LENGTH = 14.3
    RHO = 2804.0
    MASS_RATIO = 379000.0


    def expected_variance(mass_ratio: float, charge: int, momentum: float) -> float:
        mass = mass_ratio * M_E
        radius = (charge * E) ** 2 / (4 * math.pi * EPS0 * mass * C * C)
        compton = HBAR / (mass * C)
        # Photon count N ~ Q^2, normalized critical energy uc/E ~ hbar/(mc),
        # and the spectrum has <(u/uc)^2> = 11/27. Their product N*(uc/E)^2*11/27
        # is (55/(24*sqrt(3)))*r_q*lambda_m*gamma^5*L/rho^3, hence Q^2.
        # Passing P0 instead of gamma matches the source's ultrarelativistic approximation.
        return 55 / (24 * math.sqrt(3)) * radius * compton * momentum**5 * LENGTH / RHO**3


    def case(mass_ratio: float, charge_ratio: int) -> float:
        commands = f"""&change_particle name=custom,mass_ratio={mass_ratio},charge_ratio={charge_ratio} &end
    &run_setup lattice=lattice.lte,use_beamline=BL,p_central={P0},random_number_seed=13579 &end
    &moments_output filename=out.mom,matched=0,equilibrium=0,radiation=1,emit_x=0,emit_y=0,emit_z=0,beta_x=2,beta_y=3,beta_z=1 &end
    &run_control &end
    &bunched_beam n_particles_per_bunch=1,emit_x=0,emit_y=0,sigma_dp=0,sigma_s=0 &end
    &track &end
    """
        lattice = f"""B: CSBEND,L={LENGTH},ANGLE={LENGTH / RHO:.17g},N_SLICES=12,INTEGRATION_ORDER=4,EXPAND_HAMILTONIAN=0,EDGE1_EFFECTS=0,EDGE2_EFFECTS=0,SYNCH_RAD=1,ISR=1,USE_RAD_DIST=0
    END: MARK
    BL: LINE=(B,END)
    """
        out = run({"run.ele": commands, "lattice.lte": lattice}, outputs=("out.mom",))["out.mom"]
        sigma = np.asarray(out[0, "s6"])
        if len(sigma) < 2 or not np.all(np.isfinite(sigma)) or sigma[0] != 0 or sigma[-1] <= 0:
            raise RuntimeError("Invalid covariance output or nonzero initial spread")
        return float(sigma[-1] ** 2)


    def main() -> None:
        heavy = {}
        gamma = math.hypot(1, P0)
        for mass, charge in [
            (1.0, 1),
            (MASS_RATIO, 1),
            (MASS_RATIO, 2),
            (MASS_RATIO, 10),
            (MASS_RATIO, 82),
            (MASS_RATIO, -82),
        ]:
            obs = case(mass, charge)
            expected = expected_variance(mass, charge, gamma)
            matched = expected_variance(mass, charge, P0)
            print(
                f"serial mass_ratio={mass:g} charge_ratio={charge}: "
                f"observed variance={obs:.12g} expected={expected:.12g} ratio={obs / expected:.12g} "
                "(55/(24 sqrt(3)) * r_q * hbar/(m*c) * gamma^5 * L/rho^3); "
                f"expected with (beta*gamma)^5={matched:.12g}"
            )
            if abs(charge) == 1 and not math.isclose(obs, matched, rel_tol=1e-7):
                raise RuntimeError("The singly charged analytic control failed")
            if mass == MASS_RATIO:
                heavy[charge] = obs
        defects = []
        for charge in (2, 10, 82):
            ratio = heavy[charge] / heavy[1]
            print(
                f"same-mass Q={charge}/Q=1 variance: observed={ratio:.12g} expected={charge**2} "
                f"(photon count scales as Q^2; Q^4={charge**4})"
            )
            correct = math.isclose(ratio, charge**2, rel_tol=1e-10)
            faulty = math.isclose(ratio, charge**4, rel_tol=1e-10)
            if not correct and not faulty:
                raise RuntimeError("Charge scaling matches neither Q^2 nor Q^4")
            defects.append(faulty)
        sign_error = abs(heavy[82] / heavy[-82] - 1)
        print(f"charge-sign control: observed relative difference={sign_error:.12g} expected=0")
        if sign_error > 1e-13:
            raise RuntimeError("Charge-sign invariance failed")
        verdict(all(defects))


    if __name__ == "__main__":
        main()

    ```

## Provenance

| Role | Model | Effort | Date | Code examined |
|---|---|---|---|---|
| Found | `gpt-6-astra` | xhigh | 2026-11-04 | master (`0355dd6aca12`), source with SDDS a3f730a14c49, CPU, serial |
| Confirmed | `gpt-6.1-sol` | xhigh | 2026-11-05 | master (`0355dd6aca12`), source with SDDS a3f730a14c49, CPU, serial |

