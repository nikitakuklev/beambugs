---
title: "TRK-52: ENERGY tracking adds one to the specified total Lorentz factor"
hide:
  - toc
---

# TRK-52 — ENERGY tracking adds one to the specified total Lorentz factor

| | |
|---|---|
| Code | [Elegant](../index.md) · Tracking |
| Priority | P3 |
| Severity | Wrong physics |
| Evidence | Reproduced, independently rechecked |
| Last reproduced | 2026-11-06 on Elegant commit [`0355dd6`](https://github.com/rtsoliday/elegant/commit/0355dd6aca12), committed 2026-11-03 |
| Source | [src/track_data.c:1010–1013](https://github.com/rtsoliday/elegant/blob/0355dd6aca12c861971fbf8f3ae78a0ded991734/src/track_data.c#L1010-L1013)<br>[src/do_tracking.c:2010–2028](https://github.com/rtsoliday/elegant/blob/0355dd6aca12c861971fbf8f3ae78a0ded991734/src/do_tracking.c#L2010-L2028)<br>[src/compute_matrices.c:1936–1943](https://github.com/rtsoliday/elegant/blob/0355dd6aca12c861971fbf8f3ae78a0ded991734/src/compute_matrices.c#L1936-L1943)<br>[src/do_tracking.c:3866–3905](https://github.com/rtsoliday/elegant/blob/0355dd6aca12c861971fbf8f3ae78a0ded991734/src/do_tracking.c#L3866-L3905)<br>[src/multipole.c:791–795](https://github.com/rtsoliday/elegant/blob/0355dd6aca12c861971fbf8f3ae78a0ded991734/src/multipole.c#L791-L795) |

Setting ENERGY.CENTRAL_ENERGY to a total Lorentz factor γ makes particle tracking select the reference momentum for γ + 1, while the parameter definition and optics use γ. For electrons at γ = 5.000, tracking selects 5.916 mc instead of 4.899 mc, a 20.76% excess. A later ENERGY with MATCH_PARTICLES = 1 transfers this error to the beam; a following 0.7000 m drift then takes 2.368 ns instead of 2.383 ns. Explicit CENTRAL_MOMENTUM gives the correct result, while rebasing alone preserves particle momentum and field-free flight time.

Run `python TRK-52.py` with [_diagnostics.py](../helpers/_diagnostics.py), [_elegant.py](../helpers/_elegant.py) and [_refmomentum.py](../helpers/_refmomentum.py) in the same directory.

??? example "Reproducer: TRK-52.py"

    [Download TRK-52.py](TRK-52.py)

    ```python
    """TRK-52: ENERGY tracking adds one to the specified total Lorentz factor

    Setting ENERGY.CENTRAL_ENERGY to a total Lorentz factor γ makes particle tracking select the reference momentum for γ + 1, while the parameter definition and optics use γ. For electrons at γ = 5.000, tracking selects 5.916 mc instead of 4.899 mc, a 20.76% excess. A later ENERGY with MATCH_PARTICLES = 1 transfers this error to the beam; a following 0.7000 m drift then takes 2.368 ns instead of 2.383 ns. Explicit CENTRAL_MOMENTUM gives the correct result, while rebasing alone preserves particle momentum and field-free flight time.

    Inject six electrons at momentum 3 mc and compare ENERGY's gamma and momentum inputs against the mass shell, native optics and a 0.7 m drift, with and without a separate particle match. Symmetric quadrupole rays compare normalized and physical-field inputs against the linear Lorentz force. Pass --mpi to repeat the energy and matching checks in three-rank Pelegant.
    """

    import math

    import numpy as np
    from _diagnostics import conversion_defect, scalar
    from _elegant import modes, particles, verdict
    from _refmomentum import CHARGE, MASS, RAYS, C, close, quadrupole_slope, response, track


    def main():
        right, wrong = math.sqrt(24), math.sqrt(35)
        defect = False
        for mode in modes():
            for label, specification in (
                ("gamma", "CENTRAL_ENERGY=5"),
                ("momentum", f"CENTRAL_MOMENTUM={right:.17g}"),
            ):
                for matched in (False, True):
                    lattice = (
                        f"E: ENERGY,{specification}\n"
                        'W: WATCH,FILENAME="reference.sdds",MODE="coordinates"\n'
                        "M: ENERGY,MATCH_PARTICLES=1\nD: EDRIFT,L=0.7\n"
                        f"BL: LINE=(E,W,{'M,' if matched else ''}D)\n"
                    )
                    result = track(lattice, 3.0, mode=mode, twiss=not matched, watch=True)
                    ref = scalar(result["reference.sdds"], "pCentral")
                    faulty = wrong if label == "gamma" else right
                    defect |= conversion_defect(ref, right, faulty, rtol=4e-14)
                    data = result["particles"]
                    close(data["p"], ref if matched else 3.0)
                    # E^2 = (pc)^2 + (mc^2)^2, hence t = L*sqrt(1+p^2)/(p*c).
                    p = ref if matched else 3.0
                    expected_p = right if matched else 3.0
                    expected_time = 0.7 * math.hypot(1, expected_p) / (C * expected_p)
                    actual_time = 0.7 * math.hypot(1, p) / (C * p)
                    close(data["t"], actual_time)
                    optics = ""
                    if not matched:
                        twiss = particles(result["optics.sdds"])
                        mask = twiss["ElementName"] == "E"
                        if np.count_nonzero(mask) != 1:
                            raise RuntimeError("Expected one ENERGY row in optics output")
                        optical_ref = float(twiss["pCentral0"][mask][0])
                        close(optical_ref, right)
                        optics = f"; optics P0={optical_ref:.16g}"
                    print(
                        f"{mode}/{label}/match={int(matched)}: P0={ref:.16g} expected={right:.16g} mc"
                        f"{optics}; p={data['p'][0]:.16g} expected={expected_p:.16g} mc; "
                        f"t={data['t'][0]:.16g} expected={expected_time:.16g} s "
                        "(P0=sqrt(gamma^2-1), t=L*sqrt(1+p^2)/(p*c))"
                    )
            for specification in (
                "CENTRAL_ENERGY=0",
                "CENTRAL_ENERGY=5,MATCH_PARTICLES=1",
                "CENTRAL_ENERGY=5,MATCH_BEAMLINE=1",
            ):
                result = track(
                    f"E: ENERGY,{specification}\n"
                    'W: WATCH,FILENAME="reference.sdds",MODE="coordinates"\n'
                    "D: EDRIFT,L=0.7\nBL: LINE=(E,W,D)\n",
                    3.0,
                    mode=mode,
                    watch=True,
                )
                close(scalar(result["reference.sdds"], "pCentral"), 3.0)
                close(result["particles"]["p"], 3.0)
                close(result["particles"]["t"], 0.7 * math.sqrt(10) / (3 * C))
            print(
                f"{mode}/bypass controls: zero energy and either same-element match give P0=p=3 mc; drift t=L*sqrt(10)/(3*c) passes"
            )

        expected = quadrupole_slope(1.6 * right / 3, 0.25)
        field = 1.6 * right * MASS * C / CHARGE * 0.012
        for label, specification, magnet, faulty in (
            ("momentum", f"CENTRAL_MOMENTUM={right:.17g}", "K1=1.6", expected),
            ("gamma", "CENTRAL_ENERGY=5", "K1=1.6", quadrupole_slope(1.6 * wrong / 3, 0.25)),
            ("fixed field", "CENTRAL_ENERGY=5", f"B={field:.17g},BORE=0.012", expected),
        ):
            result = track(
                f"E: ENERGY,{specification}\nQ: KQUAD,L=0.25,{magnet},N_SLICES=96,"
                "INTEGRATION_ORDER=6,EDGE1_EFFECTS=0,EDGE2_EFFECTS=0,SYNCH_RAD=0,ISR=0\n"
                "BL: LINE=(E,Q)\n",
                3.0,
                rays=RAYS,
            )
            measured = response(result["particles"])
            defect |= conversion_defect(measured, expected, faulty, rtol=2e-12)
            print(
                f"serial/{label} quadrupole: R21={measured:.16g} expected={expected:.16g} m^-1 (-sqrt(k)*sin(sqrt(k)*L), k=K1*P0/p)"
            )
        verdict(defect)


    if __name__ == "__main__":
        main()

    ```

## Provenance

| Role | Model | Effort | Date | Code examined |
|---|---|---|---|---|
| Found | `gpt-6-astra` | xhigh | 2026-11-06 | master (`0355dd6aca12`), source with SDDS a3f730a14c49, CPU, double, serial elegant and elegantto, one thread |
| Confirmed | `gpt-6.1-sol` | xhigh | 2026-11-06 | master (`0355dd6aca12`), source with SDDS a3f730a14c49, CPU, double, serial and 3-rank Pelegant, one thread |

