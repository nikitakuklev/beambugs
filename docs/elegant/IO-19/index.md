---
title: "IO-19: CWIGGLER FIELD_OUTPUT entrance row writes canonical rather than kinetic px"
hide:
  - toc
---

# IO-19 — CWIGGLER FIELD_OUTPUT entrance row writes canonical rather than kinetic px

| | |
|---|---|
| Code | [Elegant](../index.md) · Lattices, particles and I/O |
| Priority | P4 |
| Failure | Silent wrong result |
| Evidence | Reproduced, independently rechecked |
| Last reproduced | 2026-11-07 on Elegant commit [`0355dd6`](https://github.com/rtsoliday/elegant/commit/0355dd6aca12), committed 2026-11-03 |
| Tags | <span class="bb-tag">partly-source-only</span> [?](../../methodology.md#tags) |
| Source | [src/GWigSymplecticPass.c:356–368](https://github.com/rtsoliday/elegant/blob/0355dd6aca12c861971fbf8f3ae78a0ded991734/src/GWigSymplecticPass.c#L356-L368)<br>[src/gwig.c:88–91](https://github.com/rtsoliday/elegant/blob/0355dd6aca12c861971fbf8f3ae78a0ded991734/src/gwig.c#L88-L91)<br>[src/gwig.c:148–151](https://github.com/rtsoliday/elegant/blob/0355dd6aca12c861971fbf8f3ae78a0ded991734/src/gwig.c#L148-L151)<br>[src/gwig.c:93–111](https://github.com/rtsoliday/elegant/blob/0355dd6aca12c861971fbf8f3ae78a0ded991734/src/gwig.c#L93-L111)<br>[src/gwig.c:155–168](https://github.com/rtsoliday/elegant/blob/0355dd6aca12c861971fbf8f3ae78a0ded991734/src/gwig.c#L155-L168)<br>[src/gwig.c:652–655](https://github.com/rtsoliday/elegant/blob/0355dd6aca12c861971fbf8f3ae78a0ded991734/src/gwig.c#L652-L655)<br>[src/GWigSymplecticPass.c:479–486](https://github.com/rtsoliday/elegant/blob/0355dd6aca12c861971fbf8f3ae78a0ded991734/src/GWigSymplecticPass.c#L479-L486)<br>[src/gwig.c:66–83](https://github.com/rtsoliday/elegant/blob/0355dd6aca12c861971fbf8f3ae78a0ded991734/src/gwig.c#L66-L83)<br>[src/GWigSymplecticPass.c:566–587](https://github.com/rtsoliday/elegant/blob/0355dd6aca12c861971fbf8f3ae78a0ded991734/src/GWigSymplecticPass.c#L566-L587)<br>[doc/cwiggler.tex:14–21](https://github.com/rtsoliday/elegant/blob/0355dd6aca12c861971fbf8f3ae78a0ded991734/doc/cwiggler.tex#L14-L21)<br>[doc/cwiggler.tex:30–41](https://github.com/rtsoliday/elegant/blob/0355dd6aca12c861971fbf8f3ae78a0ded991734/doc/cwiggler.tex#L30-L41)<br>[src/track_data.c:2876–2877](https://github.com/rtsoliday/elegant/blob/0355dd6aca12c861971fbf8f3ae78a0ded991734/src/track_data.c#L2876-L2877)<br>[src/GWigSymplecticPass.c:439–444](https://github.com/rtsoliday/elegant/blob/0355dd6aca12c861971fbf8f3ae78a0ded991734/src/GWigSymplecticPass.c#L439-L444) |

With CWIGGLER FIELD_OUTPUT and a nonzero horizontal vector potential at the entrance, the first px sample contains canonical momentum while later samples contain kinetic momentum, both normalized to reference momentum. In a 0.1000 m-period, 1.000 T wiggler at 3.066 GeV/c with a π/3 harmonic phase and FORCE_MATCHED=0, it reports zero instead of −1.348×10⁻³; a derivative of tracked positions agrees with the kinetic value within 7.177×10⁻¹². FORCE_MATCHED=1 also permits the mismatch when a phased higher harmonic remains active at entry. Both integration orders are affected. Enabling the diagnostic leaves particle tracking unchanged.

Run `python IO-19.py` with [_cwiggler.py](../helpers/_cwiggler.py), [_diagnostics.py](../helpers/_diagnostics.py) and [_elegant.py](../helpers/_elegant.py) in the same directory.

??? example "Reproducer: IO-19.py"

    [Download IO-19.py](IO-19.py)

    ```python
    """IO-19: CWIGGLER FIELD_OUTPUT entrance row writes canonical rather than kinetic px

    With CWIGGLER FIELD_OUTPUT and a nonzero horizontal vector potential at the entrance, the first px sample contains canonical momentum while later samples contain kinetic momentum, both normalized to reference momentum. In a 0.1000 m-period, 1.000 T wiggler at 3.066 GeV/c with a π/3 harmonic phase and FORCE_MATCHED=0, it reports zero instead of −1.348×10⁻³; a derivative of tracked positions agrees with the kinetic value within 7.177×10⁻¹². FORCE_MATCHED=1 also permits the mismatch when a phased higher harmonic remains active at entry. Both integration orders are affected. Enabling the diagnostic leaves particle tracking unchanged.

    The script tracks one on-axis particle at +1% momentum through a four-period, 1 T horizontal-plane CWIGGLER given by a harmonic table, with 2048 steps per period and radiation off, for both integration orders. It compares the entrance px sample with the analytic kinetic momentum −a_x(0), with a one-sided derivative of the stored x positions, and with a cubic extrapolation of the following px samples to the entrance. Zero-phase and clipped FORCE_MATCHED=1 fields, which have no entrance potential, the analytic interior samples and FIELD_OUTPUT on/off tracking are controls.
    """

    import math

    from _cwiggler import FIELD_RIGIDITY, close
    from _diagnostics import run

    # isort: split
    import _elegant as native
    import numpy as np

    P0 = 6000.0  # reference momentum, units of m c (3.066 GeV/c)
    DELTA = 0.01
    PERIOD = 0.1
    PERIODS = 4
    STEPS = 2048
    H = PERIOD / STEPS
    KW = 2 * math.pi / PERIOD
    # Normalized vector-potential amplitude e B0 / (m c kw P) for B0 = 1 T. CWIGGLER takes its
    # rigidity from sqrt(P0^2 - 1) (it treats p0/(m c) as gamma); using the executed normalization
    # here keeps this field-output check separate from that rigidity error.
    A = FIELD_RIGIDITY / (math.sqrt(P0 * P0 - 1) * KW)
    # (label, FORCE_MATCHED, harmonics as (Cmn, n, phase)); FORCE_MATCHED acts on the first harmonic.
    CASES = (
        ("phase pi/3, FORCE_MATCHED=0", 0, ((1.0, 1, math.pi / 3),)),
        ("phase 0 + C3=0.3 at pi/2, FORCE_MATCHED=1", 1, ((1.0, 1, 0.0), (0.3, 3, math.pi / 2))),
    )
    CONTROLS = (
        ("control phase 0, FORCE_MATCHED=0", 0, ((1.0, 1, 0.0),), True),
        (
            "control phase pi/2, FORCE_MATCHED=1 (field clipped to a crest)",
            1,
            ((1.0, 1, math.pi / 2),),
            False,
        ),
    )
    DERIVATIVE_TOL = 2e-9  # float32 positions over 6h, plus the order-2 integration error
    EXTRAPOLATION_TOL = 2e-9  # 15 float32 half-spacings of the px samples
    BODY_TOL = 8e-8 * A  # interior float32 px samples against the analytic kinetic momentum


    def harmonics(rows):
        """BY_FILE table: kx = 0, ky = kz = n kw, so the field is independent of x."""
        body = "".join(f"{c:.17g} 0 {n} {n} {phase:.17g}\n" for c, n, phase in rows)
        columns = "".join(
            f"&column name={name}, type=double &end\n"
            for name in ("Cmn", "KxOverKw", "KyOverKw", "KzOverKw", "Phase")
        )
        return f"SDDS1\n{columns}&data mode=ascii &end\n{len(rows)}\n{body}"


    def track(order, matched, rows, field_output):
        field = ',FIELD_OUTPUT="field.sdds"' if field_output else ""
        lattice = (
            f'W:CWIGGLER,L={PERIODS * PERIOD:.17g},PERIODS={PERIODS},B_MAX=1,BY_FILE="harmonics.sdds",'
            f"FORCE_MATCHED={matched},STEPS_PER_PERIOD={STEPS},INTEGRATION_ORDER={order},"
            f"SYNCH_RAD=0,ISR=0{field}\nBL:LINE=(W)\n"
        )
        beam = {name: np.zeros(1) for name in ("x", "xp", "y", "yp")}
        beam.update(p=np.array([P0 * (1 + DELTA)]), particleID=np.array([1]))
        commands = (
            f"&run_setup lattice=line.lte,use_beamline=BL,p_central={P0},"
            "default_order=2,output=out.sdds,random_number_seed=271828 &end\n"
            "&run_control n_steps=1 &end\n&sdds_beam input=beam.sdds &end\n&track soft_failure=0 &end\n"
        )
        files = {"line.lte": lattice, "run.ele": commands, "harmonics.sdds": harmonics(rows)}
        outputs = ("out.sdds", "field.sdds") if field_output else ("out.sdds",)
        result = run(files, "serial", {"beam.sdds": beam}, outputs)
        data = native.particles(result["out.sdds"])
        if not np.array_equal(data["particleID"], [1]):
            raise RuntimeError("Particle lost or particle ID changed")
        exit_ray = np.array([data[k][0] for k in ("x", "xp", "y", "yp", "t", "p")])
        if not field_output:
            return exit_ray, None
        sdds = result["field.sdds"]
        if sdds.n_pages != 1:
            raise RuntimeError("Expected one field-output page")
        samples = {name: np.asarray(sdds[0, name]) for name in ("x", "z", "px", "By")}
        if len(samples["z"]) != PERIODS * STEPS + 1 or not all(
            np.isfinite(v).all() for v in samples.values()
        ):
            raise RuntimeError("Unexpected field-output rows")
        return exit_ray, samples


    def kinetic(z, rows):
        """Kinetic px/p0 on the plane y = 0 with zero canonical qx.

        The field does not depend on x, so canonical qx is conserved (zero here) and the kinetic
        momentum is -a_x(z), where a_x(z) = A sum (Cn/n) sin(n kw z + phase_n) integrates the
        documented cosine expansion of By."""
        return -A * sum(c / n * np.sin(n * KW * z + phase) for c, n, phase in rows)


    def entrance(field):
        x = field["x"].astype(float)
        if x[0] != 0:
            raise RuntimeError("Entrance position is not on axis")
        # One-sided third-order derivative at z = 0 from stored positions; in the paraxial
        # Hamiltonian (1 + delta) dx/dz is the normalized kinetic momentum.
        slope = (1 + DELTA) * (-11 * x[0] + 18 * x[1] - 9 * x[2] + 2 * x[3]) / (6 * H)
        # Cubic extrapolation of the next four px samples (all kinetic) back to z = 0.
        px = field["px"].astype(float)
        extrapolated = 4 * px[1] - 6 * px[2] + 4 * px[3] - px[4]
        return slope, extrapolated


    def main():
        faults = []
        for order in (2, 4):
            for label, matched, rows in CASES:
                exit_on, field = track(order, matched, rows, True)
                exit_off, _ = track(order, matched, rows, False)
                assert field is not None
                close(exit_on, exit_off, 0.0, f"{label} order {order}: FIELD_OUTPUT changed tracking")
                z = np.arange(len(field["z"])) * H
                close(field["px"][1:], kinetic(z[1:], rows), BODY_TOL, f"{label} interior px")
                expected = float(kinetic(0.0, rows))
                slope, extrapolated = entrance(field)
                close(slope, expected, DERIVATIVE_TOL, f"{label} order {order} position derivative")
                close(extrapolated, expected, EXTRAPOLATION_TOL, f"{label} order {order} extrapolation")
                observed = float(field["px"][0])
                faults.append(abs(observed - expected) > 1e3 * DERIVATIVE_TOL)
                print(
                    f"order={order} {label}: entrance By={float(field['By'][0]):.6g} T; entrance "
                    f"px={observed:.10e} expected={expected:.10e} (-a_x(0), kinetic); position "
                    f"derivative={slope:.10e} (|diff|={abs(slope - expected):.4g}); px extrapolation="
                    f"{extrapolated:.10e} (|diff|={abs(extrapolated - expected):.4g}); tracking with "
                    "FIELD_OUTPUT off identical"
                )
        for label, matched, rows, active in CONTROLS:
            _, field = track(4, matched, rows, True)
            assert field is not None
            slope, extrapolated = entrance(field)
            expected = float(kinetic(0.0, rows)) + 0.0 if active else 0.0  # + 0.0 drops a -0 sign
            observed = float(field["px"][0])
            close(observed, expected, EXTRAPOLATION_TOL, label)
            print(
                f"order=4 {label}: entrance px={observed:.4g} expected={expected:.4g}; position "
                f"derivative={slope:.4g}; px extrapolation={extrapolated:.4g}"
            )
        native.verdict(all(faults))


    if __name__ == "__main__":
        main()

    ```

## Provenance

| Role | Model | Effort | Date | Code examined |
|---|---|---|---|---|
| Found | `claude-opus-5-5` | high | 2026-11-06 | master (`0355dd6aca12`), source inspection |
| Confirmed | `gpt-6.1-sol` | xhigh | 2026-11-07 | master (`0355dd6aca12`), source with SDDS a3f730a14c49, CPU, double, serial, one thread |
| Confirmed<br><small>extrapolation of the interior px samples</small> | `claude-opus-5-5` | high | 2026-11-07 | master (`0355dd6aca12`), source with SDDS a3f730a14c49, CPU, double, serial, one thread |

