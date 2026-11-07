---
title: "IO-18: CWIGGLER FIELD_OUTPUT writes vertical position in the py momentum column"
hide:
  - toc
---

# IO-18 — CWIGGLER FIELD_OUTPUT writes vertical position in the py momentum column

| | |
|---|---|
| Code | [Elegant](../index.md) · Lattices, particles and I/O |
| Priority | P4 |
| Failure | Silent wrong result |
| Evidence | Reproduced, independently rechecked |
| Last reproduced | 2026-11-06 on Elegant commit [`0355dd6`](https://github.com/rtsoliday/elegant/commit/0355dd6aca12), committed 2026-11-03 |
| Source | [src/gwig.c:652–657](https://github.com/rtsoliday/elegant/blob/0355dd6aca12c861971fbf8f3ae78a0ded991734/src/gwig.c#L652-L657)<br>[src/GWigSymplecticPass.c:356–368](https://github.com/rtsoliday/elegant/blob/0355dd6aca12c861971fbf8f3ae78a0ded991734/src/GWigSymplecticPass.c#L356-L368)<br>[src/GWigSymplecticPass.c:479–486](https://github.com/rtsoliday/elegant/blob/0355dd6aca12c861971fbf8f3ae78a0ded991734/src/GWigSymplecticPass.c#L479-L486)<br>[src/gwig.c:76–79](https://github.com/rtsoliday/elegant/blob/0355dd6aca12c861971fbf8f3ae78a0ded991734/src/gwig.c#L76-L79)<br>[src/gwig.c:88–104](https://github.com/rtsoliday/elegant/blob/0355dd6aca12c861971fbf8f3ae78a0ded991734/src/gwig.c#L88-L104)<br>[src/gwig.c:148–166](https://github.com/rtsoliday/elegant/blob/0355dd6aca12c861971fbf8f3ae78a0ded991734/src/gwig.c#L148-L166)<br>[src/track_data.c:2877–2877](https://github.com/rtsoliday/elegant/blob/0355dd6aca12c861971fbf8f3ae78a0ded991734/src/track_data.c#L2877-L2877) |

A CWIGGLER with FIELD_OUTPUT writes a field-sample file whose py column repeats the vertical position y instead of the vertical momentum normalized to the reference momentum. The px column correctly holds the horizontal momentum, so the two momentum columns disagree in meaning. For a ray entering a 2 m, 1 T planar wiggler at 300.0 MeV/c with y = 1 mm and y′ = 0.5 mrad, the first row reports py = 1.000×10⁻³ instead of 5.000×10⁻⁴, and the last 8.524×10⁻⁴ instead of −6.225×10⁻⁴; every row equals the y column. Tracked momenta at whole periods and the slope of the y column establish the correct values. Tracking itself is unaffected.

Run `python IO-18.py` with [_diagnostics.py](../helpers/_diagnostics.py) and [_elegant.py](../helpers/_elegant.py) in the same directory.

??? example "Reproducer: IO-18.py"

    [Download IO-18.py](IO-18.py)

    ```python
    """IO-18: CWIGGLER FIELD_OUTPUT writes vertical position in the py momentum column

    A CWIGGLER with FIELD_OUTPUT writes a field-sample file whose py column repeats the vertical position y instead of the vertical momentum normalized to the reference momentum. The px column correctly holds the horizontal momentum, so the two momentum columns disagree in meaning. For a ray entering a 2 m, 1 T planar wiggler at 300.0 MeV/c with y = 1 mm and y′ = 0.5 mrad, the first row reports py = 1.000×10⁻³ instead of 5.000×10⁻⁴, and the last 8.524×10⁻⁴ instead of −6.225×10⁻⁴; every row equals the y column. Tracked momenta at whole periods and the slope of the y column establish the correct values. Tracking itself is unaffected.

    Track a fixed off-axis ray through a 20-period sinusoidal wiggler and through each whole-period truncation in fresh processes. Compare FIELD_OUTPUT momenta with the particle output where the vector potential vanishes, and independently compare the y-column derivative with vertical momentum. Check both integration orders and momentum offsets 0 and +1%, with x, y and px as controls.
    """

    from _diagnostics import run

    # isort: split
    import _elegant as native
    import numpy as np

    native.TIMEOUT = 290  # ty: ignore[invalid-assignment]  # Per-native-process deadline.
    P0 = 587.0
    PERIOD = 0.1
    PERIODS = 20
    STEPS = 12
    RAY = dict(x=0.0005, xp=0.0003, y=0.001, yp=0.0005)


    def track(periods, order, delta, field_output=False):
        field = ',FIELD_OUTPUT="field.sdds"' if field_output else ""
        lattice = (
            f"W:CWIGGLER,L={periods * PERIOD:.17g},PERIODS={periods},B_MAX=1,SINUSOIDAL=1,"
            f"STEPS_PER_PERIOD={STEPS},INTEGRATION_ORDER={order},SYNCH_RAD=0,ISR=0{field}\n"
            "BL:LINE=(W)\n"
        )
        beam = {name: np.array([value]) for name, value in RAY.items()}
        beam.update(p=np.array([P0 * (1 + delta)]), particleID=np.array([1]))
        commands = (
            f"&run_setup lattice=line.lte,use_beamline=BL,p_central={P0},"
            "default_order=2,output=out.sdds,random_number_seed=271828 &end\n"
            "&run_control n_steps=1 &end\n&sdds_beam input=beam.sdds &end\n&track soft_failure=0 &end\n"
        )
        outputs = ("out.sdds", "field.sdds") if field_output else ("out.sdds",)
        result = run({"line.lte": lattice, "run.ele": commands}, "serial", {"beam.sdds": beam}, outputs)
        data = native.particles(result["out.sdds"])
        if not np.array_equal(data["particleID"], [1]):
            raise RuntimeError("Particle lost or particle ID changed")
        exit_ray = {key: float(data[key][0]) for key in ("x", "xp", "y", "yp", "p")}
        samples = None
        if field_output:
            sdds = result["field.sdds"]
            if sdds.n_pages != 1:
                raise RuntimeError("Expected one field-output page")
            samples = {
                name: np.asarray(sdds[0, name], dtype=float) for name in ("x", "y", "z", "px", "py")
            }
            if not all(np.isfinite(values).all() for values in samples.values()):
                raise RuntimeError("Nonfinite field output")
        return exit_ray, samples


    def main():
        faults = []
        for order in (2, 4):
            for delta in (0.0, 0.01):
                _, field = track(PERIODS, order, delta, True)
                if field is None or len(field["z"]) != PERIODS * STEPS + 1:
                    raise RuntimeError("Unexpected field-output row count")
                points = [dict(RAY, p=P0 * (1 + delta))]
                points += [track(n, order, delta)[0] for n in range(1, PERIODS + 1)]
                indices = np.arange(PERIODS + 1) * STEPS
                np.testing.assert_allclose(
                    field["z"][indices], np.arange(PERIODS + 1) * PERIOD, rtol=0, atol=1.2e-7
                )
                xp, yp = np.array([[p["xp"], p["yp"]] for p in points]).T
                # At whole periods the vector potential vanishes: q=(p/p0)*(xp,yp)/sqrt(1+xp²+yp²).
                momentum = np.array([p["p"] for p in points]) / P0
                qx, qy = momentum * np.array([xp, yp]) / np.sqrt(1 + xp * xp + yp * yp)
                control_error = 0.0
                for key, expected in (
                    ("x", [p["x"] for p in points]),
                    ("y", [p["y"] for p in points]),
                    ("px", qx),
                ):
                    values = field[key][indices]
                    np.testing.assert_allclose(values, expected, rtol=6e-8, atol=2e-15)
                    control_error = max(control_error, float(np.max(np.abs(values - expected))))
                equal_y = np.array_equal(field["py"], field["y"])
                if not equal_y:
                    np.testing.assert_allclose(field["py"][indices], qy, rtol=6e-8, atol=2e-15)
                faults.append(equal_y)
                # Expanded-Hamiltonian drift dy/dz=qy/(1+delta), independently of the py column.
                i = indices[1:-1]
                derivative = (field["y"][i + 1] - field["y"][i - 1]) / (
                    field["z"][i + 1] - field["z"][i - 1]
                )
                slope_error = float(np.max(np.abs((1 + delta) * derivative - qy[1:-1])))
                if slope_error > 3.1e-8:
                    raise RuntimeError(f"Vertical slope check failed: {slope_error}")
                error = float(np.max(np.abs(field["py"][indices] - qy)))
                print(
                    f"order={order} delta={delta:+.2f}: rows={len(field['z'])} expected=241; entrance py={field['py'][0]:.9e} expected={qy[0]:.9e}; exit py={field['py'][-1]:.9e} expected={qy[-1]:.9e} (tracked kinetic momentum)"
                )
                print(
                    f"order={order} delta={delta:+.2f}: py=y in every row={equal_y}; max|py-qy|={error:.3e}; x/y/px control residual={control_error:.3e} (float32 rounding); max|(1+delta)dy/dz-qy|={slope_error:.3e} (centred y-column derivative)"
                )
        native.verdict(any(faults))


    if __name__ == "__main__":
        main()

    ```

## Provenance

| Role | Model | Effort | Date | Code examined |
|---|---|---|---|---|
| Found | `gpt-6.1-sol` | xhigh | 2026-11-06 | master (`0355dd6aca12`), source inspection |
| Confirmed | `claude-opus-5-5` | high | 2026-11-06 | master (`0355dd6aca12`), source with SDDS a3f730a14c49, CPU, double, serial, one thread |

