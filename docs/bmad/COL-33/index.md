---
title: "COL-33: ibs_rates0.out uses one thousand times the requested population"
hide:
  - toc
---

# COL-33 — ibs_rates0.out uses one thousand times the requested population

| | |
|---|---|
| Code | [Bmad](../index.md) · Collective effects |
| Priority | P4 |
| Severity | Silent wrong result |
| Evidence | Reproduced, independently rechecked |
| Last reproduced | 2026-11-05 on Bmad 20261031-0 (`c4c629f`, head build) |
| Source | [current input units, bsim/ibs_ring/ibs_ring.f90:86–93](https://github.com/bmad-sim/bmad-ecosystem/blob/c4c629fc9071d3a56cb1ba5bc714baba65e9c5d3/bsim/ibs_ring/ibs_ring.f90#L86-L93)<br>[ibs_rates0.out population assignment, bsim/ibs_ring/ibs_ring.f90:280–290](https://github.com/bmad-sim/bmad-ecosystem/blob/c4c629fc9071d3a56cb1ba5bc714baba65e9c5d3/bsim/ibs_ring/ibs_ring.f90#L280-L290)<br>[correct initial conversion, bsim/ibs_ring/ibs_ring.f90:210–213](https://github.com/bmad-sim/bmad-ecosystem/blob/c4c629fc9071d3a56cb1ba5bc714baba65e9c5d3/bsim/ibs_ring/ibs_ring.f90#L210-L213)<br>[scan conversion, bsim/ibs_ring/ibs_ring.f90:269–274](https://github.com/bmad-sim/bmad-ecosystem/blob/c4c629fc9071d3a56cb1ba5bc714baba65e9c5d3/bsim/ibs_ring/ibs_ring.f90#L269-L274)<br>[final-output conversion, bsim/ibs_ring/ibs_ring.f90:353–357](https://github.com/bmad-sim/bmad-ecosystem/blob/c4c629fc9071d3a56cb1ba5bc714baba65e9c5d3/bsim/ibs_ring/ibs_ring.f90#L353-L357) |

When ibs_ring writes ibs_rates0.out, it converts the requested current to bunch population without converting milliamperes to amperes. That output therefore uses 1000 times the requested population. At 0.001 mA in a 75.20 m ring, it uses 1.566e9 particles instead of 1.566e6. Other current-conversion paths include the missing factor.

Run `python COL-33.py` with [_bmad.py](../helpers/_bmad.py) and [_collective_contracts.py](../helpers/_collective_contracts.py) in the same directory.

??? example "Reproducer: COL-33.py"

    [Download COL-33.py](COL-33.py)

    ```python
    """COL-33: ibs_rates0.out uses one thousand times the requested population

    When ibs_ring writes ibs_rates0.out, it converts the requested current to bunch population without converting milliamperes to amperes. That output therefore uses 1000 times the requested population. At 0.001 mA in a 75.20 m ring, it uses 1.566e9 particles instead of 1.566e6. Other current-conversion paths include the missing factor.

    The script runs the production ibs_ring executable on an inline ring and then recompiles the
    matching driver with one added population-print statement. It requires both rate files from the
    diagnostic build to equal the production files before comparing the measured population with the
    current-to-charge conversion.
    """

    import io

    import numpy as np
    from _bmad import compare, fortran, run, verdict
    from _collective_contracts import rows, source_text

    FILES = {
        "ring.bmad": """parameter[particle] = electron
    parameter[p0c] = 1e9
    bmad_com[radiation_damping_on] = f
    bmad_com[radiation_fluctuations_on] = f

    q1: quad, l = 0.5, k1 = 0.6, tilt = 0.05
    q2: quad, l = 0.5, k1 = -0.5

    d: drift, l = 0.3
    b: sbend, l = 3, angle = pi/8, e1 = 0.07, e2 = 0.07

    rf: rfcavity, l = 1.2, rf_frequency = 250e6, voltage = 1e6

    sector: line = (d, q1, d, b, d, q2, d, rf, b)
    ring: line = (8*sector)

    use, ring
    """,
        "run.in": """&parameters
    lat_file='ring.bmad'
    a_emit=1e-9
    b_emit=1e-11
    energy_spread=0.001
    fake_3HC=-1
    high_current=0.001
    low_current=0.001
    delta_current=0.001
    ibs_formula='bjmt'
    clog_to_use=1
    eqb_method='der'
    ratio=0
    initial_blow_up=3,3,3
    set_dispersion=.false.
    eta_set=0
    etap_set=0
    inductance=0
    granularity=-1
    /
    """,
    }
    production = run("ibs_ring", ["run.in"], files=FILES, outputs=["ibs_rates0.out", "ibs_rates.out"])
    if "Step          1 of          1 complete!" not in production.log:
        raise RuntimeError(production.log)
    text = source_text("bsim/ibs_ring/ibs_ring.f90")
    # Insert only a diagnostic before the first ring-rate evaluation. All input,
    # equilibrium, unit conversion and file-output statements stay unchanged.
    anchor = "call ibs_rates1turn(lat,ibs_sim_params,rates,granularity)"
    if text.count(anchor) != 1:
        raise RuntimeError("Expected one initial ring-rate call")
    text = text.replace(
        anchor,
        """write(*,'(a,4es25.16)') 'RESULT population ', lat%param%n_part, high_current, &
     lat%param%total_length, e_charge*abs(charge_of(lat%param%particle))*c_light
    """
        + anchor,
        1,
    )
    diagnostic = fortran(text, ["run.in"], files=FILES, outputs=["ibs_rates0.out", "ibs_rates.out"])
    for name in production.outputs:
        actual = np.loadtxt(io.BytesIO(diagnostic.outputs[name]))
        control = np.loadtxt(io.BytesIO(production.outputs[name]))
        if (
            actual.shape != control.shape
            or not np.isfinite(actual).all()
            or not np.array_equal(actual, control)
        ):
            raise RuntimeError(f"Diagnostic build changed production output: {name}")
    print("production/diagnostic control: both rate files identical")
    n, current, circumference, denominator = rows(diagnostic.log)["population"]
    # I[mA]*1e-3 = N*|q|*c/C; denominator is |q|*c, returned by Bmad.
    expected = 1e-3 * current * circumference / denominator
    defect = compare(
        "ibs_rates0.out population, N=I[mA]*1e-3*C/(|q|*c)", n, expected, atol=0, rtol=2e-14
    )
    print(
        f"current={current:.16g} mA, circumference={circumference:.16g} m, population factor={n / expected:.16g}"
    )
    verdict(defect)

    ```

## Provenance

| Role | Model | Effort | Date | Code examined |
|---|---|---|---|---|
| Found<br><small>source check</small> | `gemini-4-argon-high` |  | 2026-10-20 | 20261020-0 (`19b8afb46a40`) |
| Confirmed<br><small>source check</small> | `gpt-6-astra` | xhigh | 2026-10-20 | 20261020-0 (`19b8afb46a40`) |
| Confirmed | `gpt-6.1-sol` | xhigh | 2026-11-05 | 20261031-0 (`c4c629fc9071`), source, production, GNU Fortran 15.3.0, CPU, double, serial, OpenMP=1; driver rebuilt with read-only diagnostics and reference calculations |

