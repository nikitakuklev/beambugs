---
title: "COL-26: The default disabled envelope excitation injects negative diffusion"
hide:
  - toc
---

# COL-26 — The default disabled envelope excitation injects negative diffusion

| | |
|---|---|
| Code | [Bmad](../index.md) · Collective effects |
| Priority | P2 |
| Severity | Wrong physics |
| Evidence | Reproduced, independently rechecked |
| Last reproduced | 2026-11-05 on Bmad release [20261031-0](https://github.com/bmad-sim/bmad-ecosystem/tree/20261031-0) (`c4c629f`) |
| Source | [envelope_ibs default, bsim/envelope_ibs/envelope_ibs.f90:51–60](https://github.com/bmad-sim/bmad-ecosystem/blob/c4c629fc9071d3a56cb1ba5bc714baba65e9c5d3/bsim/envelope_ibs/envelope_ibs.f90#L51-L60)<br>[excitation normalization, bsim/envelope_ibs/envelope_ibs.f90:200–207](https://github.com/bmad-sim/bmad-ecosystem/blob/c4c629fc9071d3a56cb1ba5bc714baba65e9c5d3/bsim/envelope_ibs/envelope_ibs.f90#L200-L207) |

Leaving the artificial vertical excitation at its disabled default makes the envelope IBS program inject negative diffusion. The target vertical emittance parameter ey0 defaults to −1, which is used to scale the excitation instead of disabling it. With IBS and turn tracking disabled, the three equilibrium emittances are −0.03500 m, −1.000 m and −0.0007606 m. The disabled setting should match ey0 = 0, which gives positive emittances in the same ring.

Run `python COL-26.py` with [_bmad.py](../helpers/_bmad.py) in the same directory.

??? example "Reproducer: COL-26.py"

    [Download COL-26.py](COL-26.py)

    ```python
    """COL-26: The default disabled envelope excitation injects negative diffusion

    Leaving the artificial vertical excitation at its disabled default makes the envelope IBS program inject negative diffusion. The target vertical emittance parameter ey0 defaults to −1, which is used to scale the excitation instead of disabling it. With IBS and turn tracking disabled, the three equilibrium emittances are −0.03500 m, −1.000 m and −0.0007606 m. The disabled setting should match ey0 = 0, which gives positive emittances in the same ring.

    The script runs the envelope_ibs executable in BMAD_DIST twice with an inline eight-sector ring. It compares the omitted ey0 setting with explicit zero excitation and checks that the zero-excitation control has finite positive emittances.
    """

    import numpy as np
    from _bmad import compare, run, verdict

    LATTICE = """parameter[p0c] = 1e9
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
    """
    INIT = "&envelope_tracker\nlat_file='ring.bmad'\ninclude_ibs=.false.\nnturns=-1\nbend_slice_length=.2\nslice_length=.2\n{}\n/\n"
    values = []
    for label, setting in [("disabled default", ""), ("zero excitation", "ey0=0")]:
        result = run(
            "envelope_ibs",
            ["run.init"],
            files={"ring.bmad": LATTICE, "run.init": INIT.format(setting)},
            outputs=["emit_from_integrals.out"],
        )
        rows = [
            line.split(":", 1)[1]
            for line in result.outputs["emit_from_integrals.out"].decode().splitlines()
            if line.lstrip().startswith("Emittance (SR only):")
        ]
        if len(rows) != 1:
            raise RuntimeError("Expected one SR-only emittance record")
        numbers = np.array([float(x.replace("D", "E")) for x in rows[0].split()])
        print(label, numbers)
        if not np.isfinite(numbers).all():
            raise RuntimeError("Nonfinite envelope output")
        values.append(numbers)
    if not (values[1] > 0).all():
        raise RuntimeError("Zero-excitation control must have positive emittances")
    verdict(
        compare(
            "disabled excitation emittances (m), equal to ey0=0",
            values[0],
            values[1],
            atol=0,
            rtol=2e-5,
        )
    )

    ```

## Provenance

| Role | Model | Effort | Date | Code examined |
|---|---|---|---|---|
| Found | `gpt-6.1-sol` | xhigh | 2026-10-21 | 20261020-0 (`19b8afb46a40`), source, production, CPU, double, serial |
| Confirmed | `gpt-6-astra` | xhigh | 2026-10-21 | 20261020-0 (`19b8afb46a40`), source, production, CPU, double, serial |

