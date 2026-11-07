---
title: "COL-58: Gaussian table smoothing omits the squared index"
hide:
  - toc
---

# COL-58 — Gaussian table smoothing omits the squared index

| | |
|---|---|
| Code | [Bmad](../index.md) · Collective effects |
| Priority | P2 |
| Severity | Wrong physics |
| Evidence | Reproduced, independently rechecked |
| Last reproduced | 2026-10-20 on Bmad release [20261020-0](https://github.com/bmad-sim/bmad-ecosystem/tree/20261020-0) (`19b8afb`) |
| Pull request | [[gh:https://github.com/bmad-sim/bmad-ecosystem/pull/2225]], fix |
| Source | [`parser_read_sr_wake`, bmad/parsing/bmad_parser_mod.f90:1625–1658](https://github.com/bmad-sim/bmad-ecosystem/blob/c4c629fc9071d3a56cb1ba5bc714baba65e9c5d3/bmad/parsing/bmad_parser_mod.f90#L1625-L1658)<br>[`historical parser_read_sr_wake`, bmad/parsing/bmad_parser_mod.f90:1625–1641](https://github.com/bmad-sim/bmad-ecosystem/blob/19b8afb46a400d12a53a92d616e02359b3b2b71f/bmad/parsing/bmad_parser_mod.f90#L1625-L1641)<br>[`Short-Range Longitudinal Wake With a Wake Table (s:sr.wake.z.long)`, bmad/doc/attributes.tex:3123–3139](https://github.com/bmad-sim/bmad-ecosystem/blob/c4c629fc9071d3a56cb1ba5bc714baba65e9c5d3/bmad/doc/attributes.tex#L3123-L3139) |

Applying Gaussian smoothing to a tabulated short-range wake uses an asymmetric exponential when the grid index is not squared. A symmetric impulse must produce equal kicks at equally spaced witnesses on either side, but the faulty kernel gives an outer-kick ratio of 54.60 for a smoothing width of half a grid spacing. A normalized Gaussian convolution and an unsmoothed self-bin control establish the expected symmetry and magnitude.

Run `python COL-58.py` with [_bmad.py](../helpers/_bmad.py) and [_wake_tables_and_bbu.py](../helpers/_wake_tables_and_bbu.py) in the same directory.

??? example "Reproducer: COL-58.py"

    [Download COL-58.py](COL-58.py)

    ```python
    """COL-58: Gaussian table smoothing omits the squared index

    Applying Gaussian smoothing to a tabulated short-range wake uses an asymmetric exponential when the grid index is not squared. A symmetric impulse must produce equal kicks at equally spaced witnesses on either side, but the faulty kernel gives an outer-kick ratio of 54.60 for a smoothing width of half a grid spacing. A normalized Gaussian convolution and an unsmoothed self-bin control establish the expected symmetry and magnitude.

    The script parses a symmetric impulse table with and without Gaussian smoothing, then compares three tracked kicks against a normalized Gaussian. The self-bin is halved before smoothing, as required by beam loading.
    """

    import numpy as np
    from _bmad import compare, fortran, verdict
    from _wake_tables_and_bbu import TABLE_WAKE_SOURCE, require, rows

    table = ", ".join(f"{i * 0.01:.2f} {1e18 if i == 0 else 0:.1e}" for i in range(-4, 5))
    lattice = (
        "parameter[particle]=electron\nparameter[p0c]=1e12\n"
        "d: lcavity,l=0.6,sr_wake={scale_with_length=F,z_long={smoothing_sigma=0.005,w={"
        + table
        + ", }}}\nl: line=(d)\nuse,l\n"
    )
    control = rows(
        fortran(
            TABLE_WAKE_SOURCE,
            files={"lat.bmad": lattice.replace("smoothing_sigma=0.005", "smoothing_sigma=0")},
        ).log
    )
    require("unsmoothed delta; half self-bin", control["table"], [0, -1e-6, 0], rtol=2e-13, atol=1e-22)
    actual = rows(fortran(TABLE_WAKE_SOURCE, files={"lat.bmad": lattice}).log)
    require("raw impulse (V/C)", actual["raw"], [0, 0, 0, 0, 1e18, 0, 0, 0, 0])
    # sigma/dz=0.5; nint(3*sigma/dz)=2. Half the impulse enters the convolution.
    kernel = np.exp(-0.5 * (np.arange(-2, 3) / 0.5) ** 2)
    expected = -1e-6 * kernel[1:4] / kernel.sum()
    defect = compare(
        "smoothed delta; normalized Gaussian of half self-bin",
        actual["table"],
        expected,
        rtol=2e-13,
        atol=1e-22,
    )
    defect |= compare(
        "outer-kick ratio; Gaussian symmetry",
        actual["table"][2] / actual["table"][0],
        1,
        rtol=2e-13,
        atol=0,
    )
    verdict(defect)

    ```

## Provenance

| Role | Model | Effort | Date | Code examined |
|---|---|---|---|---|
| Found | `gpt-6-astra` | xhigh | 2026-10-20 | 20261020-0 (`19b8afb46a40`), source, production, CPU, double, serial |
| Confirmed | `gpt-6.1-sol` | xhigh | 2026-10-20 | 20261020-0 (`19b8afb46a40`), source, production, CPU, double, serial |

