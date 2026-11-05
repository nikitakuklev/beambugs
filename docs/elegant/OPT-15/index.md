---
title: "OPT-15: Individual matrix output keeps nonlinear coefficients of preceding elements"
hide:
  - toc
---

# OPT-15 — Individual matrix output keeps nonlinear coefficients of preceding elements

| | |
|---|---|
| Code | [Elegant](../index.md) · Optics |
| Priority | P4 |
| Severity | Silent wrong result |
| Evidence | Reproduced, independently rechecked |
| Last reproduced | 2026-11-04 on Elegant master (`0355dd6`) |
| Source | [`null_matrices`, matrix.c:183–200](https://github.com/rtsoliday/elegant/blob/0355dd6aca12c861971fbf8f3ae78a0ded991734/src/matrix.c#L183-L200)<br>[`initialize_matrices`, matrix.c:130–177](https://github.com/rtsoliday/elegant/blob/0355dd6aca12c861971fbf8f3ae78a0ded991734/src/matrix.c#L130-L177)<br>[`run_matrix_output`, matrix_output.c:422–429](https://github.com/rtsoliday/elegant/blob/0355dd6aca12c861971fbf8f3ae78a0ded991734/src/matrix_output.c#L422-L429)<br>[`SDDS_set_matrices`, matrix_output.c:733–752](https://github.com/rtsoliday/elegant/blob/0355dd6aca12c861971fbf8f3ae78a0ded991734/src/matrix_output.c#L733-L752)<br>[`individual_matrices` writes the individual on-trajectory matrices of the elements, elegant.tex:5362–5363](https://github.com/rtsoliday/elegant/blob/0355dd6aca12c861971fbf8f3ae78a0ded991734/doc/elegant.tex#L5362-L5363) |

Elegant's `matrix_output` with `individual_matrices=1` should write the map of each element on its own trajectory, but the second- and third-order coefficients of an element carry leftover values from earlier elements. A 1 m drift after a 0.4 m quadrupole with K1 = 0.7 m⁻² is written with T161 = 0.3243, T262 = 0.05496, U1661 = −0.3204 and U2662 = −0.05457, while a drift does not depend on the relative momentum offset δ and all four are 0. Putting the drift first or switching the quadrupole off gives zeros.

Download these shared helpers into the same directory as the reproducer scripts:

- [Download _elegant.py](../helpers/_elegant.py) · [View source](../helpers/index.md#_elegant)

??? example "Reproducer: OPT-15.py"

    [Download OPT-15.py](OPT-15.py)

    ```python
    """OPT-15: Individual matrix output keeps nonlinear coefficients of preceding elements

    Elegant's `matrix_output` with `individual_matrices=1` should write the map of each element on its
    own trajectory, but the second- and third-order coefficients of an element carry leftover values
    from earlier elements. A 1 m drift after a 0.4 m quadrupole with K1 = 0.7 m⁻² is written with T161 =
    0.3243, T262 = 0.05496, U1661 = −0.3204 and U2662 = −0.05457, while a drift does not depend on the
    relative momentum offset δ and all four are 0. Putting the drift first or switching the quadrupole
    off gives zeros.

    The script writes the individual-element matrices of a quadrupole followed by a drift to an SDDS
    file, in serial Elegant (and with `--mpi` in 3-rank Pelegant), and compares the drift's
    coefficients with the exact drift map. The controls are the drift placed before the quadrupole and
    a quadrupole of zero strength; both give exact zeros. Second-order output, without the third-order
    coefficients, shows the same error.
    """

    import sys

    import numpy as np
    from _elegant import modes, run, verdict

    DRIFT = 1.0  # m
    QUAD = 0.4  # m
    ATOL = 1e-15  # the exact drift coefficients are zero; the control values are exactly zero


    def commands(order):
        return (
            f"&run_setup lattice=lattice.lte, use_beamline=BL, p_central=200, default_order={order}"
            " &end\n"
            "&run_control n_steps=1 &end\n"
            f"&matrix_output SDDS_output=mat.sdds, SDDS_output_order={order}, individual_matrices=1"
            " &end\n"
            "&bunched_beam n_particles_per_bunch=12 &end\n"
            "&track &end\n"
        )


    def drift_row(mode, k1, quad_first, order):
        """R11, R12, T161, T262 (and U1661, U2662 at third order) written for the drift."""
        names = ["R11", "R12", "T161", "T262"] + (["U1661", "U2662"] if order == 3 else [])
        sequence = "Q,D" if quad_first else "D,Q"
        lattice = (
            f"Q: QUAD,L={QUAD!r},K1={k1!r},ORDER={order}\n"
            f"D: DRIF,L={DRIFT!r},ORDER={order}\n"
            f"BL: LINE=({sequence})\n"
        )
        result = run({"lattice.lte": lattice, "run.ele": commands(order)}, mode, outputs=("mat.sdds",))
        matrices = result["mat.sdds"]
        row = list(matrices[0, "ElementName"]).index("D")
        return np.array([float(matrices[0, name][row]) for name in names])


    def fmt(values):
        return "[" + ", ".join(f"{v:.16g}" for v in np.atleast_1d(values)) + "]"


    def compare(label, observed, expected):
        """Print one case; return True when observed and expected disagree."""
        bad = not np.allclose(observed, expected, rtol=0, atol=ATOL)
        print(f"{label}: observed={fmt(observed)} expected={fmt(expected)}")
        return bad


    defect = False
    print("columns: drift R11, R12, T161, T262, U1661, U2662 (second order: first four)")
    for mode in modes():
        for order in (3, 2):
            # A drift is x1 = x + L x', x'1 = x' for every momentum offset, so R11 = R12 = 1 (L = 1 m).
            # The transverse drift relations have no nonlinear coefficients, so T161, T262, U1661 and
            # U2662 are zero; T161 and T262 would be the momentum derivatives of R11 and R22 (the
            # coefficients of x delta and x' delta).
            expected = np.array([1, DRIFT, 0, 0, 0, 0][: 6 if order == 3 else 4])
            for label, k1, quad_first in (
                ("control, drift before quadrupole", 0.7, False),
                ("control, K1=0", 0.0, True),
            ):
                if compare(
                    f"{mode} order={order} {label}", drift_row(mode, k1, quad_first, order), expected
                ):
                    sys.exit("A control has nonzero coefficients: the comparison is not valid.")
            observed = drift_row(mode, 0.7, True, order)
            # The target row must be the drift (finite, R11 = 1, R12 = L) before the verdict, which
            # then rests on the nonlinear coefficients alone.
            if not np.isfinite(observed).all() or np.any(np.abs(observed[:2] - expected[:2]) > ATOL):
                sys.exit("The drift row is not finite with R11 = 1 and R12 = L: invalid comparison.")
            compare(f"{mode} order={order} K1=0.7, drift after quadrupole", observed, expected)
            defect |= bool(np.any(np.abs(observed[2:] - expected[2:]) > ATOL))
    print(f"comparison: absolute tolerance {ATOL:g}")
    verdict(defect)

    ```

## Provenance

| Role | Model | Effort | Date | Code examined |
|---|---|---|---|---|
| Found | `gpt-6.1-sol` | xhigh | 2026-11-03 | master (`0355dd6aca12`), source with SDDS a3f730a14c49, CPU, serial and 3-rank Pelegant |
| Confirmed | `gpt-6-astra` | xhigh | 2026-11-03 | master (`0355dd6aca12`), source with SDDS a3f730a14c49, CPU, serial and 3-rank Pelegant |

