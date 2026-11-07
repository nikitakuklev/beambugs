---
title: "OPT-06: Analytic QUAD cubic chromatic coefficients are wrong"
hide:
  - toc
---

# OPT-06 — Analytic QUAD cubic chromatic coefficients are wrong

| | |
|---|---|
| Code | [Elegant](../index.md) · Optics |
| Priority | P2 |
| Severity | Wrong physics |
| Evidence | Probe-pinned |
| Last reproduced | 2026-11-04 on Elegant commit [`0355dd6`](https://github.com/rtsoliday/elegant/commit/0355dd6aca12), committed 2026-11-03 |
| Tags | <span class="bb-tag">partly-source-only</span> [?](../../methodology.md#tags) |
| Source | [`quadrupole_matrix`, focusing-branch δ² coefficients of both transverse blocks, quad_matrix.c:162–199](https://github.com/rtsoliday/elegant/blob/0355dd6aca12c861971fbf8f3ae78a0ded991734/src/quad_matrix.c#L162-L199)<br>[the corresponding coefficients, quad_matrix.c:259–296](https://github.com/rtsoliday/elegant/blob/0355dd6aca12c861971fbf8f3ae78a0ded991734/src/quad_matrix.c#L259-L296) |

Elegant's aligned, hard-edge QUAD writes incorrect third-order matrix coefficients for transverse transport proportional to δ², where δ is the relative momentum offset. For length L = 0.5 m and strength K₁ = 0.6 m⁻², exact transport has R₁₁(δ) = cos(u/√(1+δ)), with u = √(K₁) L, giving U₁₆₆₁ = −0.07222; Elegant writes −0.07245, with relative errors across both planes of 0.1853% to 0.6598%. In a one-period FODO cell using cubic lens maps, d²νₓ/dδ² and d³νₓ/dδ³ are 0.1300 and −0.07925 instead of 0.1268 and −0.08637, where νₓ is the horizontal tune.

Run `python OPT-06.py` with [_cubic_optics.py](../helpers/_cubic_optics.py) and [_elegant.py](../helpers/_elegant.py) in the same directory.

??? example "Reproducer: OPT-06.py"

    [Download OPT-06.py](OPT-06.py)

    ```python
    """OPT-06: Analytic QUAD cubic chromatic coefficients are wrong

    Elegant's aligned, hard-edge QUAD writes incorrect third-order matrix coefficients for transverse
    transport proportional to δ², where δ is the relative momentum offset. For length L = 0.5 m and
    strength K₁ = 0.6 m⁻², exact transport has R₁₁(δ) = cos(u/√(1+δ)), with u = √(K₁) L, giving U₁₆₆₁ =
    −0.07222; Elegant writes −0.07245, with relative errors across both planes of 0.1853% to 0.6598%. In
    a one-period FODO cell using cubic lens maps, d²νₓ/dδ² and d³νₓ/dδ³ are 0.1300 and −0.07925 instead
    of 0.1268 and −0.08637, where νₓ is the horizontal tune.

    The script builds `QUAD,L=0.5,K1=0.6` with a third-order map in serial Elegant (and with `--mpi` in
    3-rank Pelegant) and compares the transverse 2x2 blocks of the written map, as Taylor series in δ,
    with the exact hard-edge transport. The same quadrupole as a tracking-derived `KQUAD` matrix, and
    the zeroth- and first-order coefficients of the analytic matrix, are the controls. The one-period
    FODO cell then gives the horizontal and vertical chromaticities, compared with the same exact maps
    truncated at second order in δ.
    """

    import sys

    import numpy as np
    from _cubic_optics import fodo_tune_derivatives, quad_matrix, taylor
    from _elegant import modes, run, verdict

    K1, LENGTH = 0.6, 0.5
    CONTROL_RTOL = 1e-12  # analytic R and T coefficients agree with the oracle to rounding
    TRACKING_RTOL, TRACKING_ATOL = 1e-5, 1e-6  # tracking-derived matrix: finite-difference accuracy
    RTOL, ATOL = 1e-9, 1e-12

    COMMANDS = (
        "&run_setup lattice=lattice.lte, use_beamline=BL, p_central=1000, default_order=3 &end\n"
        "&run_control n_steps=1, n_passes=1 &end\n"
        "&matrix_output SDDS_output=map.sdds, SDDS_output_order=3 &end\n"
        "&bunched_beam n_particles_per_bunch=12 &end\n"
        "&track &end\n"
    )
    FODO_COMMANDS = (
        "&run_setup lattice=lattice.lte, use_beamline=BL, p_central=1000, default_order=3 &end\n"
        "&run_control n_steps=1, n_passes=1 &end\n"
        "&twiss_output filename=twiss.sdds, concat_order=3, higher_order_chromaticity=1,\n"
        " higher_order_chromaticity_points=11, higher_order_chromaticity_range=0.01 &end\n"
        "&bunched_beam n_particles_per_bunch=12 &end\n"
        "&track &end\n"
    )
    FODO_LATTICE = "QF: QUAD,L=0.5,K1=0.6\nQD: QUAD,L=0.5,K1=-0.54\nD: DRIF,L=1\nBL: LINE=(QF,D,QD,D)\n"
    QUAD = f"QUAD,L={LENGTH!r},K1={K1!r}"
    KQUAD = f"KQUAD,L={LENGTH!r},K1={K1!r},N_SLICES=128,INTEGRATION_ORDER=4,TRACKING_MATRIX=3"


    def map_columns(mode, element):
        out = run(
            {"lattice.lte": f"E: {element}\nBL: LINE=(E)\n", "run.ele": COMMANDS},
            mode,
            outputs=("map.sdds",),
        )
        exported = out["map.sdds"]
        return lambda name: exported[0, name][-1]  # last row: the map through the element


    def coefficients(read, plane):
        """Elegant's R, T (x delta) and U (x delta^2) entries of one transverse block, as arrays [3][2][2].

        Entry (i, j) of order n is R_ij, T_i6j or U_i66j with 1-based indices offset by the plane.
        """
        first = 1 if plane == "x" else 3
        names = ("R{i}{j}", "T{i}6{j}", "U{i}66{j}")
        return np.array(
            [
                [[read(name.format(i=first + a, j=first + b)) for b in range(2)] for a in range(2)]
                for name in names
            ]
        )


    def exact(plane):
        """Taylor coefficients c_0, c_1, c_2 in delta of the exact hard-edge block.

        The gradient scales as K1/(1 + delta) (rigidity); the quadrupole defocuses in y. With
        u = sqrt(K1) L the x block has R11(delta) = cos(u/sqrt(1 + delta)), whose delta^2 coefficient
        is -(3 u sin u + u^2 cos u)/8. All entries are analytic in delta, so Cauchy's formula on a
        circle in the complex delta plane gives the coefficients to rounding.
        """
        sign = 1 if plane == "x" else -1
        return np.array([c.real for c in taylor(lambda z: quad_matrix(sign * K1 / (1 + z), LENGTH))])


    def fmt(values):
        return "[" + ", ".join(f"{v:.10g}" for v in np.ravel(values)) + "]"


    def compare(label, observed, expected, rtol=RTOL, atol=ATOL):
        """Print one case; return True when observed and expected disagree."""
        observed, expected = np.asarray(observed), np.asarray(expected)
        bad = not np.allclose(observed, expected, rtol=rtol, atol=atol)
        print(f"{label}: observed={fmt(observed)} expected={fmt(expected)}")
        return bad


    defect = False
    print("columns per block: R_ab, T_a6b, U_a66b with a, b over the plane's two coordinates")
    for mode in modes():
        analytic, tracked = map_columns(mode, QUAD), map_columns(mode, KQUAD)
        for plane in "xy":
            expected = exact(plane)
            got, control = coefficients(analytic, plane), coefficients(tracked, plane)
            # Control 1: orders 0 and 1 of the analytic matrix are right.
            if compare(
                f"{mode} {plane} control, R and T of QUAD", got[:2], expected[:2], CONTROL_RTOL, 1e-14
            ):
                sys.exit("The R/T control disagrees with the oracle: the comparison is not valid.")
            # Control 2: the tracking-derived KQUAD matrix reproduces the oracle at order 2.
            if compare(
                f"{mode} {plane} control, U of KQUAD",
                control[2],
                expected[2],
                TRACKING_RTOL,
                TRACKING_ATOL,
            ):
                sys.exit("The KQUAD control disagrees with the oracle: the comparison is not valid.")
            first = 1 if plane == "x" else 3
            names = [f"U{first + a}66{first + b}" for a in range(2) for b in range(2)]
            print(f"{mode} {plane} U entries: {names}")
            defect |= compare(f"{mode} {plane} U of QUAD", got[2], expected[2])

        # Consequence: higher chromaticities of the one-period FODO cell (cubic maps, analytic QUAD).
        twiss = run(
            {"lattice.lte": FODO_LATTICE, "run.ele": FODO_COMMANDS}, mode, outputs=("twiss.sdds",)
        )["twiss.sdds"]
        for plane in "xy":
            observed = [twiss[f"dnu{plane}/dp{order}"][0] for order in (2, 3)]
            expected = fodo_tune_derivatives(plane, thin=False)
            defect |= compare(
                f"{mode} FODO dnu{plane}/dp2, dnu{plane}/dp3", observed, expected, 1e-9, 1e-12
            )
    print(f"comparison: rtol={RTOL:g}, atol={ATOL:g}")
    verdict(defect)

    ```

## Provenance

| Role | Model | Effort | Date | Code examined |
|---|---|---|---|---|
| Found | `gpt-6-astra` | xhigh | 2026-10-15 | 2026.4.0 (`d177ff5347c7`), source with SDDS 3fa650ee9a1c, CPU, serial |
| Confirmed | `gpt-6-astra` | xhigh | 2026-10-15 | 2026.4.0 (`d177ff5347c7`), source with SDDS 3fa650ee9a1c, CPU, serial |
| Confirmed | `gpt-6.1-sol` | xhigh | 2026-10-15 | 2026.4.0 (`d177ff5347c7`), source with SDDS 3fa650ee9a1c, CPU, serial |
| Confirmed | `gpt-6-astra` | high | 2026-10-25 | master (`75a2e1a1a58b`), source with SDDS a60b9bbed7cc, CPU, serial and 3-rank Pelegant |
| Confirmed | `gpt-6.1-sol` | high | 2026-10-25 | master (`75a2e1a1a58b`), source with SDDS a60b9bbed7cc, CPU, serial and 3-rank Pelegant |

