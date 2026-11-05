---
title: "OPT-07: Higher chromaticities mix cell and repeated-period phases"
hide:
  - toc
---

# OPT-07 — Higher chromaticities mix cell and repeated-period phases

| | |
|---|---|
| Code | [Elegant](../index.md) · Optics |
| Priority | P1 |
| Severity | Wrong physics |
| Evidence | Probe-pinned |
| Last reproduced | 2026-11-04 on Elegant master (`0355dd6`) |
| Tags | <span class="bb-tag">partly-source-only</span> [?](../../methodology.md#tags) |
| Source | [`computeHigherOrderChromaticities`, one-cell map construction and trace, chrom.c:928–942](https://github.com/rtsoliday/elegant/blob/0355dd6aca12c861971fbf8f3ae78a0ded991734/src/chrom.c#L928-L942)<br>[trace fit and chromaticity formulas, chrom.c:949–956](https://github.com/rtsoliday/elegant/blob/0355dd6aca12c861971fbf8f3ae78a0ded991734/src/chrom.c#L949-L956)<br>[`propagate_twiss_parameters` has already multiplied by `n_periods`, twiss.cc:806–807](https://github.com/rtsoliday/elegant/blob/0355dd6aca12c861971fbf8f3ae78a0ded991734/src/twiss.cc#L806-L807)<br>[multiplies the resulting `chrom2` and `chrom3` by `n_periods` again, twiss.cc:2296–2299](https://github.com/rtsoliday/elegant/blob/0355dd6aca12c861971fbf8f3ae78a0ded991734/src/twiss.cc#L2296-L2299)<br>[the same scaling, twiss.cc:2344–2347](https://github.com/rtsoliday/elegant/blob/0355dd6aca12c861971fbf8f3ae78a0ded991734/src/twiss.cc#L2344-L2347)<br>[`dnux/dp2` and `dnux/dp3` as d²ν/dδ² and d³ν/dδ³, doc/elegant.tex:8228–8231](https://github.com/rtsoliday/elegant/blob/0355dd6aca12c861971fbf8f3ae78a0ded991734/doc/elegant.tex#L8228-L8231)<br>[`n_periods` field, src/twiss.nl:77](https://github.com/rtsoliday/elegant/blob/0355dd6aca12c861971fbf8f3ae78a0ded991734/src/twiss.nl#L77) |

With `higher_order_chromaticity=1` and `n_periods=2`, Elegant's `twiss_output` converts single-cell trace derivatives using the repeated lattice's phase, producing wrong second and third chromaticities. On a continuous phase branch, a stable cell's tune ν and every momentum derivative must scale with the period count. For 0.5 m `KQUAD`s with K1 = 0.6 and −0.54 m⁻² and 1 m drifts, one period gives d²νₓ/dδ² = 0.1268 and d³νₓ/dδ³ = −0.08637, where δ is the relative momentum offset. Two periods require 0.2537 and −0.1727, but the output gives 0.1865 and −0.3286; the tune and first chromaticity scale correctly.

Download these shared helpers into the same directory as the reproducer scripts:

- [Download _cubic_optics.py](../helpers/_cubic_optics.py) · [View source](../helpers/index.md#_cubic_optics)
- [Download _elegant.py](../helpers/_elegant.py) · [View source](../helpers/index.md#_elegant)

??? example "Reproducer: OPT-07.py"

    [Download OPT-07.py](OPT-07.py)

    ```python
    """OPT-07: Higher chromaticities mix cell and repeated-period phases

    With `higher_order_chromaticity=1` and `n_periods=2`, Elegant's `twiss_output` converts single-cell
    trace derivatives using the repeated lattice's phase, producing wrong second and third
    chromaticities. On a continuous phase branch, a stable cell's tune ν and every momentum derivative
    must scale with the period count. For 0.5 m `KQUAD`s with K1 = 0.6 and −0.54 m⁻² and 1 m drifts, one
    period gives d²νₓ/dδ² = 0.1268 and d³νₓ/dδ³ = −0.08637, where δ is the relative momentum offset.
    Two periods require 0.2537 and −0.1727, but the output gives 0.1865 and −0.3286; the tune and first
    chromaticity scale correctly.

    The script computes the tune and its first three momentum derivatives of a FODO cell of
    tracking-derived `KQUAD` quadrupoles with `n_periods` 1 and 2 in serial Elegant (and with `--mpi` in
    3-rank Pelegant). It compares the two-period values with the exact scaling by the period count, and
    the one-period values with the exact hard-edge maps truncated at second order in δ. The tune and the
    first chromaticity are the controls, since they scale correctly.
    """

    import sys

    import numpy as np
    from _cubic_optics import fodo_tune_derivatives
    from _elegant import modes, run, verdict

    RTOL, ATOL = 1e-9, 1e-12  # exact scaling by the period count
    ORACLE_RTOL = 1e-5  # one-period values against the oracle: finite-difference accuracy of Elegant

    LATTICE = (
        "QF: KQUAD,N_SLICES=128,INTEGRATION_ORDER=4,TRACKING_MATRIX=3,L=0.5,K1=0.6,TILT=0\n"
        "QD: KQUAD,N_SLICES=128,INTEGRATION_ORDER=4,TRACKING_MATRIX=3,L=0.5,K1=-0.54,TILT=0\n"
        "D: DRIF,L=1\nBL: LINE=(QF,D,QD,D)\n"
    )
    NAMES = ("nu{}", "dnu{}/dp", "dnu{}/dp2", "dnu{}/dp3")  # tune and its derivatives with respect to δ


    def commands(periods):
        return (
            "&run_setup lattice=lattice.lte, use_beamline=BL, p_central=1000, default_order=3 &end\n"
            "&run_control n_steps=1, n_passes=1 &end\n"
            f"&twiss_output filename=twiss.sdds, concat_order=3, n_periods={periods},\n"
            " higher_order_chromaticity=1, higher_order_chromaticity_points=11,\n"
            " higher_order_chromaticity_range=0.01 &end\n"
            "&bunched_beam n_particles_per_bunch=12 &end\n"
            "&track &end\n"
        )


    def derivatives(mode, periods, plane):
        twiss = run(
            {"lattice.lte": LATTICE, "run.ele": commands(periods)}, mode, outputs=("twiss.sdds",)
        )["twiss.sdds"]
        return np.array([twiss[name.format(plane)][0] for name in NAMES])


    def fmt(values):
        return "[" + ", ".join(f"{v:.15g}" for v in np.ravel(values)) + "]"


    def compare(label, observed, expected, rtol=RTOL, atol=ATOL):
        """Print one case; return True when observed and expected disagree."""
        observed, expected = np.asarray(observed), np.asarray(expected)
        if not (np.isfinite(observed).all() and np.isfinite(expected).all()):
            sys.exit(f"{label}: nonfinite value, the comparison is not valid.")
        bad = not np.allclose(observed, expected, rtol=rtol, atol=atol)
        print(f"{label}: observed={fmt(observed)} expected={fmt(expected)}")
        return bad


    defect = False
    print("columns: nu, dnu/dp, dnu/dp2, dnu/dp3 (cycles); expected = n_periods x one-period value")
    for mode in modes():
        for plane in "xy":
            one, two = derivatives(mode, 1, plane), derivatives(mode, 2, plane)
            # Control: the one-period second and third derivatives agree with the exact hard-edge
            # maps truncated at second order in delta, so the one-period calculation is sound.
            oracle = fodo_tune_derivatives(plane, thin=False)
            if compare(
                f"{mode} {plane} control, one period vs exact maps", one[2:], oracle, ORACLE_RTOL, 1e-6
            ):
                sys.exit("The one-period values disagree with the oracle: the comparison is not valid.")
            # The period count multiplies the tune and every momentum derivative of a stable cell on a
            # fixed phase branch. Control: tune and first chromaticity do scale.
            if compare(f"{mode} {plane} control, nu and dnu/dp, two periods", two[:2], 2 * one[:2]):
                sys.exit("The tune/chromaticity control does not scale: the comparison is not valid.")
            defect |= compare(f"{mode} {plane} two periods, dnu/dp2 and dnu/dp3", two[2:], 2 * one[2:])
            # The same invariant against the oracle with n_periods = 2.
            compare(
                f"{mode} {plane} two periods vs exact maps",
                two[2:],
                fodo_tune_derivatives(plane, thin=False, periods=2),
                ORACLE_RTOL,
                1e-6,
            )
    print(f"comparison: rtol={RTOL:g}, atol={ATOL:g}")
    verdict(defect)

    ```

## Provenance

| Role | Model | Effort | Date | Code examined |
|---|---|---|---|---|
| Found | `gpt-6.1-sol` | xhigh | 2026-10-15 | 2026.4.0 (`d177ff5347c7`), source with SDDS 3fa650ee9a1c, CPU, serial |
| Confirmed | `gpt-6-astra` | xhigh | 2026-10-15 | 2026.4.0 (`d177ff5347c7`), source with SDDS 3fa650ee9a1c, CPU, serial |
| Confirmed | `gpt-6.1-sol` | xhigh | 2026-10-15 | 2026.4.0 (`d177ff5347c7`), source with SDDS 3fa650ee9a1c, CPU, serial |
| Confirmed | `gpt-6-astra` | high | 2026-10-25 | master (`75a2e1a1a58b`), source with SDDS a60b9bbed7cc, CPU, serial and 3-rank Pelegant |
| Confirmed | `gpt-6.1-sol` | high | 2026-10-25 | master (`75a2e1a1a58b`), source with SDDS a60b9bbed7cc, CPU, serial and 3-rank Pelegant |

