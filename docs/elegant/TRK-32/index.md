---
title: "TRK-32: CCBEND with ETILT as its only error skips the entrance misalignment transform"
hide:
  - toc
---

# TRK-32 — CCBEND with ETILT as its only error skips the entrance misalignment transform

| | |
|---|---|
| Code | [Elegant](../index.md) · Tracking |
| Priority | P2 |
| Severity | Wrong physics |
| Evidence | Reproduced, independently rechecked |
| Last reproduced | 2026-11-04 on Elegant master (`7514db3`, head build) |
| Source | [`track_through_ccbend_impl`, ccbend.c:518–529](https://github.com/rtsoliday/elegant/blob/0355dd6aca12c861971fbf8f3ae78a0ded991734/src/ccbend.c#L518-L529)<br>[ccbend.c:637–642](https://github.com/rtsoliday/elegant/blob/0355dd6aca12c861971fbf8f3ae78a0ded991734/src/ccbend.c#L637-L642)<br>[parameter table, track_data.c:3818–3823](https://github.com/rtsoliday/elegant/blob/0355dd6aca12c861971fbf8f3ae78a0ded991734/src/track_data.c#L3818-L3823)<br>[manual, ccbend.tex:89–95](https://github.com/rtsoliday/elegant/blob/0355dd6aca12c861971fbf8f3ae78a0ded991734/doc/ccbend.tex#L89-L95) |

A CCBEND whose only error is the roll ETILT, with DX, DY, DZ, EPITCH and EYAW all zero, applies the misalignment transform at the exit but not at the entrance. A nonzero DX, even 10⁻¹⁶ m, restores the full transform, so the result jumps at zero displacement. For a 0.9 m, 0.12 rad bend rolled by 2 mrad, an on-axis electron leaves at y = 8.262×10⁻¹⁹ m and y′ = −1.199×10⁻⁴ instead of y = −1.079×10⁻⁴ m and y′ = −2.390×10⁻⁴, the limit from both sides of DX = 0, which the small-angle estimates for a rolled dipole confirm. A roll-only error thus gives half the vertical kick and no vertical displacement.

Run `python TRK-32.py` with [_elegant.py](../helpers/_elegant.py) in the same directory.

??? example "Reproducer: TRK-32.py"

    [Download TRK-32.py](TRK-32.py)

    ```python
    """TRK-32: CCBEND with ETILT as its only error skips the entrance misalignment transform

    A CCBEND whose only error is the roll ETILT, with DX, DY, DZ, EPITCH and EYAW all zero, applies the
    misalignment transform at the exit but not at the entrance. A nonzero DX, even 10⁻¹⁶ m, restores the
    full transform, so the result jumps at zero displacement. For a 0.9 m, 0.12 rad bend rolled by 2
    mrad, an on-axis electron leaves at y = 8.262×10⁻¹⁹ m and y′ = −1.199×10⁻⁴ instead of y =
    −1.079×10⁻⁴ m and y′ = −2.390×10⁻⁴, the limit from both sides of DX = 0, which the small-angle
    estimates for a rolled dipole confirm. A roll-only error thus gives half the vertical kick and no
    vertical displacement.

    The script tracks on-axis electrons through a CCBEND whose only error is ETILT and through the same
    magnet with an added displacement DX = ±10⁻¹⁶ m and 10⁻¹⁰ m, which cannot change a finite roll
    response. The two-sided limit at DX = 0 is the expected exit orbit; a small-angle estimate for a
    rolled dipole checks its size, and the ETILT = 0 runs are the control.
    """

    import sys

    import numpy as np
    from _elegant import modes, particles, run, verdict

    P0 = 6000.0  # reference momentum in m_e c (3.066 GeV electrons)
    LENGTH, ANGLE, SLICES = 0.9, 0.12, 64
    N = 12  # identical on-axis particles, so that every MPI worker tracks some
    # DX = +1e-16 and -1e-16 agree to about 1e-18 m; a displacement of 1e-16 m can move the exit orbit
    # by at most about that much, so the limit is known far better than TOL.
    TOL = 1e-12
    COLS = ("x", "xp", "y", "yp")


    def fmt(values):
        return "[" + ", ".join(f"{v:.12g}" for v in np.atleast_1d(values)) + "]"


    def exit_orbit(mode, etilt, dx):
        """Exit (x, xp, y, yp) of on-axis electrons through one CCBEND with default settings."""
        lattice = (
            f"B: CCBEND,L={LENGTH!r},ANGLE={ANGLE!r},N_SLICES={SLICES},ETILT={etilt!r},DX={dx!r}\n"
            "BL: LINE=(B)\n"
        )
        commands = (
            f"&run_setup lattice=lattice.lte, use_beamline=BL, p_central={P0!r}, output=out.sdds &end\n"
            "&run_control n_steps=1, n_passes=1 &end\n"
            "&sdds_beam input=beam.sdds &end\n"
            "&track &end\n"
        )
        beam = {"p": np.full(N, P0)}
        out = particles(
            run({"lattice.lte": lattice, "run.ele": commands}, mode, {"beam.sdds": beam})["out.sdds"]
        )
        if len(out["p"]) != N:
            sys.exit(f"{mode}: particles were lost")
        q = np.array([out[c] for c in COLS])
        if np.max(np.abs(q - q[:, :1])) > 0:
            sys.exit(f"{mode}: identical particles leave with different coordinates")
        return q[:, 0]


    defect = []
    print(f"columns: {', '.join(COLS)} at the exit; L={LENGTH} m, ANGLE={ANGLE} rad, p0={P0:g} m_e c")
    for mode in modes():
        for etilt in (0.002, -0.002, 0.0):
            plus, minus = exit_orbit(mode, etilt, 1e-16), exit_orbit(mode, etilt, -1e-16)
            if np.max(np.abs(plus - minus)) > TOL:
                sys.exit(f"{mode} ETILT={etilt}: the two sides of DX=0 disagree: invalid")
            # The response is continuous in DX, so the limit DX -> 0 from both sides is the expected
            # exit orbit at DX = 0.
            expected = (plus + minus) / 2
            tiny = exit_orbit(mode, etilt, 1e-10)
            if np.max(np.abs(tiny - expected)) > 1e-9:
                sys.exit(f"{mode} ETILT={etilt}: DX=1e-10 is not continuous with the limit: invalid")
            observed = exit_orbit(mode, etilt, 0.0)
            diff = float(np.max(np.abs(observed - expected)))
            label = f"{mode} ETILT={etilt:g}"
            print(f"{label} DX=+-1e-16 (expected): {fmt(plus)} / {fmt(minus)}")
            print(f"{label} DX=1e-10: {fmt(tiny)}")
            print(
                f"{label} DX=0: observed={fmt(observed)} expected={fmt(expected)} max|diff|={diff:.4g}"
            )
            if etilt == 0.0:
                if diff > TOL:
                    sys.exit(f"{label}: the error-free control depends on DX: invalid")
                continue
            # Size check: a dipole rolled by e about its axis bends the orbit vertically by -ANGLE*e
            # spread over the magnet, so to leading order y = -L*ANGLE*e/2 and yp = -ANGLE*e.
            estimate = np.array([-LENGTH * ANGLE * etilt / 2, -ANGLE * etilt])
            if np.max(np.abs(expected[2:] / estimate - 1)) > 0.01:
                sys.exit(f"{label}: the limit disagrees with the rolled-dipole estimate: invalid")
            print(f"{label} rolled-dipole estimate (y, yp): {fmt(estimate)}")
            defect.append(diff > TOL)
    print(f"comparison: abs tolerance {TOL:g} on x, xp, y, yp")
    verdict(bool(defect) and all(defect))

    ```

## Provenance

| Role | Model | Effort | Date | Code examined |
|---|---|---|---|---|
| Found | `gemini-4-argon-medium` |  | 2026-11-04 | master (`7514db36f764`), source with SDDS a3f730a14c49, CPU, serial |
| Found<br><small>entrance-transform omission and rolled-dipole comparison</small> | `claude-opus-5-5` | high | 2026-11-04 | master (`7514db36f764`), source with SDDS a3f730a14c49, CPU, serial |
| Confirmed | `gpt-6.1-sol` | xhigh | 2026-11-04 | master (`7514db36f764`), source with SDDS a3f730a14c49, CPU, serial |

