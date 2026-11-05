---
title: "OPT-21: Tracking tune shift with amplitude ignores nuy_roi_width and the amplitude scaling factors"
hide:
  - toc
---

# OPT-21 — Tracking tune shift with amplitude ignores nuy_roi_width and the amplitude scaling factors

| | |
|---|---|
| Code | [Elegant](../index.md) · Optics |
| Priority | P4 |
| Severity | Crash or API hazard |
| Evidence | Reproduced, independently rechecked |
| Last reproduced | 2026-11-04 on Elegant master (`7514db3`) |
| Source | [`computeTuneShiftWithAmplitude`, peak windows, src/twiss.cc:3619–3622](https://github.com/rtsoliday/elegant/blob/0355dd6aca12c861971fbf8f3ae78a0ded991734/src/twiss.cc#L3619-L3622)<br>[the amplitude retry, src/twiss.cc:3666–3684](https://github.com/rtsoliday/elegant/blob/0355dd6aca12c861971fbf8f3ae78a0ded991734/src/twiss.cc#L3666-L3684)<br>[the namelist, src/twiss.nl:34–38](https://github.com/rtsoliday/elegant/blob/0355dd6aca12c861971fbf8f3ae78a0ded991734/src/twiss.nl#L34-L38)<br>[`nux_roi_width`, `nuy_roi_width` are the widths of the regions of interest for the x and y tunes, doc/elegant.tex:8468–8473](https://github.com/rtsoliday/elegant/blob/0355dd6aca12c861971fbf8f3ae78a0ded991734/doc/elegant.tex#L8468-L8473)<br>[a tune shift above `scale_down_limit` decreases the amplitude by `scale_down_factor`, one below `scale_up_limit` increases it by `scale_up_factor`, doc/elegant.tex:8474–8481](https://github.com/rtsoliday/elegant/blob/0355dd6aca12c861971fbf8f3ae78a0ded991734/doc/elegant.tex#L8474-L8481) |

In tracking mode, Elegant's `tune_shift_with_amplitude` searches the vertical tune peak within `nux_roi_width` and never reads `nuy_roi_width`, `scale_down_factor`, `scale_up_factor` or `scale_up_limit`, all documented in the manual. When the tune change exceeds `scale_down_limit`, it sets the largest amplitude to one third of itself plus two thirds of the smallest, whatever factor was requested, and it never scales amplitudes up. For eight FODO cells with a thin octupole, changing any of these controls (vertical window 0.02 to 10⁻¹⁰, scale-up factor 2 to 5, scale-up limit 0.1 to 10⁻¹⁰, scale-down factor 2 to 10) leaves every tracked tune and fitted coefficient bit-identical; both scale-down factors reduce the largest amplitude from 1.999 mm to 0.6657 mm.

Download these shared helpers into the same directory as the reproducer scripts:

- [Download _elegant.py](../helpers/_elegant.py) · [View source](../helpers/index.md#_elegant)

??? example "Reproducer: OPT-21.py"

    [Download OPT-21.py](OPT-21.py)

    ```python
    """OPT-21: Tracking tune shift with amplitude ignores nuy_roi_width and the amplitude scaling factors

    In tracking mode, Elegant's `tune_shift_with_amplitude` searches the vertical tune peak within
    `nux_roi_width` and never reads `nuy_roi_width`, `scale_down_factor`, `scale_up_factor` or
    `scale_up_limit`, all documented in the manual. When the tune change exceeds `scale_down_limit`, it
    sets the largest amplitude to one third of itself plus two thirds of the smallest, whatever factor
    was requested, and it never scales amplitudes up. For eight FODO cells with a thin octupole,
    changing any of these controls (vertical window 0.02 to 10⁻¹⁰, scale-up factor 2 to 5, scale-up
    limit 0.1 to 10⁻¹⁰, scale-down factor 2 to 10) leaves every tracked tune and fitted coefficient
    bit-identical; both scale-down factors reduce the largest amplitude from 1.999 mm to 0.6657 mm.

    The script runs serial Elegant `tune_shift_with_amplitude` in tracking mode (4096 turns, a 4×4 grid
    from 1 μm to 2 mm, `tune_output`) on eight FODO cells with one thin octupole, changing one
    documented control at a time: `nuy_roi_width` from 0.02 to 10⁻¹⁰, `scale_up_factor` from 2 to 5,
    `scale_up_limit` from 0.1 to 10⁻¹⁰, and, with `scale_down_limit` = 10⁻⁵, `scale_down_factor` from
    2 to 10. It compares the tracked grids (x, y, νx, νy) and the fitted coefficients bit for bit. The
    largest tune change on the grid is printed as a control: it exceeds the narrow vertical window by
    five orders of magnitude and stays far below the scale-up limit of 0.1.
    """

    import numpy as np
    from _elegant import run, verdict

    LATTICE = """F: QUAD,L=0.25,K1=1.1,ORDER=1
    G: QUAD,L=0.25,K1=-1.0,ORDER=1
    D: DRIF,L=0.7
    E: DRIF,L=0.9
    O: MULT,ORDER=3,KNL=3.0
    C: LINE=(F,D,G,E)
    A: LINE=(O,8*C)
    """
    COMMON = "turns=4096, x0=1e-6, y0=1e-6, x1=0.002, y1=0.002, grid_size=4, order=1"
    COEFFICIENTS = ["dnux/dAx", "dnux/dAy", "dnuy/dAx", "dnuy/dAy"]


    def tswa(options):
        """Tracked grid (x, y, nux, nuy) and fitted coefficients for one set of options."""
        commands = (
            "&run_setup lattice=lattice.lte, use_beamline=A, p_central_mev=2400, default_order=3 &end\n"
            f"&tune_shift_with_amplitude {COMMON}, tune_output=tunes.sdds, {options} &end\n"
            "&twiss_output filename=twiss.sdds, matched=1, output_at_each_step=1, concat_order=3 &end\n"
            "&run_control &end\n&bunched_beam &end\n&track &end\n"
        )
        result = run(
            {"lattice.lte": LATTICE, "run.ele": commands}, outputs=("tunes.sdds", "twiss.sdds")
        )
        grid = np.array(
            [np.asarray(result["tunes.sdds"][0, c], float) for c in ("x", "y", "nux", "nuy")]
        )
        coefficients = np.array([float(result["twiss.sdds"][name][0]) for name in COEFFICIENTS])
        return grid, coefficients


    def identical(a, b):
        return a[0].shape == b[0].shape and np.array_equal(a[0], b[0]) and np.array_equal(a[1], b[1])


    defect = False
    up = "nuy_roi_width=0.02, scale_up_limit=0.1, scale_up_factor=2"
    base = tswa(up)
    spread = np.ptp(base[0][3])
    print(
        f"control: vertical tune range on the grid = {spread:.4g}, largest tune change = "
        f"{max(np.ptp(base[0][2]), spread):.4g}; amplitude range x = {base[0][0].min():.4g} to "
        f"{base[0][0].max():.10g} m"
    )
    print(f"base coefficients {', '.join(COEFFICIENTS)} (1/m): {np.array2string(base[1], precision=8)}")
    # Each pair differs in one documented control; the manual says each changes the algorithm here.
    pairs = [
        (
            "nuy_roi_width 0.02 -> 1e-10 (vertical peak search window narrower than the tune change)",
            up.replace("nuy_roi_width=0.02", "nuy_roi_width=1e-10"),
        ),
        (
            "scale_up_factor 2 -> 5 (tune change below scale_up_limit=0.1: amplitudes scaled up)",
            up.replace("scale_up_factor=2", "scale_up_factor=5"),
        ),
        (
            "scale_up_limit 0.1 -> 1e-10 (scale-up no longer triggered)",
            up.replace("scale_up_limit=0.1", "scale_up_limit=1e-10"),
        ),
    ]
    for label, options in pairs:
        same = identical(base, tswa(options))
        print(f"{label}: observed={'identical' if same else 'different'} expected=different")
        defect |= same
    down2 = tswa("scale_down_limit=1e-5, scale_down_factor=2")
    down10 = tswa("scale_down_limit=1e-5, scale_down_factor=10")
    # The manual: a tune change above scale_down_limit decreases the amplitude by scale_down_factor.
    # The source instead sets x1 = (x1 + 2 x0)/3 on every retry, whatever the factor.
    print(f"scale_down_factor=2: largest x on the grid = {down2[0][0].max():.10g} m")
    print(f"scale_down_factor=10: largest x on the grid = {down10[0][0].max():.10g} m")
    same = identical(down2, down10)
    print(
        f"scale_down_factor 2 -> 10: observed={'identical' if same else 'different'} expected=different"
    )
    defect |= same
    verdict(defect)

    ```

## Provenance

| Role | Model | Effort | Date | Code examined |
|---|---|---|---|---|
| Found | `claude-opus-5-5` | high | 2026-11-04 | master (`7514db36f764`), source with SDDS a3f730a14c49, CPU, serial |
| Confirmed<br><small>nuy_roi_width, scale_down_factor, scale_up_factor and scale_up_limit</small> | `gpt-6.1-sol` | xhigh | 2026-11-04 | master (`7514db36f764`), source with SDDS a3f730a14c49, CPU, serial |

