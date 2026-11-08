---
title: "COL-08: Broadband ZTRANSVERSE doubles its DC impedance"
hide:
  - toc
---

# COL-08 — Broadband ZTRANSVERSE doubles its DC impedance

| | |
|---|---|
| Code | [Elegant](../index.md) · Collective effects |
| Priority | P2 |
| Failure | Wrong physics |
| Evidence | Reproduced, independently rechecked |
| Last reproduced | 2026-11-04 on Elegant commit [`0355dd6`](https://github.com/rtsoliday/elegant/commit/0355dd6aca12), committed 2026-11-03 |
| Source | [DC coefficient, ztransverse.c:498–499](https://github.com/rtsoliday/elegant/blob/0355dd6aca12c861971fbf8f3ae78a0ded991734/src/ztransverse.c#L498-L499)<br>[interior coefficients, ztransverse.c:500–508](https://github.com/rtsoliday/elegant/blob/0355dd6aca12c861971fbf8f3ae78a0ded991734/src/ztransverse.c#L500-L508)<br>[impedance manual, ztransverse.tex:3–18](https://github.com/rtsoliday/elegant/blob/0355dd6aca12c861971fbf8f3ae78a0ded991734/doc/ztransverse.tex#L3-L18)<br>[spectrum product, ztransverse.c:291–301](https://github.com/rtsoliday/elegant/blob/0355dd6aca12c861971fbf8f3ae78a0ded991734/src/ztransverse.c#L291-L301)<br>[real FFT packing, fftpackC.c:133–156](https://github.com/rtsoliday/SDDS/blob/a3f730a14c49ddd4f4d6347b42d60d34aa7cadc6/fftpack/fftpackC.c#L133-L156)<br>[file DC convention, ztransverse.c:575–581](https://github.com/rtsoliday/elegant/blob/0355dd6aca12c861971fbf8f3ae78a0ded991734/src/ztransverse.c#L575-L581)<br>[manual note, ztransverse.tex:33–37](https://github.com/rtsoliday/elegant/blob/0355dd6aca12c861971fbf8f3ae78a0ded991734/doc/ztransverse.tex#L33-L37) |

Broadband ZTRANSVERSE sets its zero-frequency impedance to 2Rₛ/Q instead of the documented Rₛ/Q, where Rₛ is the shunt impedance and Q the resonator quality factor. This adds a constant transverse wake over the finite Fourier-transform window for beams with a net transverse offset. For 12 electron macroparticles at 50 mₑc representing 2 nC total charge and 1 mm horizontal offset in adjacent 50 ps bins, with Rₛ = 1×10⁶ Ω/m, Q = 1, resonant frequency 1 GHz and 8192 bins, the leading group's kick is 6.960×10⁻⁶ rad instead of the independently summed 6.769×10⁻⁶ rad. The excess is 1.911×10⁻⁷ rad per particle and halves when the window doubles.

Run `python COL-08.py` with [_elegant.py](../helpers/_elegant.py) and [_ztransverse.py](../helpers/_ztransverse.py) in the same directory.

??? example "Reproducer: COL-08.py"

    [Download COL-08.py](COL-08.py)

    ```python
    """COL-08: Broadband ZTRANSVERSE doubles its DC impedance

    Broadband ZTRANSVERSE sets its zero-frequency impedance to 2Rₛ/Q instead of the documented Rₛ/Q,
    where Rₛ is the shunt impedance and Q the resonator quality factor. This adds a constant
    transverse wake over the finite Fourier-transform window for beams with a net transverse offset.
    For 12 electron macroparticles at 50 mₑc representing 2 nC total charge and 1 mm horizontal
    offset in adjacent 50 ps bins, with Rₛ = 1×10⁶ Ω/m, Q = 1, resonant frequency 1 GHz and 8192
    bins, the leading group's kick is 6.960×10⁻⁶ rad instead of the independently summed 6.769×10⁻⁶
    rad. The excess is 1.911×10⁻⁷ rad per particle and halves when the window doubles.

    The script tracks the beam through the broadband element and through a table of the same impedance on the same frequency grid, and compares both with the kick of the analytic impedance, summed as a Fourier series without an FFT. The table is the control: it agrees with the series to rounding, so the kick difference of the broadband element isolates its DC coefficient. Adjacent bins with equal offsets cancel the Nyquist source moment, leaving the DC term as the only difference. The excess is compared with the kick of an extra Rs/Q coefficient, for windows of 8192 and 16384 bins.
    """

    import sys

    import numpy as np
    from _elegant import modes, verdict
    from _ztransverse import CHARGE, COUNT, DT, ELECTRON_EV, RS, P, Q, kicks, oracle, show

    ATOL = 1e-18  # rad; kicks are ~1e-5 rad, rounding of the sums ~1e-20 rad
    MIN_ERROR = 1e-12  # rad; the DC excess is ~1e-7 rad
    X = np.full(COUNT, 1e-3)  # m, equal offsets: the Nyquist source moment cancels

    defect = False
    for mode in modes():
        for n_bins in (8192, 16384):
            # Series kick of the analytic impedance (DC coefficient Rs/Q), the oracle.
            expected = oracle(X, n_bins)
            table = kicks(mode, X, n_bins, broadband=False)
            show(mode, f"N={n_bins} impedance table", table, expected)
            if not np.all(np.isfinite(expected)) or not np.all(np.isfinite(table)):
                sys.exit("Nonfinite oracle or impedance-table kicks.")
            if not np.allclose(table, expected, rtol=0, atol=ATOL):
                sys.exit("The table control disagrees with the Fourier-series oracle.")
            broadband = kicks(mode, X, n_bins, broadband=True)
            if not np.all(np.isfinite(broadband)):
                sys.exit("Nonfinite broadband kicks.")
            show(mode, f"N={n_bins} broadband", broadband, expected)
            # A DC coefficient larger by Rs/Q adds the same slope to every particle:
            #   q (Rs/Q) sum(x) / (T p0 m c^2/e), T = N dt the window, q the macroparticle charge.
            predicted = CHARGE / COUNT * (RS / Q) * X.sum() / (n_bins * DT * P * ELECTRON_EV)
            excess = broadband - expected
            show(mode, f"N={n_bins} broadband excess", excess, np.full(2, predicted))
            defect |= bool(np.all(np.abs(excess - predicted) < ATOL) and np.all(excess > MIN_ERROR))
    print(f"tolerance {ATOL:g} rad; defect if the excess exceeds {MIN_ERROR:g} rad")
    verdict(defect)

    ```

## Provenance

| Role | Model | Effort | Date | Code examined |
|---|---|---|---|---|
| Found | <span class="bb-nowrap">`gpt-6-astra`</span> | xhigh | <span class="bb-nowrap">2026-10-15</span> | 2026.4.0 (`d177ff5347c7`), source reading, not executed for this claim |
| Confirmed | <span class="bb-nowrap">`gpt-6-astra`</span> | xhigh | <span class="bb-nowrap">2026-10-15</span> | 2026.4.0 (`d177ff5347c7`), source with SDDS 3fa650ee9a1c, CPU, serial |
| Confirmed | <span class="bb-nowrap">`gpt-6.1-sol`</span> | xhigh | <span class="bb-nowrap">2026-10-15</span> | 2026.4.0 (`d177ff5347c7`), source with SDDS 3fa650ee9a1c, CPU, serial |
| Confirmed | <span class="bb-nowrap">`gpt-6-astra`</span> | high | <span class="bb-nowrap">2026-10-25</span> | master (`75a2e1a1a58b`), source with SDDS a60b9bbed7cc, CPU, serial and 3-rank Pelegant |
| Confirmed | <span class="bb-nowrap">`gpt-6.1-sol`</span> | high | <span class="bb-nowrap">2026-10-25</span> | master (`75a2e1a1a58b`), source with SDDS a60b9bbed7cc, CPU, serial and 3-rank Pelegant |

