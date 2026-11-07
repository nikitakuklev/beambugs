---
title: "COL-09: Broadband ZTRANSVERSE uses the wrong Nyquist frequency and sign"
hide:
  - toc
---

# COL-09 — Broadband ZTRANSVERSE uses the wrong Nyquist frequency and sign

| | |
|---|---|
| Code | [Elegant](../index.md) · Collective effects |
| Priority | P3 |
| Failure | Wrong physics |
| Evidence | Reproduced, independently rechecked |
| Last reproduced | 2026-11-04 on Elegant commit [`0355dd6`](https://github.com/rtsoliday/elegant/commit/0355dd6aca12), committed 2026-11-03 |
| Source | [frequency count, ztransverse.c:492](https://github.com/rtsoliday/elegant/blob/0355dd6aca12c861971fbf8f3ae78a0ded991734/src/ztransverse.c#L492)<br>[Nyquist coefficient, ztransverse.c:509–513](https://github.com/rtsoliday/elegant/blob/0355dd6aca12c861971fbf8f3ae78a0ded991734/src/ztransverse.c#L509-L513)<br>[interior coefficients, ztransverse.c:500–508](https://github.com/rtsoliday/elegant/blob/0355dd6aca12c861971fbf8f3ae78a0ded991734/src/ztransverse.c#L500-L508)<br>[real FFT packing, fftpackC.c:133–156](https://github.com/rtsoliday/SDDS/blob/a3f730a14c49ddd4f4d6347b42d60d34aa7cadc6/fftpack/fftpackC.c#L133-L156)<br>[spectrum product, ztransverse.c:291–301](https://github.com/rtsoliday/elegant/blob/0355dd6aca12c861971fbf8f3ae78a0ded991734/src/ztransverse.c#L291-L301)<br>[impedance manual, ztransverse.tex:3–18](https://github.com/rtsoliday/elegant/blob/0355dd6aca12c861971fbf8f3ae78a0ded991734/doc/ztransverse.tex#L3-L18) |

Broadband ZTRANSVERSE evaluates its highest resolved (Nyquist) impedance at the next frequency index and reverses the real part's sign. With 50 ps bins, a resonator of shunt impedance 1×10⁶ Ω/m, quality factor 1 and resonant frequency 1 GHz should have −9999 Ω/m at the 10 GHz endpoint; the element uses +9994 Ω/m. For a 2 nC electron beam at 50 mₑc, six particles at +1 mm horizontal offset and six at −1 mm in adjacent bins, the 8192-bin calculation adds errors of ±3.821×10⁻⁹ rad relative to an independent Fourier sum and impedance-table control. The error alternates between bins and shrinks with a longer window.

Run `python COL-09.py` with [_elegant.py](../helpers/_elegant.py) and [_ztransverse.py](../helpers/_ztransverse.py) in the same directory.

??? example "Reproducer: COL-09.py"

    [Download COL-09.py](COL-09.py)

    ```python
    """COL-09: Broadband ZTRANSVERSE uses the wrong Nyquist frequency and sign

    Broadband ZTRANSVERSE evaluates its highest resolved (Nyquist) impedance at the next frequency index
    and reverses the real part's sign. With 50 ps bins, a resonator of shunt impedance 1×10⁶ Ω/m,
    quality factor 1 and resonant frequency 1 GHz should have −9999 Ω/m at the 10 GHz endpoint; the
    element uses +9994 Ω/m. For a 2 nC electron beam at 50 mₑc, six particles at +1 mm horizontal offset
    and six at −1 mm in adjacent bins, the 8192-bin calculation adds errors of ±3.821×10⁻⁹ rad relative
    to an independent Fourier sum and impedance-table control. The error alternates between bins and
    shrinks with a longer window.

    The script tracks the beam through the broadband element and through a table of the same impedance on the same frequency grid, and compares both with the kick of the analytic impedance, summed as a Fourier series without an FFT. The table is the control: it agrees with the series to rounding. Opposite offsets in adjacent bins cancel the DC source moment, leaving the endpoint as the only difference; evaluating the series with the element's endpoint value reproduces the broadband kick.
    """

    import sys

    import numpy as np
    from _elegant import modes, verdict
    from _ztransverse import DT, FR, RS, Q, impedance, kicks, oracle, show

    ATOL = 1e-18  # rad; kicks are ~1e-5 rad, rounding of the sums ~1e-20 rad
    MIN_ERROR = 1e-12  # rad; the endpoint error is ~4e-9 rad
    N = 8192
    X = np.repeat([1e-3, -1e-3], 6)  # m, opposite offsets: the DC source moment cancels

    # The correct endpoint is the analytic real impedance at the Nyquist frequency N/2 / (N dt).
    correct = float(impedance(0.5 / DT / FR).real)
    # The element evaluates r = (N/2 + 1)/(N dt fr) with the sign of the real part reversed:
    #   term = Q (1/r - r), Re Z_code = -term Rs / r / (1 + term^2).
    r_code = (N // 2 + 1) / (N * DT) / FR
    term = Q * (1 / r_code - r_code)
    code = -term * RS / r_code / (1 + term**2)
    print(f"Nyquist real impedance: analytic={correct!r} Ohm/m, element={code!r} Ohm/m")

    defect = False
    for mode in modes():
        expected = oracle(X, N)
        table = kicks(mode, X, N, broadband=False)
        show(mode, "impedance table", table, expected)
        if not np.all(np.isfinite(expected)) or not np.all(np.isfinite(table)):
            sys.exit("Nonfinite oracle or impedance-table kicks.")
        if not np.allclose(table, expected, rtol=0, atol=ATOL):
            sys.exit("The table control disagrees with the Fourier-series oracle.")
        broadband = kicks(mode, X, N, broadband=True)
        if not np.all(np.isfinite(broadband)):
            sys.exit("Nonfinite broadband kicks.")
        show(mode, "broadband", broadband, expected)
        # The same series with the element's endpoint value predicts the broadband kick.
        predicted = oracle(X, N, nyquist_re=code)
        if not np.all(np.isfinite(predicted)):
            sys.exit("Nonfinite model kicks.")
        show(mode, "broadband error", broadband - expected, predicted - expected)
        defect |= bool(
            np.max(np.abs(broadband - predicted)) < ATOL
            and np.max(np.abs(broadband - expected)) > MIN_ERROR
        )
    print(f"tolerance {ATOL:g} rad; defect if the broadband kick is off by more than {MIN_ERROR:g} rad")
    verdict(defect)

    ```

## Provenance

| Role | Model | Effort | Date | Code examined |
|---|---|---|---|---|
| Found<br><small>Nyquist term evaluated at index N/2+1</small> | `gpt-6-astra` | xhigh | 2026-10-15 | 2026.4.0 (`d177ff5347c7`), source reading, not executed for this claim |
| Found<br><small>reversed real-part sign of the Nyquist term</small> | `gpt-6.1-sol` | xhigh | 2026-10-15 | 2026.4.0 (`d177ff5347c7`), source reading, not executed for this claim |
| Confirmed | `gpt-6-astra` | xhigh | 2026-10-15 | 2026.4.0 (`d177ff5347c7`), source with SDDS 3fa650ee9a1c, CPU, serial |
| Confirmed | `gpt-6.1-sol` | xhigh | 2026-10-15 | 2026.4.0 (`d177ff5347c7`), source with SDDS 3fa650ee9a1c, CPU, serial |
| Confirmed | `gpt-6-astra` | high | 2026-10-25 | master (`75a2e1a1a58b`), source with SDDS a60b9bbed7cc, CPU, serial and 3-rank Pelegant |
| Confirmed | `gpt-6.1-sol` | high | 2026-10-25 | master (`75a2e1a1a58b`), source with SDDS a60b9bbed7cc, CPU, serial and 3-rank Pelegant |

