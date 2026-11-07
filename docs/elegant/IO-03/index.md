---
title: "IO-03: SDDS complete elliptic integral E(k) returns π/4 at the endpoints k = ±1"
hide:
  - toc
---

# IO-03 — SDDS complete elliptic integral E(k) returns π/4 at the endpoints k = ±1

| | |
|---|---|
| Code | [Elegant](../index.md) · Lattices, particles and I/O |
| Priority | P4 |
| Failure | Silent wrong result |
| Evidence | Reproduced, independently rechecked |
| Last reproduced | 2026-11-04 on SDDS commit [`a3f730a`](https://github.com/rtsoliday/SDDS/commit/a3f730a14c49), committed 2026-10-31 |
| Source | [`K * (1 - sum / 2)`, elliptic.c:72–98](https://github.com/rtsoliday/SDDS/blob/a3f730a14c49ddd4f4d6347b42d60d34aa7cadc6/mdbmth/elliptic.c#L72-L98)<br>[`KE_cei`, elliptic.c:100–130](https://github.com/rtsoliday/SDDS/blob/a3f730a14c49ddd4f4d6347b42d60d34aa7cadc6/mdbmth/elliptic.c#L100-L130)<br>[rpn_cei2, rpns/code/math.c:744–756](https://github.com/rtsoliday/SDDS/blob/a3f730a14c49ddd4f4d6347b42d60d34aa7cadc6/rpns/code/math.c#L744-L756)<br>[cei1 and cei2, SDDSaps/doc/rpn.tex:110–113](https://github.com/rtsoliday/SDDS/blob/a3f730a14c49ddd4f4d6347b42d60d34aa7cadc6/SDDSaps/doc/rpn.tex#L110-L113) |

Evaluating the complete elliptic integral of the second kind, E(k) = ∫₀^{π/2} √(1 − k² sin²θ) dθ, with `sddsprocess` (rpn `cei2`) or the SDDS routines `E_cei` and `KE_cei` returns π/4 = 0.7854 at k = ±1. At those endpoints the integrand is cos θ and E = 1, so the result is 21.46% too small, without a utility warning. At the six tested interior moduli, results agree with an independent implementation within 2×10⁻¹⁵; at k = 0.8, E = 1.276.

Run `python IO-03.py` with [_elegant.py](../helpers/_elegant.py) and [_sdds.py](../helpers/_sdds.py) in the same directory.

??? example "Reproducer: IO-03.py"

    [Download IO-03.py](IO-03.py)

    ```python
    """IO-03: SDDS complete elliptic integral E(k) returns π/4 at the endpoints k = ±1

    Evaluating the complete elliptic integral of the second kind, E(k) = ∫₀^{π/2} √(1 − k² sin²θ) dθ, with `sddsprocess` (rpn `cei2`) or the SDDS routines `E_cei` and `KE_cei` returns π/4 = 0.7854 at k = ±1. At those endpoints the integrand is cos θ and E = 1, so the result is 21.46% too small, without a utility warning. At the six tested interior moduli, results agree with an independent implementation within 2×10⁻¹⁵; at k = 0.8, E = 1.276.

    Setup: `gcc`, and `sddsprocess` on `PATH` from an SDDS build tree, because the script compiles small C programs against that tree's `lib/Linux-x86_64/libmdbmth.a` and `libmdblib.a`.

    The script evaluates E at k = −1, 0, 0.25, 0.8, 0.99, 1 − 2⁻³⁰, the largest double below 1, and 1 with the library routines `E_cei` and `KE_cei` in a small C program and with `sddsprocess`. It compares with SciPy's Cephes `ellipe`, an independent implementation, and with the exact value 1 at the endpoints. The interior moduli are the control.
    """

    import sys

    import numpy as np
    from _elegant import verdict
    from _sdds import c_check, process
    from scipy.special import ellipe

    MODULI = np.array([-1.0, 0.0, 0.25, 0.8, 0.99, 1 - 2.0**-30, np.nextafter(1.0, 0.0), 1.0])
    ATOL = 5e-15  # interior agreement with Cephes; the largest difference seen is 1.1e-15
    C_SOURCE = r"""
    #include <math.h>
    #include <stdio.h>
    extern double E_cei(double);
    extern double *KE_cei(double, double *);
    int main(void) {
      const double moduli[] = {MODULI};
      for (unsigned i = 0; i < sizeof(moduli) / sizeof(moduli[0]); ++i) {
        double values[2];
        KE_cei(moduli[i], values);
        printf("%.17g %.17g\n", E_cei(moduli[i]), values[1]);
      }
      return 0;
    }
    """.replace("MODULI", ", ".join(repr(float(k)) for k in MODULI))


    # E(k) = int_0^{pi/2} sqrt(1 - k^2 sin^2 t) dt is the Cephes ellipe at parameter m = k^2, whose
    # value at m = 1 is exactly 1 (the integrand is cos t).
    expected = ellipe(MODULI**2)
    api = np.array(c_check(C_SOURCE), dtype=float)
    results = {
        "E_cei": api[:, 0],
        "KE_cei.E": api[:, 1],
        "sddsprocess cei2": process({"k": MODULI}, "-define=column,E,k cei2")["E"],
    }
    # Wrong-shaped or nonfinite output is an invalid run: NaN interior values would pass an
    # "error > ATOL" test, and infinite endpoint values would satisfy the defect threshold.
    for name, observed in results.items():
        if observed.shape != MODULI.shape or not np.all(np.isfinite(observed)):
            sys.exit(
                f"{name}: the output is not {MODULI.size} finite values: the comparison is invalid"
            )
    if not np.all(np.isfinite(expected)):
        sys.exit("the Cephes expected values are not finite")
    for i, k in enumerate(MODULI):
        values = " ".join(f"{name}={observed[i]:.17g}" for name, observed in results.items())
        print(f"k={k:.17g}: observed: {values}; expected={expected[i]:.17g}")
    endpoint = np.abs(MODULI) == 1
    defect = True
    for name, observed in results.items():
        if not np.allclose(observed[~endpoint], expected[~endpoint], rtol=0, atol=ATOL):
            sys.exit(f"{name}: an interior control failed: the comparison is invalid")
        defect &= bool(np.all(np.abs(observed[endpoint] - 1) > 0.2))
    print(f"comparison: interior atol={ATOL:g}; endpoints exact value 1")
    verdict(defect)

    ```

## Provenance

| Role | Model | Effort | Date | Code examined |
|---|---|---|---|---|
| Found | `gpt-6-astra` | xhigh | 2026-10-28 | SDDS master (`a60b9bbed7cc`), source with Elegant 75a2e1a1a58b, CPU, serial utility and native library API |
| Confirmed | `gpt-6.1-sol` | xhigh | 2026-10-28 | SDDS master (`a60b9bbed7cc`), source with Elegant 75a2e1a1a58b, CPU, serial utility and native library API |

