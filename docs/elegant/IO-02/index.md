---
title: "IO-02: SDDS Gill–Miller integral of a three-sample table is zero and leaves a result unset"
hide:
  - toc
---

# IO-02 — SDDS Gill–Miller integral of a three-sample table is zero and leaves a result unset

| | |
|---|---|
| Code | [Elegant](../index.md) · Lattices, particles and I/O |
| Priority | P4 |
| Failure | Silent wrong result |
| Evidence | Reproduced, independently rechecked |
| Last reproduced | 2026-11-04 on SDDS commit [`a3f730a`](https://github.com/rtsoliday/SDDS/commit/a3f730a14c49), committed 2026-10-31 |
| Source | [contract, GillMillerIntegration.c:34–40](https://github.com/rtsoliday/SDDS/blob/a3f730a14c49ddd4f4d6347b42d60d34aa7cadc6/mdbmth/GillMillerIntegration.c#L34-L40)<br>[gmintegral, SDDSaps/doc/sddsprocess.tex:320–322](https://github.com/rtsoliday/SDDS/blob/a3f730a14c49ddd4f4d6347b42d60d34aa7cadc6/SDDSaps/doc/sddsprocess.tex#L320-L322)<br>[setup, GillMillerIntegration.c:54–60](https://github.com/rtsoliday/SDDS/blob/a3f730a14c49ddd4f4d6347b42d60d34aa7cadc6/mdbmth/GillMillerIntegration.c#L54-L60)<br>[loop, GillMillerIntegration.c:64](https://github.com/rtsoliday/SDDS/blob/a3f730a14c49ddd4f4d6347b42d60d34aa7cadc6/mdbmth/GillMillerIntegration.c#L64)<br>[final slot, GillMillerIntegration.c:145–149](https://github.com/rtsoliday/SDDS/blob/a3f730a14c49ddd4f4d6347b42d60d34aa7cadc6/mdbmth/GillMillerIntegration.c#L145-L149)<br>[wrapper, SDDSaps/SDDSaps.c:4165–4171](https://github.com/rtsoliday/SDDS/blob/a3f730a14c49ddd4f4d6347b42d60d34aa7cadc6/SDDSaps/SDDSaps.c#L4165-L4171)<br>[caller, SDDSaps.c:1849–1851](https://github.com/rtsoliday/SDDS/blob/a3f730a14c49ddd4f4d6347b42d60d34aa7cadc6/SDDSaps/SDDSaps.c#L1849-L1851) |

When `sddsprocess` computes a `gmintegral` from three samples, or a program calls the SDDS routine `GillMillerIntegration` with three samples, the final integral silently becomes zero. For f = 1 at x = 0, 1, 2 the correct integral is 2; for f = x at x = 0.2, 0.7, 2.3 it is 2.625, yet the utility returns 0 for both without warning. Direct library calls also leave the middle cumulative result unwritten, while four-sample constant and nonuniform cubic controls agree with exact antiderivatives to double precision roundoff.

Run `python IO-02.py` with [_elegant.py](../helpers/_elegant.py) and [_sdds.py](../helpers/_sdds.py) in the same directory.

??? example "Reproducer: IO-02.py"

    [Download IO-02.py](IO-02.py)

    ```python
    """IO-02: SDDS Gill–Miller integral of a three-sample table is zero and leaves a result unset

    When `sddsprocess` computes a `gmintegral` from three samples, or a program calls the SDDS routine `GillMillerIntegration` with three samples, the final integral silently becomes zero. For f = 1 at x = 0, 1, 2 the correct integral is 2; for f = x at x = 0.2, 0.7, 2.3 it is 2.625, yet the utility returns 0 for both without warning. Direct library calls also leave the middle cumulative result unwritten, while four-sample constant and nonuniform cubic controls agree with exact antiderivatives to double precision roundoff.

    Setup: `gcc`, and `sddsprocess` on `PATH` from an SDDS build tree, because the script compiles small C programs against that tree's `lib/Linux-x86_64/libmdbmth.a` and `libmdblib.a`.

    The script integrates constant and linear three-sample tables with `sddsprocess` and calls the library on uniform constant three- and four-sample tables prefilled with a sentinel. It compares with the exact antiderivative; the controls are the trapezoid rule of `sddsprocess` and four-sample constant and cubic tables.
    """

    import sys

    import numpy as np
    from _elegant import verdict
    from _sdds import c_check, process

    SENTINEL = -12345.0
    RTOL = 1e-14  # four-sample controls agree with the antiderivative to double roundoff
    C_SOURCE = r"""
    #include <stdio.h>
    extern int GillMillerIntegration(double *, double *, double *, double *, long);
    int main(void) {
      double x[] = {0, 1, 2, 3}, f[] = {1, 1, 1, 1};
      for (long n = 3; n <= 4; n++) {
        double integral[4] = {-12345, -12345, -12345, -12345};
        double error[4] = {-12345, -12345, -12345, -12345};
        int status = GillMillerIntegration(integral, error, f, x, n);
        printf("%ld %d", n, status);
        for (long j = 0; j < n; j++) printf(" %.17g", integral[j]);
        for (long j = 0; j < n; j++) printf(" %.17g", error[j]);
        printf("\n");
      }
      return 0;
    }
    """


    def fmt(values):
        return "[" + ", ".join(f"{v:.17g}" for v in np.ravel(values)) + "]"


    def compare(label, observed, expected):
        print(f"{label}: observed={fmt(observed)} expected={fmt(expected)}")


    defect = False

    # Library routine: f = 1 on x = 0, 1, 2, (3). The cumulative integral from x[0] is x - x[0].
    for line in c_check(C_SOURCE):
        n, status = int(line[0]), int(line[1])
        integral = np.array(line[2 : 2 + n], dtype=float)
        error = np.array(line[2 + n :], dtype=float)
        expected = np.arange(n, dtype=float)
        compare(f"API n={n} status={status} integral", integral, expected)
        print(f"API n={n} error estimates: {fmt(error)} (unwritten entries keep {SENTINEL:g})")
        if status != 0:
            sys.exit("the routine reported failure: the comparison is invalid")
        if n == 4:
            if not np.allclose(integral, expected, rtol=RTOL, atol=1e-15):
                sys.exit("the four-sample control failed: the comparison is invalid")
        else:
            # Success status, wrong final integral, and the middle entries never written.
            defect |= not np.isclose(integral[-1], expected[-1], rtol=RTOL, atol=0)
            defect |= bool(integral[1] == SENTINEL and error[1] == SENTINEL)

    # Utility: exact integral of x**power from the first to the last sample.
    for label, x, power in (
        ("constant-three", [0.0, 1.0, 2.0], 0),
        ("linear-three-nonuniform", [0.2, 0.7, 2.3], 1),
        ("constant-four", [0.0, 1.0, 2.0, 3.0], 0),
        ("cubic-four-nonuniform", [0.2, 0.7, 1.4, 2.3], 3),
    ):
        x = np.array(x)
        out = process(
            {"x": x, "f": x**power},
            "-process=f,gmintegral,GM,functionOf=x",
            "-process=f,integral,Trap,functionOf=x",
        )
        expected = (x[-1] ** (power + 1) - x[0] ** (power + 1)) / (power + 1)
        compare(f"sddsprocess {label} GM", out["GM"], expected)
        compare(f"sddsprocess {label} trapezoid", out["Trap"], expected)
        if len(x) == 3:
            defect |= not np.isclose(out["GM"][0], expected, rtol=RTOL, atol=0)
        elif not np.isclose(out["GM"][0], expected, rtol=RTOL, atol=0):
            sys.exit("the four-sample control failed: the comparison is invalid")
        if power == 0 or len(x) == 3:
            # The trapezoid rule is exact for these tables, so it validates the expected value.
            if not np.isclose(out["Trap"][0], expected, rtol=RTOL, atol=0):
                sys.exit("the trapezoid control failed: the expected value is invalid")
    print(f"comparison: rtol={RTOL:g}")
    verdict(defect)

    ```

## Provenance

| Role | Model | Effort | Date | Code examined |
|---|---|---|---|---|
| Found | `gpt-6-astra` | xhigh | 2026-10-28 | SDDS master (`a60b9bbed7cc`), source with Elegant 75a2e1a1a58b, CPU, serial utility and native library API |
| Confirmed | `gpt-6.1-sol` | xhigh | 2026-10-28 | SDDS master (`a60b9bbed7cc`), source with Elegant 75a2e1a1a58b, CPU, serial utility and native library API |

