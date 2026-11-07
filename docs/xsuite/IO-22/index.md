---
title: "IO-22: Physics manual omits the squared reference speed in the exact speed equation"
hide:
  - toc
---

# IO-22 — Physics manual omits the squared reference speed in the exact speed equation

| | |
|---|---|
| Code | [Xsuite](../index.md) · Lattices, particles and I/O |
| Priority | P4 |
| Severity | Documentation or example mismatch |
| Evidence | Reproduced, independently rechecked |
| Last reproduced | 2026-11-07 on xsuite commit [`fb7df34`](https://github.com/xsuite/xsuite/commit/fb7df34faa78), committed 2026-11-02 |
| Source | [particle-coordinate definitions and exact conversions, xtrack.tex:334–358](https://github.com/xsuite/xsuite/blob/e0913e0a18028b2fce06991aebed6150876124b3/docs/physics_manual/xtrack.tex#L334-L358)<br>[small-offset expansion, lines 383–390](https://github.com/xsuite/xsuite/blob/e0913e0a18028b2fce06991aebed6150876124b3/docs/physics_manual/xtrack.tex#L383-L390)<br>[physics_man.tex:143](https://github.com/xsuite/xsuite/blob/e0913e0a18028b2fce06991aebed6150876124b3/docs/physics_manual/physics_man.tex#L143)<br>[particles.py:1740–1777](https://github.com/xsuite/xtrack/blob/ad5e808cdc5b9a881eac2f41e32a0b7d11cc7028/xtrack/particles/particles.py#L1740-L1777) |

The physics manual’s exact conversion from normalized energy deviation pτ to particle speed β uses 1−β₀ where 1−β₀² is required, with β₀ the reference speed in units of c. At pτ = 0 and β₀ = 0.6000, the printed equation gives β = 0.7746 instead of 0.6000, an excess of 29.10%. The relativistic mass shell requires the squared reference speed and agrees with native particle state. The error is confined to the documented equation; native energy-to-speed conversion is correct.

Run `python IO-22.py` with [_prebuilt.py](../helpers/_prebuilt.py) and [_common.py](../helpers/_common.py) in the same directory.

??? example "Reproducer: IO-22.py"

    [Download IO-22.py](IO-22.py)

    ```python
    """IO-22: Physics manual omits the squared reference speed in the exact speed equation

    The physics manual’s exact conversion from normalized energy deviation pτ to particle speed β uses 1−β₀ where 1−β₀² is required, with β₀ the reference speed in units of c. At pτ = 0 and β₀ = 0.6000, the printed equation gives β = 0.7746 instead of 0.6000, an excess of 29.10%. The relativistic mass shell requires the squared reference speed and agrees with native particle state. The error is confined to the documented equation; native energy-to-speed conversion is correct.

    Setup: `XSUITE_SOURCE` set to a clone of xsuite/xsuite at release 0.66.3 (`e0913e0a1802`); the
    script reads the physics manual's `docs/physics_manual/xtrack.tex` from it.

    Read the uniquely identified β(pτ) equality from the manual and evaluate its numerator as written
    for nine fixed proton states. Compare with a 70-digit relativistic mass-shell calculation and native
    beta0*rvv, checking the squared-numerator formula as a control. All states and offsets are fixed,
    and particle conversion runs in serial CPU Python.
    """

    import importlib.metadata
    import math
    import os
    import re
    from pathlib import Path

    import mpmath as mp
    from _prebuilt import CTX, identity, report, xt


    def main():
        source = Path(os.environ["XSUITE_SOURCE"]) / "docs/physics_manual/xtrack.tex"
        equations = [
            (number, line)
            for number, line in enumerate(source.read_text().splitlines(), 1)
            if r"&\beta = \frac{P c}{E}" in line
        ]
        assert len(equations) == 1, equations
        number, equation = equations[0]
        numerator = re.search(
            r"\\frac\{1 - \\beta_0(?P<square>\^2)?\}"
            r"\{\\left\(1 \+ \\beta_0 p_\\tau\\right\)\^2\}",
            equation,
        )
        assert numerator is not None, equation
        power = 2 if numerator.group("square") else 1
        identity()
        print("PACKAGE mpmath", importlib.metadata.version("mpmath"))
        print("AFFINITY", sorted(os.sched_getaffinity(0)))
        print(f"SOURCE_EQUATION {source}:{number} {equation.strip()}")
        print(f"NUMERATOR_POWER {power}; EXECUTION Python particle conversion, no tracking kernel")
        print("COMPONENTS observed=[documented_beta, corrected_beta, native_beta]")
        mp.mp.dps = 70
        tolerance = 1.5e-15
        cases = {}
        corrected_errors = []
        native_errors = []
        documented_mismatches = 0
        for beta0 in (0.6, 0.8, 0.99):
            for ptau in (0.0, -0.1, 0.1):
                particle = xt.Particles(
                    _context=CTX, beta0=beta0, mass0=xt.PROTON_MASS_EV, ptau=ptau
                )
                b = float(particle.beta0[0])
                t = float(particle.ptau[0])
                # p_tau = (E-E0)/(p0*c); use the independent mass shell
                # (P*c)^2 = E^2-(m*c^2)^2, then beta = P*c/E.
                energy = mp.mpf(float(particle.energy0[0])) + mp.mpf(t) * mp.mpf(
                    float(particle.p0c[0])
                )
                rest_energy = mp.mpf(float(particle.mass0))
                expected = float(mp.sqrt(energy**2 - rest_energy**2) / energy)
                documented = math.sqrt(1 - (1 - b**power) / (1 + b * t) ** 2)
                # gamma = gamma0*(1+beta0*p_tau) and gamma0^-2 = 1-beta0^2.
                corrected = math.sqrt(1 - (1 - b * b) / (1 + b * t) ** 2)
                native = float(particle.beta0[0] * particle.rvv[0])
                corrected_errors.append(abs(corrected - expected))
                native_errors.append(abs(native - expected))
                assert abs(corrected - expected) < tolerance
                assert abs(native - expected) < tolerance
                if t == 0:
                    assert abs(expected - b) < tolerance
                documented_mismatches += abs(documented - expected) > tolerance
                relative_error = 100 * (documented / expected - 1)
                label = (
                    f"beta0={b:g} ptau={t:+g} relative_error_percent={relative_error:.12g} "
                    "derivation=mass-shell Pc/E"
                )
                cases[label] = ([documented, corrected, native], [expected] * 3)
        print(
            f"CONTROLS corrected_max_abs_error={max(corrected_errors):.17g} "
            f"native_max_abs_error={max(native_errors):.17g}"
        )
        print(f"DOCUMENTED_MISMATCHES {documented_mismatches}/{len(cases)}")
        report(cases, rtol=0, atol=tolerance)


    if __name__ == "__main__":
        main()

    ```

## Provenance

| Role | Model | Effort | Date | Code examined |
|---|---|---|---|---|
| Found | `claude-opus-5-5` | high | 2026-11-06 | xsuite 0.66.3 (`e0913e0a1802`), LaTeX source inspection |
| Confirmed | `gpt-6.1-sol` | xhigh | 2026-11-06 | xsuite 0.66.3 (`e0913e0a1802`), LaTeX source; installed xtrack 0.118.2 (ad5e808cdc5b), Python particle conversion, CPU, double, serial; mpmath 1.3.0 at 70 digits |

