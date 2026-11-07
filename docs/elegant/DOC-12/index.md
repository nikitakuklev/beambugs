---
title: "DOC-12: SHRFDF manual reverses the squared radial coefficient sign"
hide:
  - toc
---

# DOC-12 — SHRFDF manual reverses the squared radial coefficient sign

| | |
|---|---|
| Code | [Elegant](../index.md) · Tracking |
| Priority | P4 |
| Severity | Documentation or example mismatch |
| Evidence | Reproduced, independently rechecked |
| Last reproduced | 2026-11-06 on Elegant master (`0355dd6`, committed 2026-11-03) |
| Tags | <span class="bb-tag">partly-source-only</span> [?](../../methodology.md#tags) |
| Source | [doc/shrfdf.tex:1–31](https://github.com/rtsoliday/elegant/blob/0355dd6aca12c861971fbf8f3ae78a0ded991734/doc/shrfdf.tex#L1-L31)<br>[src/shrfdf.c:65–90](https://github.com/rtsoliday/elegant/blob/0355dd6aca12c861971fbf8f3ae78a0ded991734/src/shrfdf.c#L65-L90) |

SHRFDF's manual combines a positive cubic radial expansion with the equation αₙ² = k₀² − kₙ², giving the opposite radial curvature from the vacuum field equation. The consistent relation for that expansion is αₙ² = kₙ² − k₀². For a 30.00 mm period, phase advance π/2 and first higher harmonic, the normalized transverse kick at x = 1.200 mm is 1.036 instead of the printed model's 0.9645. Independent modified-Bessel analysis and native transverse kicks establish the sign. The fundamental is unaffected, and the longitudinal-kick sign is a separate defect.

Run `python DOC-12.py` with [_harmonic_rf.py](../helpers/_harmonic_rf.py), [_slicing.py](../helpers/_slicing.py) and [_elegant.py](../helpers/_elegant.py) in the same directory.

??? example "Reproducer: DOC-12.py"

    [Download DOC-12.py](DOC-12.py)

    ```python
    """DOC-12: SHRFDF manual reverses the squared radial coefficient sign

    SHRFDF's manual combines a positive cubic radial expansion with the equation αₙ² = k₀² − kₙ², giving the opposite radial curvature from the vacuum field equation. The consistent relation for that expansion is αₙ² = kₙ² − k₀². For a 30.00 mm period, phase advance π/2 and first higher harmonic, the normalized transverse kick at x = 1.200 mm is 1.036 instead of the printed model's 0.9645. Independent modified-Bessel analysis and native transverse kicks establish the sign. The fundamental is unaffected, and the longitudinal-kick sign is a separate defect.

    Setup: `ELEGANT_SOURCE` set to a clone of rtsoliday/elegant; the script reads the manual's LaTeX source from it.

    Read the radial equation from the checkout's `doc/shrfdf.tex` and derive its curvature independently from the vacuum Helmholtz equation. Compare normalized native transverse kicks in modes 0, 1, 3 and 9 with the retained cubic polynomial and the full modified-Bessel derivative.
    """

    import os
    from pathlib import Path

    from _harmonic_rf import kick
    from _slicing import close

    # isort: split
    import _elegant as native
    import numpy as np
    from scipy.special import ivp


    def main():
        source = os.environ.get("ELEGANT_SOURCE")
        if not source:
            raise RuntimeError("Set ELEGANT_SOURCE to the elegant source checkout")
        text = (Path(source) / "doc/shrfdf.tex").read_text()
        wrong = r"\alpha_n^2 + k_n^2 = k_0^2" in text
        if not wrong and r"\alpha_n^2 = k_n^2 - k_0^2" not in text:
            raise RuntimeError("Unrecognized SHRFDF radial equation")
        print(
            f"manual alpha_n² relation: {'k0²-kn²' if wrong else 'kn²-k0²'}, expected=kn²-k0² (vacuum Helmholtz separation)"
        )
        for mode in (0, 1, 3, 9):
            radius = 0.0012 / max(mode, 1)
            rays = np.zeros((5, 6))
            rays[:, 0] = [0, radius, -radius, 0, 0]
            rays[:, 2] = [0, 0, 0, radius, -radius]
            px = kick(mode, np.pi / 2, rays)
            ratios = px / px[0]
            k0 = (np.pi / 2) / 0.03
            kn = (np.pi / 2 + 2 * np.pi * mode) / 0.03
            alpha2 = kn * kn - k0 * k0
            # Laplacian[x(x²+y²)]=8x gives F=x[1+alpha²(x²+y²)/8]; dF/dx fixes the profile.
            expected = 1 + alpha2 * (3 * rays[:, 0] ** 2 + rays[:, 2] ** 2) / 8
            printed = 2 - expected if wrong else expected
            close(ratios, expected, 2e-15, "retained cubic profile")
            bessel = float(2 * ivp(1, np.sqrt(alpha2) * radius)) if mode else 1.0
            print(
                f"mode={mode} r={radius:.17g} m alpha²={alpha2:.17g} m^-2: native={ratios.tolist()}, expected cubic={expected.tolist()}, printed={printed.tolist()} (normalized vacuum profile); full 2I1prime={bessel:.17g}"
            )
        native.verdict(wrong)


    if __name__ == "__main__":
        main()

    ```

## Provenance

| Role | Model | Effort | Date | Code examined |
|---|---|---|---|---|
| Found | `gpt-6-astra` | xhigh | 2026-11-06 | master (`0355dd6aca12`), source with SDDS a3f730a14c49, CPU, double, serial, one thread |
| Confirmed | `gpt-6.1-sol` | xhigh | 2026-11-06 | master (`0355dd6aca12`), source with SDDS a3f730a14c49, CPU, double, serial, one thread |

