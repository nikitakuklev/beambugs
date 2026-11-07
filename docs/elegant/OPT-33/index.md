---
title: "OPT-33: Analytic bend matrices replace weak focusing and violate subdivision invariance"
hide:
  - toc
---

# OPT-33 — Analytic bend matrices replace weak focusing and violate subdivision invariance

| | |
|---|---|
| Code | [Elegant](../index.md) · Optics |
| Priority | P3 |
| Severity | Wrong physics |
| Evidence | Reproduced, independently rechecked |
| Last reproduced | 2026-11-06 on Elegant master (`0355dd6`, committed 2026-11-03) |
| Tags | <span class="bb-tag">partly-source-only</span> [?](../../methodology.md#tags) |
| Source | [src/bend_matrix.c:626–669](https://github.com/rtsoliday/elegant/blob/0355dd6aca12c861971fbf8f3ae78a0ded991734/src/bend_matrix.c#L626-L669)<br>[src/bend_matrix.c:687–705](https://github.com/rtsoliday/elegant/blob/0355dd6aca12c861971fbf8f3ae78a0ded991734/src/bend_matrix.c#L687-L705)<br>[src/compute_matrices.c:1450–1471](https://github.com/rtsoliday/elegant/blob/0355dd6aca12c861971fbf8f3ae78a0ded991734/src/compute_matrices.c#L1450-L1471)<br>[src/compute_matrices.c:1801–1823](https://github.com/rtsoliday/elegant/blob/0355dd6aca12c861971fbf8f3ae78a0ded991734/src/compute_matrices.c#L1801-L1823)<br>[src/compute_matrices.c:1878–1893](https://github.com/rtsoliday/elegant/blob/0355dd6aca12c861971fbf8f3ae78a0ded991734/src/compute_matrices.c#L1878-L1893)<br>[src/compute_matrices.c:1768–1784](https://github.com/rtsoliday/elegant/blob/0355dd6aca12c861971fbf8f3ae78a0ded991734/src/compute_matrices.c#L1768-L1784)<br>[src/bend_matrix.c:51–65](https://github.com/rtsoliday/elegant/blob/0355dd6aca12c861971fbf8f3ae78a0ded991734/src/bend_matrix.c#L51-L65)<br>[doc/sben.tex:23–37](https://github.com/rtsoliday/elegant/blob/0355dd6aca12c861971fbf8f3ae78a0ded991734/doc/sben.tex#L23-L37)<br>[src/track_data.c:2016–2016](https://github.com/rtsoliday/elegant/blob/0355dd6aca12c861971fbf8f3ae78a0ded991734/src/track_data.c#L2016-L2016) |

Analytic bend matrices replace weak physical focusing coefficients with a positive numerical floor, suppressing normal quadrupole errors and making the result depend on subdivision. For a 10.00 m, 1.000 mrad bend, K₁ = 1.000×10⁻⁶ m⁻² represents a one-unit field error at 10 mm. The vertical slope response R₄₃ is −2.154×10⁻¹² m⁻¹ instead of +1.000×10⁻⁵ m⁻¹, and an illustrative periodic cell's phase differs by 1.765×10⁻⁴ turns. Dividing the zero-gradient bend into centimetre pieces amplifies horizontal focusing 21.54 times. Independent oscillator series confirm the errors in SBEN and analytic CSBEND/CSRCSBEND matrices; numerical bend tracking and tracking-based CSBEND matrices preserve the physical gradient.

Run `python OPT-33.py` with [_slicing.py](../helpers/_slicing.py) and [_elegant.py](../helpers/_elegant.py) in the same directory.

??? example "Reproducer: OPT-33.py"

    [Download OPT-33.py](OPT-33.py)

    ```python
    """OPT-33: Analytic bend matrices replace weak focusing and violate subdivision invariance

    Analytic bend matrices replace weak physical focusing coefficients with a positive numerical floor, suppressing normal quadrupole errors and making the result depend on subdivision. For a 10.00 m, 1.000 mrad bend, K₁ = 1.000×10⁻⁶ m⁻² represents a one-unit field error at 10 mm. The vertical slope response R₄₃ is −2.154×10⁻¹² m⁻¹ instead of +1.000×10⁻⁵ m⁻¹, and an illustrative periodic cell's phase differs by 1.765×10⁻⁴ turns. Dividing the zero-gradient bend into centimetre pieces amplifies horizontal focusing 21.54 times. Independent oscillator series confirm the errors in SBEN and analytic CSBEND/CSRCSBEND matrices; numerical bend tracking and tracking-based CSBEND matrices preserve the physical gradient.

    The script compares native analytic bend matrices with 70-digit oscillator power series at weak and control gradients. It checks numerical particle tracking, tracking-based CSBEND matrices, periodic-cell phase and centimetre subdivision of a weak bend.
    """

    from _slicing import close, discrepancy, native, track

    # isort: split
    # _slicing sets numerical thread limits before importing the native runner and NumPy.
    import mpmath as mp
    import numpy as np

    mp.mp.dps = 70


    def oscillator(length, coefficient):
        # Entire oscillator series: C=sum((-a)^j L^(2j)/(2j)!),
        # S=sum((-a)^j L^(2j+1)/(2j+1)!), M=[[C,S],[-a*S,C]].
        ell, a = mp.mpf(str(length)), mp.mpf(str(coefficient))
        c = mp.fsum((-a) ** j * ell ** (2 * j) / mp.factorial(2 * j) for j in range(60))
        s = mp.fsum((-a) ** j * ell ** (2 * j + 1) / mp.factorial(2 * j + 1) for j in range(60))
        return np.array([[float(c), float(s)], [float(-a * s), float(c)]])


    def body(kind, length, angle, k1, *, pieces=1, tracking=0):
        options = ""
        if kind == "CSBEND":
            options = f",N_SLICES=256,INTEGRATION_ORDER=6,TRACKING_MATRIX={tracking}"
        elif kind == "CSRCSBEND":
            options = ",N_SLICES=256,CSR=0,BINS=16"
        return (
            f"B: {kind},L={length / pieces:.17g},ANGLE={angle / pieces:.17g},K1={k1:.17g},"
            f"EDGE1_EFFECTS=0,EDGE2_EFFECTS=0{options}\nBL: LINE=({pieces}*B)\n"
        )


    def simulate(lattice):
        rays = np.zeros((2, 6))
        rays[:, 2] = [1e-6, -1e-6]
        return track("bend matrix", lattice, rays, p0=1000, matrix=True)


    def main():
        faults = []
        epsilon = (1e-16) ** (1 / 3)
        for kind, tracking in (("SBEN", 0), ("CSBEND", 0), ("CSBEND", 1), ("CSRCSBEND", 0)):
            for k1 in (-1e-6, 0.0, 1e-6):
                result = simulate(body(kind, 10, 0.001, k1, tracking=tracking))
                expected = oscillator(10, -k1)
                observed = result["R"][2:4, 2:4]
                derivative = (result["q"][0, 3] - result["q"][1, 3]) / 2e-6
                if tracking:
                    close(observed, expected, 1e-12, "Tracking-based matrix")
                else:
                    faulty = oscillator(10, (epsilon / 10) ** 2)
                    faults.append(discrepancy(observed, expected, faulty, 1e-12))
                if kind != "SBEN":
                    close(derivative, expected[1, 0], 2e-13, "Numerical tracker gradient")
                else:
                    close(derivative, observed[1, 0], 1e-17, "SBEN matrix tracking")
                print(
                    f"{kind} TRACKING_MATRIX={tracking} K1={k1:g}: R33={observed[0, 0]:.16g} "
                    f"expected={expected[0, 0]:.16g}; R34={observed[0, 1]:.16g} "
                    f"expected={expected[0, 1]:.16g} m; R43={observed[1, 0]:.16g} "
                    f"expected={expected[1, 0]:.16g} m^-1 (70-digit series); particle R43={derivative:.16g}"
                )
        for kind in ("SBEN", "CSBEND", "CSRCSBEND"):
            for k1 in (-5e-6, -4e-6, 4e-6, 5e-6):
                result = simulate(body(kind, 1, 0.1, k1))
                expected = oscillator(1, -k1)
                observed = result["R"][2:4, 2:4]
                if abs(k1) >= 5e-6:
                    close(observed, expected, 3e-14, "Above-threshold control")
                else:
                    faults.append(discrepancy(observed, expected, oscillator(1, epsilon**2), 3e-14))
                print(
                    f"threshold {kind} K1={k1:g}: R43={observed[1, 0]:.16g} "
                    f"expected={expected[1, 0]:.16g} m^-1 (70-digit series)"
                )
        for pieces in (1, 10, 100, 1000):
            result = simulate(body("SBEN", 10, 0.001, 0, pieces=pieces))
            expected_x = oscillator(10, 1e-8)
            observed = result["R"]
            ax = (epsilon * pieces / 10) ** 2 if 0.001 / pieces < epsilon else 1e-8
            faults.append(discrepancy(observed[:2, :2], expected_x, oscillator(10, ax), 3e-12))
            faults.append(
                discrepancy(
                    observed[2:4, 2:4],
                    oscillator(10, 0),
                    oscillator(10, (epsilon * pieces / 10) ** 2),
                    3e-12,
                )
            )
            print(
                f"zero-gradient pieces={pieces} piece_L={10 / pieces:g} m: R21={observed[1, 0]:.16g} "
                f"expected={expected_x[1, 0]:.16g} m^-1; ratio={observed[1, 0] / expected_x[1, 0]:.16g}; "
                f"R43={observed[3, 2]:.16g} expected=0 m^-1 (same exact body flow)"
            )
        for pieces in (1, 10, 100):
            observed = simulate(body("SBEN", 1, 0.1, 1, pieces=pieces))["R"][2:4, 2:4]
            expected = oscillator(1, -1)
            error = close(observed, expected, 3e-14, "Strong-gradient subdivision")
            print(
                f"strong-gradient pieces={pieces}: vertical block error={error:.3g} expected=0 (series)"
            )
        for k1 in (0.0, 1e-6):
            lattice = (
                f"B: SBEN,L=10,ANGLE=.001,K1={k1},EDGE1_EFFECTS=0,EDGE2_EFFECTS=0\n"
                "QF: QUAD,L=1,K1=.01\nQD: QUAD,L=1,K1=-.01\nD: DRIF,L=20\n"
                "BL: LINE=(B,D,QF,D,QD,D)\n"
            )
            observed = simulate(lattice)["R"][2:4, 2:4]
            expected = (
                oscillator(20, 0)
                @ oscillator(1, 0.01)
                @ oscillator(20, 0)
                @ oscillator(1, -0.01)
                @ oscillator(20, 0)
                @ oscillator(10, -k1)
            )
            fixed = (
                oscillator(20, 0)
                @ oscillator(1, 0.01)
                @ oscillator(20, 0)
                @ oscillator(1, -0.01)
                @ oscillator(20, 0)
            )
            faulty = fixed @ oscillator(10, (epsilon / 10) ** 2)
            faults.append(discrepancy(observed, expected, faulty, 2e-12))
            phase = float(np.arccos(np.trace(observed) / 2) / (2 * np.pi))
            exact_phase = float(np.arccos(np.trace(expected) / 2) / (2 * np.pi))
            if not k1:
                close(phase, exact_phase, 4e-11, "Zero-gradient cell phase")
            print(
                f"periodic cell K1={k1:g}: phase={phase:.16g} expected={exact_phase:.16g} turns/cell "
                f"(product of series flows); error={phase - exact_phase:.16g} turns/cell"
            )
        native.verdict(any(faults))


    if __name__ == "__main__":
        main()

    ```

## Provenance

| Role | Model | Effort | Date | Code examined |
|---|---|---|---|---|
| Found | `gpt-6-astra` | xhigh | 2026-11-06 | master (`0355dd6aca12`), source with SDDS a3f730a14c49, CPU, double, serial, one thread |
| Confirmed | `gpt-6.1-sol` | xhigh | 2026-11-06 | master (`0355dd6aca12`), source with SDDS a3f730a14c49, CPU, double, serial, one thread |

