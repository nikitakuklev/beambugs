---
title: "IO-10: SAD export reverses quadrupole and sextupole roll"
hide:
  - toc
---

# IO-10 — SAD export reverses quadrupole and sextupole roll

| | |
|---|---|
| Code | [Elegant](../index.md) · Lattices, particles and I/O |
| Priority | P4 |
| Failure | Wrong physics |
| Evidence | Reproduced, independently rechecked |
| Last reproduced | 2026-11-06 on Elegant commit [`0355dd6`](https://github.com/rtsoliday/elegant/commit/0355dd6aca12), committed 2026-11-03 |
| Source | [src/convert_to_mad8.c:248–257](https://github.com/rtsoliday/elegant/blob/0355dd6aca12c861971fbf8f3ae78a0ded991734/src/convert_to_mad8.c#L248-L257)<br>[src/convert_to_mad8.c:288–297](https://github.com/rtsoliday/elegant/blob/0355dd6aca12c861971fbf8f3ae78a0ded991734/src/convert_to_mad8.c#L288-L297)<br>[src/elegantto.c:21–29](https://github.com/rtsoliday/elegant/blob/0355dd6aca12c861971fbf8f3ae78a0ded991734/src/elegantto.c#L21-L29)<br>[src/elegantto.c:284–294](https://github.com/rtsoliday/elegant/blob/0355dd6aca12c861971fbf8f3ae78a0ded991734/src/elegantto.c#L284-L294)<br>[doc/elegant.tex:9587–9623](https://github.com/rtsoliday/elegant/blob/0355dd6aca12c861971fbf8f3ae78a0ded991734/doc/elegant.tex#L9587-L9623)<br>[src/tilt_matrices.c:199–202](https://github.com/rtsoliday/elegant/blob/0355dd6aca12c861971fbf8f3ae78a0ded991734/src/tilt_matrices.c#L199-L202)<br>[src/tturn.f:355–359](https://github.com/KatsOide/SAD/blob/d082c7c5161ea111132e12c0cf9288fb892c23ba/src/tturn.f#L355-L359)<br>[src/tffs.f:1310–1314](https://github.com/KatsOide/SAD/blob/d082c7c5161ea111132e12c0cf9288fb892c23ba/src/tffs.f#L1310-L1314)<br>[src/tquad.f:38–40](https://github.com/KatsOide/SAD/blob/d082c7c5161ea111132e12c0cf9288fb892c23ba/src/tquad.f#L38-L40)<br>[src/tsol.f:150–159](https://github.com/KatsOide/SAD/blob/d082c7c5161ea111132e12c0cf9288fb892c23ba/src/tsol.f#L150-L159)<br>[src/tffs.f:1328–1332](https://github.com/KatsOide/SAD/blob/d082c7c5161ea111132e12c0cf9288fb892c23ba/src/tffs.f#L1328-L1332)<br>[src/inc/TENT.inc:16–29](https://github.com/KatsOide/SAD/blob/d082c7c5161ea111132e12c0cf9288fb892c23ba/src/inc/TENT.inc#L16-L29)<br>[src/tquad.f:170–188](https://github.com/KatsOide/SAD/blob/d082c7c5161ea111132e12c0cf9288fb892c23ba/src/tquad.f#L170-L188) |

Exporting a rolled QUAD, KQUAD, SEXT, or KSEXT with elegantto's SAD output writes TILT unchanged into ROTATE, reversing the physical roll because SAD defines the opposite angle sign. A 0.4200 m quadrupole with normalized gradient K₁ = 1.700 m⁻² and TILT = π/4 exports a horizontal-slope response to vertical position of +0.7145 m⁻¹ instead of −0.7145 m⁻¹. Native tracking and an independent Hamiltonian matrix agree; SAD's manual and source establish the required ROTATE = −TILT conversion. Sextupole skew terms reverse similarly, while zero-roll and MAD8 controls pass.

Run `python IO-10.py` with [_misalign.py](../helpers/_misalign.py), [_elegant.py](../helpers/_elegant.py) and [_diagnostics.py](../helpers/_diagnostics.py) in the same directory.

??? example "Reproducer: IO-10.py"

    [Download IO-10.py](IO-10.py)

    ```python
    """IO-10: SAD export reverses quadrupole and sextupole roll

    Exporting a rolled QUAD, KQUAD, SEXT, or KSEXT with elegantto's SAD output writes TILT unchanged into ROTATE, reversing the physical roll because SAD defines the opposite angle sign. A 0.4200 m quadrupole with normalized gradient K₁ = 1.700 m⁻² and TILT = π/4 exports a horizontal-slope response to vertical position of +0.7145 m⁻¹ instead of −0.7145 m⁻¹. Native tracking and an independent Hamiltonian matrix agree; SAD's manual and source establish the required ROTATE = −TILT conversion. Sextupole skew terms reverse similarly, while zero-roll and MAD8 controls pass.

    Setup: `elegantto` of the Elegant build on `PATH`; the script exports lattices with it.

    Compare native quadrupole Jacobians with a laboratory Hamiltonian matrix exponential and evaluate exported SAD maps with physical roll equal to minus ROTATE. Measure the native sextupole second derivative with symmetric rays and Richardson extrapolation, including both roll signs, both strength signs, symmetry angles and MAD8 controls.
    """

    import math

    from _misalign import close, coefficient, export, fault, lattice, quad_map, track

    # isort: split
    # The helper sets numerical thread limits before NumPy initializes.
    import numpy as np
    from _elegant import verdict


    def main():
        defect = False
        h = 1e-7
        rays = np.concatenate((np.eye(4) * h, -np.eye(4) * h))
        for kind in ("QUAD", "KQUAD"):
            for k in (1.7, -1.7):
                for tilt in (0.0, 0.23, -0.23, math.pi / 4, math.pi / 2):
                    source = lattice(kind, 0.42, k, tilt)
                    native = track(source, rays)
                    jacobian = ((native[:4] - native[4:]) / (2 * h)).T
                    expected = quad_map(k, 0.42, tilt)
                    close(jacobian, expected, 8e-14)
                    sad = export(source, "sad")
                    length = coefficient(sad, "L")
                    strength = coefficient(sad, "K1") / length
                    angle = coefficient(sad, "ROTATE")
                    close(strength, k, 2e-14)
                    # SAD entrance x_local=c*x-s*y, y_local=s*x+c*y:
                    # its physical roll is -ROTATE, opposite elegant's TILT.
                    observed = quad_map(strength, length, -angle)
                    defect |= fault(observed, expected, quad_map(k, 0.42, -tilt), 8e-14)
                    mad = export(source, "mad8")
                    control = quad_map(
                        coefficient(mad, "K1"), coefficient(mad, "L"), coefficient(mad, "TILT")
                    )
                    close(control, expected, 8e-14)
                    close(quad_map(strength, length, tilt), expected, 8e-14)
                    print(
                        f"{kind} K1={k:g} tilt={tilt:.16g}: native R23={jacobian[1, 2]:.16g}; "
                        f"SAD R23={observed[1, 2]:.16g} expected={expected[1, 2]:.16g} m^-1; "
                        f"native max error={np.max(np.abs(jacobian - expected)):.4g}; MAD8=pass, ROTATE=-TILT repair=pass "
                        "(M=exp(L J Hessian(H)), SAD physical roll=-ROTATE)"
                    )
        for kind in ("SEXT", "KSEXT"):
            cases = [(9.0, t) for t in (0.0, 0.23, -0.23, math.pi / 6, math.pi / 3)]
            cases.append((-9.0, 0.23))
            for k, tilt in cases:
                h = 1e-5
                rays = np.zeros((5, 4))
                rays[:, 0] = [0, h, -h, h / 2, -h / 2]
                source = lattice(kind, 0.28, k, tilt)
                native = track(source, rays)
                full = (native[1, 3] + native[2, 3] - 2 * native[0, 3]) / h**2
                half = (native[3, 3] + native[4, 3] - 2 * native[0, 3]) / (h / 2) ** 2
                derivative = float((4 * half - full) / 3)
                # At the axis, Hamilton's equations give d²y'_out/dx_in²=-K2*L*sin(3t).
                expected = -k * 0.28 * math.sin(3 * tilt)
                close(derivative, expected, 3e-12)
                sad = export(source, "sad")
                observed = coefficient(sad, "K2") * math.sin(3 * coefficient(sad, "ROTATE"))
                defect |= fault(observed, expected, -expected, 3e-14)
                close(-coefficient(sad, "K2") * math.sin(3 * tilt), expected, 3e-14)
                mad = export(source, "mad8")
                control = (
                    -coefficient(mad, "K2")
                    * coefficient(mad, "L")
                    * math.sin(3 * coefficient(mad, "TILT"))
                )
                close(control, expected, 3e-14)
                print(
                    f"{kind} K2={k:g} tilt={tilt:.16g}: native D={derivative:.16g}; "
                    f"SAD D={observed:.16g} expected={expected:.16g} m^-2; MAD8=pass, ROTATE=-TILT repair=pass "
                    "(D=d^2(yprime_out)/dx_in^2=-K2*L*sin(3*TILT))"
                )
        verdict(defect)


    if __name__ == "__main__":
        main()

    ```

## Provenance

| Role | Model | Effort | Date | Code examined |
|---|---|---|---|---|
| Found | <span class="bb-nowrap">`gpt-6-astra`</span> | xhigh | <span class="bb-nowrap">2026-11-06</span> | master (`0355dd6aca12`), source with SDDS a3f730a14c49, CPU, double, serial elegant and elegantto, one thread |
| Confirmed | <span class="bb-nowrap">`gpt-6.1-sol`</span> | xhigh | <span class="bb-nowrap">2026-11-06</span> | master (`0355dd6aca12`), source with SDDS a3f730a14c49, CPU, double, serial elegant and elegantto, one thread |

