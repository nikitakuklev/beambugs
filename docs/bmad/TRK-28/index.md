---
title: "TRK-28: Magnus solenoid spin omits the tracked/reference charge ratio"
hide:
  - toc
---

# TRK-28 — Magnus solenoid spin omits the tracked/reference charge ratio

| | |
|---|---|
| Code | [Bmad](../index.md) · Tracking, RF, fields and spin |
| Priority | P3 |
| Failure | Wrong physics |
| Evidence | Reproduced, independently rechecked |
| Last reproduced | 2026-11-05 on Bmad release [20261031-0](https://github.com/bmad-sim/bmad-ecosystem/tree/20261031-0) (`c4c629f`) |
| Source | [magnus_omega solenoid generator, bmad/spin/track1_spin_integration.f90:252–327](https://github.com/bmad-sim/bmad-ecosystem/blob/c4c629fc9071d3a56cb1ba5bc714baba65e9c5d3/bmad/spin/track1_spin_integration.f90#L252-L327)<br>[valid_spin_tracking_method, bmad/spin/valid_spin_tracking_method.f90:81–84](https://github.com/bmad-sim/bmad-ecosystem/blob/c4c629fc9071d3a56cb1ba5bc714baba65e9c5d3/bmad/spin/valid_spin_tracking_method.f90#L81-L84)<br>[Bmad Manual, Equations of Motion, `orpt`, bmad/doc/spin.tex:25–35](https://github.com/bmad-sim/bmad-ecosystem/blob/c4c629fc9071d3a56cb1ba5bc714baba65e9c5d3/bmad/doc/spin.tex#L25-L35)<br>[Bmad Manual, Solenoid, bmad/doc/elements.tex:5270–5279](https://github.com/bmad-sim/bmad-ecosystem/blob/c4c629fc9071d3a56cb1ba5bc714baba65e9c5d3/bmad/doc/elements.tex#L5270-L5279) |

Forward Magnus spin tracking through a solenoid gives equal-momentum electrons and positrons the same rotation in the same physical field. The normalized solenoid strength needs the tracked-to-reference charge ratio in the Thomas–BMT generator. For a positron at 3 MeV/c in an electron-reference 1.3 m solenoid of strength −1.1 m⁻¹, spin (0.36, 0.48, 0.8) becomes (−0.4254, 0.4231, 0.8) instead of (0.5253, −0.2899, 0.8). Matching-charge cases and the two ordinary integration methods agree with the charge-aware expectation.

Run `python TRK-28.py` with [_bmad.py](../helpers/_bmad.py) and [_rf_and_spin.py](../helpers/_rf_and_spin.py) in the same directory.

??? example "Reproducer: TRK-28.py"

    [Download TRK-28.py](TRK-28.py)

    ```python
    """TRK-28: Magnus solenoid spin omits the tracked/reference charge ratio

    Forward Magnus spin tracking through a solenoid gives equal-momentum electrons and positrons the same rotation in the same physical field. The normalized solenoid strength needs the tracked-to-reference charge ratio in the Thomas–BMT generator. For a positron at 3 MeV/c in an electron-reference 1.3 m solenoid of strength −1.1 m⁻¹, spin (0.36, 0.48, 0.8) becomes (−0.4254, 0.4231, 0.8) instead of (0.5253, −0.2899, 0.8). Matching-charge cases and the two ordinary integration methods agree with the charge-aware expectation.

    The script sweeps equal-mass electron and positron spins in both reference-charge lattices using BMAD_DIST. It compares forward Magnus transport with the charge-aware axial Thomas–BMT rotation and validates ordinary integration, Runge–Kutta, matching-charge, zero-field and longitudinal-spin controls.
    """

    import numpy as np
    from _bmad import compare, verdict
    from _rf_and_spin import control, magnus_records, rotate

    records = magnus_records()
    same_error = opposite_error = 0.0
    for ident, same_charge, theta, seed, forward, returned, backward in records:
        if ident[-1] != 1:
            continue
        error = float(np.max(np.abs(forward - rotate(seed, theta))))
        if same_charge:
            same_error = max(same_error, error)
        else:
            opposite_error = max(opposite_error, error)
        if ident == (1, 2, 2, 2, 1, 1):
            print(
                f"3 MeV/c positron: forward={forward.tolist()} expected={rotate(seed, theta).tolist()}"
            )
    control("same-charge forward Magnus T-BMT maximum residual", same_error, 0, atol=3e-15)
    verdict(
        compare("opposite-charge forward Magnus T-BMT maximum residual", opposite_error, 0, atol=3e-15)
    )

    ```

## Provenance

| Role | Model | Effort | Date | Code examined |
|---|---|---|---|---|
| Found | <span class="bb-nowrap">`gpt-6-astra`</span> | xhigh | <span class="bb-nowrap">2026-10-29</span> | 20261020-0 (`19b8afb46a40`), source-matched 20261021-0 (d00a8f17e55a) libraries, production, CPU, double, serial |
| Confirmed<br><small class="bb-scope">equal-mass electron/positron relative-charge spin scaling</small> | <span class="bb-nowrap">`gpt-6.1-sol`</span> | xhigh | <span class="bb-nowrap">2026-10-29</span> | 20261020-0 (`19b8afb46a40`), source-matched 20261021-0 (d00a8f17e55a) libraries, production, CPU, double, serial |

