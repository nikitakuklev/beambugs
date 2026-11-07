---
title: "TRK-27: Magnus solenoid spin fails backward-time inversion"
hide:
  - toc
---

# TRK-27 — Magnus solenoid spin fails backward-time inversion

| | |
|---|---|
| Code | [Bmad](../index.md) · Tracking, RF, fields and spin |
| Priority | P2 |
| Severity | Wrong physics |
| Evidence | Reproduced, independently rechecked |
| Last reproduced | 2026-11-05 on Bmad release [20261031-0](https://github.com/bmad-sim/bmad-ecosystem/tree/20261031-0) (`c4c629f`) |
| Source | [magnus_omega solenoid generator, bmad/spin/track1_spin_integration.f90:252–327](https://github.com/bmad-sim/bmad-ecosystem/blob/c4c629fc9071d3a56cb1ba5bc714baba65e9c5d3/bmad/spin/track1_spin_integration.f90#L252-L327)<br>[ordinary spin integration, bmad/spin/track1_spin_integration.f90:225–230](https://github.com/bmad-sim/bmad-ecosystem/blob/c4c629fc9071d3a56cb1ba5bc714baba65e9c5d3/bmad/spin/track1_spin_integration.f90#L225-L230)<br>[Bmad Manual, time direction, bmad/doc/tracking.tex:75–77](https://github.com/bmad-sim/bmad-ecosystem/blob/c4c629fc9071d3a56cb1ba5bc714baba65e9c5d3/bmad/doc/tracking.tex#L75-L77)<br>[Bmad Manual, Magnus Spin Tracking, `s:magnus`, bmad/doc/charged-tracking.tex:1281–1313](https://github.com/bmad-sim/bmad-ecosystem/blob/c4c629fc9071d3a56cb1ba5bc714baba65e9c5d3/bmad/doc/charged-tracking.tex#L1281-L1313) |

Backward-time Magnus spin tracking through an axial solenoid repeats the forward rotation instead of undoing it. The constant-field spin generator requires opposite rotation angles for the two time directions. At 3 MeV/c, a 1.3 m solenoid with strength −1.1 m⁻¹ maps spin (0.36, 0.48, 0.8) to (−0.4780, −0.3626, 0.8) after a round trip instead of restoring it. Ordinary spin integration and Runge–Kutta tracking restore the spin to roundoff.

Run `python TRK-27.py` with [_bmad.py](../helpers/_bmad.py) and [_rf_and_spin.py](../helpers/_rf_and_spin.py) in the same directory.

??? example "Reproducer: TRK-27.py"

    [Download TRK-27.py](TRK-27.py)

    ```python
    """TRK-27: Magnus solenoid spin fails backward-time inversion

    Backward-time Magnus spin tracking through an axial solenoid repeats the forward rotation instead of undoing it. The constant-field spin generator requires opposite rotation angles for the two time directions. At 3 MeV/c, a 1.3 m solenoid with strength −1.1 m⁻¹ maps spin (0.36, 0.48, 0.8) to (−0.4780, −0.3626, 0.8) after a round trip instead of restoring it. Ordinary spin integration and Runge–Kutta tracking restore the spin to roundoff.

    The script sweeps both reference charges, two momenta, three solenoids and three momentum offsets using BMAD_DIST. Matching-charge Magnus round trips and fresh backward passes are compared with Cartesian Thomas–BMT rotations, with ordinary integration, Runge–Kutta, zero-field and longitudinal-spin controls.
    """

    import numpy as np
    from _bmad import compare, verdict
    from _rf_and_spin import control, magnus_records, rotate

    records = magnus_records()
    forward_error = return_error = backward_error = 0.0
    for ident, same_charge, theta, seed, forward, returned, backward in records:
        config, ie, isp, idelta, ispin, method = ident
        if method != 1 or not same_charge:
            continue
        forward_error = max(forward_error, float(np.max(np.abs(forward - rotate(seed, theta)))))
        return_error = max(return_error, float(np.max(np.abs(returned - seed))))
        backward_error = max(backward_error, float(np.max(np.abs(backward - rotate(seed, -theta)))))
        if ident == (1, 2, 1, 2, 1, 1):
            print(
                f"3 MeV/c example: forward angle={theta:.16g} rad; round trip={returned.tolist()} expected={seed.tolist()}"
            )
    control("same-charge forward Magnus T-BMT maximum residual", forward_error, 0, atol=3e-15)
    a = compare("same-charge Magnus round-trip maximum residual", return_error, 0, atol=3e-15)
    b = compare(
        "same-charge Magnus fresh backward T-BMT maximum residual", backward_error, 0, atol=3e-15
    )
    verdict(a or b)

    ```

## Provenance

| Role | Model | Effort | Date | Code examined |
|---|---|---|---|---|
| Found | `gpt-6-astra` | xhigh | 2026-10-29 | 20261020-0 (`19b8afb46a40`), source-matched 20261021-0 (d00a8f17e55a) libraries, production, CPU, double, serial |
| Confirmed<br><small>constant-generator axial-solenoid backward-time inversion</small> | `gpt-6.1-sol` | xhigh | 2026-10-29 | 20261020-0 (`19b8afb46a40`), source-matched 20261021-0 (d00a8f17e55a) libraries, production, CPU, double, serial |

