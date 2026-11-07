---
title: "COL-22: S-based cathode_fft_3d tracking silently applies no space charge"
hide:
  - toc
---

# COL-22 — S-based cathode_fft_3d tracking silently applies no space charge

| | |
|---|---|
| Code | [Bmad](../index.md) · Collective effects |
| Priority | P4 |
| Failure | Silent wrong result |
| Evidence | Reproduced, independently rechecked |
| Last reproduced | 2026-11-05 on Bmad release [20261031-0](https://github.com/bmad-sim/bmad-ecosystem/tree/20261031-0) (`c4c629f`) |
| Source | [track1_bunch dispatch, bmad/multiparticle/beam_mod.f90:283–320](https://github.com/bmad-sim/bmad-ecosystem/blob/c4c629fc9071d3a56cb1ba5bc714baba65e9c5d3/bmad/multiparticle/beam_mod.f90#L283-L320)<br>[mesh-kick dispatch, bmad/space_charge/csr_and_space_charge_mod.f90:1646–1650](https://github.com/bmad-sim/bmad-ecosystem/blob/c4c629fc9071d3a56cb1ba5bc714baba65e9c5d3/bmad/space_charge/csr_and_space_charge_mod.f90#L1646-L1650)<br>[CSR and Space Charge Methods, s:csr.sc.meth, bmad/doc/methods.tex:648–650](https://github.com/bmad-sim/bmad-ecosystem/blob/c4c629fc9071d3a56cb1ba5bc714baba65e9c5d3/bmad/doc/methods.tex#L648-L650) |

Selecting cathode_fft_3d with s-based tracking silently transports the bunch without space charge. This method is documented for time-based tracking only, so the incompatible configuration should be rejected. In a 1 cm drift, the exit position–momentum covariance is 1.000e−13 m and the longitudinal covariance is zero, exactly matching space charge off. The ordinary FFT method gives nonzero collective kicks for the same bunch.

Run `python COL-22.py` with [_bmad.py](../helpers/_bmad.py) and [_collective_fields.py](../helpers/_collective_fields.py) in the same directory.

??? example "Reproducer: COL-22.py"

    [Download COL-22.py](COL-22.py)

    ```python
    """COL-22: S-based cathode_fft_3d tracking silently applies no space charge

    Selecting cathode_fft_3d with s-based tracking silently transports the bunch without space charge. This method is documented for time-based tracking only, so the incompatible configuration should be rejected. In a 1 cm drift, the exit position–momentum covariance is 1.000e−13 m and the longitudinal covariance is zero, exactly matching space charge off. The ordinary FFT method gives nonzero collective kicks for the same bunch.

    The script tracks fresh identical Tao bunches with space charge off, ordinary FFT and cathode FFT
    under s-based tracking. The ordinary FFT case controls that collective effects are active; the
    cathode case must either report its unsupported configuration or reveal the silent space-charge-off
    result.
    """

    import numpy as np
    from _bmad import compare, verdict
    from _collective_fields import tao_covariance

    off, rejected = tao_covariance("off")
    if rejected:
        raise RuntimeError("Space-charge-off control rejected")
    fft, rejected = tao_covariance("fft_3d")
    if rejected or not np.isfinite(fft).all() or np.max(abs(fft - off)) < 1e-12:
        raise RuntimeError("Ordinary FFT collective-force control failed")
    print(f"Space-charge-off sigma_12, sigma_56 (m): {off.tolist()}")
    print(f"Ordinary FFT sigma_12, sigma_56 (m): {fft.tolist()}")
    cathode, rejected = tao_covariance("cathode_fft_3d")
    if rejected:
        print("S-based cathode configuration: rejected as required by the method contract")
        verdict(False)
    else:
        # Rejection is the contract. Equality with SC off identifies the silent bypass.
        bypass = not compare(
            "s-based cathode sigma_12, sigma_56 (m), space-charge-off signature",
            cathode,
            off,
            atol=1e-25,
            rtol=2e-13,
        )
        if not bypass:
            raise RuntimeError("Accepted cathode configuration has a different tracking signature")
        print("S-based cathode configuration: observed=accepted without diagnostic; expected=rejected")
        verdict(True)

    ```

## Provenance

| Role | Model | Effort | Date | Code examined |
|---|---|---|---|---|
| Found | `gpt-6.1-sol` | xhigh | 2026-10-22 | fft-sc-endpoint-kicks (`f8173e0253f1`), source, production, CPU, double, serial |
| Confirmed | `claude-opus-5-5` | high | 2026-10-22 | 20261021-0 (`d00a8f17e55a`), source, production, CPU, double, serial |

