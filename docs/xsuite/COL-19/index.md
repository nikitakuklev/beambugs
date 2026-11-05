---
title: "COL-19: B&M applies an extra beta factor to momentum dispersion"
hide:
  - toc
---

# COL-19 — B&M applies an extra beta factor to momentum dispersion

| | |
|---|---|
| Code | [Xsuite](../index.md) · Collective effects |
| Priority | P2 |
| Severity | Wrong physics |
| Evidence | Reproduced, independently rechecked |
| Last reproduced | 2026-10-29 on xfields main (`3f06c87`) |
| Source | [`BjorkenMtingwaIBS._a`, _analytical.py:898–921](https://github.com/xsuite/xfields/blob/223b4fa134a1435be9b45ef21ae8903f96c5f685/xfields/ibs/_analytical.py#L898-L921)<br>[`_constants`](https://github.com/xsuite/xfields/blob/223b4fa134a1435be9b45ef21ae8903f96c5f685/xfields/ibs/_analytical.py#L1418-L1458)<br>[segment map](https://github.com/xsuite/xtrack/blob/ad5e808cdc5b9a881eac2f41e32a0b7d11cc7028/xtrack/beam_elements/elements_src/linesegmentmap.h#L44-L81)<br>[off-momentum differences](https://github.com/xsuite/xtrack/blob/ad5e808cdc5b9a881eac2f41e32a0b7d11cc7028/xtrack/twiss/chromatic_functions.py#L124-L135)<br>[manual's collision matrices](https://github.com/xsuite/xsuite/blob/e0913e0a18028b2fce06991aebed6150876124b3/docs/physics_manual/xfields.tex#L2637-L2700) |

`BjorkenMtingwaIBS`, the Bjorken–Mtingwa model of intrabeam-scattering growth rates, multiplies the dispersions and their slopes in both planes by the relativistic reference speed β₀. Xtrack supplies them as derivatives with respect to the relative momentum deviation δ, which the manual's collision matrices use unscaled; integrating the collision matrix with Carlson elliptic integrals gives the correct rates and agrees with Xfields' Nagaitsev model at zero vertical dispersion. For protons at β₀ = 0.45 with beta function βₓ = 8 m, dispersion Dₓ = 1.7 m and slope Dₓ′ = 0.11, the longitudinal cooling rate is −2.890 × 10⁻³ s⁻¹ instead of −2.596 × 10⁻³ s⁻¹, 11.36% too large in magnitude; at β₀ = 0.75 the error is 22.43%.

Download these shared helpers into the same directory as the reproducer scripts:

- [Download _ibs.py](../helpers/_ibs.py) · [View source](../helpers/index.md#_ibs)
- [Download _prebuilt.py](../helpers/_prebuilt.py) · [View source](../helpers/index.md#_prebuilt)
- [Download _common.py](../helpers/_common.py) · [View source](../helpers/index.md#_common)

??? example "Reproducer: COL-19.py"

    [Download COL-19.py](COL-19.py)

    ```python
    """COL-19: B&M applies an extra beta factor to momentum dispersion

    `BjorkenMtingwaIBS`, the Bjorken–Mtingwa model of intrabeam-scattering growth rates, multiplies the
    dispersions and their slopes in both planes by the relativistic reference speed β₀. Xtrack supplies
    them as derivatives with respect to the relative momentum deviation δ, which the manual's collision
    matrices use unscaled; integrating the collision matrix with Carlson elliptic integrals gives the
    correct rates and agrees with Xfields' Nagaitsev model at zero vertical dispersion. For protons at
    β₀ = 0.45 with beta function βₓ = 8 m, dispersion Dₓ = 1.7 m and slope Dₓ′ = 0.11, the longitudinal
    cooling rate is −2.890 × 10⁻³ s⁻¹ instead of −2.596 × 10⁻³ s⁻¹, 11.36% too large in magnitude; at
    β₀ = 0.75 the error is 22.43%.

    The shared helper `_ibs.py` builds each periodic 59 m segment from `LineSegmentMap` elements,
    confirms with tracked off-momentum closed orbits that the Twiss dispersion is dx/dδ, and computes
    the collision-matrix rates independently of Xfields at the same Coulomb logarithm. The script runs
    six cases (β₀ = 0.45, 0.75 and 0.999, plus reversed, vertical, varying and very small dispersion)
    and checks that Xfields' B&M rates equal that calculation with the dispersions multiplied by β₀ and
    differ from it without the factor. With very small dispersion the factor no longer matters, so there
    the rates must also match the unscaled calculation. It requires the installed prebuilt kernels
    (compilation is forbidden).
    """

    from _ibs import PARAMS, run_case
    from _prebuilt import identity, kernels, np

    identity()
    print("ORACLE L=Lx+Ly+Lz; phi=Dprime+alpha*D/beta; H=D^2/beta+beta*phi^2")
    print("Rest-frame momentum delta=gamma*u_parallel; L uses gamma*D_delta")
    print("Eigenvalue/Carlson integral: J_i=2*(sum(R)-3*R_i)/(3*sqrt(det(L)))")
    print("Amplitude rates C/2*trace(L_u*J); shared Coulomb log held fixed")

    for label, beta, optics in [
        ("low", 0.45, {}),
        ("mid", 0.75, dict(dx=-1.7, dpx=-0.11)),
        ("high", 0.999, {}),
        ("vertical", 0.45, dict(dy=0.5, dpy=-0.04)),
        ("varying", 0.75, dict(varying=True)),
        ("tiny", 0.45, dict(dx=1e-7, dpx=0)),
    ]:
        record, _ = run_case(label, beta, params=PARAMS, **optics)
        np.testing.assert_allclose(
            record["actual"], record["beta_scaled_oracle"], rtol=3e-8, atol=2e-13
        )
        if label == "tiny":
            np.testing.assert_allclose(record["actual"], record["oracle"], rtol=1e-9, atol=2e-13)
        else:
            assert np.max(abs(np.array(record["relative_error"]))) > 1e-4

    kernels()
    print("DEFECT REPRODUCED")

    ```

## Provenance

| Role | Model | Effort | Date | Code examined |
|---|---|---|---|---|
| Found | `gpt-6-astra` | xhigh | 2026-10-29 | xfields 0.27.5 (`223b4fa134a1`), installed wheel, prebuilt kernels, compilation forbidden, CPU, serial |
| Confirmed | `gpt-6.1-sol` | xhigh | 2026-10-29 | xfields 0.27.5 (`223b4fa134a1`), installed wheel, prebuilt kernels, compilation forbidden, CPU, serial |

