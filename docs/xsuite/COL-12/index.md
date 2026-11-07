---
title: "COL-12: ElectronCloud applies inconsistent reference-charge factors"
hide:
  - toc
---

# COL-12 — ElectronCloud applies inconsistent reference-charge factors

| | |
|---|---|
| Code | [Xsuite](../index.md) · Collective effects |
| Priority | P3 |
| Severity | Wrong physics |
| Evidence | Reproduced, independently rechecked |
| Last reproduced | 2026-10-29 on xfields main (`3f06c87`, head build) |
| Source | [`ElectronCloud_track_local_particle`, electroncloud.h:45–57](https://github.com/xsuite/xfields/blob/223b4fa134a1435be9b45ef21ae8903f96c5f685/xfields/beam_elements/electroncloud_src/electroncloud.h#L45-L57) |

`ElectronCloud` multiplies its longitudinal kick by the reference charge `q0` but not its transverse kicks, so for `q0` ≠ 1 the kicks no longer derive from one scalar potential and the map is not symplectic. Kicks from a single potential have equal mixed derivatives, ∂(Δpx)/∂ζ = ∂(Δpζ)/∂x, and the canonical energy coordinate `pzeta` (pζ) carries no charge factor. For a 3 GeV/c reference in a 0.7 m element with potential 0.031·xζ − 0.017·yζ, ∂(Δpx)/∂ζ is −0.02170, while ∂(Δpζ)/∂x is −0.04340 for `q0` = 2 and +0.04340 for `q0` = −2 instead of −0.02170. A negative reference charge reverses only the longitudinal kick; references with `q0` = 1 are unaffected.

Run `python COL-12.py` with [_prebuilt.py](../helpers/_prebuilt.py) and [_common.py](../helpers/_common.py) in the same directory.

??? example "Reproducer: COL-12.py"

    [Download COL-12.py](COL-12.py)

    ```python
    """COL-12: ElectronCloud applies inconsistent reference-charge factors

    `ElectronCloud` multiplies its longitudinal kick by the reference charge `q0` but not its transverse
    kicks, so for `q0` ≠ 1 the kicks no longer derive from one scalar potential and the map is not
    symplectic. Kicks from a single potential have equal mixed derivatives, ∂(Δpx)/∂ζ = ∂(Δpζ)/∂x, and
    the canonical energy coordinate `pzeta` (pζ) carries no charge factor. For a 3 GeV/c reference in a
    0.7 m element with potential 0.031·xζ − 0.017·yζ, ∂(Δpx)/∂ζ is −0.02170, while ∂(Δpζ)/∂x is −0.04340
    for `q0` = 2 and +0.04340 for `q0` = −2 instead of −0.02170. A negative reference charge reverses
    only the longitudinal kick; references with `q0` = 1 are unaffected.

    The script loads exact tricubic Taylor data for that potential and tracks a 3 GeV/c reference
    particle with `charge_ratio` = 1 from (x, y, ζ) = (0.12, −0.08, 0.18) m for `q0` = 1, 2 and −2. It
    compares the kicks with −L·∇φ and measures the mixed derivatives by central differences with step
    10⁻⁴ m; `q0` = 1 is the control.
    """

    import xfields as xf
    from _prebuilt import CTX, identity, kernels, np, xt

    identity()

    # Exact tricubic data for phi(x,y,zeta) = A*x*zeta + B*y*zeta.
    a, b, length = 0.031, -0.017, 0.7
    grid = np.linspace(-0.6, 0.6, 7)
    fmap = xf.TriCubicInterpolatedFieldMap(_context=CTX, x_grid=grid, y_grid=grid, z_grid=grid)
    z, y, x = np.meshgrid(grid, grid, grid, indexing="ij")
    h = grid[1] - grid[0]
    taylor = np.stack(
        (
            a * x * z + b * y * z,
            a * z * h,
            b * z * h,
            (a * x + b * y) * h,
            np.zeros_like(x),
            np.full_like(x, a * h * h),
            np.full_like(x, b * h * h),
            np.zeros_like(x),
        ),
        axis=-1,
    )
    # Xobjects generates this array field from the class field specification.
    fmap._phi_taylor[:] = taylor.ravel()  # ty: ignore[unresolved-attribute]
    cloud = xf.ElectronCloud(_buffer=fmap._buffer, fieldmap=fmap, length=length)


    def kick(q, xx=0.12, zz=0.18):
        p = xt.Particles(_context=CTX, p0c=3e9, q0=q, x=xx, y=-0.08, zeta=zz, pzeta=0.015)
        # These particle properties return linked arrays despite scalar annotations.
        assert p.chi[0] == 1 and p.charge_ratio[0] == 1  # ty: ignore[not-subscriptable]
        before = float(p.pzeta[0])
        cloud.track(p)
        assert p.state[0] > 0
        return np.array([p.px[0], p.py[0], p.pzeta[0] - before])


    expected = -length * np.array([a * 0.18, b * 0.18, a * 0.12 + b * (-0.08)])
    for q in [1, 2, -2]:
        observed = kick(q)
        implemented = expected.copy()
        implemented[2] *= q
        np.testing.assert_allclose(observed, implemented, rtol=0, atol=5e-16)
        step = 1e-4
        dpx_dz = (kick(q, zz=0.18 + step)[0] - kick(q, zz=0.18 - step)[0]) / (2 * step)
        dpz_dx = (kick(q, xx=0.12 + step)[2] - kick(q, xx=0.12 - step)[2]) / (2 * step)
        # Scalar-potential kicks give dpx/dzeta = dpzeta/dx = -L*a; the extra longitudinal q0
        # leaves the reciprocity error |(q0 - 1)*L*a| instead of zero.
        expected_residual = abs((q - 1) * length * a)
        np.testing.assert_allclose(abs(dpx_dz - dpz_dx), expected_residual, rtol=0, atol=2e-12)
        print(f"q0={q} kicks={observed.tolist()} scalar_gradient_expected={expected.tolist()}")
        print(
            f"q0={q} dpx/dzeta={dpx_dz:.14g} dpzeta/dx={dpz_dx:.14g} reciprocity_error={abs(dpx_dz - dpz_dx):.14g}"
        )
    kernels()
    print("DEFECT REPRODUCED")

    ```

## Provenance

| Role | Model | Effort | Date | Code examined |
|---|---|---|---|---|
| Found | `gpt-6-astra` | xhigh | 2026-10-28 | xfields 0.27.5 (`223b4fa134a1`), installed wheel, prebuilt kernels, compilation forbidden, CPU, serial |
| Confirmed | `gpt-6.1-sol` | xhigh | 2026-10-28 | xfields 0.27.5 (`223b4fa134a1`), installed wheel, prebuilt kernels, compilation forbidden, CPU, serial |

