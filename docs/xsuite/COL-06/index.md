---
title: "COL-06: 2.5D FFT space charge: the longitudinal kick of a nonuniform bunch violates geometric similarity"
hide:
  - toc
---

# COL-06 — 2.5D FFT space charge: the longitudinal kick of a nonuniform bunch violates geometric similarity

| | |
|---|---|
| Code | [Xsuite](../index.md) · Collective effects |
| Priority | P2 |
| Severity | Wrong physics |
| Evidence | Reproduced, independently rechecked |
| Last reproduced | 2026-10-29 on xfields release [v0.27.7](https://github.com/xsuite/xfields/tree/v0.27.7) (`3f06c87`) |
| Tags | <span class="bb-tag">contract-unsettled</span> [?](../../methodology.md#tags) |
| Source | [`primitive_func_2p5d`, fftsolvers.py:321–327](https://github.com/xsuite/xfields/blob/223b4fa134a1435be9b45ef21ae8903f96c5f685/xfields/solvers/fftsolvers.py#L321-L327)<br>[:173](https://github.com/xsuite/xfields/blob/223b4fa134a1435be9b45ef21ae8903f96c5f685/xfields/solvers/fftsolvers.py#L173)<br>[:230](https://github.com/xsuite/xfields/blob/223b4fa134a1435be9b45ef21ae8903f96c5f685/xfields/solvers/fftsolvers.py#L230)<br>[`SpaceCharge3D_track_local_particle`, spacecharge3d.h:55–60](https://github.com/xsuite/xfields/blob/223b4fa134a1435be9b45ef21ae8903f96c5f685/xfields/beam_elements/spacecharge_src/spacecharge3d.h#L55-L60)<br>[manual's modulated-2D kick](https://github.com/xsuite/xsuite/blob/e0913e0a18028b2fce06991aebed6150876124b3/docs/physics_manual/xfields.tex#L302-L310)<br>[xfields.tex:3907–3936](https://github.com/xsuite/xsuite/blob/e0913e0a18028b2fce06991aebed6150876124b3/docs/physics_manual/xfields.tex#L3907-L3936) |

With either 2.5D FFT solver and `apply_z_kick=True`, `SpaceCharge3D` differentiates slice potentials whose transverse reference length is fixed at 1 m. Changing this reference adds a longitudinal force wherever the line density varies. Doubling every length of an electron bunch with Lorentz factor γ = 10 should leave the scaled longitudinal kick unchanged by free-space Coulomb similarity, but changes it by 7.868%. For a γ = 7 electron bunch with transverse RMS size 0.4 mm and longitudinal RMS 20 mm, the on-axis kick in the relative momentum deviation δ at 0.75 longitudinal RMS is 9.177 × 10⁻¹⁰, versus 6.680 × 10⁻¹⁰ from the exact 3D field, a 37.38% discrepancy. Transverse kicks are unaffected; constant line density or `apply_z_kick=False` removes it.

Run `python COL-06.py` with [_common.py](../helpers/_common.py) in the same directory.

??? example "Reproducer: COL-06.py"

    [Download COL-06.py](COL-06.py)

    ```python
    """COL-06: 2.5D FFT space charge: the longitudinal kick of a nonuniform bunch violates geometric
    similarity

    With either 2.5D FFT solver and `apply_z_kick=True`, `SpaceCharge3D` differentiates slice potentials
    whose transverse reference length is fixed at 1 m. Changing this reference adds a longitudinal force
    wherever the line density varies. Doubling every length of an electron bunch with Lorentz factor γ =
    10 should leave the scaled longitudinal kick unchanged by free-space Coulomb similarity, but changes
    it by 7.868%. For a γ = 7 electron bunch with transverse RMS size 0.4 mm and longitudinal RMS 20 mm,
    the on-axis kick in the relative momentum deviation δ at 0.75 longitudinal RMS is 9.177 × 10⁻¹⁰,
    versus 6.680 × 10⁻¹⁰ from the exact 3D field, a 37.38% discrepancy. Transverse kicks are unaffected;
    constant line density or `apply_z_kick=False` removes it.

    The script applies one `SpaceCharge3D` kick (`FFTSolver2p5D`, 64³ mesh, kick length 1 mm) to a
    20,000-macroparticle, 1 pC electron bunch at γ = 10 with transverse RMS sizes 0.1 mm and
    longitudinal RMS 1 mm, then repeats it with the bunch, mesh ranges and kick length all doubled.
    Scaling every length by a at fixed charge and energy scales Coulomb fields by 1/a² and the kick
    length by a, so a times each kick must not change: the transverse kick passes as the control, and
    the longitudinal kick shows the bug.
    """

    import numpy as np
    import xfields as xf
    import xtrack as xt

    N_PARTICLES, CHARGE = 20000, 1e-12  # macroparticles, bunch charge in C


    def scaled_kicks(xyz, a):
        """a*(px, delta) after one kick of a bunch with every length multiplied by a."""
        p = xt.Particles(
            mass0=xt.ELECTRON_MASS_EV,
            gamma0=10.0,
            x=a * xyz[:, 0],
            y=a * xyz[:, 1],
            zeta=a * xyz[:, 2],
            weight=CHARGE / 1.602176634e-19 / len(xyz),
        )
        sc = xf.SpaceCharge3D(
            length=1e-3 * a,
            update_on_track=True,
            apply_z_kick=True,
            x_range=(-8e-4 * a, 8e-4 * a),
            y_range=(-8e-4 * a, 8e-4 * a),
            z_range=(-8e-3 * a, 8e-3 * a),
            nx=64,
            ny=64,
            nz=64,
            solver="FFTSolver2p5D",
        )
        sc.track(p)
        assert np.all(p.state == 1)
        return a * np.stack([p.px, p.delta], axis=1)


    def evaluate():
        rng = np.random.default_rng(1)
        xyz = rng.normal(size=(N_PARTICLES, 3)) * [1e-4, 1e-4, 1e-3]  # sigma x, y, zeta in m
        xyz -= xyz.mean(axis=0)
        k1, k2 = scaled_kicks(xyz, 1.0), scaled_kicks(xyz, 2.0)
        change = np.linalg.norm(k2 - k1, axis=0) / np.linalg.norm(k1, axis=0)
        return {
            "a*px, relative change for a = 1 -> 2": (change[0], 0.0),
            "a*delta, relative change for a = 1 -> 2": (change[1], 0.0),
        }


    if __name__ == "__main__":
        from _common import prepare, report

        prepare(random_seed=1)
        print(
            "Expected: 0. Scaling every length by a at fixed charge and energy scales Coulomb "
            "fields by 1/a**2 and the kick length by a, so a*(kick) is invariant."
        )
        report(evaluate(), rtol=0, atol=1e-12)

    ```

## Provenance

| Role | Model | Effort | Date | Code examined |
|---|---|---|---|---|
| Found | `claude-opus-5-5` | high | 2026-10-27 | xfields 0.27.5 (`223b4fa134a1`), installed wheel, CPU, serial |
| Confirmed<br><small>2.5D longitudinal reference-length dependence; source only</small> | `gpt-6-astra` | xhigh | 2026-10-27 | xfields 0.27.5 (`223b4fa134a1`) |
| Confirmed<br><small>both 2.5D solvers' free-space longitudinal kick, nonuniform line density</small> | `gpt-6.1-sol` | xhigh | 2026-10-30 | xfields 0.27.5 (`223b4fa134a1`), installed packages, kernels compiled from installed source, CPU, double, serial |

