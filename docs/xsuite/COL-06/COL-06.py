"""COL-06: 2.5D FFT space charge: the longitudinal kick of a nonuniform bunch violates geometric
similarity

With either 2.5D FFT solver and `apply_z_kick=True`, `SpaceCharge3D` differentiates slice potentials
whose transverse reference length is fixed at 1 m. The manual explicitly chooses this reference, but
changing it adds a longitudinal force wherever the line density varies. Doubling every length of an
electron bunch with Lorentz factor γ = 10 should leave the scaled longitudinal kick unchanged by
free-space Coulomb similarity, but changes it by 7.868%. For a γ = 7 electron bunch with transverse
RMS size 0.4 mm and longitudinal RMS 20 mm, the on-axis kick in the relative momentum deviation δ at
0.75 longitudinal RMS is 9.177 × 10⁻¹⁰, versus 6.680 × 10⁻¹⁰ from the exact 3D field, a 37.38%
discrepancy. Transverse kicks are unaffected; constant line density or `apply_z_kick=False` removes
the effect.

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
