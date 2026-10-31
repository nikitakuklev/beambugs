"""COL-03: Particle-to-mesh deposition omits source charge ratios

When `update_from_particles` deposits an Xtrack `Particles` object onto a particle-in-cell mesh,
each macroparticle contributes `weight`·`q0`·e without its `charge_ratio` (q/`q0`, its charge
relative to the reference charge). Cloud-in-cell deposition must conserve the physical charge e·Σ
`weight`·`q0`·`charge_ratio`, as the explicit-array input path does. For live particles of weights 3
and 7 on a 9³ mesh with `q0` = 1, charge ratios 0.5, 1 and 2 all deposit 10 elementary charges
instead of 5, 10 and 20. Mesh space-charge and beam-beam solvers receive this wrong source density;
particles with the reference charge are unaffected, and a source comment acknowledges the missing
charge ratios.

The script deposits two live particles (weights 3 and 7) and one lost particle (weight 11) on a 9 ×
9 × 9 mesh spanning [−1, 1] on each axis, all strictly inside, for `charge_ratio` = 0.5, 1 and 2.
The total deposited charge must equal e·Σ `weight`·`q0`·`charge_ratio` over the live particles. The
explicit-array path and the same species encoded in `q0` instead of `charge_ratio` both give that
charge and serve as controls.
"""

import numpy as np
import xfields as xf
import xtrack as xt

RTOL, ATOL = 4e-15, 2e-14
QELEM = 1.60217662e-19  # Charge constant used by the deposition C kernel.


def evaluate():
    fieldmap = xf.TriLinearInterpolatedFieldMap(
        x_range=(-1, 1), y_range=(-1, 1), z_range=(-1, 1), nx=9, ny=9, nz=9
    )
    volume = fieldmap.dx * fieldmap.dy * fieldmap.dz
    cases = {}
    for charge_ratio in (0.5, 1.0, 2.0):
        particles = xt.Particles(
            p0c=1e9,
            x=[0.13, -0.31, 0.11],
            y=[-0.15, 0.27, 0.12],
            zeta=[0.17, -0.14, 0.21],
            weight=[3.0, 7.0, 11.0],
            state=[1, 1, 0],
            mass_ratio=1,
            charge_ratio=charge_ratio,
        )
        expected_charge = float(
            np.sum(particles.weight * particles.charge_ratio * (particles.state > 0))
        )
        fieldmap.update_from_particles(particles, update_phi=False)
        particle_density = fieldmap.rho.copy()
        observed_charge = float(np.sum(particle_density) * volume / QELEM)
        # The public explicit-array path carries physical charge in ncharges_p.
        fieldmap.update_from_particles(
            x_p=particles.x,
            y_p=particles.y,
            z_p=particles.zeta,
            ncharges_p=particles.weight * particles.charge_ratio,
            state_p=particles.state,
            q0_coulomb=QELEM,
            update_phi=False,
        )
        array_density = fieldmap.rho.copy()
        np.testing.assert_allclose(
            np.sum(array_density) * volume / QELEM, expected_charge, rtol=RTOL, atol=ATOL
        )
        # An equivalent homogeneous species encoded in q0 instead of charge_ratio
        # must give the same physical charge density, including the lost-row mask.
        equivalent = xt.Particles(
            p0c=1e9,
            q0=charge_ratio,
            x=particles.x,
            y=particles.y,
            zeta=particles.zeta,
            weight=particles.weight,
            state=particles.state,
        )
        fieldmap.update_from_particles(equivalent, update_phi=False)
        np.testing.assert_allclose(fieldmap.rho, array_density, rtol=RTOL, atol=1e-32)
        if charge_ratio == 1:
            np.testing.assert_allclose(particle_density, array_density, rtol=RTOL, atol=1e-32)
        cases[f"charge_ratio={charge_ratio}, deposited charge / e"] = (
            observed_charge,
            expected_charge,
        )
    return cases


if __name__ == "__main__":
    from _common import prepare, report

    prepare()
    print(
        "Expected: sum(rho)*cell_volume/e = sum(alive weight*q0*charge_ratio); all particles lie strictly inside the mesh."
    )
    print(
        "Explicit-array deposition and an equivalent q0 representation independently check the expected mesh."
    )
    report(evaluate(), rtol=RTOL, atol=ATOL)
