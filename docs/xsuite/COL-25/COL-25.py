"""COL-25: Rounded CIC indices discard strictly interior upper-edge particles

Cloud-in-cell deposition and trilinear field gathering can classify a strictly interior particle as outside the mesh when rounded index arithmetic selects the last node as the lower neighbor. On nine-node axes spanning −1 to 1 m, the representable value immediately below the upper endpoint deposits zero instead of its full charge and gathers zero instead of a constant field. With a 1 V/m tangential field over 0.6 m, a γ₀=2 proton receives zero normalized kick instead of 1.066×10⁻¹⁰. Exact nodal basis weights establish the valid stencil. Full PIC tracking also loses the source; averaged 2.5D normalization produces NaNs if no deposited source remains.

Evaluate tensor-product nodal hat functions at 80 digits on exact stored binary64 inputs, without computing cell indices. Compare both deposition APIs, public gathering and frozen-map kicks on all upper faces using explicit and range/count grids, with endpoint and padding controls. Supply the independent density to each native PIC solver to isolate the source omission from its discrete field approximation.
"""

import hashlib
from pathlib import Path

# Configure threads and forbid compilation before importing Xfields.
from _prebuilt import CTX, identity, kernels, np, report, xt

# isort: split
import mpmath as mp
import xfields as xf

mp.mp.dps = 80
QE = 1.60217662e-19  # Charge constant used by the native kernel, in coulombs.
RTOL = 5e-14


def exact(value):
    """Convert a stored binary64 input without decimal rounding."""
    numerator, denominator = float(value).as_integer_ratio()
    return mp.mpf(numerator) / denominator


def density(grid, point, charge):
    """Tensor nodal basis, independent of the native floor/index algorithm."""
    h = exact(grid[1]) - exact(grid[0])
    hats = [
        [max(mp.mpf(0), 1 - abs(exact(value) - exact(node)) / h) for node in grid]
        for value in point
    ]
    assert all(mp.fsum(axis) == 1 for axis in hats)
    # Each hat integrates to one grid spacing; dividing by h³ gives C/m³.
    return np.array([
        float(exact(charge) * wx * wy * wz / h**3)
        for wx in hats[0] for wy in hats[1] for wz in hats[2]
    ]).reshape((len(grid),) * 3)


def boundary_cases(cases):
    grid = np.arange(-4, 5, dtype=float) / 4
    coordinates = np.meshgrid(grid, grid, grid, indexing="ij")
    for axis in range(3):
        point = np.zeros(3)
        point[axis] = np.nextafter(1.0, 0.0)
        oracle_rho = density(grid, point, 1.0)
        expected_charge = float(oracle_rho.sum() * 0.25**3)
        assert expected_charge == 1.0
        # The derivative is tangent to the boundary face: its own finite-
        # difference boundary planes are elsewhere, so their taper is irrelevant.
        tangent = 1 if axis == 0 else 0
        phi = 7 - coordinates[tangent]
        for use_range in (False, True):
            geometry = (
                dict(x_range=(-1, 1), y_range=(-1, 1), z_range=(-1, 1), nx=9, ny=9, nz=9)
                if use_range else dict(x_grid=grid, y_grid=grid, z_grid=grid)
            )
            fmap = xf.TriLinearInterpolatedFieldMap(_context=CTX, phi=phi, **geometry)
            for raw in (False, True):
                particle = xt.Particles(
                    _context=CTX, gamma0=2.0, x=point[0], y=point[1], zeta=point[2]
                )
                if raw:
                    fmap.update_from_particles(
                        x_p=point[:1].copy(), y_p=point[1:2].copy(), z_p=point[2:].copy(),
                        ncharges_p=np.ones(1), q0_coulomb=1.0, update_phi=False,
                    )
                    charge_unit = 1.0
                else:
                    fmap.update_from_particles(particles=particle, update_phi=False)
                    charge_unit = QE
                charge = float(fmap.rho.sum() * 0.25**3 / charge_unit)
                values = fmap.get_values_at_points(
                    *[np.array([value]) for value in point], return_rho=False
                )
                lens = xf.SpaceCharge3D(
                    fieldmap=fmap, length=0.6, update_on_track=False, apply_z_kick=False
                )
                lens.track(particle)
                kick = float((particle.px, particle.py)[tangent][0])
                # Lorentz force integrated over L/(β₀c), normalized by p₀:
                # Δp_t = L E_t/[mass0(eV) γ₀³ β₀²], with E_t=1 and β₀²=3/4.
                expected_kick = float(exact(0.6) / (exact(particle.mass0) * 8 * mp.mpf(".75")))
                assert int(particle.state[0]) == 1
                label = (f"upper-face={'xyz'[axis]} grid={'range' if use_range else 'explicit'} "
                         f"deposit={'arrays' if raw else 'particles'} "
                         "[charge/q,phi(V),tangent_gradient(V/m),kick]; "
                         "hat partition=1, affine phi=7-t, Lorentz integral")
                cases[label] = (
                    [charge, float(values[0][0]), float(values[1 + tangent][0]), kick],
                    [expected_charge, 7.0, -1.0, expected_kick],
                )


def controls():
    grid = np.arange(-4, 5, dtype=float) / 4
    fixtures = [
        ("two_steps_inside", grid, np.nextafter(np.nextafter(1.0, 0.0), 0.0), 1.0),
        ("upper_endpoint", grid, 1.0, 0.0),
        ("lower_endpoint", grid, -1.0, 1.0),
        ("below_lower_endpoint", grid, np.nextafter(-1.0, -np.inf), 0.0),
        ("ordinary_interior", grid, 0.875, 1.0),
        ("whole_cell_padding", np.arange(-5, 6, dtype=float) / 4, 1.0, 1.0),
        ("zero_origin", np.arange(9, dtype=float) / 8, np.nextafter(1.0, 0.0), 1.0),
    ]
    for label, nodes, coordinate, expected_charge in fixtures:
        fmap = xf.TriLinearInterpolatedFieldMap(
            _context=CTX, phi=np.full((len(nodes),) * 3, 7.0),
            x_grid=nodes, y_grid=nodes, z_grid=nodes,
        )
        position = np.array([coordinate])
        fmap.update_from_particles(
            x_p=position, y_p=np.zeros(1), z_p=np.zeros(1),
            ncharges_p=np.ones(1), q0_coulomb=1.0, update_phi=False,
        )
        charge = float(fmap.rho.sum() * fmap.dx * fmap.dy * fmap.dz)
        (potential,) = fmap.get_values_at_points(
            position, np.zeros(1), np.zeros(1), return_rho=False,
            return_dphi_dx=False, return_dphi_dy=False, return_dphi_dz=False,
        )
        observed = [charge, float(potential[0])]
        expected = [expected_charge, 7 * expected_charge]
        assert np.allclose(observed, expected, rtol=RTOL, atol=0), label
        print(f"CONTROL {label}: observed={observed} expected={expected} "
              "derivation=complete-stencil hull convention, constant interpolation")


def pic_cases(cases):
    grid = np.arange(-4, 5, dtype=float) / 4096
    predecessor = np.nextafter(grid[-1], 0.0)
    volume = float(exact(grid[1] - grid[0]) ** 3)
    for solver in ("FFTSolver3D", "FFTSolver2p5D", "FFTSolver2p5DAveraged"):
        for label, source_x in (
            ("upper_predecessor", predecessor),
            ("two_steps_inside", np.nextafter(predecessor, 0.0)),
        ):
            rho = density(grid, (source_x, 0.0, 0.0), QE)
            assert abs(float(rho.sum() * volume / QE) - 1) < 2e-16
            settings = dict(
                _context=CTX, length=0.6, apply_z_kick=False, update_on_track=True,
                x_grid=grid, y_grid=grid, z_grid=grid, solver=solver, gamma0=2.0,
            )
            oracle = xf.SpaceCharge3D(rho=rho, **settings)
            oracle.update_on_track = False
            witness = xt.Particles(_context=CTX, gamma0=2.0, x=-grid[5])
            oracle.track(witness)
            expected_kick = float(witness.px[0])
            assert np.isfinite(expected_kick) and expected_kick != 0.0
            lens = xf.SpaceCharge3D(**settings)
            particles = xt.Particles(
                _context=CTX, gamma0=2.0, x=[source_x, -grid[5]],
                y=[0.0, 0.0], zeta=[0.0, 0.0], weight=[1.0, 0.0],
            )
            # A zero-weight witness samples the source's field without depositing.
            lens.track(particles)
            fieldmap = lens.fieldmap  # ty: ignore[unresolved-attribute]  # Xobjects-generated field.
            charge = float(fieldmap.rho.sum() * volume / QE)
            kick = float(particles.px[1])
            assert np.all(particles.state == 1)
            print(f"PIC {solver} {label}: source_x={source_x:.17g} "
                  f"observed_charge/e={charge:.17g} expected_charge/e=1 "
                  f"observed_px={kick:.17g} expected_px={expected_kick:.17g} "
                  "derivation=80-digit hat density through same discrete solver; states=[1,1]")
            if label == "two_steps_inside":
                assert abs(charge - 1) < 4e-15
                assert np.isclose(kick, expected_kick, rtol=RTOL, atol=0), solver
            else:
                cases[f"{solver} upper predecessor [charge/e,kick]; independent hat source"] = (
                    [charge, kick], [1.0, expected_kick]
                )


def main():
    identity()
    print("mpmath", mp.__version__, "; nodal basis precision=80 decimal digits")
    cases = {}
    boundary_cases(cases)
    controls()
    pic_cases(cases)
    kernels()
    libraries = {
        Path(line.split()[-1])
        for line in Path("/proc/self/maps").read_text().splitlines()
        if "/xsuite/lib/" in line and ".so" in line
    }
    for path in sorted(libraries):
        print("KERNEL_SHA256", path, hashlib.sha256(path.read_bytes()).hexdigest())
    report(cases, rtol=RTOL, atol=0)

if __name__ == "__main__":
    main()
