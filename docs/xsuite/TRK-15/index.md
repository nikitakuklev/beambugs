---
title: "TRK-15: Zero horizontal focusing loses accelerated-path terms"
hide:
  - toc
---

# TRK-15 — Zero horizontal focusing loses accelerated-path terms

| | |
|---|---|
| Code | [Xsuite](../index.md) · Tracking |
| Priority | P2 |
| Severity | Wrong physics |
| Evidence | Reproduced, independently rechecked |
| Last reproduced | 2026-10-29 on xtrack main (`f09e0fb`) |
| Source | [`track_expanded_combined_dipole_quad_single_particle`, track_magnet_drift.h:202–235](https://github.com/xsuite/xtrack/blob/ad5e808cdc5b9a881eac2f41e32a0b7d11cc7028/xtrack/beam_elements/elements_src/track_magnet_drift.h#L202-L235)<br>[expanded-map Hamiltonian](https://github.com/xsuite/xsuite/blob/e0913e0a18028b2fce06991aebed6150876124b3/docs/physics_manual/xtrack.tex#L1105-L1119)<br>[displayed degenerate path length](https://github.com/xsuite/xsuite/blob/e0913e0a18028b2fce06991aebed6150876124b3/docs/physics_manual/xtrack.tex#L1220-L1226) |

A straight `Bend` with zero quadrupole strength, a nonzero distributed dipole strength `k0` and `model='mat-kick-mat'` omits path-length terms from its longitudinal coordinate ζ when the horizontal focusing is zero. Integrating the squared transverse slope in the stated Hamiltonian gives a finite flight-time contribution and an invertible map. For a 2.3 GeV/c proton with zero initial slopes and momentum deviation, a 1.7 m bend with `k0` = 0.03 m⁻¹ and one uniform kick gives ζ = −0.2764 mm instead of −0.7370 mm. Adding more kicks, which carry no field here, changes the error, and forward-then-backward tracking does not return to the start. The transverse map and the zero-field case are correct.

Download these shared helpers into the same directory as the reproducer scripts:

- [Download _prebuilt.py](../helpers/_prebuilt.py) · [View source](../helpers/index.md#_prebuilt)
- [Download _common.py](../helpers/_common.py) · [View source](../helpers/index.md#_common)

??? example "Reproducer: TRK-15.py"

    [Download TRK-15.py](TRK-15.py)

    ```python
    """TRK-15: Zero horizontal focusing loses accelerated-path terms

    A straight `Bend` with zero quadrupole strength, a nonzero distributed dipole strength `k0` and
    `model='mat-kick-mat'` omits path-length terms from its longitudinal coordinate ζ when the
    horizontal focusing is zero. Integrating the squared transverse slope in the stated Hamiltonian
    gives a finite flight-time contribution and an invertible map. For a 2.3 GeV/c proton with zero
    initial slopes and momentum deviation, a 1.7 m bend with `k0` = 0.03 m⁻¹ and one uniform kick gives
    ζ = −0.2764 mm instead of −0.7370 mm. Adding more kicks, which carry no field here, changes the
    error, and forward-then-backward tracking does not return to the start. The transverse map and the
    zero-field case are correct.

    The script tracks three protons with different slopes and relative momentum deviations δ = 0, −0.13
    and 0.27 through thirty straight bends (L = 0.4 and 1.7 m, `k0` from −0.08 to 0.08 m⁻¹, 1, 3 or 7
    uniform kicks, edges off), compares them with the analytic straight-body map and backtracks each to
    its start. The ζ error must equal the missing slope-integral terms summed over the integrator's
    half-body steps, and the zero-field bends must pass. As continuity controls, bends with small
    nonzero quadrupole strength must agree with an independent DOP853 integration of Hamilton's
    equations. It runs serially on the installed prebuilt CPU kernels with kernel compilation forbidden.
    """

    from _prebuilt import CTX, identity, kernels, np, xt
    from scipy.integrate import solve_ivp


    def line(*elements):
        result = xt.Line(elements=list(elements))
        result.build_tracker(_context=CTX)
        return result


    def coordinates(p):
        return np.array([getattr(p, key) for key in ("x", "px", "y", "py", "zeta", "delta")])


    identity()
    print("ORACLE px(s)=px_in-k0*s; integrate (px(s)^2+py_in^2)/(2*(1+delta)^2*rvv)")
    print("rvv=beta/beta0; zeta=s-beta0*c*t; h=k1=0; chi=1")
    initial = dict(
        x=[0.001, -0.004, 0.008],
        px=[0.0, 0.017, -0.011],
        y=[0.0, 0.003, -0.002],
        py=[0.0, -0.006, 0.004],
        delta=[0.0, -0.13, 0.27],
        zeta=[0.0, 0.002, -0.001],
    )
    base = xt.Particles(_context=CTX, p0c=2.3e9, **initial)
    start = coordinates(base)
    rvv = np.asarray(base.rvv).copy()
    r = 1 + start[5]
    rows = []


    def magnet(length, k0, k1=0, kicks=1, model="mat-kick-mat"):
        return xt.Bend(
            length=length,
            angle=0,
            k0=k0,
            k1=k1,
            model=model,
            integrator="uniform",
            num_multipole_kicks=kicks,
            edge_entry_active=False,
            edge_exit_active=False,
        )


    # Delta zeta = L (1 - 1/rvv) - ((px_in^2 + py_in^2) L - px_in k0 L^2 + k0^2 L^3/3)/(2 r^2 rvv),
    # with r = 1 + delta and rvv = beta/beta0.
    def analytic(length, k0):
        out = start.copy()
        out[0] += (start[1] * length - k0 * length**2 / 2) / r
        out[1] -= k0 * length
        out[2] += start[3] * length / r
        integral = (
            (start[1] ** 2 + start[3] ** 2) * length - start[1] * k0 * length**2 + k0**2 * length**3 / 3
        )
        out[4] += length * (1 - 1 / rvv) - integral / (2 * rvv * r**2)
        return out


    for length in (0.4, 1.7):
        for k0 in (-0.08, -0.03, 0.0, 0.03, 0.08):
            oracle = analytic(length, k0)
            for kicks in (1, 3, 7):
                q = base.copy()
                lattice = line(magnet(length, k0, kicks=kicks))
                lattice.track(q)
                actual = coordinates(q)
                assert np.all(q.state == 1)
                np.testing.assert_allclose(
                    actual[[0, 1, 2, 3, 5]], oracle[[0, 1, 2, 3, 5]], atol=3e-15, rtol=0
                )
                # The Kx = 0 branch drops A B L^2/2 + A^2 L^3/6 of the slope integral, with
                # A = -Kx x - chi k0/(1+delta) + h and B = px/(1+delta).
                # Each empty-kick interval uses its entering slope for path length.
                # Uniform retains two separate half-body calls per kick, including
                # adjacent calls around different kicks; the faulty map cannot merge them.
                steps = [length / (2 * kicks)] * (2 * kicks)
                predicted_error = np.zeros(3)
                px_enter = start[1].copy()
                for step in steps:
                    predicted_error += (-px_enter * k0 * step**2 / 2 + k0**2 * step**3 / 6) / (
                        rvv * r**2
                    )
                    px_enter -= k0 * step
                np.testing.assert_allclose(actual[4] - oracle[4], predicted_error, atol=2e-15, rtol=0)
                forward_error = float(np.max(np.abs(actual[4] - oracle[4])))
                lattice.track(q, backtrack=True)
                inverse_error = float(np.max(np.abs(coordinates(q) - start)))
                if k0 == 0:
                    assert forward_error < 2e-15 and inverse_error < 2e-15
                else:
                    assert forward_error > 1e-6
                rows.append(
                    dict(
                        L=length,
                        k0=k0,
                        kicks=kicks,
                        zeta0=float(actual[4, 0]),
                        oracle0=float(oracle[4, 0]),
                        error0=float(actual[4, 0] - oracle[4, 0]),
                        max_zeta_error=forward_error,
                        inverse_error=inverse_error,
                    )
                )
    print("SWEEP", len(rows), "max_prediction_error_below", 2e-15)
    for row in rows:
        if row["L"] == 1.7 and row["k0"] in (0.03, 0.08) and row["kicks"] in (1, 7):
            print("MATRIX", row)

    # A finite focusing field switches branches. Integrate Hamilton's equations
    # independently, including both transverse planes and the longitudinal clock.
    continuity = []
    for k1 in (-0.02, -0.002, 0.0, 0.002, 0.02):
        length, k0 = 0.9, 0.06
        expected = np.empty_like(start)
        for i in range(3):

            def rhs(_s, state, i=i, k0=k0, k1=k1):
                x, px, y, py, _z, _d = state
                return [
                    px / r[i],
                    -k0 - k1 * x,
                    py / r[i],
                    k1 * y,
                    1 - (1 + (px**2 + py**2) / (2 * r[i] ** 2)) / rvv[i],
                    0.0,
                ]

            solution = solve_ivp(rhs, (0, length), start[:, i], method="DOP853", rtol=3e-13, atol=2e-15)
            assert solution.success
            expected[:, i] = solution.y[:, -1]
        q = base.copy()
        line(magnet(length, k0, k1=k1)).track(q)
        errors = np.max(np.abs(coordinates(q) - expected), axis=1)
        if k1:
            assert max(errors) < 1e-14
        else:
            assert errors[4] > 2e-4
        row = dict(
            k1=k1,
            max_coordinate_errors=errors.tolist(),
            actual_zeta0=float(q.zeta[0]),
            oracle_zeta0=float(expected[4, 0]),
        )
        continuity.append(row)
        print("CONTINUITY", row)

    # The geometric exact dipole differs from the paraxial oracle at fourth order.
    scaling = []
    for k0 in (0.001, 0.002, 0.004):
        length = 1.0
        values = []
        for model in ("mat-kick-mat", "bend-kick-bend"):
            q = base.copy()
            line(magnet(length, k0, model=model)).track(q)
            values.append(float(q.zeta[0] - analytic(length, k0)[4, 0]))
        scaling.append(dict(k0=k0, matrix_error=values[0], geometric_error=values[1]))
    print("SMALL_SLOPE_SCALING", scaling)
    print("CONTROLS zero-field forward/inverse < 2e-15; nonzero-focusing DOP853 < 1e-14")
    kernels()
    print("DEFECT REPRODUCED")

    ```

## Provenance

| Role | Model | Effort | Date | Code examined |
|---|---|---|---|---|
| Found | `gpt-6-astra` | xhigh | 2026-10-28 | xtrack 0.118.2 (`ad5e808cdc5b`), installed packages, prebuilt kernels (compilation forbidden), CPU, serial |
| Confirmed | `gpt-6.1-sol` | xhigh | 2026-10-28 | xtrack 0.118.2 (`ad5e808cdc5b`), installed packages, prebuilt kernels (compilation forbidden), CPU, serial |

