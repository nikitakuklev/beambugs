---
title: "OPT-39: Displaced-reference radiation maps omit the constant bend-face wedge loss"
hide:
  - toc
---

# OPT-39 — Displaced-reference radiation maps omit the constant bend-face wedge loss

| | |
|---|---|
| Code | [Bmad](../index.md) · Optics |
| Priority | P2 |
| Failure | Wrong physics |
| Evidence | Reproduced, independently rechecked |
| Last reproduced | 2026-11-07 on Bmad release [20261031-0](https://github.com/bmad-sim/bmad-ecosystem/tree/20261031-0) (`c4c629f`) |
| Tags | <span class="bb-tag">partly-source-only</span> [?](../../methodology.md#tags) |
| Source | [inclined-face dispatch, bmad/modules/rad_6d_mod.f90:561–582](https://github.com/bmad-sim/bmad-ecosystem/blob/c4c629fc9071d3a56cb1ba5bc714baba65e9c5d3/bmad/modules/rad_6d_mod.f90#L561-L582)<br>[body path factor and mean vector, bmad/modules/rad_6d_mod.f90:638–666](https://github.com/bmad-sim/bmad-ecosystem/blob/c4c629fc9071d3a56cb1ba5bc714baba65e9c5d3/bmad/modules/rad_6d_mod.f90#L638-L666)<br>[add_bend_edge_to_damp_mat, bmad/modules/rad_6d_mod.f90:691–716](https://github.com/bmad-sim/bmad-ecosystem/blob/c4c629fc9071d3a56cb1ba5bc714baba65e9c5d3/bmad/modules/rad_6d_mod.f90#L691-L716)<br>[affine tracking action, bmad/modules/radiation_mod.f90:111–119](https://github.com/bmad-sim/bmad-ecosystem/blob/c4c629fc9071d3a56cb1ba5bc714baba65e9c5d3/bmad/modules/radiation_mod.f90#L111-L119)<br>[The supported reference API, bmad/modules/radiation_mod.f90:153–171](https://github.com/bmad-sim/bmad-ecosystem/blob/c4c629fc9071d3a56cb1ba5bc714baba65e9c5d3/bmad/modules/radiation_mod.f90#L153-L171)<br>[explicit/cached reference selection, bmad/modules/radiation_mod.f90:199–219](https://github.com/bmad-sim/bmad-ecosystem/blob/c4c629fc9071d3a56cb1ba5bc714baba65e9c5d3/bmad/modules/radiation_mod.f90#L199-L219)<br>[make_mat6 cache assignment, bmad/code/make_mat6.f90:100–107](https://github.com/bmad-sim/bmad-ecosystem/blob/c4c629fc9071d3a56cb1ba5bc714baba65e9c5d3/bmad/code/make_mat6.f90#L100-L107)<br>[Both half maps, bmad/modules/radiation_mod.f90:231–246](https://github.com/bmad-sim/bmad-ecosystem/blob/c4c629fc9071d3a56cb1ba5bc714baba65e9c5d3/bmad/modules/radiation_mod.f90#L231-L246)<br>[tracking_rad_map_setup, bmad/low_level/tracking_rad_map_setup.f90:45–69](https://github.com/bmad-sim/bmad-ecosystem/blob/c4c629fc9071d3a56cb1ba5bc714baba65e9c5d3/bmad/low_level/tracking_rad_map_setup.f90#L45-L69)<br>[Tao ordinary and sliced bunch tracking, tao/code/tao_lattice_calc_mod.f90:560–585](https://github.com/bmad-sim/bmad-ecosystem/blob/c4c629fc9071d3a56cb1ba5bc714baba65e9c5d3/tao/code/tao_lattice_calc_mod.f90#L560-L585)<br>[arbitrary map reference orbits, bmad/doc/coordinates.tex:87–95](https://github.com/bmad-sim/bmad-ecosystem/blob/c4c629fc9071d3a56cb1ba5bc714baba65e9c5d3/bmad/doc/coordinates.tex#L87-L95)<br>[constructs radiation maps about them, bmad/doc/synchrotron-radiation.tex:93–103](https://github.com/bmad-sim/bmad-ecosystem/blob/c4c629fc9071d3a56cb1ba5bc714baba65e9c5d3/bmad/doc/synchrotron-radiation.tex#L93-L103)<br>[actual field-filled path, bmad/doc/synchrotron-radiation.tex:12–45](https://github.com/bmad-sim/bmad-ecosystem/blob/c4c629fc9071d3a56cb1ba5bc714baba65e9c5d3/bmad/doc/synchrotron-radiation.tex#L12-L45)<br>[constant-free relative map in the manual, bmad/doc/synchrotron-radiation.tex:143–145](https://github.com/bmad-sim/bmad-ecosystem/blob/c4c629fc9071d3a56cb1ba5bc714baba65e9c5d3/bmad/doc/synchrotron-radiation.tex#L143-L145) |

Tracking with radiation damping through a bend with inclined faces gives a wrong mean energy loss when the radiation map is expanded about a displaced orbit. The field wedges that inclined faces add to or remove from the path enter the map's derivative matrix but not its mean-loss vector. Maps about a supplied orbit, such as Tao's bunch centroid, or about the orbit cached by the last transfer-matrix calculation are affected. For 3 GeV/c electrons in a 1 m rectangular bend of curvature 0.2 m⁻¹, parallel orbits displaced horizontally by ±1 mm differ in loss by 18.29 eV (18.31 eV with 64 slices) instead of zero; integration to the physical face planes confirms equal path lengths in the field.

Run `python OPT-39.py` with [_bend_radiation.py](../helpers/_bend_radiation.py) and [_bmad.py](../helpers/_bmad.py) in the same directory.

??? example "Reproducer: OPT-39.py"

    [Download OPT-39.py](OPT-39.py)

    ```python
    """OPT-39: Displaced-reference radiation maps omit the constant bend-face wedge loss

    Tracking with radiation damping through a bend with inclined faces gives a wrong mean energy loss when the radiation map is expanded about a displaced orbit. The field wedges that inclined faces add to or remove from the path enter the map's derivative matrix but not its mean-loss vector. Maps about a supplied orbit, such as Tao's bunch centroid, or about the orbit cached by the last transfer-matrix calculation are affected. For 3 GeV/c electrons in a 1 m rectangular bend of curvature 0.2 m⁻¹, parallel orbits displaced horizontally by ±1 mm differ in loss by 18.29 eV (18.31 eV with 64 slices) instead of zero; integration to the physical face planes confirms equal path lengths in the field.

    The script compares explicit, cached and centered native radiation maps for parallel displaced trajectories and independently integrates Lorentz motion to physical face planes. It checks subdivision, shrinking displacement stencils and perpendicular faces, then exercises Tao bunch-centroid tracking.
    """

    import re

    import numpy as np
    from _bend_radiation import difference, local, slope
    from _bmad import compare, run, verdict
    from scipy.integrate import solve_ivp


    def ray_length(x: float) -> float:
        # The face planes are Z=0 and Z=2*rho*sin(e). Translation in Cartesian X
        # leaves both planes and the uniform-field Lorentz equations invariant.
        # Thus every ray's field-filled length is 2*rho*e=1 m, exactly.
        rho, edge = 5.0, 0.1

        def rhs(s, u):
            return [np.sin(u[2]), np.cos(u[2]), 1 / rho]

        class ExitFace:
            terminal = True
            direction = 1

            def __call__(self, s, u):
                return u[1] - 2 * rho * np.sin(edge)

        sol = solve_ivp(
            rhs, [0, 2], [x, 0, -edge], events=ExitFace(), method="DOP853", rtol=2.3e-14, atol=2e-15
        )
        if not sol.success or len(sol.t_events[0]) != 1:
            raise RuntimeError("Lorentz integration did not intersect the exit face")
        return float(sol.t_events[0][0])


    def tao(n: int, sign: int) -> float:
        if n == 1:
            elements = "b:sbend,l=1,g=.2,e1=.1,e2=.1,fringe_type=full\nbeamline:line=(b)"
        else:
            elements = (
                f"b:sbend,l={1 / n:.17g},g=.2,fringe_type=full\n"
                + "\n".join(
                    f"b{i}:b,e1={0.1 if i == 1 else 0},e2={0.1 if i == n else 0}"
                    for i in range(1, n + 1)
                )
                + "\nbeamline:line=("
                + ",".join(f"b{i}" for i in range(1, n + 1))
                + ")"
            )
        lattice = f"""no_digested
    parameter[geometry]=open
    parameter[particle]=electron
    parameter[p0c]=3e9
    beginning[beta_a]=10
    beginning[beta_b]=10
    bmad_com[radiation_damping_on]=T
    bmad_com[radiation_fluctuations_on]=F
    bmad_com[radiation_zero_average]=F
    bmad_com[synch_rad_scale]=.001
    {elements}
    use,beamline
    """
        init = f"""&tao_beam_init
    beam_init%n_particle=1
    beam_init%n_bunch=1
    beam_init%bunch_charge=1e-12
    beam_init%a_emit=0
    beam_init%b_emit=0
    beam_init%sig_z=0
    beam_init%sig_pz=0
    beam_init%center(1)={sign * 0.001}
    beam_init%renorm_center=T
    beam_init%renorm_sigma=F
    beam_init%random_sigma_cutoff=0
    beam_saved_at='*'
    /
    """
        log = run(
            "tao",
            ["-noplot", "-lat", "lat.bmad", "-init", "tao.init", "-startup", "commands.tao"],
            files={
                "lat.bmad": lattice,
                "tao.init": init,
                "commands.tao": "set global track_type = beam\nshow beam end\nexit\n",
            },
        ).log
        if (
            "[ERROR" in log
            or "[ABORT" in log
            or "Particles surviving:        1" not in log
            or "Particles lost:             0" not in log
        ):
            raise RuntimeError(f"Tao tracking failed:\n{log}")
        centroids = re.findall(r"^  Centroid:\s+(.+)$", log, re.MULTILINE)
        if len(centroids) != 1 or len(centroids[0].split()) != 7:
            raise RuntimeError(f"Missing or malformed Tao centroid:\n{log}")
        delta = float(centroids[0].split()[5])
        if not np.isfinite(delta):
            raise RuntimeError("Nonfinite Tao centroid")
        return delta


    def main() -> None:
        defect = False
        lengths = [ray_length(x) for x in [-0.001, 0.0, 0.001]]
        np.testing.assert_allclose(lengths, 1.0, rtol=0, atol=3e-14)
        print(f"Cartesian Lorentz face-intersection lengths/m: observed={lengths} expected=[1, 1, 1]")
        frozen = {}
        shrinking = []
        for edge, h in [(0.0, 0.001), (0.1, 0.001), (0.1, 0.0005), (0.1, 0.00025)]:
            kd, _, rows = local(edge, h)
            # A cache populated by make_mat6 must have the same effect as explicitly
            # supplying that reference. Check every axis, sign and subdivision.
            for n in [1, 8, 64]:
                for axis in [1, 3]:
                    for sign in [-1, 1]:
                        np.testing.assert_allclose(
                            rows[n, 1, axis, sign], rows[n, 2, axis, sign], rtol=0, atol=2e-17
                        )
            if edge == 0.1:
                shrinking.append(slope(rows, kd, 64, 1, h))
                if h == 0.001:
                    for n in [1, 8, 64]:
                        frozen[n] = 3e9 * difference(rows, n, 1, 1)
                        defect |= compare(
                            f"N={n} displaced-reference loss(+x)-loss(-x)/eV (equal paths)",
                            frozen[n],
                            0.0,
                            atol=6e-8,
                        )
                    print(
                        f"N=64 explicit-reference losses(+x,-x)="
                        f"{rows[64, 1, 1, 1]:.17g}, {rows[64, 1, 1, -1]:.17g}; cached controls pass"
                    )
                else:
                    print(
                        f"stencil=+/-{h:g} m: normalized loss slope={shrinking[-1]:.17g} m^-2; physical expected=0"
                    )
            if h != 0.001:
                continue
            # Centered-map error is O(N^-2); displaced-map body quadrature can also
            # contain 1/N. Extrapolate both before applying exact path controls.
            centered = (64 * slope(rows, kd, 64, 0) - slope(rows, kd, 8, 0)) / 63
            ns = np.array([1, 8, 64])
            displaced = np.linalg.solve(
                np.stack([ns * 0 + 1, 1 / ns, 1 / ns**2], axis=1),
                [slope(rows, kd, int(n), 1) for n in ns],
            )[0]
            expected = 0.04 * np.sin(0.2) if edge == 0 else 0.0
            np.testing.assert_allclose(centered, expected, rtol=0, atol=2e-12)
            if edge == 0:
                np.testing.assert_allclose(displaced, expected, rtol=0, atol=3e-11)
            print(
                f"edge={edge:g} extrapolated centered slope={centered:.17g}; "
                f"displaced={displaced:.17g}; physical expected={expected:.17g} m^-2"
            )
            if edge == 0.1:
                # Without its cancelling wedges, the curved body gives 2*g^2*tan(e).
                print(
                    f"rectangular body-only slope={0.08 * np.tan(0.1):.17g} m^-2; "
                    f"Lorentz-oracle loss difference={3e9 * kd * 0.04 * (lengths[2] - lengths[0]):.6g} eV"
                )
        if np.ptp(shrinking) >= 2e-12:
            raise RuntimeError("Shrinking-stencil slopes do not agree")
        for n in [1, 64]:
            minus, plus = tao(n, -1), tao(n, 1)
            loss = -(plus - minus) * 3e9 / 0.001
            # Tao prints nine significant digits. Its centroid quantization gives
            # 0.0006 eV tolerance after dividing by the weak-radiation scale.
            np.testing.assert_allclose(loss, frozen[n], rtol=0, atol=0.0006)
            print(
                f"Tao N={n}: delta(-x)={minus:.12g} delta(+x)={plus:.12g}; "
                "one particle survives, zero losses per case"
            )
            defect |= compare(
                f"Tao N={n} scale-normalized loss difference/eV (equal paths)", loss, 0.0, atol=0.0006
            )
        verdict(defect)


    if __name__ == "__main__":
        main()

    ```

## Provenance

| Role | Model | Effort | Date | Code examined |
|---|---|---|---|---|
| Found | <span class="bb-nowrap">`gpt-6-astra`</span> | xhigh | <span class="bb-nowrap">2026-11-06</span> | 20261031-0 (`c4c629fc9071d3a56cb1ba5bc714baba65e9c5d3`), source, production, GNU Fortran 15.3.0, CPU, double, OpenMP one thread, MPI disabled |
| Confirmed | <span class="bb-nowrap">`gpt-6.1-sol`</span> | xhigh | <span class="bb-nowrap">2026-11-06</span> | 20261031-0 (`c4c629fc9071d3a56cb1ba5bc714baba65e9c5d3`), source, production, GNU Fortran 15.3.0, CPU, double, OpenMP one thread, MPI disabled |
| Confirmed | <span class="bb-nowrap">`claude-opus-5-5`</span> | high | <span class="bb-nowrap">2026-11-07</span> | 20261031-0 (`c4c629fc9071d3a56cb1ba5bc714baba65e9c5d3`), source, production, GNU Fortran 15.3.0, CPU, double, OpenMP one thread, MPI disabled |

