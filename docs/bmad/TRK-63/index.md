---
title: "TRK-63: Combined-function bends lose the zero-focusing and small-phase limits"
hide:
  - toc
---

# TRK-63 — Combined-function bends lose the zero-focusing and small-phase limits

| | |
|---|---|
| Code | [Bmad](../index.md) · Tracking, RF, fields and spin |
| Priority | P3 |
| Failure | Wrong physics |
| Evidence | Reproduced, independently rechecked |
| Last reproduced | 2026-11-06 on Bmad release [20261031-0](https://github.com/bmad-sim/bmad-ecosystem/tree/20261031-0) (`c4c629f`) |
| Source | [`sbend_body_with_k1_map`, bmad/low_level/track_a_bend.f90:524–558](https://github.com/bmad-sim/bmad-ecosystem/blob/c4c629fc9071d3a56cb1ba5bc714baba65e9c5d3/bmad/low_level/track_a_bend.f90#L524-L558)<br>[orbit-center shift and momentum derivatives, same file:573–616](https://github.com/bmad-sim/bmad-ecosystem/blob/c4c629fc9071d3a56cb1ba5bc714baba65e9c5d3/bmad/low_level/track_a_bend.f90#L573-L616)<br>[longitudinal Jacobian and final orbit, same file:628–657](https://github.com/bmad-sim/bmad-ecosystem/blob/c4c629fc9071d3a56cb1ba5bc714baba65e9c5d3/bmad/low_level/track_a_bend.f90#L628-L657)<br>[`track1_bmad`, bmad/low_level/track1_bmad.f90:208–209](https://github.com/bmad-sim/bmad-ecosystem/blob/c4c629fc9071d3a56cb1ba5bc714baba65e9c5d3/bmad/low_level/track1_bmad.f90#L208-L209)<br>[native body dispatch and subdivision, track_a_bend.f90:98–118](https://github.com/bmad-sim/bmad-ecosystem/blob/c4c629fc9071d3a56cb1ba5bc714baba65e9c5d3/bmad/low_level/track_a_bend.f90#L98-L118)<br>[small-integrated-quadrupole guard and exact-multipole bypass, same file:58–73](https://github.com/bmad-sim/bmad-ecosystem/blob/c4c629fc9071d3a56cb1ba5bc714baba65e9c5d3/bmad/low_level/track_a_bend.f90#L58-L73)<br>[supported bend attributes, bmad/doc/elements.tex:505–519](https://github.com/bmad-sim/bmad-ecosystem/blob/c4c629fc9071d3a56cb1ba5bc714baba65e9c5d3/bmad/doc/elements.tex#L505-L519)<br>[exact-multipole semantics, same file:608–639](https://github.com/bmad-sim/bmad-ecosystem/blob/c4c629fc9071d3a56cb1ba5bc714baba65e9c5d3/bmad/doc/elements.tex#L608-L639)<br>[documented body Hamiltonian, bmad/doc/charged-tracking.tex:543–593](https://github.com/bmad-sim/bmad-ecosystem/blob/c4c629fc9071d3a56cb1ba5bc714baba65e9c5d3/bmad/doc/charged-tracking.tex#L543-L593)<br>[reference-clock contract, bmad/low_level/low_energy_z_correction.f90:2–8](https://github.com/bmad-sim/bmad-ecosystem/blob/c4c629fc9071d3a56cb1ba5bc714baba65e9c5d3/bmad/low_level/low_energy_z_correction.f90#L2-L8)<br>[implementation, same file:38–54](https://github.com/bmad-sim/bmad-ecosystem/blob/c4c629fc9071d3a56cb1ba5bc714baba65e9c5d3/bmad/low_level/low_energy_z_correction.f90#L38-L54) |

When a native combined-function bend has zero horizontal focusing, k₁+g²=0, where k₁ is quadrupole strength and g is curvature, its body map returns NaN coordinates instead of finite transport. For length 0.5000 m and curvature 0.1250 m⁻¹, an initial horizontal position of 1.000 mm and normalized momentum 0.002000 should give 2.000 mm at exit. Just below the small-phase switch, longitudinal position is wrong by 62.50 µm and its momentum derivative by 0.03125 m; cancellation above the switch leaves nanometre errors. Independent integration of the documented paraxial Hamiltonian establishes these results. Exact-multipole, Runge–Kutta and PTC tracking avoid this body routine.

Run `python TRK-63.py` with [_bmad.py](../helpers/_bmad.py) in the same directory.

??? example "Reproducer: TRK-63.py"

    [Download TRK-63.py](TRK-63.py)

    ```python
    """TRK-63: Combined-function bends lose the zero-focusing and small-phase limits

    When a native combined-function bend has zero horizontal focusing, k₁+g²=0, where k₁ is quadrupole strength and g is curvature, its body map returns NaN coordinates instead of finite transport. For length 0.5000 m and curvature 0.1250 m⁻¹, an initial horizontal position of 1.000 mm and normalized momentum 0.002000 should give 2.000 mm at exit. Just below the small-phase switch, longitudinal position is wrong by 62.50 µm and its momentum derivative by 0.03125 m; cancellation above the switch leaves nanometre errors. Independent integration of the documented paraxial Hamiltonian establishes these results. Exact-multipole, Runge–Kutta and PTC tracking avoid this body routine.

    The script uses BMAD_DIST and the shared runner to track fixed electrons through fringe-free bends, both with and without a requested matrix. It integrates the documented Hamiltonian and all variational equations independently with DOP853, halves the integration step, and checks exact-zero, signed near-zero, off-momentum and short positive-step cases. Ordinary focusing and alternative tracking paths supply controls.
    """

    from __future__ import annotations

    import numpy as np
    from _bmad import fortran, verdict
    from scipy.integrate import solve_ivp

    PROGRAM = """
    program probe
    use bmad
    implicit none
    type(lat_struct),target::lat
    type(coord_struct) a,b
    real(rp) v(6),m(6,6)
    integer j,mode
    logical err
    call bmad_parser('lat.bmad',lat,err_flag=err)
    if(err) error stop 'parser'
    open(30,file='input')
    read(30,*)v
    close(30)
    write(*,'(a,es26.17)') 'MASS ',mass_of(electron$)
    write(*,'(a,3es26.17)') 'PARAM ',lat%ele(1)%value(l$),lat%ele(1)%value(g$),lat%ele(1)%value(k1$)
    do mode=0,1
     call init_coord(a,v,lat%ele(1),upstream_end$)
     call mat_make_unit(m)
     if(mode==1 .and. lat%ele(1)%tracking_method==bmad_standard$) then
      b=a
      call track1_bmad(b,lat%ele(1),lat%param,err_flag=err,mat6=m,make_matrix=.true.)
     else
      call track1(a,lat%ele(1),lat%param,b,err_flag=err)
     endif
     if(err) error stop 'track1'
     write(*,'(a,i0,6es26.17)') 'V',mode,b%vec
     if(mode==1) then
      do j=1,6
       write(*,'(a,i0,6es26.17)') 'M',j,m(j,:)
      enddo
     endif
    enddo
    end program
    """


    def oracle(v, length, g, k, mass, max_step):
        # Canonical equations for the manual's paraxial Hamiltonian, with P=1+delta.
        # z is positive for early arrival, hence dz/ds = dH/ddelta.
        def rhs(s, u):
            x, px, y, py, _, d = u[:6]
            p = 1 + d
            f = np.array(
                [
                    px / p,
                    g * d - (k + g * g) * x,
                    py / p,
                    k * y,
                    -g * x - (px * px + py * py) / (2 * p * p),
                    0.0,
                ]
            )
            a = np.zeros((6, 6))
            a[0, 1], a[0, 5] = 1 / p, -px / p**2
            a[1, 0], a[1, 5] = -(k + g * g), g
            a[2, 3], a[2, 5] = 1 / p, -py / p**2
            a[3, 2] = k
            a[4, 0], a[4, 1], a[4, 3] = -g, -px / p**2, -py / p**2
            a[4, 5] = (px * px + py * py) / p**3
            return np.r_[f, (a @ u[6:].reshape(6, 6)).ravel()]

        sol = solve_ivp(
            rhs,
            (0, length),
            np.r_[v, np.eye(6).ravel()],
            method="DOP853",
            atol=1e-16,
            rtol=3e-14,
            max_step=max_step,
        )
        if not sol.success:
            raise RuntimeError(sol.message)
        w = sol.y[:, -1].copy()
        # Add the exact relativistic reference-clock term L*(beta/beta_ref-1).
        # Rationalizing the square roots avoids subtracting two nearly equal speeds.
        p = 1 + v[5]
        q2 = (mass / 1e9) ** 2
        energy = np.sqrt(p * p + q2)
        ref_energy = np.sqrt(1 + q2)
        w[4] += length * q2 * v[5] * (2 + v[5]) / (energy * (p * ref_energy + energy))
        w[6:].reshape(6, 6)[4, 5] += length * ref_energy * q2 / energy**3
        return w[:6], w[6:].reshape(6, 6)


    def native(length, g, k, v, option):
        lattice = (
            "no_digested\nparameter[geometry]=open\nparameter[particle]=electron\n"
            "beginning[p0c]=1e9\n"
            f"e: sbend,l={length:.17g},g={g:.17g},k1={k:.17g},"
            f"fringe_type=none{option}\nlat: line=(e)\nuse,lat\n"
        )
        log = fortran(PROGRAM, files={"lat.bmad": lattice, "input": " ".join(map(str, v))}).log
        rows = {}
        for tag, count in [("MASS", 1), ("PARAM", 3), ("V0", 6), ("V1", 6)] + [
            (f"M{j}", 6) for j in range(1, 7)
        ]:
            found = [line.split()[1:] for line in log.splitlines() if line.startswith(tag + " ")]
            if len(found) != 1 or len(found[0]) != count:
                raise RuntimeError(f"Missing or malformed {tag} record:\n{log}")
            rows[tag] = np.array([float(t) for t in found[0]])
        np.testing.assert_allclose(rows["PARAM"], [length, g, k], atol=0, rtol=0)
        np.testing.assert_allclose(rows["V1"], rows["V0"], atol=0, rtol=0, equal_nan=True)
        mass = rows["MASS"][0]
        if not np.isfinite(mass) or mass <= 0:
            raise RuntimeError("Invalid native electron mass")
        return rows["V1"], np.array([rows[f"M{j}"] for j in range(1, 7)]), mass


    def main():
        g, length = 0.125, 0.5
        cases = [("normal", length, 0.8, 0.0), ("zero", length, -g * g, 0.0)]
        for sign in [1, -1]:
            for phase in [0.999e-6, 1.001e-6]:
                cases.append(
                    (f"kx{sign:+d}_phase{phase:g}", length, -g * g + sign * (phase / length) ** 2, 0.0)
                )
        for delta in [-0.001, 0.001]:
            cases.append((f"normal_delta{delta:+g}", length, 0.8, delta))
            cases.append(
                (f"negative_kx_delta{delta:+g}", length, -g * g - (0.999e-6 / length) ** 2, delta)
            )
        cases.append(("short_positive_step", 5e-7, 0.8, 0.0))
        defect = False
        max_refinement = max_matrix_refinement = 0.0
        for name, step, k, delta in cases:
            v = np.array([0.001, 0.002, -0.002, 0.001, 0.0, delta])
            observed, matrix, mass = native(step, g, k, v, "")
            expected, target_matrix = oracle(v, step, g, k, mass, step / 20)
            refined, refined_matrix = oracle(v, step, g, k, mass, step / 40)
            refinement = float(np.max(np.abs(expected - refined)))
            matrix_refinement = float(np.max(np.abs(target_matrix - refined_matrix)))
            max_refinement = max(max_refinement, refinement)
            max_matrix_refinement = max(max_matrix_refinement, matrix_refinement)
            np.testing.assert_allclose(expected, refined, atol=2e-16, rtol=0)
            np.testing.assert_allclose(target_matrix, refined_matrix, atol=2e-14, rtol=0)
            if not (np.isfinite(expected).all() and np.isfinite(target_matrix).all()):
                raise RuntimeError("Nonfinite independent integration")
            finite = bool(np.isfinite(observed).all() and np.isfinite(matrix).all())
            bad = not finite
            if finite:
                bad = not (
                    np.allclose(observed, expected, atol=2e-15, rtol=0)
                    and np.allclose(matrix, target_matrix, atol=2e-13, rtol=0)
                )
            if name.startswith("normal") and bad:
                raise RuntimeError(f"Ordinary-focusing control failed: {name}")
            # At kx=0 and delta=0, x=x0+L*px0 and px=px0 exactly.
            if name == "zero":
                np.testing.assert_allclose(expected[:2], [0.002, 0.002], atol=2e-17, rtol=0)
                print(
                    f"zero x,px,z; elementary drift limit plus Hamiltonian: "
                    f"observed={observed[[0, 1, 4]].tolist()} "
                    f"expected={expected[[0, 1, 4]].tolist()}"
                )
            elif name == "short_positive_step":
                # Integrating -g*x gives -g*L²*px0/(2P). The source's positive
                # coefficient differs by g*L²*px0/P even at ordinary focusing.
                sign_error = g * step**2 * v[1] / (1 + delta)
                measured = observed[4] - expected[4]
                print(
                    f"short_positive_step z error [m]; g*L^2*px/P: "
                    f"observed={measured:.12e} expected={sign_error:.12e}; "
                    f"z={observed[4]:.12e} Hamiltonian={expected[4]:.12e}"
                )
                # This small signal falls below the main orbit tolerance; compare
                # its own scale while allowing a future repaired implementation.
                bad = abs(measured) > 2e-22
                if bad:
                    np.testing.assert_allclose(measured, sign_error, atol=2e-22, rtol=0)
            else:
                if not finite:
                    raise RuntimeError(f"Unexpected nonfinite near-zero/control case: {name}")
                print(
                    f"{name} x,z,M52; Hamiltonian+variations: "
                    f"observed={observed[[0, 4]].tolist() + [float(matrix[4, 1])]} "
                    f"expected={expected[[0, 4]].tolist() + [float(target_matrix[4, 1])]}; "
                    f"max_matrix_error={np.max(np.abs(matrix - target_matrix)):.6e}"
                )
            defect |= bad
        v = np.array([0.001, 0.002, -0.002, 0.001, 0.0, 0.0])
        for option in [
            "exact_multipoles=vertically_pure",
            "tracking_method=runge_kutta",
            "tracking_method=symp_lie_ptc",
            "k1=0",
        ]:
            # Alternative methods are finiteness controls: their Hamiltonians differ.
            # For nonstandard trackers the client repeats public tracking and emits
            # identity rather than requesting a matrix from a different method.
            k = 0.0 if option == "k1=0" else -g * g
            extra = "" if option == "k1=0" else "," + option
            observed, matrix, _ = native(length, g, k, v, extra)
            finite = bool(np.isfinite(observed).all() and np.isfinite(matrix).all())
            print(f"avoidance {option}; finite transport: observed={finite} expected=True")
            if not finite:
                raise RuntimeError(f"Alternative-path control failed: {option}")
        print(
            f"oracle step halving: max_coordinate_change={max_refinement:.6e} "
            f"(bound 2e-16); max_matrix_change={max_matrix_refinement:.6e} (bound 2e-14)"
        )
        print(
            "native matrix/no-matrix orbits agree exactly in 13 cases; 2 alternative-method repeats agree"
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

