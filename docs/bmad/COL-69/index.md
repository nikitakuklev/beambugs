---
title: "COL-69: Potential-well matrix header reverses the implemented kick order"
hide:
  - toc
---

# COL-69 — Potential-well matrix header reverses the implemented kick order

| | |
|---|---|
| Code | [Bmad](../index.md) · Collective effects |
| Priority | P4 |
| Failure | Documentation or example mismatch |
| Evidence | Reproduced, independently rechecked |
| Last reproduced | 2026-11-05 on Bmad release [20261031-0](https://github.com/bmad-sim/bmad-ecosystem/tree/20261031-0) (`c4c629f`) |
| Source | [`pwd_mat`, bmad/multiparticle/longitudinal_profile_mod.f90:658–699](https://github.com/bmad-sim/bmad-ecosystem/blob/c4c629fc9071d3a56cb1ba5bc714baba65e9c5d3/bmad/multiparticle/longitudinal_profile_mod.f90#L658-L699)<br>[`map_ref_orb_out = %mat6 * map_ref_orbit_in + %vec0`](https://github.com/bmad-sim/bmad-ecosystem/blob/c4c629fc9071d3a56cb1ba5bc714baba65e9c5d3/bmad/doc/ele-struct.tex#L540-L550)<br>[“Potential well distortion (pwd)”](https://github.com/bmad-sim/bmad-ecosystem/blob/c4c629fc9071d3a56cb1ba5bc714baba65e9c5d3/bsim/ibs_ring/doc/ibs_ring.tex#L218-L240)<br>[`set_pwd_ele` header](https://github.com/bmad-sim/bmad-ecosystem/blob/c4c629fc9071d3a56cb1ba5bc714baba65e9c5d3/bmad/multiparticle/longitudinal_profile_mod.f90#L533-L538)<br>[calculation](https://github.com/bmad-sim/bmad-ecosystem/blob/c4c629fc9071d3a56cb1ba5bc714baba65e9c5d3/bmad/multiparticle/longitudinal_profile_mod.f90#L611-L619)<br>[`transfer_matrix_calc` starts at the end of its indexed element](https://github.com/bmad-sim/bmad-ecosystem/blob/c4c629fc9071d3a56cb1ba5bc714baba65e9c5d3/bmad/code/transfer_matrix_calc.f90#L1-L36)<br>[`residual_pwd_sig_z`](https://github.com/bmad-sim/bmad-ecosystem/blob/c4c629fc9071d3a56cb1ba5bc714baba65e9c5d3/bmad/multiparticle/ibs_mod.f90#L934-L940)<br>[ibs_ring driver disables PWD and retains its element-1 insertion code](https://github.com/bmad-sim/bmad-ecosystem/blob/c4c629fc9071d3a56cb1ba5bc714baba65e9c5d3/bsim/ibs_ring/ibs_ring.f90#L109-L134) |

Calling pwd_mat according to its documented matrix product gives a different
observation point from the returned map. The header specifies M K, where M is
the ring map and K is the potential-well kick, but the routine returns K M.
For a stable 3 GeV electron example, a particle starting at longitudinal position
z = 1 mm with zero relative momentum deviation returns at 0.9922 mm under the
documented convention and 0.9800 mm under the implemented convention. The two
maps describe points immediately before and after the kick; their equilibrium
bunch lengths and tunes agree.

Run `python COL-69.py` with [_bmad.py](../helpers/_bmad.py) in the same directory.

??? example "Reproducer: COL-69.py"

    [Download COL-69.py](COL-69.py)

    ```python
    """COL-69: Potential-well matrix header reverses the implemented kick order

    Calling pwd_mat according to its documented matrix product gives a different
    observation point from the returned map. The header specifies M K, where M is
    the ring map and K is the potential-well kick, but the routine returns K M.
    For a stable 3 GeV electron example, a particle starting at longitudinal position
    z = 1 mm with zero relative momentum deviation returns at 0.9922 mm under the
    documented convention and 0.9800 mm under the implemented convention. The two
    maps describe points immediately before and after the kick; their equilibrium
    bunch lengths and tunes agree.

    The script builds a stable linear Taylor ring and a copy with an explicit kick at its beginning,
    then tracks from each side of that kick. Scalar kick equations and analytic matched covariances
    check the returned matrices independently. Zero-inductance, identity-map and bunch-length controls
    isolate the order mismatch;
    """

    from __future__ import annotations

    import numpy as np
    from _bmad import compare, fortran, verdict
    from numpy.typing import ArrayLike, NDArray

    PROGRAM = """
    program probe
    use bmad
    use ibs_mod
    implicit none
    type(lat_struct), target :: bare, inserted
    type(normal_modes_struct) mode
    type(ibs_sim_param_struct) params
    type(coord_struct), allocatable :: orbit(:)
    real(rp) m(6,6), got(6,6), before(6,6), after(6,6), sigma(6,6), zero(6,6), seed(6), sig_z
    logical err
    integer state

    call bmad_parser('bare.bmad', bare, err_flag=err)
    if (err) error stop 'bare parse'
    call bmad_parser('inserted.bmad', inserted, err_flag=err)
    if (err) error stop 'inserted parse'
    bare%param%n_part = 1e10_rp
    call lat_make_mat6(bare, err_flag=err)
    if (err) error stop 'bare matrices'
    call lat_make_mat6(inserted, err_flag=err)
    if (err) error stop 'inserted matrices'
    call transfer_matrix_calc(bare, m, ix1=0, one_turn=.true.)
    call transfer_matrix_calc(inserted, before, ix1=0, one_turn=.true.)
    call transfer_matrix_calc(inserted, after, ix1=1, one_turn=.true.)
    got = pwd_mat(bare, m, 20e-9_rp, 0.005_rp)
    zero = pwd_mat(bare, m, 0.0_rp, 0.005_rp)
    call rows('BARE', m)
    call rows('PWD', got)
    call rows('BEFORE', before)
    call rows('AFTER', after)
    call rows('ZERO', zero)
    call mat_make_unit(m)
    m = pwd_mat(bare, m, 20e-9_rp, 0.005_rp)
    call rows('IDENTITY', m)

    seed = [1e-4_rp, 0.0_rp, 2e-4_rp, 0.0_rp, 1e-3_rp, 0.0_rp]
    call reallocate_coord(orbit, inserted%n_ele_track)
    call init_coord(orbit(0), seed, inserted%ele(0), downstream_end$)
    call track_many(inserted, orbit, 0, 0, 1, track_state=state)
    if (state /= moving_forward$) error stop 'before tracking'
    write(*,'(a,6es25.16)') 'TRACK_BEFORE ', orbit(0)%vec
    call init_coord(orbit(1), seed, inserted%ele(1), downstream_end$)
    call track_many(inserted, orbit, 1, 1, 1, track_state=state)
    if (state /= moving_forward$) error stop 'after tracking'
    write(*,'(a,6es25.16)') 'TRACK_AFTER ', orbit(1)%vec
    seed(6) = m(6,5)*seed(5)
    call init_coord(orbit(1), seed, inserted%ele(1), downstream_end$)
    call track_many(inserted, orbit, 1, 1, 1, track_state=state)
    if (state /= moving_forward$) error stop 'conjugated tracking'
    write(*,'(a,6es25.16)') 'TRACK_CONJUGATED ', orbit(1)%vec

    mode%a%emittance = 1e-9_rp
    mode%b%emittance = 2e-11_rp
    mode%z%emittance = 1.25e-6_rp
    mode%a%tune = atan2(0.8_rp,0.6_rp)
    mode%b%tune = atan2(0.6_rp,0.8_rp)
    mode%z%tune = -acos(0.98_rp)
    call make_smat_from_abc(before, mode, sigma, err)
    if (err) error stop 'before sigma'
    call rows('SIGMA_BEFORE', sigma)
    call make_smat_from_abc(after, mode, sigma, err)
    if (err) error stop 'after sigma'
    call rows('SIGMA_AFTER', sigma)
    call make_smat_from_abc(got, mode, sigma, err)
    if (err) error stop 'pwd sigma'
    call rows('SIGMA_PWD', sigma)
    mode%sig_z = 0.005_rp
    mode%sigE_E = 0.00025_rp
    params%inductance = 20e-9_rp
    call bl_via_mat(bare, params, mode, sig_z)
    write(*,'(a,es25.16)') 'BUNCH_LENGTH ', sig_z
    contains
    subroutine rows(label, mat)
    character(*) label
    real(rp) mat(6,6)
    integer i
    do i=1,6
    write(*,'(a,6es25.16)') trim(label)//' ', mat(i,:)
    enddo
    end subroutine
    end program
    """

    SIGMA_Z = 0.005  # m
    SIGMA_P = 0.00025
    EMITTANCE_Z = SIGMA_Z * SIGMA_P


    def slope(sigma_z: float) -> float:
        # The ibs_ring manual's inductive defocusing coefficient for unit charge.
        # SI constants are exact; E is the numerical energy in eV, as in Bmad.
        return float(
            -20e-9 * 1e10 * 1.602176634e-19 * 299792458.0**2 / (np.sqrt(2 * np.pi) * sigma_z**3 * 3e9)
        )


    def kick(vector: NDArray[np.float64], strength: float) -> NDArray[np.float64]:
        result = vector.copy()
        result[5] += strength * result[4]
        return result


    def covariance(block: NDArray[np.float64]) -> NDArray[np.float64]:
        # For a stable symplectic 2x2 map, beta=B/sin(mu), alpha=(A-D)/(2 sin(mu)),
        # gamma=-C/sin(mu). Choose the negative phase so beta is positive here.
        cosine = float(np.trace(block) / 2)
        if abs(cosine) >= 1:
            raise RuntimeError("The longitudinal test map must be stable")
        sine = -np.sqrt(1 - cosine**2)
        beta = block[0, 1] / sine
        alpha = (block[0, 0] - block[1, 1]) / (2 * sine)
        gamma = -block[1, 0] / sine
        return EMITTANCE_Z * np.array([[beta, -alpha], [-alpha, gamma]])


    def require_close(actual: ArrayLike, expected: ArrayLike, *, atol: float = 2e-14) -> None:
        observed, target = np.asarray(actual, dtype=float), np.asarray(expected, dtype=float)
        if observed.shape != target.shape or not (
            np.isfinite(observed).all() and np.isfinite(target).all()
        ):
            raise RuntimeError("Control values must have matching shapes and be finite")
        np.testing.assert_allclose(observed, target, atol=atol, rtol=2e-14, equal_nan=False)


    def main() -> None:
        cosine = 0.98
        sine = -np.sqrt(1 - cosine**2)
        matrix = np.zeros((6, 6))
        matrix[:2, :2] = [[0.6, 8.0], [-0.08, 0.6]]
        matrix[2:4, 2:4] = [[0.8, 9.0], [-0.04, 0.8]]
        matrix[4:, 4:] = [
            [cosine, 20 * sine],
            [-sine / 20, cosine],
        ]
        strength = slope(SIGMA_Z)
        unit = np.eye(6)
        # Apply the scalar kick to each basis vector before/after the ring. This
        # oracle follows the documented physical sequence instead of pwd_mat.
        before = np.column_stack([matrix @ kick(e, strength) for e in unit])
        after = np.column_stack([kick(matrix @ e, strength) for e in unit])
        kick_matrix = np.column_stack([kick(e, strength) for e in unit])
        if np.max(np.abs(before - after)) < 0.01:
            raise RuntimeError("The test must distinguish the two multiplication orders")

        arc_terms = ", ".join(
            f"tt{i + 1}{j + 1}={matrix[i, j]:.17g}" for i, j in zip(*np.nonzero(matrix), strict=True)
        )
        kick_terms = ", ".join(f"tt{i}{i}=1" for i in range(1, 7))
        header = "parameter[particle]=electron\nparameter[e_tot]=3e9\nparameter[geometry]=closed\n"
        elements = (
            f"arc: taylor, l=100, {arc_terms}\nkick: taylor, {kick_terms}, tt65={strength:.17g}\n"
        )
        files = {
            "bare.bmad": header + elements + "ring: line=(arc)\nuse,ring\n",
            "inserted.bmad": header + elements + "ring: line=(kick,arc)\nuse,ring\n",
        }
        result = fortran(PROGRAM, files=files)

        def read(name: str, shape: tuple[int, ...]) -> NDArray[np.float64]:
            values = [
                float(value)
                for line in result.log.splitlines()
                if line.startswith(name + " ")
                for value in line.split()[1:]
            ]
            if len(values) != int(np.prod(shape)) or not np.isfinite(values).all():
                raise RuntimeError(f"Missing or invalid {name} records:\n{result.log}")
            return np.array(values).reshape(shape)

        got = read("PWD", (6, 6))
        require_close(read("BARE", (6, 6)), matrix)
        require_close(read("BEFORE", (6, 6)), before)
        require_close(read("AFTER", (6, 6)), after)
        require_close(read("ZERO", (6, 6)), matrix)
        require_close(read("IDENTITY", (6, 6)), kick_matrix)
        seed = np.array([1e-4, 0.0, 2e-4, 0.0, 1e-3, 0.0])
        track_before = read("TRACK_BEFORE", (6,))
        track_after = read("TRACK_AFTER", (6,))
        require_close(track_before, matrix @ kick(seed, strength), atol=2e-18)
        require_close(track_after, kick(matrix @ seed, strength), atol=2e-18)
        # The same physical initial state crosses K when moving the observation
        # point. Transforming it removes the apparent tracking disagreement.
        require_close(read("TRACK_CONJUGATED", (6,)), kick(track_before, strength), atol=2e-18)

        sigma_before = read("SIGMA_BEFORE", (6, 6))
        sigma_after = read("SIGMA_AFTER", (6, 6))
        sigma_got = read("SIGMA_PWD", (6, 6))
        require_close(sigma_before[4:, 4:], covariance(before[4:, 4:]), atol=1e-19)
        require_close(sigma_after[4:, 4:], covariance(after[4:, 4:]), atol=1e-19)
        require_close(sigma_after, kick_matrix @ sigma_before @ kick_matrix.T, atol=1e-19)
        require_close(sigma_before, before @ sigma_before @ before.T, atol=1e-19)
        require_close(sigma_after, after @ sigma_after @ after.T, atol=1e-19)
        require_close(sigma_got[4, 4], sigma_before[4, 4], atol=1e-19)
        require_close(np.trace(before[4:, 4:]), np.trace(after[4:, 4:]))
        print(
            "Controls: zero inductance, identity, two observation points, tracking and covariance PASS"
        )
        print(f"Kick slope [m^-1]: {strength:.16g}")

        defect = compare(
            "pwd_mat longitudinal block; documented kick then ring",
            got[4:, 4:],
            before[4:, 4:],
            atol=2e-14,
            rtol=2e-14,
        )
        # Identify order reversal as the cause. A future header-conforming result
        # also passes the controls and ends with NOT REPRODUCED.
        require_close(got, after if defect else before)
        particle_defect = compare(
            "Same input coordinates, final z [m]; explicit kick then ring",
            (got @ seed)[4],
            track_before[4],
            atol=2e-18,
            rtol=2e-14,
        )
        if particle_defect != defect:
            raise RuntimeError("Matrix and particle comparisons disagree")
        print(
            "Matched sigma_z [m], returned/before/after: "
            f"{np.sqrt(sigma_got[4, 4]):.16g} / "
            f"{np.sqrt(sigma_before[4, 4]):.16g} / {np.sqrt(sigma_after[4, 4]):.16g}"
        )

        # bl_via_mat sets epsilon_z = z*sigma_p and solves z=sqrt(epsilon_z*beta).
        # Hence z=beta(z)*sigma_p. The trace and B coefficient are identical for
        # MK and KM, so this scalar equation independently checks either order.
        def residual(z: float) -> float:
            trace_half = cosine + matrix[4, 5] * slope(z) / 2
            beta = matrix[4, 5] / -np.sqrt(1 - trace_half**2)
            return float(beta * SIGMA_P - z)

        lo, hi = SIGMA_Z, 1.5 * SIGMA_Z
        if not residual(lo) > 0 > residual(hi):
            raise RuntimeError("The analytic bunch-length root must be bracketed")
        for _ in range(60):
            mid = (lo + hi) / 2
            if residual(mid) > 0:
                lo = mid
            else:
                hi = mid
        expected_length = (lo + hi) / 2
        if abs(residual(expected_length)) > 2e-17:
            raise RuntimeError("Analytic bunch-length root did not converge")
        observed_length = read("BUNCH_LENGTH", (1,))[0]
        # The production super_brent call requests 1e-5 relative accuracy.
        if compare(
            "bl_via_mat sigma_z [m]; scalar fixed-point equation, both orders",
            observed_length,
            expected_length,
            atol=0,
            rtol=2e-5,
        ):
            raise RuntimeError("Bunch-length control failed")
        verdict(defect)


    if __name__ == "__main__":
        main()

    ```

## Provenance

| Role | Model | Effort | Date | Code examined |
|---|---|---|---|---|
| Found | `gpt-6.1-sol` | xhigh | 2026-11-05 | 20261031-0 (`c4c629fc9071`), source, production, GNU Fortran 15.3.0, CPU, double, serial, OpenMP=1 |
| Confirmed<br><small>matrix kick order and bunch-length invariance</small> | `gpt-6-astra` | xhigh | 2026-11-05 | 20261031-0 (`c4c629fc9071`), source, production, GNU Fortran 15.3.0, CPU, double, serial, OpenMP=1 |

