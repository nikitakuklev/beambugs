---
title: "OPT-43: Downstream tracking radiation kick omits the inverse half-element transport"
hide:
  - toc
---

# OPT-43 — Downstream tracking radiation kick omits the inverse half-element transport

| | |
|---|---|
| Code | [Bmad](../index.md) · Optics |
| Priority | P2 |
| Severity | Wrong physics |
| Evidence | Reproduced, independently rechecked |
| Last reproduced | 2026-11-06 on Bmad release [20261031-0](https://github.com/bmad-sim/bmad-ecosystem/tree/20261031-0) (`c4c629f`) |
| Tags | <span class="bb-tag">partly-source-only</span> [?](../../methodology.md#tags) |
| Source | [radiation_map_setup half maps, bmad/modules/radiation_mod.f90:222–246](https://github.com/bmad-sim/bmad-ecosystem/blob/c4c629fc9071d3a56cb1ba5bc714baba65e9c5d3/bmad/modules/radiation_mod.f90#L222-L246)<br>[rad1_damp_and_stoc_mats output basis, bmad/modules/rad_6d_mod.f90:573–579](https://github.com/bmad-sim/bmad-ecosystem/blob/c4c629fc9071d3a56cb1ba5bc714baba65e9c5d3/bmad/modules/rad_6d_mod.f90#L573-L579)<br>[tracking_rad_map_setup edge conversion, bmad/low_level/tracking_rad_map_setup.f90:54–67](https://github.com/bmad-sim/bmad-ecosystem/blob/c4c629fc9071d3a56cb1ba5bc714baba65e9c5d3/bmad/low_level/tracking_rad_map_setup.f90#L54-L67)<br>[track1_radiation, bmad/modules/radiation_mod.f90:105–114](https://github.com/bmad-sim/bmad-ecosystem/blob/c4c629fc9071d3a56cb1ba5bc714baba65e9c5d3/bmad/modules/radiation_mod.f90#L105-L114)<br>[track1 at the start edge, bmad/code/track1.f90:198–202](https://github.com/bmad-sim/bmad-ecosystem/blob/c4c629fc9071d3a56cb1ba5bc714baba65e9c5d3/bmad/code/track1.f90#L198-L202)<br>[at the end edge, bmad/code/track1.f90:316–320](https://github.com/bmad-sim/bmad-ecosystem/blob/c4c629fc9071d3a56cb1ba5bc714baba65e9c5d3/bmad/code/track1.f90#L316-L320)<br>[bmad/code/track1.f90:144–147](https://github.com/bmad-sim/bmad-ecosystem/blob/c4c629fc9071d3a56cb1ba5bc714baba65e9c5d3/bmad/code/track1.f90#L144-L147)<br>[transport map with radiation, bmad/doc/synchrotron-radiation.tex:116–150](https://github.com/bmad-sim/bmad-ecosystem/blob/c4c629fc9071d3a56cb1ba5bc714baba65e9c5d3/bmad/doc/synchrotron-radiation.tex#L116-L150) |

Tracking with radiation damping applies each element's downstream-half radiation kick at the element exit, using a matrix built for coordinates at the element's midpoint. At the exit the matrix must be multiplied on the right by the inverse transport of that half, as independent linear transport and quadrature of the half-element corrections confirm. For 3 GeV/c electrons in a 2 m bend with curvature 0.1 m⁻¹ and quadrupole strength k₁ = 0.3 m⁻², the kick's momentum response to a 1 mrad exit angle is zero instead of 66.03 eV/c. With exact half-element corrections, the tracked derivative of relative momentum deviation δ with respect to entrance angle becomes −5.175×10⁻⁵ instead of −4.178×10⁻⁵. Perpendicular faces do not avoid the error.

Run `python OPT-43.py` with [_bmad.py](../helpers/_bmad.py) in the same directory.

??? example "Reproducer: OPT-43.py"

    [Download OPT-43.py](OPT-43.py)

    ```python
    """OPT-43: Downstream tracking radiation kick omits the inverse half-element transport

    Tracking with radiation damping applies each element's downstream-half radiation kick at the element exit, using a matrix built for coordinates at the element's midpoint. At the exit the matrix must be multiplied on the right by the inverse transport of that half, as independent linear transport and quadrature of the half-element corrections confirm. For 3 GeV/c electrons in a 2 m bend with curvature 0.1 m⁻¹ and quadrupole strength k₁ = 0.3 m⁻², the kick's momentum response to a 1 mrad exit angle is zero instead of 66.03 eV/c. With exact half-element corrections, the tracked derivative of relative momentum deviation δ with respect to entrance angle becomes −5.175×10⁻⁵ instead of −4.178×10⁻⁵. Perpendicular faces do not avoid the error.

    The script links an inline client against the production build selected by BMAD_DIST,
    extracts the two half-element transport matrices and the stored upstream and downstream
    radiation kicks of a bend, and differentiates ordinary track1 at radiation scales +1 and -1.
    Closed-form linear transport with Gauss-Legendre quadrature independently rebuilds the
    production half-element correction, the exact correction and their composition rule;
    the upstream kick, the transport and the composition rule are passing controls.
    """

    from __future__ import annotations

    import os

    for _name in (
        "OMP_NUM_THREADS",
        "MKL_NUM_THREADS",
        "OPENBLAS_NUM_THREADS",
        "NUMEXPR_NUM_THREADS",
        "NPROC",
        "PJRT_NPROC",
    ):
        os.environ[_name] = "1"

    from collections.abc import Callable

    import numpy as np
    from _bmad import compare, fortran, verdict
    from scipy.linalg import expm

    PROGRAM = r"""program rad_basis_client
    use bmad
    use radiation_mod, only: radiation_map_setup
    implicit none
    type(lat_struct), target :: lat
    type(ele_struct), pointer :: ele
    type(ele_struct), target :: half1, half2
    type(coord_struct) c0, c1, c2, start, finish
    real(rp) gamma0, eps, jac(6,6,2)
    integer i, j, k, is
    logical err

    call bmad_parser('case.bmad', lat, err_flag=err)
    if (err) error stop 'parse'
    ele => lat%ele(1)
    call convert_pc_to(ele%value(p0c$), ele%ref_species, gamma=gamma0)
    write(*, '(a,3(1x,es25.16))') 'PARAM', classical_radius(ele%ref_species), gamma0, ele%value(p0c$)
    bmad_com%radiation_damping_on = .false.
    bmad_com%radiation_fluctuations_on = .false.
    call init_coord(c0, [0.0_rp, 0.0_rp, 0.0_rp, 0.0_rp, 0.0_rp, 0.0_rp], ele, upstream_end$)
    call make_mat6(ele, lat%param, c0, c2, err)
    if (err) error stop 'full matrix'
    ! The two half elements exactly as radiation_map_setup builds them.
    call create_element_slice(half1, ele, 0.5_rp*ele%value(l$), 0.0_rp, lat%param, .true., .false., err, &
                              pointer_to_next_ele(ele, -1))
    if (err) error stop 'slice 1'
    call make_mat6(half1, lat%param, c0, c1, err)
    if (err) error stop 'half matrix 1'
    call create_element_slice(half2, ele, 0.5_rp*ele%value(l$), 0.5_rp*ele%value(l$), lat%param, .false., .true., &
                              err, half1)
    if (err) error stop 'slice 2'
    call make_mat6(half2, lat%param, c1, c2, err)
    if (err) error stop 'half matrix 2'
    call radiation_map_setup(ele, err)
    if (err) error stop 'radiation map'
    do j = 1, 6
      write(*, '(a,i0,6(1x,es25.16))') 'T', j, ele%mat6(j,:)
      write(*, '(a,i0,6(1x,es25.16))') 'H', j, half1%mat6(j,:)
      write(*, '(a,i0,6(1x,es25.16))') 'G', j, half2%mat6(j,:)
      write(*, '(a,i0,6(1x,es25.16))') 'U', j, ele%rad_map%rm0%damp_dmat(j,:)
      write(*, '(a,i0,6(1x,es25.16))') 'D', j, ele%rad_map%rm1%damp_dmat(j,:)
    enddo
    ! Central differences of ordinary track1 at radiation scales +1 (P) and -1 (N), zero mean loss.
    bmad_com%radiation_damping_on = .true.
    bmad_com%radiation_zero_average = .true.
    eps = 1e-6_rp
    jac = 0
    do is = 1, 2
      bmad_com%synch_rad_scale = 3 - 2*is
      do k = 1, 6
        do i = 1, 2
          call init_coord(start, [0.0_rp, 0.0_rp, 0.0_rp, 0.0_rp, 0.0_rp, 0.0_rp], ele, upstream_end$)
          start%vec(k) = (3 - 2*i) * eps
          call track1(start, ele, lat%param, finish, err_flag=err)
          if (err .or. finish%state /= alive$) error stop 'track'
          jac(:,k,is) = jac(:,k,is) + (3 - 2*i) * finish%vec / (2*eps)
        enddo
      enddo
    enddo
    do j = 1, 6
      write(*, '(a,i0,6(1x,es25.16))') 'P', j, jac(j,:,1)
      write(*, '(a,i0,6(1x,es25.16))') 'N', j, jac(j,:,2)
    enddo
    write(*, '(a)') 'CLIENT COMPLETED'
    end program
    """

    CASES = (
        ("combined", 2.0, 0.1, 0.3),
        ("sector", 2.0, 0.1, 0.0),
        ("negative", 2.0, -0.1, -0.2),
    )
    COLUMNS = [
        0,
        1,
        5,
    ]  # x, px, delta columns of the delta row; the other δ-row entries vanish


    def lattice(length: float, g: float, k1: float) -> str:
        return f"""no_digested
    parameter[particle] = electron
    parameter[p0c] = 3e9
    parameter[geometry] = open
    b: sbend, l={length:.17g}, g={g:.17g}, k1={k1:.17g}, e1=0, e2=0, &
       tracking_method=bmad_standard, mat6_calc_method=auto
    beamline: line = (b)
    use, beamline
    """


    def execute(lat: str) -> dict[str, np.ndarray]:
        """One fresh native process; complete 6x6 records or an exception."""
        log = fortran(PROGRAM, files={"case.bmad": lat}).log
        if log.splitlines().count("CLIENT COMPLETED") != 1:
            raise RuntimeError(f"Missing native completion marker:\n{log}")
        rows: dict[str, list[list[float]]] = {}
        for line in log.splitlines():
            words = line.split()
            if words and (words[0] == "PARAM" or (len(words[0]) == 2 and words[0][0] in "THGUDPN")):
                rows.setdefault(words[0][0] if words[0] != "PARAM" else "PARAM", []).append(
                    [float(w) for w in words[1:]]
                )
        out = {k: np.array(v) for k, v in rows.items()}
        for key in "THGUDPN":
            if out.get(key, np.empty(0)).shape != (6, 6):
                raise RuntimeError(f"Incomplete {key} record:\n{log}")
        if out["PARAM"].shape != (1, 3) or not all(np.isfinite(v).all() for v in out.values()):
            raise RuntimeError(f"Invalid native records:\n{log}")
        return out


    def generator(g: float, k1: float, gamma0: float) -> np.ndarray:
        """Linear sector-bend generator in (x, px, y, py, z, delta) on the centered reference."""
        a = np.zeros((6, 6))
        a[0, 1] = a[2, 3] = 1
        a[1, 0] = -(g * g + k1)
        a[1, 5] = g
        a[3, 2] = k1
        a[4, 0] = -g
        a[4, 5] = 1 / gamma0**2
        return a


    def local_damping(g: float, k1: float, kd: float) -> np.ndarray:
        """The manual's local damping matrix d on the centered orbit (px = py = delta = 0).

        The particle curvature is g_x = g + k1 x, and the path-length factor (1 + g x) adds
        g^3 to dg^2/dx, so d61 = -k_d (g^3 + 2 g k1); k_d = (2/3) r_e gamma0^3.
        """
        d = np.zeros((6, 6))
        d[1, 1] = d[3, 3] = -kd * g * g
        d[5, 0] = -kd * (g**3 + 2 * g * k1)
        d[5, 5] = -2 * kd * g * g
        return d


    def integral(fun: Callable[[float], np.ndarray], length: float, n: int = 40) -> np.ndarray:
        """Gauss-Legendre quadrature of a smooth matrix function on [0, length]."""
        nodes, weights = np.polynomial.legendre.leggauss(n)
        return (
            0.5
            * length
            * sum(w * fun(0.5 * length * (x + 1)) for x, w in zip(nodes, weights, strict=True))
        )


    def control(label: str, observed: object, expected: object, atol: float) -> None:
        if compare(label, observed, expected, atol=atol):
            raise RuntimeError(f"Control failed: {label}")


    def check(name: str, length: float, g: float, k1: float) -> bool:
        """One bend in a fresh native process; True when a defect comparison fails."""
        defect = False
        r = execute(lattice(length, g, k1))
        radius, gamma0, p0c = r["PARAM"][0]
        kd = 2 / 3 * radius * gamma0**3
        a, d, half = generator(g, k1, gamma0), local_damping(g, k1, kd), length / 2
        h, t = expm(a * half), expm(a * length)
        h_inv = np.linalg.inv(h)
        # The transfer correction W of a half element maps its entrance deviation to the
        # change at its exit. The production integrand is H T(s)^-1 d(s), without the
        # forward transport T(s) of the manual's integrand H T(s)^-1 d(s) T(s).
        w_prod = h @ integral(lambda s: expm(-a * s) @ d, half)
        w_exact = h @ integral(lambda s: expm(-a * s) @ d @ expm(a * s), half)
        z_exact = integral(lambda s: expm(a * (length - s)) @ d @ expm(a * s), length)
        # Upstream kick applied before H: H (I + U) = H + W gives U = H^-1 W.
        # Downstream kick applied after H: (I + D) H = H + W gives D = W H^-1.
        # Two halves compose as Z = H W1 + W2 H, the manual's Z for the whole element.
        u_req, d_req = h_inv @ w_prod, w_prod @ h_inv
        print(f"{name}: L={length} m, g={g} m^-1, k1={k1} m^-2, p0c={p0c:.6g} eV, k_d={float(kd)!r} m")
        control(
            f"{name} radiation-free transport max error (H1, H2, T vs exp(A s))",
            [np.abs(r["H"] - h).max(), np.abs(r["G"] - h).max(), np.abs(r["T"] - t).max()],
            [0.0, 0.0, 0.0],
            atol=2e-15,
        )
        control(
            f"{name} composition rule max error (H W1 + W2 H vs exact full integral)",
            np.abs(h @ w_exact + w_exact @ h - z_exact).max(),
            0.0,
            atol=1e-19,
        )
        control(
            f"{name} upstream kick max error (stored U vs H^-1 W)",
            np.abs(r["U"] - u_req).max(),
            0.0,
            atol=1e-12,
        )
        control(
            f"{name} stored downstream kick max error vs entrance-basis W (identifies stored D = W)",
            np.abs(r["D"] - w_prod).max(),
            0.0,
            atol=1e-12,
        )
        defect |= compare(
            f"{name} downstream kick delta row (x,px,delta) (required D = W H^-1)",
            r["D"][5, COLUMNS],
            d_req[5, COLUMNS],
            atol=1e-12,
        )
        print(
            f"{name} downstream kick energy change for a 1 mrad exit angle / eV/c: "
            f"observed={float(p0c * 1e-3 * r['D'][5, 1])!r} expected={float(p0c * 1e-3 * d_req[5, 1])!r}"
        )
        # The tracked map is (I + sD) T (I + sU); its part odd in the scale s is T U + D T.
        odd = (r["P"] - r["N"]) / 2
        control(
            f"{name} track1 delta row odd in radiation scale vs T U + D T (stored kicks)",
            odd[5, COLUMNS],
            (t @ r["U"] + r["D"] @ t)[5, COLUMNS],
            atol=1e-15,
        )
        defect |= compare(
            f"{name} track1 delta row (x,px,delta) (required H W1 + W2 H)",
            odd[5, COLUMNS],
            (h @ w_prod + w_prod @ h)[5, COLUMNS],
            atol=1e-12,
        )
        print(
            f"{name} with exact half integrals, Z62 with this kick basis = "
            f"{float((h @ w_exact + w_exact @ t)[5, 1])!r}; exact Z62 = {float(z_exact[5, 1])!r}"
        )
        return defect


    def main() -> None:
        defect = False
        for case in CASES:
            defect |= check(*case)
        verdict(defect)


    if __name__ == "__main__":
        main()

    ```

## Provenance

| Role | Model | Effort | Date | Code examined |
|---|---|---|---|---|
| Found | `claude-opus-5-5` | high | 2026-11-06 | 20261031-0 (`c4c629fc9071d3a56cb1ba5bc714baba65e9c5d3`) |
| Confirmed | `gpt-6.1-sol` | xhigh | 2026-11-06 | 20261031-0 (`c4c629fc9071d3a56cb1ba5bc714baba65e9c5d3`), source, production, GNU Fortran 15.3.0, CPU, double, OpenMP one thread, MPI disabled |
| Confirmed<br><small>bmad_standard stored-kick contract and track1 composition with a closed-form oracle</small> | `claude-opus-5-5` | high | 2026-11-06 | 20261031-0 (`c4c629fc9071d3a56cb1ba5bc714baba65e9c5d3`), source, production, GNU Fortran 15.3.0, CPU, double, OpenMP one thread, MPI disabled |

