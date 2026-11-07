"""OPT-42: Bend exit-face radiation damping uses the entrance orbit and transport

Computing a radiation damping map for a bend with an inclined downstream face applies that face's loss derivative at the bend entrance instead of the exit. The map therefore drops the dependence of the face's loss on horizontal motion accumulated through the bend, and the result depends on how finely the bend is subdivided. For 3 GeV/c electrons in a 1 m bend of curvature 0.2 m⁻¹ with exit-face angle 0.1 rad, the face reduces the momentum loss of a particle entering at a 1 mrad angle by 4.547 eV/c, but the map gives zero; 16 subdivisions give 4.266 eV/c. Independent transport and Lorentz-force integrations place this contribution at the physical exit. Ordinary tracking with radiation damping gives the correct value.

The script links inline clients against the production build selected by BMAD_DIST
and isolates exit-face actions by subtracting bends with identical bodies.
Independent variational and Cartesian Lorentz integrations locate the physical
face contribution, with entrance-face and ordinary-tracking controls.
It also compares full and half-interval curvature integrals for additivity.
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

import numpy as np
from _bmad import compare, fortran, verdict
from scipy.integrate import solve_ivp

PROGRAM = r"""program b3_client
use bmad
use rad_6d_mod
use radiation_mod, only: radiation_map_setup
implicit none
type(lat_struct), target :: lat
type(ele_struct), pointer :: ele
type(ele_struct), target :: piece
type(coord_struct) start, finish, reference
type(rad_map_struct) rm
type(normal_modes_struct) modes
type(coord_struct) trajectory(0:1)
real(rp) m(6,6), d(6,6), v(6), c(6,6), kd, gamma0, px, scale, gi(2), g2, g3
integer n, i, j, where
logical err, opening
character(200) action, arg

call get_command_argument(1, action)
call get_command_argument(2, arg)
read(arg, *) n
call get_command_argument(3, arg)
read(arg, *) px
call get_command_argument(4, arg)
read(arg, *) scale
call get_command_argument(5, arg)
read(arg, *) opening
call bmad_parser('case.bmad', lat, err_flag=err)
if (err) error stop 'parse'
ele => lat%ele(1)
call convert_pc_to(ele%value(p0c$), ele%ref_species, gamma=gamma0)
kd = (2.0_rp/3.0_rp) * classical_radius(ele%ref_species) * gamma0**3
write(*, '(a,3(1x,es25.16))') 'PARAM', kd, ele%value(p0c$), mass_of(ele%ref_species)
write(*, '(a,3(1x,i0))') 'CONFIG', ele%orientation, ele%tracking_method, ele%mat6_calc_method
bmad_com%radiation_damping_on = .false.
bmad_com%radiation_fluctuations_on = .false.
call init_coord(start, [0.0_rp, px, 0.0_rp, 0.0_rp, 0.0_rp, 0.0_rp], ele, upstream_end$)
reference = start
select case (trim(action))
case ('direct')
  call mat_make_unit(m)
  d = 0; v = 0; c = 0
  do i = 1, n
    call element_slice_iterator(ele, lat%param, i, n, piece)
    call make_mat6(piece, lat%param, start, finish, err)
  if (err) error stop 'matrix'
    if (finish%state /= alive$) stop 11
    err = .false.
    call rad1_damp_and_stoc_mats(piece, opening, start, finish, rm, 1e-12_rp, 1e-12_rp, err)
    if (err) stop 12
    d = matmul(piece%mat6, d) + matmul(rm%damp_dmat, m)
    v = matmul(piece%mat6, v) + rm%xfer_damp_vec
    c = matmul(matmul(piece%mat6, c), transpose(piece%mat6)) + rm%stoc_mat
    m = matmul(piece%mat6, m)
    start = finish
  enddo
  do j = 1, 6
    write(*, '(a,i0,6(1x,es25.16))') 'M', j, m(j,:)
    write(*, '(a,i0,6(1x,es25.16))') 'D', j, d(j,:)
    write(*, '(a,i0,6(1x,es25.16))') 'C', j, c(j,:)
  enddo
  write(*, '(a,6(1x,es25.16))') 'V', v
case ('track')
  ! Seed the production cached maps on a centered trajectory before applying radiation.
  call init_coord(reference, [0.0_rp, 0.0_rp, 0.0_rp, 0.0_rp, 0.0_rp, 0.0_rp], ele, upstream_end$)
  call make_mat6(ele, lat%param, reference, finish, err)
  if (err) error stop 'matrix'
  call radiation_map_setup(ele, err, reference)
  if (err) stop 13
  do j = 1, 6
    write(*, '(a,i0,6(1x,es25.16))') 'K', j, ele%rad_map%rm1%damp_dmat(j,:)
  enddo
  bmad_com%radiation_damping_on = (scale /= 0)
  bmad_com%radiation_zero_average = .true.
  bmad_com%synch_rad_scale = scale
  call track1(start, ele, lat%param, finish, err_flag=err)
  if (err .or. finish%state /= alive$) stop 14
  start = finish
case ('branch')
  call make_mat6(ele, lat%param, start, finish, err)
  if (err) error stop 'matrix'
  trajectory = [start, finish]
  call rad_damp_and_stoc_mats(lat%ele(0), lat%ele(1), opening, rm, modes, m, err, trajectory)
  if (err) stop 17
  do j = 1, 6
    write(*, '(a,i0,6(1x,es25.16))') 'M', j, m(j,:)
    write(*, '(a,i0,6(1x,es25.16))') 'D', j, rm%damp_dmat(j,:)
  enddo
  start = finish
case ('gints')
  call make_mat6(ele, lat%param, start, finish, err)
  if (err) error stop 'matrix'
  if (finish%state /= alive$) stop 15
  do where = 1, 3
    select case (where)
    case (1)
      j = all$
    case (2)
      j = upstream$
    case (3)
      j = downstream$
    end select
    call rad_g_integrals(ele, j, start, finish, gi, g2, g3, 1e-12_rp, 1e-12_rp, 1e-12_rp)
    write(*, '(a,i0,4(1x,es25.16))') 'G', where, gi, g2, g3
  enddo
  start = finish
case default
  stop 16
end select
write(*, '(a,6(1x,es25.16))') 'ORBIT', start%vec
write(*, '(a)') 'CLIENT COMPLETED'
end program
"""


def lattice(
    length: float,
    g: float,
    edge: float,
    *,
    entrance: bool = False,
    reverse: bool = False,
    method: str = "bmad_standard",
    k1: float = 0,
) -> str:
    e1, e2 = (edge, 0.0) if entrance != reverse else (0.0, edge)
    return f"""no_digested
parameter[particle] = electron
parameter[p0c] = 3e9
parameter[geometry] = open
parameter[default_tracking_species] = {"positron" if reverse else "electron"}
b: sbend, l={length:.17g}, g={g:.17g}, k1={k1:.17g}, e1={e1:.17g}, e2={e2:.17g}, &
   fringe_type=full, fringe_at=both_ends, tracking_method={method}, &
   mat6_calc_method=auto, ds_step=0.002
beamline: line = ({"--" if reverse else ""}b)
use, beamline
"""


def execute(
    lat: str,
    *,
    action: str = "direct",
    n: int = 1,
    px: float = 0,
    scale: float = 0,
    opening: bool = False,
) -> dict[str, np.ndarray]:
    """One isolated native simulation, with complete, finite output records."""
    log = fortran(
        PROGRAM,
        [action, str(n), str(px), str(scale), str(opening)],
        files={"case.bmad": lat},
    ).log
    if log.splitlines().count("CLIENT COMPLETED") != 1:
        raise RuntimeError(f"Missing native completion marker:\n{log}")
    shapes = {"PARAM": 3, "CONFIG": 3, "ORBIT": 6}
    letters = {"direct": "MDC", "branch": "MD", "track": "K", "gints": "G"}[action]
    shapes.update(
        {
            f"{letter}{i}": 4 if letter == "G" else 6
            for letter in letters
            for i in range(1, 4 if letter == "G" else 7)
        }
    )
    if action == "direct":
        shapes["V"] = 6
    records = {}
    for line in log.splitlines():
        words = line.split()
        if not words or words[0] not in shapes:
            continue
        key = words[0]
        if key in records or len(words) != shapes[key] + 1:
            raise RuntimeError(f"Malformed or duplicate record: {line}")
        records[key] = np.array([float(s) for s in words[1:]])
    if set(records) != set(shapes) or not all(np.isfinite(v).all() for v in records.values()):
        raise RuntimeError(f"Incomplete or nonfinite native output:\n{log}")
    if records["PARAM"][0] <= 0:
        raise RuntimeError("Invalid radiation coefficient")
    for letter in letters:
        if letter != "G":
            records[letter] = np.array([records[f"{letter}{i}"] for i in range(1, 7)])
    return records


def control(label: str, observed, expected, *, atol: float) -> None:
    if compare(label, observed, expected, atol=atol):
        raise RuntimeError(f"Failed control: {label}")


def transport(s: float, g: float, k1: float = 0) -> np.ndarray:
    # Hamilton's linear equations in Frenet coordinates, normalized to P0.
    # z=beta*c*(t_ref-t), so horizontal displacement contributes z'=-g*x.
    mu = 510998.95 / 3e9
    a = np.zeros((6, 6))
    a[0, 1] = a[2, 3] = 1
    a[1, 0] = -(g * g + k1)
    a[1, 5] = g
    a[3, 2] = k1
    a[4, 0] = -g
    a[4, 5] = mu * mu / (1 + mu * mu)
    if s == 0:
        return np.eye(6)
    result = solve_ivp(
        lambda _, flat: (a @ flat.reshape(6, 6)).ravel(),
        (0, s),
        np.eye(6).ravel(),
        method="DOP853",
        rtol=3e-14,
        atol=1e-15,
    )
    if not result.success:
        raise RuntimeError(result.message)
    return result.y[:, -1].reshape(6, 6)


def edge_matrix(g: float, edge: float) -> np.ndarray:
    f = np.eye(6)
    f[1, 0] = g * np.tan(edge)
    f[3, 2] = -g * np.tan(edge)
    return f


def lorentz_face_derivative(length: float, g: float, edge: float, h: float) -> float:
    # Entirely Cartesian Lorentz motion parameterized by physical path length.
    # No transfer map or analytical circle/plane intersection is used.
    # X is outward at the entrance; the tangent rotates toward negative X.
    theta = g * length
    endpoint = np.array([(np.cos(theta) - 1) / g, np.sin(theta) / g])
    normal = np.array([np.cos(theta), np.sin(theta)])
    tangent = np.array([-np.sin(theta), np.cos(theta)])

    def path(px: float, face: float) -> float:
        def rhs(_, u):
            return [u[2], u[3], -g * u[3], g * u[2]]

        def event(_, u):
            delta = u[:2] - endpoint
            return delta @ tangent + np.tan(face) * (delta @ normal)

        # SciPy's event protocol reads these attributes from the callable.
        event.terminal = True  # ty: ignore[unresolved-attribute]
        event.direction = 1  # ty: ignore[unresolved-attribute]
        result = solve_ivp(
            rhs,
            (0, length * 1.2),
            [0, 0, px, np.sqrt(1 - px * px)],
            events=event,
            method="DOP853",
            rtol=3e-14,
            atol=1e-15,
            max_step=length / 16,
        )
        if not result.success or len(result.t_events[0]) != 1:
            raise RuntimeError("physical face intersection failed")
        return result.t_events[0][0]

    def delta(px: float) -> float:
        return -g * g * (path(px, edge) - path(px, 0))

    return (delta(h) - delta(-h)) / (2 * h)


def main() -> None:
    # Exclude the separate vertical-column defect; all expansion orbits are centered.
    cols, rows = [0, 1, 5], [0, 1, 4, 5]
    projection = np.ix_(rows, cols)
    defect = False
    transport_errors, endpoint_errors = [], []
    for label, length, g, edge, k1 in (
        ("base", 1.0, 0.2, 0.1, 0.0),
        ("negative", 1.7, -0.16, -0.13, 0.0),
        ("combined", 0.8, 0.23, 0.17, 0.37),
    ):
        t = transport(length, g, k1)
        e = np.zeros((6, 6))
        # Positive exit inclination removes path: Δδ/kd = g² tan(e2) x_exit.
        e[5, 0] = g * g * np.tan(edge)
        expected = e @ t
        for n in (1, 3, 16):
            a = execute(lattice(length, g, 0, k1=k1), n=n)
            b = execute(lattice(length, g, edge, k1=k1), n=n)
            kd = float(a["PARAM"][0])
            f = edge_matrix(g, edge)
            # Pull back the exit fringe so identical body contributions cancel.
            # Composition is first order: D_BA = T_B D_A + D_B T_A.
            measured = (np.linalg.solve(f, b["D"]) - a["D"]) / kd
            s = length * (1 - 1 / n)
            misplaced = transport(length / n, g, k1) @ e @ transport(s, g, k1)
            transport_errors.extend(
                [
                    float(np.max(np.abs((a["M"] - t)[projection]))),
                    float(np.max(np.abs((b["M"] - f @ t)[projection]))),
                ]
            )
            if max(np.max(np.abs(a["ORBIT"])), np.max(np.abs(b["ORBIT"]))) > 2e-14:
                raise RuntimeError("Reference orbit is not centered")
            endpoint_errors.append(float(np.max(np.abs((measured - misplaced)[projection]))))
            defect |= compare(
                f"{label} N={n} face row6/kd (x,px,delta; E T(L))",
                measured[5, cols],
                expected[5, cols],
                atol=5e-13,
            )
            # pc=3 GeV and normalized entrance px=0.001 give cΔP in eV.
            defect |= compare(
                f"{label} N={n} face cDeltaP/eV (E T(L), px=0.001)",
                measured[5, 1] * kd * 3e6,
                expected[5, 1] * kd * 3e6,
                atol=6e-10,
            )
        if k1 == 0:
            d1 = lorentz_face_derivative(length, g, edge, 1e-4)
            d2 = lorentz_face_derivative(length, g, edge, 5e-5)
            control(
                f"{label} Lorentz path derivative/m^-1 (variational E T(L))",
                (4 * d2 - d1) / 3,
                expected[5, 1],
                atol=4e-12,
            )
    control(
        "radiation-free transport max error (variational equations)",
        max(transport_errors),
        0,
        atol=8e-14,
    )
    print(
        f"misplaced action T(L/N) E T(L-L/N): max projected residual={max(endpoint_errors):.17g}; "
        "physical action=E T(L)"
    )

    a = execute(lattice(1, 0.2, 0))
    b = execute(lattice(1, 0.2, 0.1, entrance=True))
    f = edge_matrix(0.2, 0.1)
    face = (b["D"] @ np.linalg.inv(f) - a["D"]) / a["PARAM"][0]
    e = np.zeros((6, 6))
    e[5, 0] = 0.2**2 * np.tan(0.1)
    control(
        "entrance face projected max error/kd (T(L) E)",
        np.max(np.abs((face - a["M"] @ e)[projection])),
        0,
        atol=5e-13,
    )

    a = execute(lattice(1, 0.2, 0), opening=True)
    b = execute(lattice(1, 0.2, 0.1), opening=True)
    expected = (e @ transport(1, 0.2))[5, 1]
    defect |= compare(
        "opening-angle option face D62/kd (E T(L))",
        (b["D"][5, 1] - a["D"][5, 1]) / a["PARAM"][0],
        expected,
        atol=5e-13,
    )
    a = execute(lattice(1, 0.2, 0), action="branch")
    b = execute(lattice(1, 0.2, 0.1), action="branch")
    defect |= compare(
        "public optics face D62/kd (E T(L))",
        (b["D"][5, 1] - a["D"][5, 1]) / a["PARAM"][0],
        expected,
        atol=5e-13,
    )

    for reverse in (False, True):
        for method in ("bmad_standard", "fixed_step_runge_kutta", "symp_lie_ptc"):
            a = execute(lattice(1, 0.2, 0, reverse=reverse, method=method))
            b = execute(lattice(1, 0.2, 0.1, reverse=reverse, method=method))
            if b["CONFIG"][0] != (-1 if reverse else 1) or np.max(np.abs(b["ORBIT"])) > 2e-14:
                raise RuntimeError("Orientation or reference-orbit control failed")
            defect |= compare(
                f"{method} orientation={-1 if reverse else 1} D62/kd (E T(L))",
                (b["D"][5, 1] - a["D"][5, 1]) / a["PARAM"][0],
                expected,
                atol=5e-13,
            )

    # Actual ordinary tracking applies the downstream cache at the output deviation.
    # Shrink deterministic radiation to suppress feedback; antisymmetrize in px.
    for method in ("bmad_standard", "fixed_step_runge_kutta", "symp_lie_ptc"):
        derivatives = []
        scale, h = 0.001, 1e-4
        for edge in (0.0, 0.1):
            points = [
                execute(
                    lattice(1, 0.2, edge, method=method), action="track", px=sign * h, scale=scale
                )["ORBIT"]
                for sign in (-1, 1)
            ]
            derivatives.append((points[1] - points[0]) / (2 * h))
        response = (np.linalg.solve(f, derivatives[1]) - derivatives[0]) / scale
        control(
            f"track1 {method} face cDeltaP/eV (Lorentz/transport, px=0.001)",
            response[5] * 3e6,
            expected * a["PARAM"][0] * 3e6,
            atol=1.2e-8,
        )

    integrals = [execute(lattice(1, 0.2, edge), action="gints", px=0.001) for edge in (0, 0.1)]
    a, b = integrals
    control(
        "perpendicular curvature partition (all=upstream+downstream)",
        a["G1"] - a["G2"] - a["G3"],
        np.zeros(4),
        atol=4e-15,
    )
    defect |= compare(
        "inclined curvature partition residual (interval additivity)",
        b["G1"] - b["G2"] - b["G3"],
        np.zeros(4),
        atol=4e-15,
    )
    edge_all, edge_down = b["G1"] - a["G1"], b["G3"] - a["G3"]
    defect |= compare(
        "exit-only curvature integrals all (same physical exit as downstream)",
        edge_all,
        edge_down,
        atol=4e-15,
    )
    verdict(defect)


if __name__ == "__main__":
    main()
