"""OPT-38: Inclined bend faces introduce a forbidden vertical radiation-damping column

Tracking with radiation damping through an untilted dipole with inclined faces breaks vertical reflection symmetry. The face damping matrix assigns a horizontal field-wedge derivative to the vertical-position column, making loss depend linearly on vertical displacement. For 3 GeV/c electrons in a 1 m bend of curvature 0.2 m⁻¹ with 0.1 rad faces, y=±1 mm gives a loss difference of 18.12 eV instead of zero. The error survives subdivision and enters six-dimensional equilibrium calculations; in a coupled 32-bend ring it makes the horizontal normal-mode emittance 12.69% low relative to the same calculation with reflection symmetry restored.

The script evaluates native bend-face maps and ordinary tracking, checking exact vertical reflection at three subdivisions and a perpendicular-face control. It independently composes a planar and a coupled ring, removes only forbidden bend damping blocks and solves their covariance equations with SciPy.
"""

import numpy as np
from _bend_radiation import difference, lattice, local
from _bmad import compare, fortran, verdict
from scipy.linalg import solve_discrete_lyapunov

TRACKING_PROGRAM = r"""program tracking_probe
use bmad
implicit none
type(lat_struct),target :: lat
type(coord_struct) p,q
integer j,iz
logical err
real(rp) v(6)
call bmad_parser('lat.bmad',lat,err_flag=err)
if(err) error stop 'parse'
bmad_com%radiation_fluctuations_on=.false.
bmad_com%radiation_damping_on=.true.
bmad_com%synch_rad_scale=.001_rp
do iz=0,1
bmad_com%radiation_zero_average=(iz==1)
do j=-1,1,2
v=0;v(3)=j*.001_rp
call init_coord(p,v,lat%ele(1),upstream_end$)
call track1(p,lat%ele(1),lat%param,q,err_flag=err)
if(err.or.q%state/=alive$) error stop 'track'
write(*,'(a,2i4,es26.17)') 'LIVE ',iz,j,q%vec(6)
enddo
enddo
end program
"""

RING_PROGRAM = r"""program ring_probe
use bmad
use rad_6d_mod
implicit none
type(lat_struct),target :: lat
type(coord_struct),allocatable,target :: orbit(:)
type(normal_modes_struct) mode
type(rad_map_struct) rm, er
real(rp) nodamp(6,6), sigma(6,6)
integer stat, opening, i
logical err
call bmad_parser('lat.bmad',lat,err_flag=err)
if(err) error stop 'parse'
bmad_com%radiation_damping_on=.false.
bmad_com%radiation_fluctuations_on=.false.
call twiss_and_track(lat,orbit,stat)
if(stat/=ok$) error stop 'twiss'
write(*,'(a,4es26.17)') 'ORBIT ',maxval(abs(orbit%vec(1))),maxval(abs(orbit%vec(3))),lat%ele(lat%n_ele_track)%a%phi,lat%ele(lat%n_ele_track)%b%phi
do opening=0,1
do i=1,lat%n_ele_track
call rad1_damp_and_stoc_mats(lat%ele(i),opening==1,orbit(i-1),orbit(i),er,1e-12_rp,1e-12_rp,err)
if(err) error stop 'element rad'
write(*,'(a,3i6)') 'ELEMENT ',opening,i,lat%ele(i)%key
call dump('E'//int_str(i)//'D',opening,er%damp_dmat)
call dump('E'//int_str(i)//'Q',opening,er%stoc_mat)
call dump('E'//int_str(i)//'T',opening,lat%ele(i)%mat6)
enddo
call rad_damp_and_stoc_mats(lat%ele(0),lat%ele(0),opening==1,rm,mode,nodamp,err,orbit)
if(err) error stop 'ring rad'
write(*,'(a,i2,es26.17)') 'ENERGY ',opening,mode%dpz_damp
call dump('M',opening,rm%xfer_damp_mat)
call dump('D',opening,rm%damp_dmat)
call dump('Q',opening,rm%stoc_mat)
call dump('T',opening,nodamp)
call emit_6d(lat%ele(0),opening==1,mode,sigma,orbit)
call dump('S',opening,sigma)
if(opening==1) then
write(*,'(a,i2,3es26.17)') 'EMIT ',opening,mode%a%emittance,mode%b%emittance,mode%z%emittance
else
write(*,'(a,i2,3es26.17)') 'EMIT ',opening,mode%a%emittance_no_vert,mode%b%emittance_no_vert,mode%z%emittance_no_vert
endif
enddo
contains
subroutine dump(label,opening,matrix)
character(*) label
integer opening,i
real(rp) matrix(6,6)
do i=1,6
write(*,'(a,1x,a,2i3,6es26.17)') 'MATRIX',label,opening,i,matrix(i,:)
enddo
end subroutine
end program
"""


def tracking(frozen: float) -> bool:
    defect = False
    for method in ["bmad_standard", "fixed_step_runge_kutta", "symp_lie_ptc"]:
        log = fortran(TRACKING_PROGRAM, files={"lat.bmad": lattice(0.1, method)}).log
        rows = {}
        for line in log.splitlines():
            t = line.split()
            if t and t[0] == "LIVE":
                key = (int(t[1]), int(t[2]))
                if len(t) != 4 or key in rows:
                    raise RuntimeError(f"Malformed live record: {line}")
                rows[key] = float(t[3])
        if set(rows) != {(z, s) for z in [0, 1] for s in [-1, 1]}:
            raise RuntimeError(f"Incomplete tracking output:\n{log}")
        for zero in [0, 1]:
            loss = -(rows[zero, 1] - rows[zero, -1]) * 3e9 / 0.001
            # The small radiation scale suppresses orbit/energy feedback. The
            # frozen action and ordinary tracking agree within 1.5e-7 eV on pin.
            np.testing.assert_allclose(loss, frozen, rtol=0, atol=2e-6)
            defect |= compare(
                f"track1 {method} zero_average={zero} reflected loss/eV (symmetry)",
                loss,
                0.0,
                atol=6e-8,
            )
    return defect


def ring(tilt: float) -> bool:
    lat = f"""no_digested
parameter[geometry]=closed
parameter[particle]=electron
parameter[p0c]=3e9
qf:quadrupole,l=.2,k1=1,tilt={tilt}
qd:quadrupole,l=.2,k1=-.9
d:drift,l=.5
b:sbend,l=1,g={2 * np.pi / 32:.17g},e1={np.pi / 32:.17g},e2={np.pi / 32:.17g},fringe_type=full
rf:rfcavity,l=0,voltage=3e6,harmon=120,phi0=0
cell:line=(qf,d,b,d,qd,d,b,d)
ring:line=(16*cell,rf)
use,ring
"""
    log = fortran(RING_PROGRAM, files={"lat.bmad": lat}).log
    raw = {}
    elements = {0: [], 1: []}
    for line in log.splitlines():
        t = line.split()
        if not t:
            continue
        if t[0] == "MATRIX":
            key = (int(t[2]), t[1], int(t[3]))
            if len(t) != 10 or key in raw:
                raise RuntimeError(f"Malformed or duplicate ring row: {line}")
            raw[key] = np.array(t[4:], float)
        elif t[0] == "ELEMENT":
            elements[int(t[1])].append((int(t[2]), int(t[3])))
    expected_labels = {"M", "D", "Q", "T", "S"} | {
        f"E{i}{s}" for i in range(1, 131) for s in ["T", "D", "Q"]
    }
    if set(raw) != {
        (o, label, r) for o in [0, 1] for label in expected_labels for r in range(1, 7)
    }:
        raise RuntimeError("Incomplete ring matrices")
    matrices = {
        (o, label): np.array([raw[o, label, r] for r in range(1, 7)])
        for o in [0, 1]
        for label in expected_labels
    }
    if not all(np.isfinite(a).all() for a in matrices.values()):
        raise RuntimeError("Nonfinite ring matrices")
    for o in [0, 1]:
        if [i for i, _ in elements[o]] != list(range(1, 131)):
            raise RuntimeError("Incomplete ring element sequence")
        if sum(key == 2 for _, key in elements[o]) != 32:
            raise RuntimeError("Expected 32 untilted bends")

    parity = np.array([1, 1, -1, -1, 1, 1])

    def compose(opening: int, restore: bool):
        m, q = np.eye(6), np.zeros((6, 6))
        for i, key in elements[opening]:
            t, d, qi = (matrices[opening, f"E{i}{s}"] for s in ["T", "D", "Q"])
            # Only the untilted normal bend has this local reflection symmetry.
            # Preserve coupled quadrupoles, body-even damping and all diffusion.
            if restore and key == 2:  # sbend$
                d = (d + parity[:, None] * d * parity[None, :]) / 2
            transport = t + d
            q = transport @ q @ transport.T + qi
            m = transport @ m
        return m, q

    defect = False
    symplectic = np.kron(np.eye(3), np.array([[0, 1], [-1, 0]]))
    for opening in [0, 1]:
        m, q, sigma = (matrices[opening, label] for label in ["M", "Q", "S"])
        built_m, built_q = compose(opening, False)
        np.testing.assert_allclose(built_m, m, rtol=2e-13, atol=2e-13)
        np.testing.assert_allclose(built_q, q, rtol=2e-13, atol=2e-20)
        cm, cq = compose(opening, True)
        if max(np.max(abs(np.linalg.eigvals(a))) for a in [m, cm]) >= 1:
            raise RuntimeError("Ring is not damped and stable")
        native = solve_discrete_lyapunov(m, q, method="direct")
        expected = solve_discrete_lyapunov(cm, cq, method="direct")
        np.testing.assert_allclose(native, sigma, rtol=3e-9, atol=3e-18)
        residual = max(
            np.max(abs(s - a @ s @ a.T - b)) for s, a, b in [(native, m, q), (expected, cm, cq)]
        )
        if residual >= 3e-18:
            raise RuntimeError("Lyapunov residual exceeds tolerance")
        # Eigenvalues of Sigma*S are +/- i*emittance. Ascending order here is b,a,z.
        actual_modes = np.sort(np.linalg.eigvals(native @ symplectic).imag)[3:]
        expected_modes = np.sort(np.linalg.eigvals(expected @ symplectic).imag)[3:]
        label = f"ring tilt={tilt:g} opening={opening}"
        if tilt == 0 and opening == 0:
            np.testing.assert_allclose(native, expected, rtol=3e-9, atol=3e-18)
            print(f"{label}: planar zero-vertical-excitation covariance control passes")
        defect |= compare(
            f"{label} eigenemittances b,a,z/m (bend reflection restored)",
            actual_modes,
            expected_modes,
            atol=1e-14,
            rtol=1e-9,
        )
        if tilt:
            print(
                f"{label}: a-mode relative error={100 * (actual_modes[1] / expected_modes[1] - 1):.12g}%; "
                f"Lyapunov residual={residual:.6g}"
            )
    return defect


def main() -> None:
    defect = False
    frozen = None
    for edge in [0.0, 0.1, -0.1]:
        _, matrix, rows = local(edge)
        defect |= compare(
            f"edge={edge:+g} whole-element D63/m^-1 (reflection)", matrix[5, 2], 0.0, atol=1e-16
        )
        for n in [1, 8, 64]:
            delta = difference(rows, n, 0, 3)
            if edge == 0:
                np.testing.assert_allclose(delta, 0.0, rtol=0, atol=2e-17)
            defect |= compare(
                f"edge={edge:+g} N={n} reflected loss difference/eV (symmetry)",
                3e9 * delta,
                0.0,
                atol=6e-8,
            )
            if edge == 0.1 and n == 64:
                frozen = 3e9 * delta
    assert frozen is not None
    defect |= tracking(frozen)
    for tilt in [0.0, 0.01]:
        defect |= ring(tilt)
    verdict(defect)


if __name__ == "__main__":
    main()
