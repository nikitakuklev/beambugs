"""TAO-22: Default radiation output labels a damped transport matrix as undamped

Running show matrix -radiation in Tao labels the full damped transport as T-Matrix without Damping and reports the separate damping factor D as the identity. For a 1 GeV/c electron passing through a 1 m sector bend of radius 2 m, the display reports a symplectic error of 7.184 × 10⁻⁶. The displayed matrix matches the damping-enabled backend, while disabling damping gives an undamped map with symplectic error below 4 × 10⁻¹⁴. An independent prescribed-map case confirms that both factorization options correctly reconstruct transport and noise covariance.

The script feeds independently prescribed transport and noise matrices to an inline Fortran driver and checks both factorizations. It also compares the actual Tao radiation display for a sector bend with backend maps computed with damping enabled and disabled. The damping-off control matches the analytic sector-bend map.
"""

import re

import numpy as np
from _bmad import compare, fortran, verdict
from _tao_contracts_normal_form import ERROR, block, tao

PROGRAM = """
program radiation_factors
use bmad
use ptc_map_with_radiation_mod
use pointer_lattice
implicit none
type(c_damap) m, identity
type(ptc_rad_map_struct) r
real(rp) aa(6,6), qq(6,6)
integer i, j
logical split
open(10, file='input.dat')
read(10,*) split
do i=1,6
  read(10,*) aa(i,:)
enddo
do i=1,6
  read(10,*) qq(i,:)
enddo
close(10)
call set_ptc(e_tot=1d9, particle=electron$, taylor_order=1)
call init_all(ptc_private%base_state+envelope0+radiation0, 1, 0)
call zhe_ini(.false.)
call alloc(m, identity)
identity=1
m=1
m%x0=0
do i=1,6
  m%v(i)=0
  do j=1,6
    m%v(i)=m%v(i)+aa(i,j)*identity%v(j)
  enddo
enddo
m%e_ij=qq
! ptc_setup_map_with_radiation defaults to sagan_gen=True. The false case
! exercises its optional factorization with nontrivial damping factors.
sagan_gen=.not. split
call fill_tree_element_line_zhe_outside_map(m, as_is=.false., stochprec=1d-10, tree_zhe=r%sub_map)
! These are the fields copied by ptc_setup_map_with_radiation and printed by Tao.
r%nodamp_mat=r%sub_map(3)%rad
do i=1,6
  r%damp_mat(:,i)=r%sub_map(1)%cc(i*6+1:i*6+6)
enddo
r%stoc_mat=matmul(matmul(r%damp_mat,r%nodamp_mat),r%sub_map(2)%rad)
write(*,'(a,36es25.17)') 'RESULT t ',r%nodamp_mat
write(*,'(a,36es25.17)') 'RESULT d ',r%damp_mat
write(*,'(a,36es25.17)') 'RESULT s ',r%stoc_mat
call ptc_kill_map_with_radiation(r)
call kill(m, identity)
end program
"""
PHYSICAL_LATTICE = """parameter[geometry]=open
parameter[particle]=electron
parameter[p0c]=1e9
beginning[beta_a]=5
beginning[beta_b]=6
b: sbend,l=1,g=0.5, fringe_type=none
lat: line=(b)
use,lat
"""
PHYSICAL_PROGRAM = """program physical_radiation
use bmad
use ptc_map_with_radiation_mod
implicit none
type(lat_struct), target :: lat
type(coord_struct), allocatable :: orbit(:)
type(ptc_rad_map_struct) r
logical err, damp
integer stat
character(1) option
call get_command_argument(1, option)
damp = option == 'T'
call bmad_parser('lat.bmad', lat)
call twiss_and_track(lat, orbit, stat)
if (stat /= ok$) error stop 'failed setup'
call lat_to_ptc_layout(lat)
call ptc_setup_map_with_radiation(r, lat%ele(0), lat%ele(1), 1, damp, orbit1=orbit(0), err_flag=err)
if (err) error stop 'radiation map failure'
write(*,'(a,36es25.17)') 'PHYSICAL_T ',r%nodamp_mat
write(*,'(a,36es25.17)') 'PHYSICAL_D ',r%damp_mat
call ptc_kill_map_with_radiation(r)
end program
"""
transport = np.eye(6)
for i, angle in enumerate((0.4, 0.7, 0.13)):
    c, s = np.cos(angle), np.sin(angle)
    transport[2 * i : 2 * i + 2, 2 * i : 2 * i + 2] = [[c, 2 * s], [-s / 2, c]]
skew = np.eye(6)
skew[1, 2] = skew[3, 0] = 0.2
damped = np.diag([0.99, 0.98, 0.97, 0.96, 0.95, 0.94]) @ skew @ transport
noise_root = np.diag(np.arange(1, 7) * 1e-3)
noise_root[1, 0], noise_root[3, 0], noise_root[5, 2] = 0.0007, -0.0002, 0.0004
covariance = noise_root @ noise_root.T
symplectic = np.kron(np.eye(3), [[0, 1], [-1, 0]])
prescribed_noncanonical = False
for split in (False, True):
    data = (
        ("T" if split else "F")
        + "\n"
        + "\n".join(" ".join(map(str, row)) for row in np.vstack([damped, covariance]))
        + "\n"
    )
    result = fortran(PROGRAM, files={"input.dat": data})
    rows = {
        f[1]: np.array(f[2:], dtype=float).reshape(6, 6, order="F")
        for line in result.log.splitlines()
        if (f := line.split()) and f[0] == "RESULT"
    }
    assert set(rows) == {"t", "d", "s"}, result.log
    t, d, s = [rows[k] for k in ("t", "d", "s")]
    assert np.allclose(d @ t, damped, atol=3e-15, rtol=3e-14)
    assert np.allclose(s @ s.T, covariance, atol=2e-20, rtol=3e-14)
    error = np.max(np.abs(t.T @ symplectic @ t - symplectic))
    print(
        f"split={split}: max |D*T-A|={np.max(np.abs(d @ t - damped)):.12g}, max |S*S^T-Q|={np.max(np.abs(s @ s.T - covariance)):.12g}, det(T)={np.linalg.det(t):.12g}, symplectic error={error:.12g}"
    )
    if not split:
        assert np.allclose(d, np.eye(6), atol=3e-15, rtol=0)
        assert np.allclose(t, damped, atol=3e-15, rtol=3e-14)
        prescribed_noncanonical = compare(
            "default T determinant, undamped canonical map requires one",
            np.linalg.det(t),
            1,
            atol=3e-14,
        )
    else:
        assert np.max(np.abs(d @ t - t @ d)) > 1e-4
        assert error < 3e-14
        assert abs(np.linalg.det(t) - 1) < 3e-14

# Execute the actual reporting path with a physical lattice, then isolate
# damping by changing only include_damping in the backend API. Equal reference
# momenta make the canonical condition T^T J T=J directly applicable.
physical_maps = []
for damping in (False, True):
    result = fortran(
        PHYSICAL_PROGRAM,
        args=["T" if damping else "F"],
        files={"lat.bmad": PHYSICAL_LATTICE},
    )
    assert not ERROR.search(result.log), result.log
    matrices = []
    for name in ("T", "D"):
        matches = re.findall(r"^PHYSICAL_" + name + r" (.+)$", result.log, re.MULTILINE)
        assert len(matches) == 1, result.log
        values = np.fromstring(matches[0], sep=" ").reshape(6, 6, order="F")
        assert np.isfinite(values).all()
        matrices.append(values)
    t, d = matrices
    assert not compare(
        f"sector bend damping={damping}, identity D control", d, np.eye(6), atol=2e-15
    )
    error = np.max(np.abs(t.T @ symplectic @ t - symplectic))
    print(
        f"sector bend damping={damping}: det(T)={np.linalg.det(t):.17g}, symplectic error={error:.17g}"
    )
    if not damping:
        # The exact hard-edge bend's horizontal map at hL=.5. The 1e-12
        # tolerance accommodates PTC Taylor-algebra/inversion roundoff and
        # resolves the radiation signal by more than six orders of magnitude.
        expected = [[np.cos(0.5), 2 * np.sin(0.5)], [-0.5 * np.sin(0.5), np.cos(0.5)]]
        assert not compare(
            "undamped sector bend horizontal map, analytic control", t[:2, :2], expected, atol=1e-12
        )
        assert error < 1e-12
    physical_maps.append(t)

command = "show matrix -radiation BEGINNING B"
display = block(tao(PHYSICAL_LATTICE, [command]).log, command)
label = "T-Matrix without Damping. Symplectic Error: "
# A changed label is a resolved reporting defect. The same matrix blocks still
# establish that the displayed matrix includes damping.
sections = display.split("\n\n")
blocks = [
    part.splitlines() for part in sections if "T-Matrix" in part or "D-Damping Matrix" in part
]
assert len(blocks) == 2, display
shown = []
for lines in blocks:
    values = np.array([line.split() for line in lines[1:]], dtype=float)
    assert values.shape == (6, 6) and np.isfinite(values).all(), display
    shown.append(values)
assert not compare(
    "displayed sector-bend T, damping-enabled backend control",
    shown[0],
    physical_maps[1],
    atol=5e-16,
    rtol=5e-9,
)
assert not compare("displayed sector-bend D, identity control", shown[1], np.eye(6), atol=0)
physical_noncanonical = compare(
    "damping-enabled sector-bend symplectic condition",
    physical_maps[1].T @ symplectic @ physical_maps[1],
    symplectic,
    atol=1e-12,
)
print(f"T-Matrix without Damping label present: {label in display}")
verdict(label in display and prescribed_noncanonical and physical_noncanonical)
