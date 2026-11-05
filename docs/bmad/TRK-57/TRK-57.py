"""TRK-57: PTC branch tracker reverses its optional error flag

Calling ptc_track_all with its optional err_flag output returns true after successful tracking and false when a particle is lost at either aperture edge. The routine documents false for success and true for loss or an error. In a 1 m drift with 10 mm aperture half-widths, a surviving particle at 2.000 mm is reported as an error, while particles lost at 20.00 mm at the entrance or exit are reported as successful.

The script builds a one-drift PTC layout using BMAD_DIST and runs each trajectory in a fresh process. Straight-line geometry supplies the expected coordinates and aperture classification independently of the error flag. It tests both initial flag values and an omitted-flag control for each trajectory.
"""

import math

import numpy as np
from _bmad import fortran, verdict

PROGRAM = """program check_flag
use bmad
use ptc_layout_mod, only: lat_to_ptc_layout, ptc_track_all, kill_ptc_layouts
use s_tracking, only: check_stable
implicit none
type(lat_struct), target :: lat
type(coord_struct), allocatable :: orbit(:)
character(20) :: case_name, mode
logical :: err
integer :: track_state, flag_value, expected_state
real(rp) :: vec(6)
call get_command_argument(1, case_name)
call get_command_argument(2, mode)
call bmad_parser('lat.bmad', lat, err_flag=err)
if (err) error stop 'parser failed'
if (lat%n_ele_track /= 1) error stop 'expected one drift'
call lat_to_ptc_layout(lat)
if (.not. associated(lat%ele(1)%ptc_fibre)) error stop 'missing fibre'
allocate(orbit(0:lat%n_ele_track))
vec = 0
expected_state = lost_pos_x$
select case(trim(case_name))
case('success')
  vec(2) = 0.002_rp
  expected_state = alive$
case('entrance_loss')
  vec(1) = 0.02_rp
case('exit_loss')
  vec(2) = 0.02_rp
case default
  error stop 'invalid case'
end select
call init_coord(orbit(0), vec, lat%ele(0), downstream_end$)
bmad_com%aperture_limit_on = .true.
flag_value = -1
select case(trim(mode))
case('false', 'true')
  err = mode == 'true'
  call ptc_track_all(lat%branch(0), orbit, track_state, err)
  flag_value = merge(1, 0, err)
case('omitted')
  call ptc_track_all(lat%branch(0), orbit, track_state)
case default
  error stop 'invalid mode'
end select
if (orbit(1)%state /= expected_state) error stop 'wrong aperture classification'
if (case_name == 'success') then
  if (.not. check_stable) error stop 'PTC unstable on successful trajectory'
  if (track_state /= moving_forward$) error stop 'wrong successful track state'
endif
write(*, '(a,4(1x,i0),7(1x,es25.16))') 'RESULT', &
  flag_value, merge(1, 0, check_stable), track_state, orbit(1)%state, orbit(1)%vec, orbit(1)%s
call kill_ptc_layouts(lat)
end program
"""

LATTICE = """no_digested
parameter[particle] = electron
parameter[p0c] = 1e9
parameter[geometry] = open
parameter[no_end_marker] = T
beginning[beta_a] = 10
beginning[beta_b] = 10
d: drift, l = 1, aperture_type = rectangular, aperture_at = both_ends, &
  x1_limit = 0.01, x2_limit = 0.01, y1_limit = 0.01, y2_limit = 0.01
ln: line = (d)
use, ln
"""


def main() -> None:
    defect = False
    for name, x0, px, distance in (
        ("success", 0.0, 0.002, 1.0),
        ("entrance_loss", 0.02, 0.0, 0.0),
        ("exit_loss", 0.0, 0.02, 1.0),
    ):
        # At delta=py=0 the longitudinal momentum/p0 is sqrt(1-px**2).
        # The constant-speed drift gives x=x0+L*px/ps and z=L*(1-1/ps).
        # The rationalized z expression avoids subtracting nearly equal numbers.
        ps = math.sqrt(1.0 - px**2)
        expected = np.array(
            [
                x0 + distance * px / ps,
                px,
                0.0,
                0.0,
                -distance * px**2 / (ps * (1 + ps)),
                0.0,
                distance,
            ]
        )
        # Rectangular apertures reject x>0.01 m. An entry loss returns before drifting.
        expected_flag = int(max(x0, expected[0]) > 0.01)
        records: dict[str, tuple[tuple[int, ...], np.ndarray]] = {}
        for mode in ("omitted", "false", "true"):
            result = fortran(PROGRAM, [name, mode], files={"lat.bmad": LATTICE})
            lines = [
                line.split()[1:] for line in result.log.splitlines() if line.startswith("RESULT ")
            ]
            if len(lines) != 1 or len(lines[0]) != 11:
                raise RuntimeError(f"Expected one complete result for {name}/{mode}:\n{result.log}")
            fields = lines[0]
            status = tuple(int(value) for value in fields[:4])
            coords = np.array([float(value) for value in fields[4:]])
            if not np.isfinite(coords).all():
                raise RuntimeError(f"Nonfinite coordinates for {name}/{mode}")
            np.testing.assert_allclose(coords, expected, atol=1e-14, rtol=0)
            records[mode] = status, coords
            if mode == "omitted":
                if status[0] != -1:
                    raise RuntimeError("Omitted-flag control returned an unexpected sentinel")
                print(
                    f"{name} omitted-flag control: state={status[3]} "
                    f"x_m={coords[0]:.16g} expected_x_m={expected[0]:.16g}; "
                    "analytic drift and aperture classification pass"
                )
                continue
            if status[0] not in (0, 1):
                raise RuntimeError(f"Invalid logical output for {name}/{mode}")
            control_status, control_coords = records["omitted"]
            if status[1:] != control_status[1:]:
                raise RuntimeError(f"Optional flag changed tracking status for {name}/{mode}")
            np.testing.assert_array_equal(coords, control_coords)
            defect |= status[0] != expected_flag
            print(
                f"{name} initial_err_flag={mode}: observed={bool(status[0])} "
                f"expected={bool(expected_flag)} (contract: true iff lost/error); "
                "coordinates and state equal omitted-flag control"
            )
    verdict(defect)


if __name__ == "__main__":
    main()
