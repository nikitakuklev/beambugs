"""OPT-30: Tilted Taylor-map Jacobians lose the coordinate-frame transformations

With taylor_map_includes_offsets set to false, the Taylor tracker returns a body-frame Jacobian for a tilted element even though its particle trajectories are transformed correctly. The laboratory Jacobian must include the entrance and exit rotations. For a 0.8 m quadrupole with normalized strength 0.7 m⁻² and tilt 0.31 rad, M₂₃ is zero instead of −0.3259 m⁻¹, where M₂₃ is the derivative of horizontal normalized momentum with respect to vertical position. Including offsets in the map, or setting the tilt to zero, gives the correct matrix.

The script builds first-order Taylor maps of a tilted quadrupole with offsets excluded, offsets included, and zero tilt. It compares the returned transverse matrix and same-map trajectory differences with the analytically rotated focusing and defocusing blocks.
"""

from _bmad import compare, fortran, verdict
from _srdt_wiggler_maps import MAP_LATTICE, MAP_PROGRAM, records

log = fortran(MAP_PROGRAM, files={"lat.bmad": MAP_LATTICE}).log
rows = records(log, "TAYLOR")
if rows.shape != (3, 6):
    raise RuntimeError("Incomplete Taylor-map result")
defect = False
for row in rows:
    i, actual, fd, analytic, fd_error, matrix_error = row
    if compare(f"case {int(i)}: finite-difference M23 (1/m)", fd, analytic, atol=2e-13):
        raise AssertionError("Rotated-quadrupole oracle failed")
    bad = compare(f"case {int(i)}: reported M23 (1/m)", actual, analytic, atol=2e-13)
    if i > 1 and bad:
        raise AssertionError("Offsets-included or untilted control failed")
    print(
        f"case {int(i)}: maximum transverse matrix error={matrix_error:.16g}; finite-difference error={fd_error:.16g}"
    )
    defect |= bad
verdict(defect)
