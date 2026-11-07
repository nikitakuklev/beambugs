"""COL-17: High-energy space-charge setup reads beyond a tightly allocated branch

When the high-energy space-charge setup receives a branch allocated exactly through its last tracking element, it reads the next element before applying the last-element length rule. That next element lies outside the supplied array. A bounds-checked build stops on index 2 for a branch whose final index is 1. The same routine succeeds when the branch has an extra allocated element.

The script runs a padded branch against the production library, then recompiles the unchanged
upstream module with bounds checks using the build's compiler and libraries. It verifies the padded
result and requires the specific out-of-bounds diagnostic for a tightly allocated branch.
"""

from _bmad import RunFailure, compare, fortran, verdict
from _collective_contracts import bounds_checked, rows, source_text

PROGRAM = """program extent
use high_energy_space_charge_mod
implicit none
type(branch_struct) branch
type(normal_modes_struct) mode
integer n, i
character(8) arg
call get_command_argument(1,arg)
read(arg,*) n
allocate(branch%ele(0:n))
branch%n_ele_track=1
branch%n_ele_max=n
branch%param%particle=electron$
do i=0,n
  branch%ele(i)%ix_ele=i
  branch%ele(i)%value(l$)=1
  branch%ele(i)%value(p0c$)=1e9_rp
  branch%ele(i)%a%beta=3
  branch%ele(i)%b%beta=5
enddo
mode%a%emittance=2e-6_rp
mode%b%emittance=1e-6_rp
mode%sig_z=0.003_rp
mode%sigE_E=0
write(*,'(a,2i4)') 'RESULT extent ', ubound(branch%ele,1), branch%n_ele_track
flush(6)
call setup_high_energy_space_charge_calc(.true.,branch,1e9_rp,mode)
write(*,'(a,es25.16)') 'RESULT kick ', branch%ele(1)%high_energy_space_charge%kick_const
end program
"""
production = rows(fortran(PROGRAM, ["2"]).log)
print(f"production padded control: kick={production['kick'][0]:.16g}")
module = source_text("bmad/space_charge/high_energy_space_charge_mod.f90")
control = rows(bounds_checked(module + "\n" + PROGRAM, ["2"]))
if compare(
    "padded bounds-checked kick vs production",
    control["kick"],
    production["kick"],
    atol=0,
    rtol=2e-14,
):
    raise RuntimeError("Bounds-checked rebuild disagrees with production control")
try:
    log = bounds_checked(module + "\n" + PROGRAM, ["1"])
except RunFailure as exc:
    if (
        "RESULT extent    1   1" not in exc.log
        or "of array 'branch%ele'" not in exc.log
        or "above upper bound of 1" not in exc.log
        or "Index '2'" not in exc.log
    ):
        raise
    print(
        "tightly allocated branch: observed index 2 above upper bound 1; expected no access beyond n_ele_track=1"
    )
    print(next(line for line in exc.log.splitlines() if "Fortran runtime error:" in line))
    verdict(True)
else:
    rows(log)["kick"]
    print("tightly allocated branch: bounds-checked call completed")
    verdict(False)
