"""COL-19: The CSR manual describes a second staggered deposition absent from the calculation

When describing one-dimensional CSR binning, the manual says that particle density is deposited again into bins shifted by half a bin width. The implementation deposits triangular particle charge once and obtains the edge density derivative from adjacent bin-charge differences. A three-particle calculation agrees with that single deposition and finite-difference rule. The manual should describe the single charge array and the derivative used for the CSR kick.

The script calls the production CSR binning routine for three fixed charged particles. It checks
the bin charges against analytic integrals of triangular particle shapes and the edge
derivatives against adjacent charge differences. It reads the disputed algorithm description
through BMAD_SOURCE and uses BMAD_DIST for the library.
"""

import numpy as np
from _bmad import compare, fortran, verdict
from _collective_contracts import rows, section, source_text

result = rows(
    fortran("""program bins
use csr_and_space_charge_mod
implicit none
type(csr_struct) c
type(ele_struct) e
type(coord_struct) p(3)
logical err
integer i
e%csr_method=one_dim$
space_charge_com%n_bin=12
space_charge_com%particle_bin_span=2
p%state=alive$
p%charge=1e-12_rp
p(1)%vec(5)=-0.001_rp
p(2)%vec(5)=0.00013_rp
p(3)%vec(5)=0.001_rp
call csr_bin_particles(e,p,c,err)
if (err) error stop 'Binning failed'
write(*,'(a,es25.16)') 'RESULT dz ',c%dz_slice
write(*,'(a,12es25.16)') 'RESULT centers ',c%slice%z_center
write(*,'(a,12es25.16)') 'RESULT charge ',c%slice%charge
write(*,'(a,12es25.16)') 'RESULT derivative ',c%slice%edge_dcharge_density_dz
end program
""").log
)
q, dz = result["charge"], result["dz"][0]
# For a triangular macroparticle with half-width h, the normalized cumulative
# charge is (x+h)^2/(2h^2) on [-h,0], and 1-(h-x)^2/(2h^2) on [0,h].
h = dz
z = np.array([-0.001, 0.00013, 0.001])


def cumulative(x):
    x = np.clip(x, -h, h)
    return np.where(x < 0, (x + h) ** 2 / (2 * h * h), 1 - (h - x) ** 2 / (2 * h * h))


def deposit(centers):
    a = centers[:, None] - dz / 2 - z
    b = centers[:, None] + dz / 2 - z
    return 1e-12 * np.sum(cumulative(b) - cumulative(a), axis=1)


expected_q = deposit(result["centers"])
if compare("single triangular deposition Q (C)", q, expected_q, atol=3e-27):
    raise RuntimeError("Triangular deposition control failed")
expected_d = np.r_[0, np.diff(q) / dz**2]
if compare(
    "edge derivative (C/m^2), adjacent Q differences divided by dz^2",
    result["derivative"],
    expected_d,
    atol=2e-20,
):
    raise RuntimeError("Finite-difference implementation changed")
manual = section(
    source_text("bmad/doc/multiparticle.tex"),
    r"\subsection{1\_Dim CSR Calculation}",
    r"\subsection{Slice Space Charge Calculation}",
)
claimed = "second set of staggered bins" in manual
print(
    f"manual promises a second staggered deposition={claimed}; execution matches single-deposition finite differences"
)
verdict(claimed)
