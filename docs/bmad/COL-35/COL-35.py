"""COL-35: Envelope radiation diffusion retains electron recoil for every species

When the envelope radiation-diffusion helper is called for a heavier species, its classical radius changes with species but its recoil coefficient keeps the electron mass. The coefficient should contain ℏ/(mc), where ℏ is the reduced Planck constant, m is the species mass and c is the speed of light. At equal Lorentz factor and curvature, the proton/electron diffusion ratio is 5.446 × 10⁻⁴ instead of 2.966 × 10⁻⁷, a factor of 1836 too large. Approximate physical constants also make the electron coefficient 0.03383% high.

The script calls the production diffusion helper for electrons, positrons and protons at equal
Lorentz factor and curvature. It checks charge-conjugate equality and compares the species ratio
with the classical-radius and recoil-mass factors, then checks the electron value against Bmad's
physical constants. BMAD_DIST selects the build.
"""

from _bmad import compare, fortran, verdict
from _collective_contracts import rows

result = rows(
    fortran("""program diffusion
use envelope_mod
implicit none
real(rp) mat(6,6)
integer species, j
integer, parameter :: kinds(3) = [electron$, positron$, proton$]
do j=1,3
  species=kinds(j)
  mat=diffusion_matrix_B(1000.0_rp,0.1_rp,species)
  write(*,'(a,i0,4es25.16)') 'RESULT species', j, mat(6,6), classical_radius(species), mass_of(species), h_bar_planck*c_light
  if (count(mat /= 0) /= 1) error stop 'Unexpected diffusion tensor shape'
enddo
end program
""").log
)
electron, positron, proton = (result[f"species{i}"] for i in range(1, 4))
if compare("electron/positron control B66 (1/m)", electron[0], positron[0], atol=0, rtol=1e-15):
    raise RuntimeError("Charge-conjugate diffusion control failed")
# At fixed gamma and curvature, B66 is proportional to r_classical*hbar/(m*c).
# For equal charge magnitude, r_classical is itself proportional to 1/m.
observed_ratio = proton[0] / electron[0]
expected_ratio = (proton[1] / electron[1]) * (electron[2] / proton[2])
defect = compare(
    "proton/electron B66 ratio; expected radius ratio times inverse mass ratio",
    observed_ratio,
    expected_ratio,
    atol=0,
    rtol=2e-13,
)
# Bmad stores the rest energy m*c^2 in eV as mass_of, and h_bar_planck in
# eV*s. Thus h_bar_planck*c_light/mass_of is the recoil length hbar/(m*c).
expected_e = 55 / (24 * 3**0.5) * electron[3] / electron[2] * electron[1] * 1000**5 * 0.1**3
defect |= compare(
    "electron B66 (1/m); established Bmad constants", electron[0], expected_e, atol=0, rtol=2e-13
)
print(f"proton coefficient excess factor={observed_ratio / expected_ratio:.16g}")
verdict(defect)
