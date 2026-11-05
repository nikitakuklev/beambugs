"""COL-28: Normal quadrupole gradient creates a fictitious vertical-energy damping term

The envelope radiation-damping matrix makes linear energy loss depend on vertical displacement in an on-axis normal combined-function field. Reflection symmetry requires the vertical derivative of B² to vanish, because Bₓ = B₁y and Bᵧ = B₀ + B₁x. At Lorentz factor 1000, curvature 0.1 m⁻¹ and gradient B₁ = B₀/(1 m), the matrix gives D₆₃ = 3.757 × 10⁻⁸ m⁻² instead of zero. Its horizontal term agrees with the field derivative.

The script compiles a damping_matrix_D driver against BMAD_DIST. It checks the horizontal derivative as a control and compares the vertical derivative with the zero required by reflection symmetry.
"""

from _bmad import compare, fortran, verdict

PROGRAM = """program probe_damping
use bmad
use envelope_mod, only: damping_matrix_D
implicit none
real(rp) dmat(6,6), g, b0, b1, gamma, prefactor
gamma=1000.0_rp
g=0.1_rp
b0=sqrt((gamma*m_electron)**2-m_electron**2)*g/c_light
b1=b0
dmat=damping_matrix_D(gamma,g,b0,b1,0.0_rp,electron$)
! Loss per unit reference length is proportional to (1+g*x)*B**2.
! Its normalized x derivative is g+2*B1/B0; its y derivative vanishes on axis.
prefactor=(2.0_rp/3.0_rp)*classical_radius(electron$)*gamma**3*g**2
write(*,'(a,es25.16)') 'D(6,3) (production): ',dmat(6,3)
write(*,'(a,es25.16)') 'D(6,3) (expected by y reflection symmetry): ',0.0_rp
write(*,'(a,es25.16)') 'D(6,1) (production): ',dmat(6,1)
write(*,'(a,es25.16)') 'D(6,1) (expected): ',prefactor*(g+2*b1/b0)
end program
"""

values = [float(line.split(":", 1)[1]) for line in fortran(PROGRAM).log.splitlines() if ":" in line]
if len(values) != 4:
    raise RuntimeError("Missing damping matrix output")
if compare(
    "horizontal damping control (m^-2), derivative of B^2",
    values[2],
    values[3],
    atol=1e-22,
    rtol=2e-14,
):
    raise RuntimeError("Horizontal control failed")
verdict(compare("vertical damping D63 (m^-2), reflection symmetry", values[0], 0, atol=1e-22))
