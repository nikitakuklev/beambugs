"""TRK-11: Elens has inverted species scaling and extra momentum dependence

The annular `Elens` kick shrinks when the particle charge grows and carries an extra dependence on
the relative momentum deviation δ. Gauss's law and the magnetic force of the counter-propagating
electron beam require the normalized kick to scale with the particle's charge-to-mass ratio relative
to the reference species, χ, and the physical velocity factor. For a 3.7 A, 1.3 m, 10 kV lens with
radii 1 and 4 mm, a 1 GeV/c particle at (x, y) = (2, 0.4) mm with twice the reference charge and the
same mass gets normalized horizontal momentum −1.173 × 10⁻⁴ instead of −4.693 × 10⁻⁴. At δ = +20%
the reference-species kick is 1.200 times too large.

The script tracks particles with several charge and mass ratios, one of them at δ = 0.2, through the
lens and compares both transverse kicks with the Lorentz force of the enclosed electron current
integrated over the transit time. It also encodes the doubled charge through `q0` instead of
`charge_ratio`, and compares the kick at δ = 0.2 with that at δ = 0 for reference particles. The
reference particle and the `q0` = 2 particle must agree; the others show the bug. It runs serially
on the installed prebuilt CPU kernels with kernel compilation forbidden.
"""

from _prebuilt import CTX, identity, kernels, np, particles, report, xt

identity()
cases = {}
c = 299792458.0
eps0 = 8.854187817620e-12  # Match the executed header, not a newer SI constant.
I, length, voltage = 3.7, 1.3, 10000.0
rin, rout = 0.001, 0.004
x, y = 0.002, 0.0004
r2 = x * x + y * y
enclosed_fraction = (r2 - rin * rin) / (rout * rout - rin * rin)
electron_mass = 510998.928
beta_e = np.sqrt(voltage * (voltage + 2 * electron_mass)) / (voltage + electron_mass)
lens = xt.Elens(
    _context=CTX,
    current=I,
    elens_length=length,
    voltage=voltage,
    inner_radius=rin,
    outer_radius=rout,
)
charge = np.array([1.0, 2.0, -1.0, 1.0, 3.0])
mass = np.array([1.0, 1.0, 1.0, 2.0, 3.0])
delta = np.array([0.0, 0.0, 0.0, 0.0, 0.2])
p = particles(x=np.full(5, x), y=np.full(5, y), delta=delta, mass_ratio=mass, charge_ratio=charge)
chi = charge / mass
beta = (1 + delta) * 1e9 / np.hypot((1 + delta) * 1e9, p.mass0)
# E_r = -I_enclosed / (2*pi*eps0*beta_e*c*r); opposite electron
# velocity gives B force in the same radial direction as E. Integrate dt=L/v.
radial_prefactor = -I * enclosed_fraction * length / (2 * np.pi * eps0 * 1e9 * c * beta_e)
expected_px = radial_prefactor * chi * (1 / beta + beta_e) * x / r2
expected_py = expected_px * y / x
lens.track(p)
print("CHI", chi)
print("PX observed", np.asarray(p.px))
print("PX Lorentz expected", expected_px)
print(
    "observed/expected",
    np.asarray(p.px) / expected_px,
    "source predicts (1+delta)/chi^2",
    (1 + delta) / chi**2,
)
np.testing.assert_allclose(p.px[0], expected_px[0], rtol=3e-15, atol=0)
np.testing.assert_allclose(p.py[0], expected_py[0], rtol=3e-15, atol=0)
cases["Elens px"] = (np.asarray(p.px), expected_px)
cases["Elens py"] = (np.asarray(p.py), expected_py)

q0_two = particles(q0=2, x=x, y=y, delta=0.0)
lens.track(q0_two)
print(
    "SAME_PHYSICAL_CHARGE q0=2:",
    q0_two.px[0],
    "q0=1 charge_ratio=2:",
    p.px[1],
    "ratio=",
    q0_two.px[0] / p.px[1],
)
np.testing.assert_allclose(q0_two.px[0], 2 * expected_px[0], rtol=3e-15, atol=0)

momentum = particles(x=[x, x], y=[y, y], delta=[0.0, 0.2])
lens.track(momentum)
b = np.array([1e9, 1.2e9]) / np.hypot([1e9, 1.2e9], momentum.mass0)
expected_ratio = (1 / b[1] + beta_e) / (1 / b[0] + beta_e)
ratio = momentum.px[1] / momentum.px[0]
print(
    f"MOMENTUM ratio={ratio:.17g} expected={expected_ratio:.17g} "
    f"excess_factor={ratio / expected_ratio:.17g}"
)
cases["momentum kick ratio"] = (ratio, expected_ratio)
cases["equivalent physical charge px"] = (p.px[1], q0_two.px[0])
kernels()
report(cases, rtol=4e-15, atol=0)
