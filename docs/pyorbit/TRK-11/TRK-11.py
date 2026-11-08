"""TRK-11: Axis-field RF gaps drop part of the radial electric impulse, so their transverse map converges to a wrong limit as the step shrinks

`AxisFieldRF_Gap` tracks an on-axis RF field E(s) as a series of thin transit-time kicks between drifts. Each kick keeps only the sin ψ part of the integrated radial impulse, ψ being the RF phase, and drops the boundary term −(x/2β)E cos ψ, nonzero between steps; summed over the steps it equals ½∫E cos ψ d(x/β), with x the offset and β the particle speed over c, and stays finite as the step shrinks. For a 2.5 MeV proton in a 402.5 MHz cos² field of ±10 mm with E₀L = 0.1 MeV at −30°, ∂pₓ,out/∂pₓ,in is 0.9928 of the Lorentz-force value at steps from 0.5 to 0.125 mm, and ∂pₓ,out/∂x_in 0.9821 to 0.9819. A static field gives no lens, where the exact ∂pₓ,out/∂pₓ,in is 1.0096.

The script builds a one-gap cavity with that field at steps of 0.5, 0.25 and 0.125 mm, tracks particles with (x, x′) = (0.1 mm, 0) and (0, 0.1 mrad) at the reference energy and phase, and compares the transverse 2×2 map in (x, pₓ) with a DOP853 integration of the paraxial Lorentz force of the same field. It repeats the comparison for a quasi-static field (1 kHz, on crest), adds the dropped term ½∫E cos ψ d(x/β) to the code's slope result to account for the difference, and uses the reference energy gain and a zero-field gap, which is a drift, as controls.
"""

import math
import os

for _v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ.setdefault(_v, "1")

from orbit.core.bunch import Bunch
from orbit.core.orbit_utils import Function
from orbit.py_linac.lattice import AxisFieldRF_Gap, BaseRF_Gap, LinacAccLattice, RF_Cavity, Sequence
from scipy.integrate import solve_ivp

C = 2.99792458e8
K0, ZMAX = 2.5e-3, 0.010  # GeV, m
M = Bunch().mass()  # GeV
A = 1.0e-4  # small amplitudes: linear map


def g(s):
    # cos^2 field shape with unit integral on [-ZMAX, ZMAX]
    return math.cos(math.pi * s / (2 * ZMAX)) ** 2 / ZMAX


def gp(s):
    return -math.pi / (2 * ZMAX * ZMAX) * math.sin(math.pi * s / ZMAX)


def build(step, e0l, freq, phi):
    lat, seq, cav = LinacAccLattice("lat"), Sequence("seq"), RF_Cavity("cav")
    cav.setFrequency(freq)
    cav.setPhase(phi)
    cav.setAmp(1.0)
    cav.setUsePhaseAtEntrance(True)  # phi is the RF phase when the reference enters the field
    base = BaseRF_Gap("gap")
    base.setParam("E0L", e0l)
    cav.addRF_GapNode(base)
    gap = AxisFieldRF_Gap(base)
    fn = Function()
    for i in range(4001):
        s = -ZMAX + 2 * ZMAX * i / 4000
        fn.add(s, g(s))
    fn.setConstStep(1)
    gap.setAxisFieldFunction(fn, step)
    cav.removeAllGapNodes()
    cav.addRF_GapNode(gap)
    seq.addNode(gap)
    seq.addRF_Cavity(cav)
    lat.addNode(gap)
    lat.initialize()
    return lat


def pyorbit_map(step, e0l, freq, phi):
    lat = build(step, e0l, freq, phi)
    b = Bunch()
    b.getSyncParticle().kinEnergy(K0)
    lat.trackDesignBunch(b)
    p_in = b.getSyncParticle().momentum()
    b.addParticle(A, 0.0, 0.0, 0.0, 0.0, 0.0)
    b.addParticle(0.0, A, 0.0, 0.0, 0.0, 0.0)
    lat.trackBunch(b)
    sp = b.getSyncParticle()
    p_out = sp.momentum()
    # map in (x [m], p_x [GeV/c]), xp = p_x/p_ref: columns from x0 = A and p_x0 = A p_in
    m = [[b.x(0) / A, b.x(1) / (A * p_in)], [b.xp(0) * p_out / A, b.xp(1) * p_out / (A * p_in)]]
    return m, sp.kinEnergy()


def lorentz(e0l, freq, phi, x0, px0):
    # Paraxial Lorentz force of the axis field E(s) cos(psi), psi = omega t + phi, s as independent variable:
    # E_x = -(x/2) E'(s) cos(psi) and c B_y = -(x omega/(2c)) E(s) sin(psi) (Maxwell, first order in x);
    # dp_x/ds = (E_x - beta_z c B_y)/beta_z, dp/ds = (W/p) E cos(psi), dt/ds = 1/(beta_z c).
    # R accumulates the term the kernel drops, 1/2 int E cos(psi) d(x/beta).
    om = 2 * math.pi * freq
    p0 = math.sqrt(K0 * (K0 + 2 * M))

    def rhs(s, y):
        t, p, x, px, _ = y
        w = math.sqrt(p * p + M * M)
        bz, gam = p / w, w / M
        psi = om * t + phi
        e = e0l * g(s)  # GV/m for unit charge
        ex = -(x / 2) * e0l * gp(s) * math.cos(psi)
        cby = -(x * om / (2 * C)) * e * math.sin(psi)
        dbeta = e * math.cos(psi) / (M * bz * gam**3)  # d beta/ds = (dK/ds)/(m beta gamma^3)
        dxb = (px / p) / bz - x * dbeta / bz**2  # d(x/beta)/ds
        return [
            1 / (bz * C),
            (w / p) * e * math.cos(psi),
            px / p,
            (ex - bz * cby) / bz,
            0.5 * e * math.cos(psi) * dxb,
        ]

    sol = solve_ivp(
        rhs,
        (-ZMAX, ZMAX),
        [0.0, p0, x0, px0, 0.0],
        method="DOP853",
        rtol=1e-13,
        atol=[1e-25, 1e-18, 1e-20, 1e-20, 1e-20],
    )
    _, p, x, px, r = sol.y[:, -1]
    return math.sqrt(p * p + M * M) - M, x, px, r


defects, controls_ok = 0, True
CASES = (
    ("402.5 MHz, E0L 0.1 MeV, -30 deg", 1.0e-4, 402.5e6, math.radians(-30.0), "case"),
    ("1 kHz (static), E0L 0.1 MeV, on crest", 1.0e-4, 1.0e3, 0.0, "case"),
    ("402.5 MHz, zero field", 0.0, 402.5e6, math.radians(-30.0), "control"),
)
for label, e0l, freq, phi, kind in CASES:
    p0 = math.sqrt(K0 * (K0 + 2 * M))
    k_ex, x1, px1, _ = lorentz(e0l, freq, phi, A, 0.0)
    _, x2, px2, r2 = lorentz(e0l, freq, phi, 0.0, A * p0)
    me = [[x1 / A, x2 / (A * p0)], [px1 / A, px2 / (A * p0)]]
    print(
        f"[{label}] Lorentz force: M11 {me[0][0]:.6f}, M21 {me[1][0]:.6e} GeV/c/m, M22 {me[1][1]:.6f}, "
        f"det {me[0][0] * me[1][1] - me[0][1] * me[1][0]:.9f}; gain {(k_ex - K0) * 1e6:.4f} keV"
    )
    for step in (5e-4, 2.5e-4, 1.25e-4):
        m, k = pyorbit_map(step, e0l, freq, phi)
        r21 = m[1][0] / me[1][0] if me[1][0] else float("nan")
        print(
            f"  step {step * 1e3:.3f} mm: M11 observed {m[0][0]:.6f}, expected {me[0][0]:.6f}; M21 observed "
            f"{m[1][0]:.6e}, expected {me[1][0]:.6e} (ratio {r21:.5f}); M22 observed {m[1][1]:.6f}, expected "
            f"{me[1][1]:.6f} (ratio {m[1][1] / me[1][1]:.5f}); gain observed - expected {(k - k_ex) * 1e9:+.3f} eV"
        )
        if step == 1.25e-4:
            off = abs(m[1][1] / me[1][1] - 1) > 1e-3
            if kind == "control":
                controls_ok &= not off and abs(m[1][0]) < 1e-15
            else:
                defects += off
                if freq > 1e6:
                    controls_ok &= (
                        abs(k - k_ex) < 1e-10
                    )  # the RF gap's energy gain converges (0.1 eV)
    if kind == "case" and freq > 1e6:
        m, _ = pyorbit_map(1.25e-4, e0l, freq, phi)
        print(
            f"  slope particle at 0.125 mm: code p_x {m[1][1] * A * p0:.9e} + dropped term 1/2 int E cos(psi) d(x/beta) "
            f"{r2:.9e} = {m[1][1] * A * p0 + r2:.9e} GeV/c; Lorentz force {px2:.9e} GeV/c"
        )

print("DEFECT REPRODUCED" if defects == 2 and controls_ok else "NOT REPRODUCED")
