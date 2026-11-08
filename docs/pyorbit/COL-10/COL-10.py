"""COL-10: `TImpedance` gives each particle the kick that belongs half a bin further along z

`TImpedance` bins the bunch's transverse moments at cell centres z_min + (j + ½)Δz, with Δz = C/N_b for ring length C and N_b bins, but reconstructs the kick with the phase 2π(z − z_min)/C, which places bin j at its lower edge. Every sideband n ± Q, with Q the tune, is therefore shifted in phase by π(n ± Q)/N_b, and a particle at z receives the kick that belongs at z + Δz/2. For a symmetric, rigidly offset Gaussian bunch with rms length 10 m in a 248 m ring and one resistive sideband at Q = 6.23, the kick should be odd about the bunch centre, but K(z) + K(−z) reaches 2 sin(πQ/N_b) of the peak kick: 30.46% with 128 bins and 60.21% with 64.

The script tracks a mirror-symmetric Gaussian bunch, all particles at x = 1 mm and x′ = 0, through `TImpedance` with one sideband line, resistive or reactive. A resistive line must give a kick that is odd about the bunch centre and a reactive line an even one, for either sign convention of the impedance or of z; the script prints the wrong-parity sum K(z) ± K(−z) next to 2 sin(πQ/N_b). As a control it compares the kick with the analytic single-line kick evaluated at z + Δz/2 and scaled by the spline form factor, which must match.
"""

import os

for _v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ.setdefault(_v, "1")

import numpy as np
from orbit.core.bunch import Bunch
from orbit.core.impedances import TImpedance
from orbit.utils.consts import mass_proton, speed_of_light

C = 248.0  # ring length and TImpedance phase length (m)
Q = 6.23  # betatron tune
BETA_T = 10.0  # Twiss beta (m), alpha = 0
X0 = 1e-3  # rigid offset (m)
MS = 1e10
SIG = 10.0  # rms bunch length (m)

rng = np.random.default_rng(7)
half = SIG * np.abs(rng.standard_normal(2000))
zs = np.concatenate([half, -half])  # mirror-symmetric about z = 0, so the bin window is [-C/2, C/2]


def run(nb, Z):
    B = Bunch()
    B.mass(mass_proton)
    B.macroSize(MS)
    B.getSyncParticle().kinEnergy(1.0)
    for z in zs:
        B.addParticle(X0, 0.0, 0.0, 0.0, z, 0.0)
    T = TImpedance(C, 1, nb, 1, 0)
    T.assignLatFuncs(Q, 0.0, BETA_T, Q, 0.0, BETA_T)
    Zp = [0j] * (nb // 2)
    Zm = [0j] * (nb // 2)
    Zp[0] = Z  # sideband at f0 Q
    Zm[0] = -np.conj(Z)  # its mirror at -f0 Q, Z(-f) = -conj(Z(f)); the wrapper mirrors only n >= 1
    T.assignImpedance("X", Zp, Zm)
    T.trackBunch(B)
    sp = B.getSyncParticle()
    # TImpedance's kick scale: macro size * 1e-9 * e * c / (beta * C * E_total[GeV])
    c2k = (
        MS * 1e-9 * 1.602176634e-19 * speed_of_light / (sp.beta() * C * (sp.kinEnergy() + B.mass()))
    )
    return np.array([B.xp(i) for i in range(B.getSize())]), c2k


bad, ctl_ok = False, True
n = len(half)
for nb in (128, 64):
    for lab, Z, odd in (("resistive", 1e6 + 0j, True), ("reactive", 1e6j, False)):
        k, c2k = run(nb, Z)
        kp, km = k[:n], k[n:]  # particle at +z and its mirror at -z
        # Single line, rigid bunch symmetric about 0: K(z) = c2k Im(Z A exp(i Q theta(z))) with
        # A = sum_i x_i exp(-i Q theta_i) and theta = 2 pi (z - z_min)/C. A exp(i Q theta(z)) depends only on
        # z - z_i, so K is odd in z for real Z and even for imaginary Z, whichever direction z points.
        wrong = (kp + km) if odd else (kp - km)
        frac = np.abs(wrong).max() / np.abs(k).max()
        th = 2 * np.pi * (zs + C / 2) / C
        A = np.sum(X0 * np.exp(-1j * Q * th))
        # Evaluated at z + dz/2 (theta + pi/N_b) and scaled by the quadratic-spline form factor
        # sinc^3(Q/N_b) of deposit and moment binning.
        ff = np.sinc(Q / nb) ** 3
        anh = ff * c2k * np.imag(Z * A * np.exp(1j * Q * (th + np.pi / nb)))
        ctl = np.abs(k - anh).max() / np.abs(anh).max()
        print(
            f"N_b = {nb:3d}, {lab} line: max|K(z) {'+' if odd else '-'} K(-z)|/peak observed {frac:.4f}, "
            f"expected 0 (parity), 2 sin(pi Q/N_b) = {2 * np.sin(np.pi * Q / nb):.4f}; "
            f"control max|K - analytic(z + dz/2)|/peak {ctl:.2e}, expected 0"
        )
        bad |= frac > 0.01
        ctl_ok &= ctl < 1e-3
print("DEFECT REPRODUCED" if bad and ctl_ok else "NOT REPRODUCED")
