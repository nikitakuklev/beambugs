"""COL-08: Broadband ZTRANSVERSE doubles its DC impedance

Broadband ZTRANSVERSE sets its zero-frequency impedance to 2Rₛ/Q instead of the documented Rₛ/Q, where Rₛ is the shunt impedance and Q the resonator quality factor. This adds a constant transverse wake over the finite Fourier-transform window for beams with a net transverse offset. For 12 electrons at 50 mₑc, 2 nC total charge and 1 mm horizontal offset in adjacent 50 ps bins, with Rₛ = 1×10⁶ Ω/m, Q = 1, resonant frequency 1 GHz and 8192 bins, the leading group's kick is 6.960×10⁻⁶ rad instead of the independently summed 6.769×10⁻⁶ rad. The excess is 1.911×10⁻⁷ rad per particle and halves when the window doubles.

The script tracks the beam through the broadband element and through a table of the same impedance on the same frequency grid, and compares both with the kick of the analytic impedance, summed as a Fourier series without an FFT. The table is the control: it agrees with the series to rounding, so the kick difference of the broadband element isolates its DC coefficient. Adjacent bins with equal offsets cancel the Nyquist source moment, leaving the DC term as the only difference. The excess is compared with the kick of an extra Rs/Q coefficient, for windows of 8192 and 16384 bins.
"""

import sys

import numpy as np
from _elegant import modes, verdict
from _ztransverse import CHARGE, COUNT, DT, ELECTRON_EV, RS, P, Q, kicks, oracle, show

ATOL = 1e-18  # rad; kicks are ~1e-5 rad, rounding of the sums ~1e-20 rad
MIN_ERROR = 1e-12  # rad; the DC excess is ~1e-7 rad
X = np.full(COUNT, 1e-3)  # m, equal offsets: the Nyquist source moment cancels

defect = False
for mode in modes():
    for n_bins in (8192, 16384):
        # Series kick of the analytic impedance (DC coefficient Rs/Q), the oracle.
        expected = oracle(X, n_bins)
        table = kicks(mode, X, n_bins, broadband=False)
        show(mode, f"N={n_bins} impedance table", table, expected)
        if not np.all(np.isfinite(expected)) or not np.all(np.isfinite(table)):
            sys.exit("Nonfinite oracle or impedance-table kicks.")
        if not np.allclose(table, expected, rtol=0, atol=ATOL):
            sys.exit("The table control disagrees with the Fourier-series oracle.")
        broadband = kicks(mode, X, n_bins, broadband=True)
        if not np.all(np.isfinite(broadband)):
            sys.exit("Nonfinite broadband kicks.")
        show(mode, f"N={n_bins} broadband", broadband, expected)
        # A DC coefficient larger by Rs/Q adds the same slope to every particle:
        #   q (Rs/Q) sum(x) / (T p0 m c^2/e), T = N dt the window, q the macroparticle charge.
        predicted = CHARGE / COUNT * (RS / Q) * X.sum() / (n_bins * DT * P * ELECTRON_EV)
        excess = broadband - expected
        show(mode, f"N={n_bins} broadband excess", excess, np.full(2, predicted))
        defect |= bool(np.all(np.abs(excess - predicted) < ATOL) and np.all(excess > MIN_ERROR))
print(f"tolerance {ATOL:g} rad; defect if the excess exceeds {MIN_ERROR:g} rad")
verdict(defect)
