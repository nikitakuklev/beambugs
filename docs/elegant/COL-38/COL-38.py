"""COL-38: CSRDRIFT spread mode with WAVELENGTH_MODE="peak-to-peak" uses an uninitialized wavelength

With SPREAD=1 and WAVELENGTH_MODE="peak-to-peak", CSRDRIFT's beam-spreading model takes its
wavelength from two wake indices that are never set: the extremum search runs on the wake array
before the bend hands it over and returns without writing them. For a 0.1 pC, 35 μm rms bunch
after a 0.6 m, 0.3 rad bend, the radiation-only spread formula with the 0.1791 mm peak-to-peak
wavelength gives an attenuation factor of 0.003415 for a 0.4 m drift, but the kick is applied
unattenuated (factor 1), and Valgrind reports the use of uninitialized values in the drift. The
default sigmaz mode differs from the formula by about 1.020×10⁻⁸ relative.

Setup: `valgrind` on `PATH`; the script runs some cases of serial Elegant under Memcheck.

The script tracks 40001 electrons (Gaussian quantiles, σ_ct = 35 μm, p = 4000 mₑc, 0.1 pC, x spread
over ±30 μm) through a one-slice 0.6 m, 0.3 rad CSRCSBEND (400 bins, STEADY_STATE=1) and a 0.4 m
CSRDRIFT with one kick, without SPREAD and with SPREAD=1, SPREAD_MODE="radiation-only" and
BUNCHLENGTH_MODE="rms" in both wavelength modes. Expected attenuation factors follow the model's
formula with the bend-exit beam size and the wavelength the mode selects; the sigmaz mode is the
control. Both spread modes are then run under Valgrind, which must report no use of uninitialized
values in the drift.
"""

import re

import numpy as np
from _csr import C, final, gaussian_beam, page, track, valgrind
from _elegant import particles, verdict

P0, SIGMA, RHO, DRIFT = 4000.0, 35e-6, 2.0, 0.4
BEND = (
    "L=0.6,ANGLE=0.3,N_SLICES=1,INTEGRATION_ORDER=6,SG_HALFWIDTH=1,SGDERIV_HALFWIDTH=3,"
    "SGDERIV_ORDER=3,BINS=400,STEADY_STATE=1"
)
BEAM = gaussian_beam(40001, SIGMA, P0)
BEAM["x"] = 3e-5 * np.cos(np.arange(40001) * 2.399963229728653)  # golden-angle spread in x
LINE = "Q,B,W,D"


def lattice(spread: str) -> str:
    drift = f"D: CSRDRIFT,L={DRIFT},N_KICKS=1" + (f",{spread}" if spread else "")
    return (
        "Q: CHARGE,TOTAL=1e-13\n"
        f'B: CSRCSBEND,{BEND},OUTPUT_FILE="wake.sdds"\n'
        'W: WATCH,FILENAME="bend.sdds",MODE="coordinates"\n' + drift
    )


def spread(mode: str) -> str:
    return f'SPREAD=1,SPREAD_MODE="radiation-only",WAVELENGTH_MODE="{mode}",BUNCHLENGTH_MODE="rms"'


def drift_kick(spread_text: str):
    result = track(
        lattice(spread_text), LINE, BEAM, P0, outputs=("out.sdds", "bend.sdds", "wake.sdds")
    )
    exit_beam = particles(result["bend.sdds"])
    return np.mean(final(result)["p"] - exit_beam["p"]), exit_beam, page(result["wake.sdds"])


def factor(wavelength: float, sigma_x: float) -> float:
    """Radiation-only spread factor of the drift kick at its middle, zTravel = L/2.

    The CSRDRIFT spread model widens the radiation by the angle
        theta = 0.5463e-3/(p 0.511e-3) (lambda_c/lambda)^(-1/3), lambda_c = 4.19 rho/p^3,
    of synchrotron radiation of wavelength lambda in a bend of radius rho, and scales the
    unattenuated wake by sigma_x/sqrt(sigma_x^2 + (zTravel theta)^2).
    """
    critical = 4.19 * RHO / P0**3
    theta = 0.5463e-3 / (P0 * 0.511e-3) / (critical / wavelength) ** (1 / 3)
    return sigma_x / np.sqrt(sigma_x**2 + (DRIFT / 2 * theta) ** 2)


base, exit_beam, (wake, info) = drift_kick("")
sigma_x = np.std(exit_beam["x"])
# The selected wavelengths: the rms bunch length (sigmaz), and twice the distance between the
# extrema of the bend's wake (peak-to-peak), from the exported wake.
lengths = {
    "sigmaz": np.std(C * exit_beam["t"]),
    "peak-to-peak": 2
    * abs(int(np.argmax(wake["DeltaGamma"])) - int(np.argmin(wake["DeltaGamma"])))
    * info["BinSize"],
}
print(f"serial no SPREAD: mean drift kick {base:.10g} m_e c; bend-exit sigma_x={sigma_x:.6g} m")
defect = False
for mode, wavelength in lengths.items():
    kick, _, _ = drift_kick(spread(mode))
    observed, expected = kick / base, factor(wavelength, sigma_x)
    good = np.isclose(observed, expected, rtol=1e-6)
    print(
        f"serial SPREAD=1 {mode} (wavelength {wavelength:.6g} m): attenuation factor "
        f"observed={observed:.10g} expected={expected:.10g}"
    )
    if mode == "sigmaz" and not good:
        raise SystemExit("The sigmaz control disagrees with the spread formula.")
    defect |= mode == "peak-to-peak" and not good
# Uninitialized values used by the drift, reported with the source line of the first use.
USE = re.compile(
    r"(?:Conditional jump or move depends on|Use of) uninitialised value.*\n(?:==\d+==\s+at .*\n)?==\d+==\s+(?:at|by) [^(]*track_through_driftCSR \(([^)]*)\)"
)
for mode in lengths:
    _, log = valgrind(lattice(spread(mode)), LINE, BEAM, P0)
    uses = list(dict.fromkeys(USE.findall(log)))
    shown = (
        f"uninitialized values used at {', '.join(uses)}"
        if uses
        else "no uninitialized value used in track_through_driftCSR"
    )
    print(f"serial SPREAD=1 {mode} under Valgrind: {shown}; expected none")
    if mode == "sigmaz" and uses:
        raise SystemExit("The sigmaz control uses uninitialized values.")
    defect |= mode == "peak-to-peak" and bool(uses)
verdict(defect)
