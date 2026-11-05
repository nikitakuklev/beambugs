"""COL-49: LSC voltage generators omit the source-charge sign and reverse positron kicks

After `change_particle name=positron`, longitudinal space-charge kicks reverse in LSCDRIFT and in the space-charge contributions of RFCW and CSRDRIFT. Reversing all source and witness charges must preserve the self-force, but both voltage generators omit the source-charge sign required by the energy-update routine. For a 100.0 pC Gaussian bunch with normalized momentum βγ = 30.00, where β is speed divided by the speed of light and γ is the Lorentz factor, a 1.000 m effective LSCDRIFT kick gives a particle one rms bunch length behind the centre +2799 eV instead of −2799 eV. A zero-charge control gives no kick.

The script tracks deterministic Gaussian longitudinal quantiles with equal transverse rms sizes through a zero-length LSCDRIFT with a one-metre effective length. It reconstructs exact energy changes particle by particle, compares the two species by charge conjugation, and checks a zero-charge control. Use --mpi to add three-rank Pelegant, with the native programs on PATH and RPN_DEFNS set.
"""

import numpy as np
from _elegant import modes, particles, run, verdict
from scipy.special import ndtri

C, REST_EV = 299792458.0, 510999.06
N, P0, SIGMA_CT, SIGMA_R = 8192, 30.0, 1e-3, 1e-4
CHARGE = 1e-10
ATOL_EV = 1e-8


def beam():
    z = SIGMA_CT * ndtri((np.arange(N) + 0.5) / N)
    phi = np.arange(N) * np.pi * (np.sqrt(5) - 1)
    x, y = np.cos(phi), np.sin(phi)
    return {
        "x": (x - x.mean()) / x.std() * SIGMA_R,
        "y": (y - y.mean()) / y.std() * SIGMA_R,
        "t": z / C,
        "p": np.full(N, P0),
        "particleID": np.arange(1, N + 1),
    }


def case(species, charge, mode):
    lattice = f"""Q: CHARGE,TOTAL={charge:.17g}
SC: LSCDRIFT,L=0,LEFFECTIVE=1,BINS=256,INTERPOLATE=1,SMOOTHING=0
BL: LINE=(Q,SC)
"""
    commands = f"""&global_settings echo_namelists=0 &end
&change_particle name={species} &end
&run_setup lattice=lattice.lte,use_beamline=BL,p_central={P0},output=out.sdds &end
&run_control &end
&sdds_beam input=beam.sdds &end
&track &end
"""
    out = particles(
        run({"lattice.lte": lattice, "run.ele": commands}, mode, {"beam.sdds": beam()})["out.sdds"]
    )
    if not np.array_equal(out["particleID"], np.arange(1, N + 1)):
        raise RuntimeError("Unexpected particle loss")
    return (np.hypot(out["p"], 1) - np.hypot(P0, 1)) * REST_EV


def main():
    defect = True
    z = C * beam()["t"]
    for mode in modes():
        electron = case("electron", CHARGE, mode)
        positron = case("positron", CHARGE, mode)
        zero = case("positron", 0, mode)
        # q_w E_source ~ q_w q_source: reversing both leaves Delta E identical.
        # This equality holds for the implemented periodic FFT model as well as for an
        # isolated bunch. It neither assumes a different boundary condition nor relies on
        # p0 approximately equalling gamma in the impedance.
        error = np.max(np.abs(positron - electron))
        reverse_error = np.max(np.abs(positron + electron))
        print(
            f"{mode} +1 sigma energy_change_eV: positron observed={np.interp(SIGMA_CT, z, positron):.15g} "
            f"expected={np.interp(SIGMA_CT, z, electron):.15g} (electron, charge conjugation)"
        )
        print(
            f"{mode} max_particle_energy_difference_eV={error:.15g} expected=0; "
            f"max_sign_reversal_residual_eV={reverse_error:.15g}"
        )
        print(
            f"{mode} zero-charge control: max_energy_change_eV observed={np.max(abs(zero)):.15g} expected=0"
        )
        if max(abs(electron)) <= ATOL_EV:
            raise RuntimeError("Electron control has no resolved LSC kick")
        if (error > ATOL_EV and reverse_error > ATOL_EV) or max(abs(zero)) > ATOL_EV:
            raise RuntimeError("Controls or sign-reversal prediction failed")
        defect &= error > ATOL_EV
    print(f"comparison: rtol=0, atol={ATOL_EV:g} eV; energy=mc^2 sqrt(1+p^2)")
    verdict(defect)


if __name__ == "__main__":
    main()
