"""COL-48: CSRCSBEND and CSRDRIFT reverse self-CSR kicks for positrons

After `change_particle name=positron`, CSRCSBEND reverses the coherent synchrotron radiation kick, making a bunch gain energy where an identical electron bunch loses it. Reversing every source and witness charge must preserve the self-force because both the source field and the witness charge reverse. A 100.0 pC Gaussian bunch at 511.0 MeV/c with 35.00 μm rms length gains 104.7 keV per particle on average in a 0.6000 m, 0.3000 rad bend instead of losing 104.7 keV. Disabling coherent radiation restores identical electron and positron transport.

The script tracks deterministic Gaussian quantiles through a single-slice steady-state CSR bend for electrons and positrons, with coherent radiation enabled and disabled. It compares every momentum kick using charge conjugation and checks a zero-charge control; incoherent radiation and random excitation are disabled. Use --mpi to add three-rank Pelegant.
"""

import numpy as np
from _elegant import modes, particles, run, verdict
from scipy.special import ndtri

P0 = 1000.0  # Absolute normalized momentum p/(m_e c), equal for electron and positron.
N = 32768
SIGMA_CT = 35e-6
REST_EV = 510999.06
C = 299792458.0
CHARGE = 1e-10  # CHARGE requires unsigned magnitude, including for positrons.
ATOL_P = 5e-12  # Absolute p/(m_e c); approx 2.6 micro-eV per particle.


def case(species, csr, charge, mode):
    beam = {
        "t": SIGMA_CT / C * ndtri((np.arange(N) + 0.5) / N),
        "p": np.full(N, P0),
        "particleID": np.arange(1, N + 1),
    }
    lattice = f"""Q: CHARGE,TOTAL={charge:.17g}
B: CSRCSBEND,L=0.6,ANGLE=0.3,N_SLICES=1,INTEGRATION_ORDER=6,BINS=256,STEADY_STATE=1,CSR={csr},SYNCH_RAD=0,ISR=0,SG_HALFWIDTH=1,SGDERIV_HALFWIDTH=3,SGDERIV_ORDER=3
BL: LINE=(Q,B)
"""
    commands = f"""&global_settings echo_namelists=0 &end
&change_particle name={species} &end
&run_setup lattice=lattice.lte,use_beamline=BL,p_central={P0},output=out.sdds &end
&run_control &end
&sdds_beam input=beam.sdds &end
&track &end
"""
    out = particles(
        run({"lattice.lte": lattice, "run.ele": commands}, mode, {"beam.sdds": beam})["out.sdds"]
    )
    if not np.array_equal(out["particleID"], np.arange(1, N + 1)):
        raise RuntimeError("Unexpected particle loss")
    return out


def mean_energy_change(on, off):
    # Reconstruct exact energy E = mc^2 sqrt(1+p^2), rather than treating momentum as energy.
    return np.mean(np.hypot(on["p"], 1) - np.hypot(off["p"], 1)) * REST_EV


def main():
    defect = True
    for mode in modes():
        electron_off = case("electron", 0, CHARGE, mode)
        positron_off = case("positron", 0, CHARGE, mode)
        for name in ("x", "xp", "y", "yp", "t", "p"):
            if not np.array_equal(electron_off[name], positron_off[name]):
                raise RuntimeError("CSR-off transport differs between species")
        print(f"{mode} CSR-off species control: max_coordinate_difference observed=0 expected=0")
        electron = case("electron", 1, CHARGE, mode)
        positron = case("positron", 1, CHARGE, mode)
        zero = case("positron", 1, 0.0, mode)
        if not np.allclose(zero["p"], positron_off["p"], rtol=0, atol=ATOL_P):
            raise RuntimeError("Zero-charge control failed")
        print(
            f"{mode} zero-charge control: max_delta_p observed={max(abs(zero['p'] - positron_off['p'])):.12g} expected=0"
        )
        electron_kick = electron["p"] - electron_off["p"]
        positron_kick = positron["p"] - positron_off["p"]
        # Maxwell fields are linear in source charge. Self-force is q*E_source ~ q^2:
        # simultaneous charge reversal preserves the entire kick. The single CSR slice
        # ensures both species deposit the same pre-kick density; the cold reference orbit
        # and fixed normalized bend angle are identical, as the off controls establish.
        error = max(abs(positron_kick - electron_kick))
        reverse_error = max(abs(positron_kick + electron_kick))
        print(
            f"{mode} mean_CSR_delta_p: positron observed={np.mean(positron_kick):.15g} "
            f"expected={np.mean(electron_kick):.15g} (equal electron self-field kick)"
        )
        print(
            f"{mode} mean_CSR_energy_change_eV: positron observed={mean_energy_change(positron, positron_off):.15g} "
            f"expected={mean_energy_change(electron, electron_off):.15g}"
        )
        print(
            f"{mode} max_particle_kick_difference={error:.15g} expected=0; "
            f"max_particle_sign_reversal_residual={reverse_error:.15g}"
        )
        if np.mean(electron_kick) >= 0 or max(abs(electron_kick)) <= ATOL_P:
            raise RuntimeError("Electron control does not have a resolved CSR loss")
        if error > ATOL_P and reverse_error > ATOL_P:
            raise RuntimeError("The identified sign-reversal cause does not explain the result")
        defect &= error > ATOL_P
    print(f"comparison: rtol=0, atol={ATOL_P:g} in p/(m_e c); species controls are bit-identical")
    verdict(defect)


if __name__ == "__main__":
    main()
