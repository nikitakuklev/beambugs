---
title: "COL-49: LSC voltage generators omit the source-charge sign and reverse positron kicks"
hide:
  - toc
---

# COL-49 — LSC voltage generators omit the source-charge sign and reverse positron kicks

| | |
|---|---|
| Code | [Elegant](../index.md) · Collective effects |
| Priority | P3 |
| Failure | Wrong physics |
| Evidence | Reproduced, independently rechecked |
| Last reproduced | 2026-11-05 on Elegant commit [`0355dd6`](https://github.com/rtsoliday/elegant/commit/0355dd6aca12), committed 2026-11-03 |
| Source | [`track_through_lscdrift`, lsc.c:319–322](https://github.com/rtsoliday/elegant/blob/0355dd6aca12c861971fbf8f3ae78a0ded991734/src/lsc.c#L319-L322)<br>[`addLSCKick`, lsc.c:616–619](https://github.com/rtsoliday/elegant/blob/0355dd6aca12c861971fbf8f3ae78a0ded991734/src/lsc.c#L616-L619)<br>[`track_through_lscdrift`, lsc.c:361–365](https://github.com/rtsoliday/elegant/blob/0355dd6aca12c861971fbf8f3ae78a0ded991734/src/lsc.c#L361-L365)<br>[`addLSCKick`, lsc.c:645–648](https://github.com/rtsoliday/elegant/blob/0355dd6aca12c861971fbf8f3ae78a0ded991734/src/lsc.c#L645-L648)<br>[`applyLongitudinalWakeKicks`, wake.c:348–360](https://github.com/rtsoliday/elegant/blob/0355dd6aca12c861971fbf8f3ae78a0ded991734/src/wake.c#L348-L360)<br>[`process_particle_command`, elegant.c:3264–3281](https://github.com/rtsoliday/elegant/blob/0355dd6aca12c861971fbf8f3ae78a0ded991734/src/elegant.c#L3264-L3281)<br>[`do_tracking`, do_tracking.c:894–897](https://github.com/rtsoliday/elegant/blob/0355dd6aca12c861971fbf8f3ae78a0ded991734/src/do_tracking.c#L894-L897)<br>[`track_through_wake`, wake.c:285–290](https://github.com/rtsoliday/elegant/blob/0355dd6aca12c861971fbf8f3ae78a0ded991734/src/wake.c#L285-L290)<br>[`trackRfCavityAddLSCKick`, simple_rfca.c:404–414](https://github.com/rtsoliday/elegant/blob/0355dd6aca12c861971fbf8f3ae78a0ded991734/src/simple_rfca.c#L404-L414)<br>[`track_through_driftCSR_Stupakov`, csbend.c:5211–5212](https://github.com/rtsoliday/elegant/blob/0355dd6aca12c861971fbf8f3ae78a0ded991734/src/csbend.c#L5211-L5212)<br>[`track_through_driftCSR_Stupakov`, csbend.c:5228–5229](https://github.com/rtsoliday/elegant/blob/0355dd6aca12c861971fbf8f3ae78a0ded991734/src/csbend.c#L5228-L5229) |

After `change_particle name=positron`, longitudinal space-charge kicks reverse in LSCDRIFT and in the space-charge contributions of RFCW and CSRDRIFT. Reversing all source and witness charges must preserve the self-force, but both voltage generators omit the source-charge sign required by the energy-update routine. For a 100.0 pC Gaussian bunch with normalized momentum βγ = 30.00, where β is speed divided by the speed of light and γ is the Lorentz factor, a 1.000 m effective LSCDRIFT kick gives a particle one rms bunch length behind the centre +2799 eV instead of −2799 eV. A zero-charge control gives no kick.

Run `python COL-49.py` with [_elegant.py](../helpers/_elegant.py) in the same directory.

??? example "Reproducer: COL-49.py"

    [Download COL-49.py](COL-49.py)

    ```python
    """COL-49: LSC voltage generators omit the source-charge sign and reverse positron kicks

    After `change_particle name=positron`, longitudinal space-charge kicks reverse in LSCDRIFT and in the space-charge contributions of RFCW and CSRDRIFT. Reversing all source and witness charges must preserve the self-force, but both voltage generators omit the source-charge sign required by the energy-update routine. For a 100.0 pC Gaussian bunch with normalized momentum βγ = 30.00, where β is speed divided by the speed of light and γ is the Lorentz factor, a 1.000 m effective LSCDRIFT kick gives a particle one rms bunch length behind the centre +2799 eV instead of −2799 eV. A zero-charge control gives no kick.

    The script tracks deterministic Gaussian longitudinal quantiles with equal transverse rms sizes through a zero-length LSCDRIFT with a one-metre effective length. It reconstructs exact energy changes particle by particle, compares the two species by charge conjugation, and checks a zero-charge control. Use --mpi to add three-rank Pelegant.
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

    ```

## Provenance

| Role | Model | Effort | Date | Code examined |
|---|---|---|---|---|
| Found | <span class="bb-nowrap">`gpt-6-astra`</span> | xhigh | <span class="bb-nowrap">2026-11-05</span> | master (`0355dd6aca12`), source with SDDS a3f730a14c49, CPU, serial and 3-rank Pelegant |
| Confirmed | <span class="bb-nowrap">`gpt-6.1-sol`</span> | xhigh | <span class="bb-nowrap">2026-11-05</span> | master (`0355dd6aca12`), source with SDDS a3f730a14c49, CPU, serial and 3-rank Pelegant |

