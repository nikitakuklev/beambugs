---
title: "TRK-56: SHRFDF reverses the longitudinal kick of its fundamental and all nine higher space harmonics"
hide:
  - toc
---

# TRK-56 — SHRFDF reverses the longitudinal kick of its fundamental and all nine higher space harmonics

| | |
|---|---|
| Code | [Elegant](../index.md) · Tracking |
| Priority | P3 |
| Severity | Wrong physics |
| Evidence | Reproduced, independently rechecked |
| Last reproduced | 2026-11-06 on Elegant commit [`0355dd6`](https://github.com/rtsoliday/elegant/commit/0355dd6aca12), committed 2026-11-03 |
| Tags | <span class="bb-tag">partly-source-only</span> [?](../../methodology.md#tags) |
| Source | [src/shrfdf.c:57–70](https://github.com/rtsoliday/elegant/blob/0355dd6aca12c861971fbf8f3ae78a0ded991734/src/shrfdf.c#L57-L70)<br>[src/shrfdf.c:73–82](https://github.com/rtsoliday/elegant/blob/0355dd6aca12c861971fbf8f3ae78a0ded991734/src/shrfdf.c#L73-L82)<br>[src/shrfdf.c:84–102](https://github.com/rtsoliday/elegant/blob/0355dd6aca12c861971fbf8f3ae78a0ded991734/src/shrfdf.c#L84-L102)<br>[doc/shrfdf.tex:1–3](https://github.com/rtsoliday/elegant/blob/0355dd6aca12c861971fbf8f3ae78a0ded991734/doc/shrfdf.tex#L1-L3)<br>[doc/shrfdf.tex:6–21](https://github.com/rtsoliday/elegant/blob/0355dd6aca12c861971fbf8f3ae78a0ded991734/doc/shrfdf.tex#L6-L21)<br>[doc/rfdf.tex:11–34](https://github.com/rtsoliday/elegant/blob/0355dd6aca12c861971fbf8f3ae78a0ded991734/doc/rfdf.tex#L11-L34)<br>[phase and coupling, src/track_rf.c:111–125](https://github.com/rtsoliday/elegant/blob/0355dd6aca12c861971fbf8f3ae78a0ded991734/src/track_rf.c#L111-L125)<br>[kicks, src/track_rf.c:198–211](https://github.com/rtsoliday/elegant/blob/0355dd6aca12c861971fbf8f3ae78a0ded991734/src/track_rf.c#L198-L211)<br>[src/track_data.c:3954–3979](https://github.com/rtsoliday/elegant/blob/0355dd6aca12c861971fbf8f3ae78a0ded991734/src/track_data.c#L3954-L3979)<br>[src/elegant.c:3264–3310](https://github.com/rtsoliday/elegant/blob/0355dd6aca12c861971fbf8f3ae78a0ded991734/src/elegant.c#L3264-L3310)<br>[src/do_tracking.c:1510–1513](https://github.com/rtsoliday/elegant/blob/0355dd6aca12c861971fbf8f3ae78a0ded991734/src/do_tracking.c#L1510-L1513)<br>[src/shrfdf.c:41–50](https://github.com/rtsoliday/elegant/blob/0355dd6aca12c861971fbf8f3ae78a0ded991734/src/shrfdf.c#L41-L50) |

Tracking through SHRFDF gives its longitudinal RF kick the opposite sign required by its transverse deflection, for the fundamental mode and all nine higher space harmonics. With a roughly 1 GeV electron beam, a 1 MV, 3 GHz fundamental mode at the transverse zero crossing adds 62.88 keV at a horizontal offset of +1.000 mm instead of removing 62.88 keV. The error is 125.8 keV. A phase-matched thin RFDF and independent electromagnetic and canonical checks establish the correct sign. The charge factors scale consistently; reversing charge does not restore the required coupling.

Run `python TRK-56.py` with [_elegant.py](../helpers/_elegant.py) in the same directory.

??? example "Reproducer: TRK-56.py"

    [Download TRK-56.py](TRK-56.py)

    ```python
    """TRK-56: SHRFDF reverses the longitudinal kick of its fundamental and all nine higher space harmonics

    Tracking through SHRFDF gives its longitudinal RF kick the opposite sign required by its transverse deflection, for the fundamental mode and all nine higher space harmonics. With a roughly 1 GeV electron beam, a 1 MV, 3 GHz fundamental mode at the transverse zero crossing adds 62.88 keV at a horizontal offset of +1.000 mm instead of removing 62.88 keV. The error is 125.8 keV. A phase-matched thin RFDF and independent electromagnetic and canonical checks establish the correct sign. The charge factors scale consistently; reversing charge does not restore the required coupling.

    Measure the transverse response of each isolated mode and infer its longitudinal impulse with a fourth-order time stencil and Simpson integration. Check four signed charges, phase-matched thin RFDF, canonical mixed derivatives and zero-strength controls using fixed particles in fresh native processes. Pass --mpi to repeat all ten electron modes in three-rank Pelegant.
    """

    import math
    import os

    for _name in (
        "OMP_NUM_THREADS",
        "MKL_NUM_THREADS",
        "OPENBLAS_NUM_THREADS",
        "NUMEXPR_NUM_THREADS",
        "NPROC",
        "PJRT_NPROC",
    ):
        os.environ[_name] = "1"

    import _elegant
    import numpy as np
    from _elegant import modes, particles, run, verdict

    _elegant.TIMEOUT = 290  # ty: ignore[invalid-assignment]  # Per-native-process deadline.
    # Use the executed native build's constants, rather than modern CODATA values.
    C = 299792458.0
    E = 1.60217733e-19
    MASS = 9.1093897e-31
    REST_EV = MASS * C**2 / E
    P0 = 2000.0
    VOLTAGE = 1e6
    K0 = 2 * math.pi * 3e9 / C
    PERIOD_PHASE = 2 * math.pi / 3
    PERIOD_LENGTH = PERIOD_PHASE / K0


    def native(element, rays, charge=-1, tilt=0.0, mode="serial", p0=P0):
        """Track fixed local-midplane rays; convert slopes into mechanical momenta."""
        u, ct = np.asarray(rays).T
        count = len(rays)
        beam = {
            "x": u * math.cos(tilt),
            "xp": np.zeros(count),
            "y": u * math.sin(tilt),
            "yp": np.zeros(count),
            "t": ct / C,
            "p": np.full(count, p0),
            "particleID": np.arange(1, count + 1, dtype=np.int32),
        }
        species = {-1: "electron", 1: "positron"}.get(charge)
        # particleCharge is minus physical charge; charge_ratio is relative to -e.
        species_command = (
            f"name={species}" if species else f"name=custom,mass_ratio=1,charge_ratio={-charge}"
        )
        files = {
            "lattice.lte": f"D: {element},TILT={tilt:.17g}\nBL: LINE=(D)\n",
            "run.ele": (
                f"&change_particle {species_command} &end\n"
                f"&run_setup lattice=lattice.lte,use_beamline=BL,p_central={p0:.17g},"
                "output=out.sdds,concat_order=0,random_number_seed=271828 &end\n"
                "&run_control n_steps=1,n_passes=1 &end\n"
                "&sdds_beam input=beam.sdds &end\n&track &end\n"
            ),
        }
        result = run(files, mode, {"beam.sdds": beam})
        out = particles(result["out.sdds"])
        np.testing.assert_array_equal(out["particleID"], beam["particleID"])
        np.testing.assert_allclose(out["t"], ct / C, rtol=0, atol=2e-25)
        pz = out["p"] / np.sqrt(1 + out["xp"] ** 2 + out["yp"] ** 2)
        out["pu"] = pz * (out["xp"] * math.cos(tilt) + out["yp"] * math.sin(tilt))
        out["pz"] = pz
        # Rationalize gamma(p)-gamma(p0) to preserve the small measured energy change.
        out["energy_eV"] = (
            (out["p"] - p0) * (out["p"] + p0) / (np.hypot(1, out["p"]) + math.hypot(1, p0)) * REST_EV
        )
        return out


    def element(harmonic, voltage=VOLTAGE, phase=0.0, factor=1.0):
        return (
            f"SHRFDF,PERIOD_LENGTH={PERIOD_LENGTH:.17g},PERIOD_PHASE={PERIOD_PHASE:.17g},"
            f"V{harmonic}={voltage:.17g},PHASE{harmonic}={phase:.17g},FACTOR={factor:.17g}"
        )


    def energy_from_impulse(dp):
        # Exact relativistic energy change for a longitudinal momentum impulse at zero px.
        return dp * (2 * P0 + dp) / (np.hypot(1, P0 + dp) + math.hypot(1, P0)) * REST_EV


    def sign_defect(observed, expected, faulty, atol):
        """Recognize the physical result or the established reversal; reject other output."""
        if math.isclose(observed, expected, rel_tol=0, abs_tol=atol):
            return False
        np.testing.assert_allclose(observed, faulty, rtol=0, atol=atol)
        return True


    def mode_case(harmonic, charge, mode="serial", tilt=0.0, factor=1.0, phase_step=1e-4):
        """Infer the longitudinal sign from the measured transverse time dependence."""
        k = K0 * (1 + 3 * harmonic)
        extent = 1e-3 if harmonic == 0 else 1e-4
        h = phase_step / k
        us = np.asarray([-extent, -extent / 2, 0, extent / 2, extent])
        cts = np.asarray([-2 * h, -h, 0, h, 2 * h])
        rays = [(u, ct) for u in us for ct in cts]
        out = native(element(harmonic, factor=factor), rays, charge, tilt, mode)
        pu = out["pu"].reshape(5, 5)
        # The arrival-time canonical pair is (ct,-gamma). Therefore the longitudinal
        # impulse is -integral[d(Delta pu)/d(ct)] du, in the model's beta->1 limit.
        # A fourth-order stencil and Simpson quadrature exactly integrate the quadratic
        # transverse profile, without consulting the implemented longitudinal impulse.
        derivative = (pu[:, 0] - 8 * pu[:, 1] + 8 * pu[:, 3] - pu[:, 4]) / (12 * h)
        inferred = -extent / 6 * (derivative[2] + 4 * derivative[3] + derivative[4])
        amplitude = -charge * VOLTAGE * E / (MASS * C**2) * factor
        alpha2 = k**2 - K0**2
        integral_shape = extent if harmonic == 0 else extent * (0.5 + alpha2 * extent**2 / 16)
        np.testing.assert_allclose(inferred, -amplitude * k * integral_shape, rtol=3e-14, atol=1e-15)
        np.testing.assert_allclose(pu[:, 2], 0, rtol=0, atol=2e-15)
        expected = float(energy_from_impulse(inferred))
        observed = float(out["energy_eV"].reshape(5, 5)[4, 2])
        faulty = float(energy_from_impulse(-inferred))
        defect = sign_defect(observed, expected, faulty, 4e-7)
        print(
            f"{mode}/n={harmonic}/Q={charge:+d}/tilt={tilt:g}/factor={factor:g}/step={phase_step:g}: "
            f"energy={observed:+.9f} eV expected={expected:+.9f} eV "
            f"(negative integral of measured dpu/d(ct)); reversed={faulty:+.9f} eV"
        )
        return defect


    def rfdf_controls(charge, phase=0.0):
        rays = [(x, ct) for x in (-1e-3, 0, 1e-3) for ct in (-2e-4, 0, 2e-4)]
        sh = native(element(0, phase=phase), rays, charge)
        rf = native(
            f"RFDF,L=0,VOLTAGE={VOLTAGE:.17g},FREQUENCY=3e9,N_KICKS=1,"
            f"MAGNETIC_DEFLECTION=1,PHASE={(phase - math.pi / 2) * 180 / math.pi:.17g}",
            rays,
            charge,
        )
        amplitude = -charge * VOLTAGE * E / (MASS * C**2)
        x, ct = np.asarray(rays).T
        transverse = amplitude * np.sin(K0 * ct + phase)
        np.testing.assert_allclose(sh["pu"], transverse, rtol=0, atol=5e-15)
        np.testing.assert_allclose(rf["pu"], transverse, rtol=0, atol=5e-15)
        # Magnetic deflection preserves |p| before the electric longitudinal impulse.
        base_pz = np.sqrt(P0**2 - transverse**2)
        dpz = -amplitude * K0 * x * np.cos(K0 * ct + phase)
        expected_p = np.sqrt(transverse**2 + (base_pz + dpz) ** 2)
        np.testing.assert_allclose(rf["p"], expected_p, rtol=0, atol=8e-13)
        transverse_error = float(np.max(np.abs(sh["pu"] - rf["pu"])))
        rf_p_error = float(np.max(np.abs(rf["p"] - expected_p)))
        print(
            f"RFDF/Q={charge:+d}/phase={phase:g}: transverse difference={transverse_error:.4g} mc "
            f"expected=0; RFDF p error={rf_p_error:.4g} mc expected=0 "
            f"(magnetic rotation plus -A*k*x*cos phase); "
            f"x=+1mm,t=0: RFDF={rf['energy_eV'][7]:+.9f} eV, "
            f"SHRFDF={sh['energy_eV'][7]:+.9f} eV; PASS"
        )


    def canonical_case(harmonic, charge):
        """Resolve the relative sign where 1-beta is below the numerical resolution."""
        p0 = 1e6
        k = K0 * (1 + 3 * harmonic)
        hx, ht = 1e-3, 1e-4 / k
        rays = [(i * hx, 0) for i in (-2, -1, 0, 1, 2)]
        rays += [(0, i * ht) for i in (-2, -1, 1, 2)]
        out = native(element(harmonic), rays, charge, p0=p0)
        pu = out["pu"]
        # Conjugate variables zeta=-ct, eta=(gamma-gamma0)/p0 and Pu=pu/p0 require
        # d(Delta Pu)/d(zeta)=d(Delta eta)/du. Paired rationalized gamma differences
        # avoid cancellation when differentiating the measured momentum at p0=1e6.
        transverse = -(pu[5] - 8 * pu[6] + 8 * pu[7] - pu[8]) / (12 * ht * p0)
        p = out["p"]
        gamma = np.hypot(1, p)
        gplus = (p[3] - p[1]) * (p[3] + p[1]) / (gamma[3] + gamma[1])
        gouter = (p[4] - p[0]) * (p[4] + p[0]) / (gamma[4] + gamma[0])
        longitudinal = (8 * gplus - gouter) / (12 * hx * p0)
        defect = sign_defect(longitudinal, transverse, -transverse, 2e-13)
        print(
            f"canonical/n={harmonic}/Q={charge:+d}/p0={p0:g}: "
            f"deta/du={longitudinal:+.15g} expected={transverse:+.15g} 1/m "
            "(dPu/dzeta, common Hamiltonian)"
        )
        return defect


    def main():
        selected_modes = modes()
        defect = False
        for charge in (-2, -1, 1, 2):
            for harmonic in range(10):
                defect |= mode_case(harmonic, charge)
            rfdf_controls(charge)
        rfdf_controls(-1, phase=0.31)
        for harmonic in (0, 9):
            defect |= mode_case(harmonic, -1, tilt=0.41)
            defect |= mode_case(harmonic, -1, factor=-0.7)
            defect |= mode_case(harmonic, -1, phase_step=5e-5)
            for charge in (-2, -1, 1, 2):
                defect |= canonical_case(harmonic, charge)
        for label, elem in (
            ("zero-voltage", element(0, voltage=0)),
            ("zero-factor", element(9, factor=0)),
        ):
            rays = [(x, ct) for x in (-1e-3, 0, 1e-3) for ct in (-1e-4, 0, 1e-4)]
            out = native(elem, rays)
            np.testing.assert_array_equal(out["pu"], 0)
            np.testing.assert_array_equal(out["energy_eV"], 0)
            print(f"{label}: transverse=0 expected=0 mc; energy=0 expected=0 eV (zero strength); PASS")
        if "mpi" in selected_modes:
            for harmonic in range(10):
                defect |= mode_case(harmonic, -1, mode="mpi")
        verdict(defect)


    if __name__ == "__main__":
        main()

    ```

## Provenance

| Role | Model | Effort | Date | Code examined |
|---|---|---|---|---|
| Found | `gpt-6-astra` | xhigh | 2026-11-06 | master (`0355dd6aca12`), source with SDDS a3f730a14c49, CPU, double, serial, one thread |
| Confirmed<br><small>all ten modes and four signed charges; electron MPI</small> | `gpt-6.1-sol` | xhigh | 2026-11-06 | master (`0355dd6aca12`), source with SDDS a3f730a14c49, CPU, double, serial and 3-rank Pelegant, one thread |

