---
title: "IO-01: Single-harmonic longitudinal generation uses proton mass for other species"
hide:
  - toc
---

# IO-01 — Single-harmonic longitudinal generation uses proton mass for other species

| | |
|---|---|
| Code | [Xsuite](../index.md) · Lattices, particles and I/O |
| Priority | P2 |
| Severity | Wrong physics |
| Evidence | Probe-pinned |
| Last reproduced | 2026-10-26 on xpart main (`2633e52`) |
| Source | [`generate_longitudinal_coordinates`, generate_longitudinal.py:450–482](https://github.com/xsuite/xpart/blob/2633e528018af2e2de24db94122fbab7956ab12e/xpart/longitudinal/generate_longitudinal.py#L450-L482) |

With `engine='single-rf-harmonic'`, `generate_longitudinal_coordinates` samples the energy coordinate p_τ (`ptau`) for the requested reference particle, but converts it to the relative momentum deviation δ (`delta`) through a temporary `Particles` built without the reference mass, so the default proton mass is used for every species. The relativistic relation δ = √(1 + p_τ² + 2p_τ/β₀) − 1, with the reference speed β₀ = v/c of the actual species, gives the correct value, and the proton case agrees with it. For a 5 MeV/c positron bunch matched to 1 kV RF at harmonic 1 in a 100 m line, one particle gets δ = 0.03520 instead of 1.919×10⁻⁴, another 0.1245 instead of 7.083×10⁻⁴. Positions `zeta` are correct.

Download these shared helpers into the same directory as the reproducer scripts:

- [Download _common.py](../helpers/_common.py) · [View source](../helpers/index.md#_common)

??? example "Reproducer: IO-01.py"

    [Download IO-01.py](IO-01.py)

    ```python
    """IO-01: Single-harmonic longitudinal generation uses proton mass for other species

    With `engine='single-rf-harmonic'`, `generate_longitudinal_coordinates` samples the energy
    coordinate p_τ (`ptau`) for the requested reference particle, but converts it to the relative
    momentum deviation δ (`delta`) through a temporary `Particles` built without the reference mass, so
    the default proton mass is used for every species. The relativistic relation δ = √(1 + p_τ² +
    2p_τ/β₀) − 1, with the reference speed β₀ = v/c of the actual species, gives the correct value, and
    the proton case agrees with it. For a 5 MeV/c positron bunch matched to 1 kV RF at harmonic 1 in a
    100 m line, one particle gets δ = 0.03520 instead of 1.919×10⁻⁴, another 0.1245 instead of
    7.083×10⁻⁴. Positions `zeta` are correct.

    The script builds a 100 m `LineSegmentMap` with 1 kV RF at harmonic 1 and momentum compaction 0.1,
    and generates eight particles with a parabolic profile and RMS bunch length 0.5 m, once for a 5
    MeV/c positron reference and once for a 1 GeV/c proton control. It replays the public
    `SingleRFHarmonicMatcher` with the same NumPy seed (12345, restored on exit) to recover the sampled
    p_τ, checks that the generated `zeta` agree, and compares the generated `delta` with the
    relativistic δ for the actual reference mass. The proton must agree; the positron shows the bug.
    """

    import numpy as np
    import xpart as xp
    import xtrack as xt
    from scipy.constants import c
    from xpart.longitudinal.single_rf_harmonic_matcher import SingleRFHarmonicMatcher

    RTOL, ATOL = 3e-13, 2e-15
    SEED = 12345


    def evaluate():
        cases = {}
        saved_rng = np.random.get_state()
        try:
            for species, mass, p0c in (
                ("proton", xt.PROTON_MASS_EV, 1e9),
                ("positron", xt.ELECTRON_MASS_EV, 5e6),
            ):
                reference = xt.Particles(mass0=mass, q0=1, p0c=p0c)
                beta = float(reference.beta0[0])
                eta = 0.1 - 1 / float(reference.gamma0[0]) ** 2
                frequency = beta * c / 100
                line = xt.Line(
                    elements=[
                        xt.LineSegmentMap(
                            length=100,
                            qx=0.2,
                            qy=0.3,
                            betx=1,
                            bety=1,
                            longitudinal_mode="linear_fixed_rf",
                            voltage_rf=1000,
                            frequency_rf=frequency,
                            phase_rf=np.pi if eta > 0 else 0,
                            momentum_compaction_factor=0.1,
                        )
                    ]
                )
                line.particle_ref = reference
                line.build_tracker()
                twiss = line.twiss()
                np.random.seed(SEED)
                zeta, observed, _ = xp.generate_longitudinal_coordinates(
                    line=line,
                    num_particles=8,
                    sigma_z=0.5,
                    engine="single-rf-harmonic",
                    distribution="parabolic",
                    return_matcher=True,
                )
                # Replay the public matcher with identical inputs and random seed to
                # recover the exact sampled energy coordinate before wrapper conversion.
                np.random.seed(SEED)
                sampler = SingleRFHarmonicMatcher(
                    q0=1,
                    voltage=1000,
                    length=100,
                    freq=frequency,
                    p0c=p0c,
                    slip_factor=twiss.slip_factor,
                    beta0=beta,
                    rms_bunch_length=0.5 / beta,
                    distribution="parabolic",
                )
                tau, ptau = sampler.sample_tau_ptau(n_particles=8)
                tau, ptau = np.asarray(tau), np.asarray(ptau)
                np.testing.assert_allclose(zeta, beta * tau, rtol=0, atol=1e-14)
                # E=E0+p0c*ptau and E**2=(pc)**2+mass0**2. Rationalization
                # avoids cancellation when recovering a small delta.
                u = ptau**2 + 2 * ptau / beta
                assert np.all(1 + u > 0)
                expected = u / (np.sqrt(1 + u) + 1)
                explicit = xt.Particles(mass0=mass, q0=1, p0c=p0c, ptau=ptau)
                np.testing.assert_allclose(explicit.delta, expected, rtol=RTOL, atol=ATOL)
                if species == "proton":
                    np.testing.assert_allclose(observed, expected, rtol=RTOL, atol=ATOL)
                else:
                    cases["positron generated delta"] = observed, expected
        finally:
            np.random.set_state(saved_rng)
        return cases


    if __name__ == "__main__":
        from _common import prepare, report

        prepare(random_seed=SEED)
        print("Expected: delta=sqrt(1+ptau**2+2*ptau/beta0)-1, using the actual reference species.")
        print(
            "A seeded public-matcher replay checks identical sampled zeta and supplies ptau; proton conversion is a control."
        )
        report(evaluate(), rtol=RTOL, atol=ATOL)

    ```

## Provenance

| Role | Model | Effort | Date | Code examined |
|---|---|---|---|---|
| Found | `gpt-6-astra` | xhigh | 2026-10-26 | xpart 0.23.23 (`2633e528018a`), installed packages, CPU, serial |
| Confirmed | `gpt-6.1-sol` | low | 2026-10-26 | xpart 0.23.23 (`2633e528018a`), installed packages, CPU, serial |

