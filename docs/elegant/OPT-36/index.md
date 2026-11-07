---
title: "OPT-36: MULT sextupoles are omitted from resonance-driving calculations"
hide:
  - toc
---

# OPT-36 — MULT sextupoles are omitted from resonance-driving calculations

| | |
|---|---|
| Code | [Elegant](../index.md) · Optics |
| Priority | P2 |
| Severity | Wrong physics |
| Evidence | Reproduced, independently rechecked |
| Last reproduced | 2026-11-07 on Elegant commit [`0355dd6`](https://github.com/rtsoliday/elegant/commit/0355dd6aca12), committed 2026-11-03 |
| Tags | <span class="bb-tag">partly-source-only</span> [?](../../methodology.md#tags) |
| Source | [src/twiss.cc:5012–5069](https://github.com/rtsoliday/elegant/blob/0355dd6aca12c861971fbf8f3ae78a0ded991734/src/twiss.cc#L5012-L5069)<br>[src/twiss.cc:5227–5274](https://github.com/rtsoliday/elegant/blob/0355dd6aca12c861971fbf8f3ae78a0ded991734/src/twiss.cc#L5227-L5274)<br>[src/twiss.cc:5316–5328](https://github.com/rtsoliday/elegant/blob/0355dd6aca12c861971fbf8f3ae78a0ded991734/src/twiss.cc#L5316-L5328)<br>[src/twiss.cc:5329–5344](https://github.com/rtsoliday/elegant/blob/0355dd6aca12c861971fbf8f3ae78a0ded991734/src/twiss.cc#L5329-L5344)<br>[src/twiss.cc:5410–5426](https://github.com/rtsoliday/elegant/blob/0355dd6aca12c861971fbf8f3ae78a0ded991734/src/twiss.cc#L5410-L5426)<br>[doc/mult.tex:3–16](https://github.com/rtsoliday/elegant/blob/0355dd6aca12c861971fbf8f3ae78a0ded991734/doc/mult.tex#L3-L16)<br>[src/twiss.cc:3128–3138](https://github.com/rtsoliday/elegant/blob/0355dd6aca12c861971fbf8f3ae78a0ded991734/src/twiss.cc#L3128-L3138)<br>[3174–3193](https://github.com/rtsoliday/elegant/blob/0355dd6aca12c861971fbf8f3ae78a0ded991734/src/twiss.cc#L3174-L3193)<br>[doc/elegant.tex:8176–8189](https://github.com/rtsoliday/elegant/blob/0355dd6aca12c861971fbf8f3ae78a0ded991734/doc/elegant.tex#L8176-L8189)<br>[src/twiss.cc:2273–2309](https://github.com/rtsoliday/elegant/blob/0355dd6aca12c861971fbf8f3ae78a0ded991734/src/twiss.cc#L2273-L2309) |

Sextupoles defined as MULT elements with ORDER=2 contribute nothing to the resonance driving terms that twiss_output computes with compute_driving_terms or s_dependent_driving_terms_file, whether the MULT is thin (zero length) or has finite length. For integrated sextupole strength K₂L = 0.003000 m⁻² in a periodic FODO cell, the coefficient h₃₀₀₀₀, which drives the horizontal third-integer resonance, is zero instead of 8.697×10⁻⁴ + 2.452×10⁻⁴i; the s-dependent f₃₀₀₀₀ and the derivative of h₃₀₀₀₀ with respect to KNL are also zero. The expected value is the Fourier component of the sextupole Hamiltonian, and an equivalent KSEXT reproduces it. Particle tracking through the same MULT applies the correct sextupole kick.

Run `python OPT-36.py` with [_slicing.py](../helpers/_slicing.py) and [_elegant.py](../helpers/_elegant.py) in the same directory.

??? example "Reproducer: OPT-36.py"

    [Download OPT-36.py](OPT-36.py)

    ```python
    """OPT-36: MULT sextupoles are omitted from resonance-driving calculations

    Sextupoles defined as MULT elements with ORDER=2 contribute nothing to the resonance driving terms that twiss_output computes with compute_driving_terms or s_dependent_driving_terms_file, whether the MULT is thin (zero length) or has finite length. For integrated sextupole strength K₂L = 0.003000 m⁻² in a periodic FODO cell, the coefficient h₃₀₀₀₀, which drives the horizontal third-integer resonance, is zero instead of 8.697×10⁻⁴ + 2.452×10⁻⁴i; the s-dependent f₃₀₀₀₀ and the derivative of h₃₀₀₀₀ with respect to KNL are also zero. The expected value is the Fourier component of the sextupole Hamiltonian, and an equivalent KSEXT reproduces it. Particle tracking through the same MULT applies the correct sextupole kick.

    Compute the periodic optics of a FODO cell from thick-quadrupole matrices and extract the sextupole Hamiltonian’s third Fourier coefficient on 64 phases. Compare global and s-dependent native coefficients for thin/finite MULT, signed/zero strengths and equivalent KSEXT controls, then check the tracked canonical kick.
    """

    from _slicing import canonical, close, discrepancy, track

    # isort: split
    import _elegant as native
    import numpy as np


    def drift(s):
        return np.array([[1.0, s], [0.0, 1.0]])


    def quad(k):
        w = np.sqrt(abs(k))
        c, s = (np.cos(w * 0.25), np.sin(w * 0.25)) if k > 0 else (np.cosh(w * 0.25), np.sinh(w * 0.25))
        return np.array([[c, s / w], [-w * s if k > 0 else w * s, c]])


    def main():
        faults = []
        strength = 0.003
        phase = np.arange(64) * (2 * np.pi / 64)
        for kind, length in (("MULT", 0.0), ("MULT", 0.14), ("KSEXT", 1e-7), ("KSEXT", 0.14)):
            matrix = drift(0.8) @ quad(-1.5) @ drift(0.8 + length) @ quad(1.5)
            mu = np.arccos(np.trace(matrix) / 2)
            beta = matrix[0, 1] / np.sin(mu)
            alpha = (matrix[0, 0] - matrix[1, 1]) / (2 * np.sin(mu))
            gamma = (1 + alpha**2) / beta

            def optics(s, beta=beta, alpha=alpha, gamma=gamma):
                m = drift(s) @ quad(1.5)
                b = m[0, 0] ** 2 * beta - 2 * m[0, 0] * m[0, 1] * alpha + m[0, 1] ** 2 * gamma
                psi = np.arctan2(m[0, 1] / beta, m[0, 0] - m[0, 1] * alpha / beta)
                return b, psi

            bm, pm = (np.array(optics(0.35)) + optics(0.35 + length)) / 2
            signed = []
            for scale in (1, -1, 0):
                kl = strength * scale
                pars = f"ORDER=2,KNL={kl}" if kind == "MULT" else f"K2={kl / length},N_KICKS=4"
                lattice = f"QF:QUAD,L=.25,K1=1.5\nQD:QUAD,L=.25,K1=-1.5\nD0:DRIF,L=.35\nS:{kind},L={length},{pars}\nD1:DRIF,L=.45\nD2:DRIF,L=.8\nO:MARK\nBL:LINE=(QF,D0,S,D1,QD,D2,O)\n"
                result = native.run(
                    {
                        "line.lte": lattice,
                        "run.ele": (
                            "&run_setup lattice=line.lte,use_beamline=BL,p_central=3200,random_number_seed=104729 &end\n"
                            "&run_control n_steps=1 &end\n"
                            "&twiss_output filename=twiss.sdds,output_at_each_step=1,matched=1,compute_driving_terms=1,leading_order_driving_terms_only=1,s_dependent_driving_terms_file=f.sdds &end\n"
                            "&bunched_beam n_particles_per_bunch=1 &end\n&track &end\n"
                        ),
                    },
                    outputs=("twiss.sdds", "f.sdds"),
                )
                tw, fs = result["twiss.sdds"], result["f.sdds"]
                h = complex(tw["Reh30000"][0], tw["Imh30000"][0])
                f = complex(fs[0, "Ref30000"][-1], fs[0, "Imf30000"][-1])
                # H3=K2L*x^3/6 on y=0. Degree-three Fourier integration is exact on this grid.
                x = np.sqrt(bm) * np.cos(phase + pm)
                expected = np.mean(kl * x**3 / 6 * np.exp(-3j * phase))
                # f30000 at the closing marker: Elegant's s-dependent sum with phase advance mu - psi.
                fe = np.exp(3j * mu) * expected.conjugate() / (1 - np.exp(3j * mu))
                if kind == "KSEXT" or scale == 0:
                    close(h, expected, 3e-17, "h control")
                    close(f, fe, 3e-17, "f control")
                else:
                    faults.extend(
                        [discrepancy(h, expected, 0.0, 3e-17), discrepancy(f, fe, 0.0, 3e-17)]
                    )
                signed.append((h, expected))
                print(
                    f"{kind} L={length:g} K2L={kl:g}: h={h:.17g}, expected={expected:.17g}; f={f:.17g}, expected={fe:.17g} (Hamiltonian Fourier/periodic-phase oracle)"
                )
            response = (signed[0][0] - signed[1][0]) / (2 * strength)
            expected_response = (signed[0][1] - signed[1][1]) / (2 * strength)
            print(
                f"{kind} L={length:g}: dh/dKNL={response:.17g}, expected={expected_response:.17g} (signed strength quotient)"
            )
        rays = np.zeros((3, 6))
        rays[:, 0] = [-0.004, 0, 0.004]
        result = track("kick", "S:MULT,L=0,ORDER=2,KNL=.003\nBL:LINE=(S)\n", rays, p0=3200)
        kick = canonical(result["q"])[:, 0]
        expected = -strength * rays[:, 0] ** 2 / 2
        close(kick, expected, 2e-22, "tracked sextupole kick")
        print(f"MULT tracking: Px/p0={kick.tolist()}, expected={expected.tolist()} (-K2L*x²/2)")
        native.verdict(any(faults))


    if __name__ == "__main__":
        main()

    ```

## Provenance

| Role | Model | Effort | Date | Code examined |
|---|---|---|---|---|
| Found | `gpt-6-astra` | xhigh | 2026-11-06 | master (`0355dd6aca12`), source with SDDS a3f730a14c49, CPU, double, serial, one thread |
| Confirmed | `gpt-6.1-sol` | xhigh | 2026-11-06 | master (`0355dd6aca12`), source with SDDS a3f730a14c49, CPU, double, serial, one thread |
| Confirmed | `claude-opus-5-5` | high | 2026-11-07 | master (`0355dd6aca12`), source with SDDS a3f730a14c49, CPU, double, serial, one thread |

