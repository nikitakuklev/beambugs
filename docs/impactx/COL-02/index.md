---
title: "COL-02: ISR uses the electron classical radius for every species"
hide:
  - toc
---

# COL-02 — ISR uses the electron classical radius for every species

| | |
|---|---|
| Code | [ImpactX](../index.md) · Collective effects |
| Priority | P2 |
| Severity | Wrong physics |
| Evidence | Reproduced, independently rechecked |
| Last reproduced | 2026-10-29 on ImpactX development (`fa0a43b`, head build) |
| Tags | <span class="bb-tag">partly-source-only</span> [?](../../methodology.md#tags) |
| Issue | [[gh:https://github.com/BLAST-ImpactX/impactx/issues/1708]] |
| Source | [`ISRPush.cpp#L36-L47`](https://github.com/BLAST-ImpactX/impactx/blob/b701a6e9d1f9e30cefb9a3e18e82ae7e02b9d943/src/particles/wakefields/ISRPush.cpp#L36-L47) |

With incoherent synchrotron radiation (ISR) enabled, ImpactX uses the electron classical radius for every species, although the radiated power scales with the species radius r_q = q²/(4πε₀mc²). Classical radiation theory gives the ultrarelativistic fractional momentum loss over a path ds at bending radius ρ as (2/3)·r_q·(βγ)³·ds/ρ². A 7 TeV (kinetic) proton reference over 100 m at ρ = 2800 m loses 9.954×10⁻⁹ of its momentum instead of 5.421×10⁻¹², 1836 times too much. Electrons are correct, but ISR accepts any species, and an ion of mass M and charge Ze radiates M/(Z²mₑ) times too much; from the source, the same radius also sets the quantum-excitation variance.

Run `python COL-02.py`.

??? example "Reproducer: COL-02.py"

    [Download COL-02.py](COL-02.py)

    ```python
    """COL-02: ISR uses the electron classical radius for every species

    With incoherent synchrotron radiation (ISR) enabled, ImpactX uses the electron classical radius for
    every species, although the radiated power scales with the species radius r_q = q²/(4πε₀mc²).
    Classical radiation theory gives the ultrarelativistic fractional momentum loss over a path ds at
    bending radius ρ as (2/3)·r_q·(βγ)³·ds/ρ². A 7 TeV (kinetic) proton reference over 100 m at
    ρ = 2800 m loses 9.954×10⁻⁹ of its momentum instead of 5.421×10⁻¹², 1836 times too much. Electrons
    are correct, but ISR accepts any species, and an ion of mass M and charge Ze radiates M/(Z²mₑ) times
    too much; from the source, the same radius also sets the quantum-excitation variance.

    The script tracks a 7 TeV proton reference through a 100 m `ExactSbend` with ρ = 2800 m, with
    first-order ISR applied to the reference particle. It compares the reference's fractional momentum
    loss with the classical formula evaluated with the proton radius.
    """
    import numpy as np
    from impactx import ImpactX, elements
    from scipy.constants import c, epsilon_0

    sim = ImpactX()
    sim.verbose = 0
    sim.tiny_profiler = False
    sim.diagnostics = False
    sim.space_charge = False
    sim.isr = True
    sim.isr_order = 1
    sim.isr_on_ref_part = True
    sim.init_grids()
    try:
        ref = sim.beam.ref
        ref.set_species("proton").set_kin_energy_MeV(7e6)  # 7 TeV
        z = np.zeros(4)
        sim.beam.add_n_particles(x=z, y=z, t=z, px=z, py=z, pt=z, qm=ref.qm_ratio_SI, bunch_charge=1e-15)
        R, ds = 2800.0, 100.0
        bg = ref.beta_gamma
        # Classical ultrarelativistic loss in a bend of radius R over ds:
        # dP/P = (2/3) r_q (beta gamma)^3 ds / R^2, species radius r_q = q^2 / (4 pi eps0 m c^2).
        # For a proton r_e / r_p = m_p / m_e = 1836.
        r_p = ref.charge**2 / (4 * np.pi * epsilon_0 * ref.mass * c**2)
        expected = 2 / 3 * r_p * bg**3 * ds / R**2
        sim.lattice.append(elements.ExactSbend(ds=ds, phi=np.degrees(ds / R)))
        sim.track_particles()
        observed = (bg - ref.beta_gamma) / bg
        print(f"proton reference dP/P observed {observed:.4g}, expected {expected:.4g}, ratio {observed / expected:.4g}")
        print("DEFECT REPRODUCED" if abs(observed / expected - 1) > 1e-3 else "NOT REPRODUCED")
    finally:
        sim.finalize()

    ```

## Provenance

| Role | Model | Effort | Date | Code examined |
|---|---|---|---|---|
| Found | `gpt-6-astra` | xhigh | 2026-10-27 | 26.10 (`615921c6867c`), impactx-noacc wheel, CPU, double, serial |
| Confirmed | `claude-opus-5-5` | high | 2026-10-27 | 26.10 (`615921c6867c`), impactx-noacc wheel, CPU, double, serial |

