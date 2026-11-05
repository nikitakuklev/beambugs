---
title: "SCT-06: Active recoil transfers use inconsistent energy normalization"
hide:
  - toc
---

# SCT-06 — Active recoil transfers use inconsistent energy normalization

| | |
|---|---|
| Code | [Elegant](../index.md) · Scattering and pressure |
| Priority | P2 |
| Severity | Wrong physics |
| Evidence | Reproduced, independently rechecked |
| Last reproduced | 2026-11-04 on Elegant master (`0355dd6`) |
| Source | [event probability, matter.c:127–128](https://github.com/rtsoliday/elegant/blob/0355dd6aca12c861971fbf8f3ae78a0ded991734/src/matter.c#L127-L128)<br>[transfer sampling, matter.c:249–250](https://github.com/rtsoliday/elegant/blob/0355dd6aca12c861971fbf8f3ae78a0ded991734/src/matter.c#L249-L250)<br>[`ELECTRON_RECOIL`, track_data.c:2143](https://github.com/rtsoliday/elegant/blob/0355dd6aca12c861971fbf8f3ae78a0ded991734/src/track_data.c#L2143) |

For relativistic electrons in MATTER with ELECTRON_RECOIL and NUCLEAR_BREMSSTRAHLUNG enabled, recoil-transfer rates are suppressed by about the momentum P, measured in mₑc, at transfers above binding energies and small compared with incident kinetic energy. The event rate contains 1/P, but sampled transfers are subtracted directly from the Lorentz factor γ. For 20000 electrons at P = 1000 crossing 0.01 m of dilute helium, the free-electron Møller integral predicts 42.22 electrons losing at least 1.022 keV; serial and three-rank parallel runs give none despite active recoil sampling. The Poisson probability of zero such transfers under that reference is 4.408×10⁻¹⁹.

Download these shared helpers into the same directory as the reproducer scripts:

- [Download _elegant.py](../helpers/_elegant.py) · [View source](../helpers/index.md#_elegant)
- [Download _matter.py](../helpers/_matter.py) · [View source](../helpers/index.md#_matter)

??? example "Reproducer: SCT-06.py"

    [Download SCT-06.py](SCT-06.py)

    ```python
    """SCT-06: Active recoil transfers use inconsistent energy normalization

    For relativistic electrons in MATTER with ELECTRON_RECOIL and NUCLEAR_BREMSSTRAHLUNG enabled,
    recoil-transfer rates are suppressed by about the momentum P, measured in mₑc, at transfers above
    binding energies and small compared with incident kinetic energy. The event rate contains 1/P, but
    sampled transfers are subtracted directly from the Lorentz factor γ. For 20000 electrons at P = 1000
    crossing 0.01 m of dilute helium, the free-electron Møller integral predicts 42.22 electrons losing
    at least 1.022 keV; serial and three-rank parallel runs give none despite active recoil sampling.
    The Poisson probability of zero such transfers under that reference is 4.408×10⁻¹⁹.

    The script tracks 20000 electrons through the helium target with recoil and bremsstrahlung enabled
    and counts those that lose at least 1.022 keV, comparing the count with the Møller-scattering
    expectation. It also checks that the number of electrons losing any energy matches the recoil
    sampler's own event rate, and shows that bremsstrahlung alone (the control) produces no such losses.
    """

    import sys

    import numpy as np
    from _elegant import modes, verdict
    from _matter import AMU, BREMS_PER_ELECTRON, BS_Y0, LENGTH, P0, RE, RHO, A, Z, beam, gamma, track

    COUNT, SEED = 20000, 90131
    QCUT = 0.002  # energy transfer in m_e c^2 (1.022 keV), far above helium's binding energy
    ACTIVE = 1e-9  # m_e c^2; a decrease above this means the recoil sampler is running
    # Bremsstrahlung alone may cost a few electrons energy (mean 0.14 events); more than 10 has
    # probability below 1e-16, so it would mean something besides recoil is acting in the active run.
    CONTROL_MAX = 10
    ELEMENT = (
        f"L={LENGTH!r},PLIMIT=0.005,ENERGY_DECAY=0,NUCLEAR_BREMSSTRAHLUNG=1,ELECTRON_RECOIL={{er}}"
    )

    # Moller scattering of the beam electron on a free electron of the helium atoms, transfers
    # T >= QCUT m_e c^2. With eps = T/[(gamma - 1) m_e c^2] and x = QCUT/(gamma - 1), the cross
    # section per atom is 2 pi Z r_e^2 / [beta^2 (gamma - 1)] times the integral over eps in [x, 1/2]
    # (the electrons are identical) of
    #   B(eps) = ((gamma - 1)/gamma)^2 + 1/eps^2 + 1/(1 - eps)^2
    #            - (2 gamma - 1)/gamma^2 [1/eps + 1/(1 - eps)],
    # which is ((gamma - 1)/gamma)^2 (1/2 - x) + 1/x - 1/(1 - x) - (2 gamma - 1)/gamma^2 ln[(1 - x)/x].
    # (Geant4 Physics Reference Manual, electron ionisation, Eq. 127.) The mean number of such
    # transfers per electron is that cross section times the target's atoms per area.
    g = float(gamma(P0))
    x = QCUT / (g - 1)
    atoms_per_area = RHO / (AMU * A) * LENGTH
    C = 2 * np.pi * Z * RE**2 * atoms_per_area
    integral = (
        ((g - 1) / g) ** 2 * (0.5 - x) + 1 / x - 1 / (1 - x) - (2 * g - 1) / g**2 * np.log((1 - x) / x)
    )
    MEAN_PHYSICAL = C / ((g * g - 1) / g**2 * (g - 1)) * integral
    EXPECTED = COUNT * -np.expm1(-MEAN_PHYSICAL)  # electrons with at least one such transfer
    SIGMA = np.sqrt(EXPECTED * np.exp(-MEAN_PHYSICAL))
    # The model's own kernel: events at rate C/P (1/y0 - 1) with y = T/(m_e c^2) sampled from
    # y0/[1 - u (1 - y0)] in [y0, 1] and subtracted from gamma, which is dN/dy = C/(P y^2); its
    # tail above QCUT is C/P (1/QCUT - 1), smaller than the Moller tail by about the momentum P.
    MEAN_MODEL = C / P0 * (1 / QCUT - 1)
    # Electrons that suffer any recoil in the model's process, in a Poisson approximation to the
    # sampler's total rate: it shows that the sampler runs, so the missing tail is not an activation
    # failure. The element has 61 independent Bernoulli sections of probability MEAN_RECOIL/61, whose
    # exact at-least-one expectation is 6913.18 with binomial sigma 67.26; both counts recorded in the
    # ledger pass either 5-sigma comparison.
    MEAN_RECOIL = C / P0 * (1 / BS_Y0 - 1)
    RECOIL_EXPECTED = COUNT * -np.expm1(-MEAN_RECOIL)
    RECOIL_SIGMA = np.sqrt(RECOIL_EXPECTED * np.exp(-MEAN_RECOIL))


    def tail(mode, er):
        """Electrons whose energy drops by at least QCUT, and those whose energy drops at all.

        Exits when any electron gains a resolvable amount of energy: neither process can do that.
        """
        out = track(mode, ELEMENT.format(er=er), SEED, beam(COUNT))
        drop = gamma(P0) - gamma(out["p"])
        if np.any(drop < -ACTIVE):
            sys.exit(f"{mode} ER={er}: {np.sum(drop < -ACTIVE)} electrons gain energy")
        return int(np.sum(drop >= QCUT)), int(np.sum(drop > ACTIVE))


    print(f"electrons losing at least {QCUT:g} m_e c^2 (1.022 keV) in one MATTER element, of {COUNT}")
    print(
        f"mean transfers above {QCUT:g} per electron: Moller {MEAN_PHYSICAL:.6g}, "
        f"sampler kernel {MEAN_MODEL:.6g} (ratio {MEAN_PHYSICAL / MEAN_MODEL:.1f})"
    )
    defect = False
    for mode in modes():
        active_tail, active = tail(mode, 1)
        control_tail, control_any = tail(mode, 0)
        if control_any > CONTROL_MAX:
            sys.exit(f"{mode}: bremsstrahlung alone makes {control_any} electrons lose energy")
        # The sampler must be running, or the tail comparison would only repeat the activation gap.
        if abs(active - RECOIL_EXPECTED) > 5 * RECOIL_SIGMA:
            sys.exit(
                f"{mode}: {active} electrons lose energy, not the sampler rate: tail check invalid"
            )
        print(
            f"{mode} recoil + bremsstrahlung, tail: observed={active_tail} expected={EXPECTED:.2f}±{SIGMA:.2f}"
        )
        print(
            f"{mode} bremsstrahlung only (control), tail: observed={control_tail} expected tail mean <= {COUNT * BREMS_PER_ELECTRON:.2f}"
        )
        print(
            f"{mode} recoil + bremsstrahlung, any loss: observed={active} expected={RECOIL_EXPECTED:.2f}±{RECOIL_SIGMA:.2f}"
        )
        defect |= active_tail < EXPECTED - 5 * SIGMA
    print(f"P(no electron above {QCUT:g} | Moller) = {np.exp(-COUNT * MEAN_PHYSICAL):.4g}")
    verdict(defect)

    ```

## Provenance

| Role | Model | Effort | Date | Code examined |
|---|---|---|---|---|
| Found | `gpt-6-astra` | xhigh | 2026-11-04 | master (`0355dd6aca12`), source with SDDS a3f730a14c49, CPU, serial |
| Confirmed | `gpt-6-astra` | high | 2026-10-25 | master (`75a2e1a1a58b`), source with SDDS a60b9bbed7cc, CPU, serial and 3-rank Pelegant |
| Confirmed | `gpt-6.1-sol` | high | 2026-10-25 | master (`75a2e1a1a58b`), source with SDDS a60b9bbed7cc, CPU, serial and 3-rank Pelegant |

