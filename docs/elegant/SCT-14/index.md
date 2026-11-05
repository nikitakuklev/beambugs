---
title: "SCT-14: FORCE_MCS removes MATTER bremsstrahlung and electron recoil"
hide:
  - toc
---

# SCT-14 — FORCE_MCS removes MATTER bremsstrahlung and electron recoil

| | |
|---|---|
| Code | [Elegant](../index.md) · Scattering and pressure |
| Priority | P2 |
| Severity | Wrong physics |
| Evidence | Reproduced, independently rechecked |
| Last reproduced | 2026-11-04 on Elegant master (`0355dd6`) |
| Source | [branch selection, matter.c:115–149](https://github.com/rtsoliday/elegant/blob/0355dd6aca12c861971fbf8f3ae78a0ded991734/src/matter.c#L115-L149)<br>[matter.c:193–213](https://github.com/rtsoliday/elegant/blob/0355dd6aca12c861971fbf8f3ae78a0ded991734/src/matter.c#L193-L213)<br>[event draws, matter.c:246–250](https://github.com/rtsoliday/elegant/blob/0355dd6aca12c861971fbf8f3ae78a0ded991734/src/matter.c#L246-L250)<br>[writeback, matter.c:258–263](https://github.com/rtsoliday/elegant/blob/0355dd6aca12c861971fbf8f3ae78a0ded991734/src/matter.c#L258-L263)<br>[`FORCE_MCS`, track_data.c:2158](https://github.com/rtsoliday/elegant/blob/0355dd6aca12c861971fbf8f3ae78a0ded991734/src/track_data.c#L2158)<br>[forced multiple scattering, matter.tex:40–71](https://github.com/rtsoliday/elegant/blob/0355dd6aca12c861971fbf8f3ae78a0ded991734/doc/matter.tex#L40-L71)<br>[energy loss, matter.tex:73–77](https://github.com/rtsoliday/elegant/blob/0355dd6aca12c861971fbf8f3ae78a0ded991734/doc/matter.tex#L73-L77)<br>[energy options, track_data.c:2142–2143](https://github.com/rtsoliday/elegant/blob/0355dd6aca12c861971fbf8f3ae78a0ded991734/src/track_data.c#L2142-L2143)<br>[matter.c:84–85](https://github.com/rtsoliday/elegant/blob/0355dd6aca12c861971fbf8f3ae78a0ded991734/src/matter.c#L84-L85) |

Setting FORCE_MCS=1 in MATTER stops all bremsstrahlung and electron-recoil energy loss, even with NUCLEAR_BREMSSTRAHLUNG and ELECTRON_RECOIL enabled; the manual describes FORCE_MCS only as a choice of angular-scattering model, without effect above 0.001 radiation lengths. For 60000 electrons at momentum 2000 mₑc crossing hydrogen of 4×10⁻⁵ radiation lengths, the source's own photon rate predicts 56.92 electrons emitting a photon, and none does (probability 1.856×10⁻²⁵); with recoil added, every electron should lose energy and none does. In lead of 0.002 radiation lengths, 463.5 of 10000 electrons should emit a photon and none does. The deterministic ENERGY_DECAY loss is unaffected.

Download these shared helpers into the same directory as the reproducer scripts:

- [Download _elegant.py](../helpers/_elegant.py) · [View source](../helpers/index.md#_elegant)
- [Download _matter_model.py](../helpers/_matter_model.py) · [View source](../helpers/index.md#_matter_model)

??? example "Reproducer: SCT-14.py"

    [Download SCT-14.py](SCT-14.py)

    ```python
    """SCT-14: FORCE_MCS removes MATTER bremsstrahlung and electron recoil

    Setting FORCE_MCS=1 in MATTER stops all bremsstrahlung and electron-recoil energy loss, even with
    NUCLEAR_BREMSSTRAHLUNG and ELECTRON_RECOIL enabled; the manual describes FORCE_MCS only as a choice
    of angular-scattering model, without effect above 0.001 radiation lengths. For 60000 electrons at
    momentum 2000 mₑc crossing hydrogen of 4×10⁻⁵ radiation lengths, the source's own photon rate
    predicts 56.92 electrons emitting a photon, and none does (probability 1.856×10⁻²⁵); with recoil
    added, every electron should lose energy and none does. In lead of 0.002 radiation lengths, 463.5 of
    10000 electrons should emit a photon and none does. The deterministic ENERGY_DECAY loss is
    unaffected.

    The script tracks on-axis electrons at 2000 mₑc through thin hydrogen (4×10⁻⁵ radiation lengths)
    with bremsstrahlung alone and with bremsstrahlung and recoil, and through lead of 0.002 radiation
    lengths with bremsstrahlung, each with FORCE_MCS=0 and FORCE_MCS=1, and counts the electrons that
    lose energy. The expected counts follow from the source's own event rates and slice count; the
    FORCE_MCS=0 runs must match them, and a FORCE_MCS=1 run with ENERGY_DECAY=1 is the control showing
    that forced multiple scattering keeps the deterministic energy loss.
    """

    import numpy as np
    from _elegant import modes, verdict
    from _matter_model import gamma_loss, model, radiation_length, track

    SEED, P0 = 135791, 2000.0  # P0 in m_e c
    THRESHOLD = 1e-10  # gamma loss; a photon takes >= 2e-5 and a recoil >= 1e-8, rounding gives < 3e-13
    RHO_PB = 11350.0  # kg/m^3
    # Densities and lengths that give the radiation depths L/X0 = 4e-5 (thin, below the 1e-3 switch of
    # matter.c:115/141) and 2e-3 (thick, where the manual says FORCE_MCS has no effect).
    RHO_H = float(4e-5 * radiation_length(1, 1.0, 1.0))  # hydrogen, L = 1 m; X0 scales as 1/rho
    L_PB = float(2e-3 * radiation_length(82, 207.2, RHO_PB))

    CASES = (  # label, Z, A, rho, L, count, options
        ("thin H, bremsstrahlung", 1, 1.0, RHO_H, 1.0, 60000, "NUCLEAR_BREMSSTRAHLUNG=1"),
        ("thin H, bremsstrahlung + recoil", 1, 1.0, RHO_H, 1.0, 20000,
         "NUCLEAR_BREMSSTRAHLUNG=1,ELECTRON_RECOIL=1"),
        ("thick Pb, bremsstrahlung", 82, 207.2, RHO_PB, L_PB, 10000, "NUCLEAR_BREMSSTRAHLUNG=1"),
    )  # fmt: skip

    defect = False
    for mode in modes():
        for label, Z, A, rho, length, count, options in CASES:
            rates = model(Z, A, rho, length, P0)
            n = rates.sections
            recoil = rates.recoil if "RECOIL" in options else 0.0
            # The source's slices are independent Bernoulli trials with probabilities brems/n and
            # recoil/n, so P(no energy loss) = (1 - brems/n)^n (1 - recoil/n)^n.
            q0 = (1 - rates.brems / n) ** n * (1 - recoil / n) ** n
            expected, sigma = count * (1 - q0), np.sqrt(count * q0 * (1 - q0))
            # Near-certain loss (recoil case) has sigma ~ 0; allow at least one electron of slack.
            tolerance = 5 * max(sigma, 1.0)
            for force in (0, 1):
                element = (
                    f"L={length!r},Z={Z},A={A!r},RHO={rho!r},ENERGY_DECAY=0,{options},FORCE_MCS={force}"
                )
                out = track(mode, element, P0, SEED, count)
                losing = int(np.sum(gamma_loss(P0, out["p"]) > THRESHOLD))
                print(
                    f"{mode} {label} (L/X0={length / radiation_length(Z, A, rho):.3g}, slices={n}, "
                    f"photons {rates.brems:.6g}, recoils {recoil:.6g} per electron), "
                    f"FORCE_MCS={force}: electrons losing energy observed={losing} "
                    f"expected={expected:.2f}±{sigma:.2f}"
                )
                if force == 0 and abs(losing - expected) > tolerance:
                    raise SystemExit(f"{mode}: the FORCE_MCS=0 run misses the model's own event rate")
                if force == 1:
                    defect |= losing < expected - tolerance
            if "RECOIL" not in options:
                # Probability of no event in the whole ensemble under the model's rates.
                print(f"  P(no electron loses energy) = {q0**count:.4g}")
        # Control: the deterministic exponential loss of ENERGY_DECAY survives FORCE_MCS=1.
        element = f"L=1.0,Z=1,A=1.0,RHO={RHO_H!r},ENERGY_DECAY=1,FORCE_MCS=1"
        out = track(mode, element, P0, SEED, 1000)
        # gamma_out = gamma_in exp(-L/X0) (matter.c:113, 265).
        expected_loss = np.sqrt(1 + P0**2) * -np.expm1(-1.0 / radiation_length(1, 1.0, RHO_H))
        loss = gamma_loss(P0, out["p"])
        print(
            f"{mode} thin H, ENERGY_DECAY=1, FORCE_MCS=1 (control): gamma loss "
            f"observed=[{loss.min():.12g}, {loss.max():.12g}] expected={expected_loss:.12g}"
        )
        if not np.allclose(loss, expected_loss, rtol=1e-9, atol=0):
            raise SystemExit(f"{mode}: the ENERGY_DECAY control misses the exponential law")
    print("comparison: 5 sigma on the counts; control rtol=1e-9")
    verdict(defect)

    ```

## Provenance

| Role | Model | Effort | Date | Code examined |
|---|---|---|---|---|
| Found | `gpt-6-astra` | xhigh | 2026-11-04 | master (`0355dd6aca12`), source with SDDS a3f730a14c49, CPU, serial |
| Found<br><small>thick-target case</small> | `claude-opus-5-5` | high | 2026-11-04 | master (`0355dd6aca12`), source with SDDS a3f730a14c49, CPU, serial |
| Confirmed | `gpt-6.1-sol` | xhigh | 2026-11-04 | master (`0355dd6aca12`), source with SDDS a3f730a14c49, CPU, serial |
| Confirmed<br><small>thick-target case</small> | `gpt-6.1-sol` | xhigh | 2026-11-04 | master (`7514db36f764`), source with SDDS a3f730a14c49, CPU, serial |

