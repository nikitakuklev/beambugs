---
title: "SCT-16: ENERGY_STRAGGLE shifts the mean energy loss and narrows its stated spread"
hide:
  - toc
---

# SCT-16 — ENERGY_STRAGGLE shifts the mean energy loss and narrows its stated spread

| | |
|---|---|
| Code | [Elegant](../index.md) · Scattering and pressure |
| Priority | P4 |
| Severity | Wrong physics |
| Evidence | Reproduced, independently rechecked |
| Last reproduced | 2026-11-04 on Elegant master (`0355dd6`, head build) |
| Source | [straggling, matter.c:264–273](https://github.com/rtsoliday/elegant/blob/0355dd6aca12c861971fbf8f3ae78a0ded991734/src/matter.c#L264-L273)<br>[simplified energy loss, matter.tex:78–94](https://github.com/rtsoliday/elegant/blob/0355dd6aca12c861971fbf8f3ae78a0ded991734/doc/matter.tex#L78-L94)<br>[`ENERGY_STRAGGLE`, track_data.c:2141](https://github.com/rtsoliday/elegant/blob/0355dd6aca12c861971fbf8f3ae78a0ded991734/src/track_data.c#L2141) |

With ENERGY_DECAY=1 and ENERGY_STRAGGLE=1, MATTER multiplies each electron's exponential energy loss by 1 + 0.5·N(0, 1), with N(0, 1) a standard normal deviate, and redraws whenever the loss would be negative. The manual states that the loss then has a standard deviation of half its mean, the exponential loss; the redraw instead gives a truncated normal distribution whose mean is 2.762% larger and whose standard deviation is 0.4581 of that mean. For 200000 electrons at 1000 mₑc crossing hydrogen of 10⁻⁴ radiation lengths, the mean loss is 2.750 ± 0.1051% above the exponential law, and the ratio is 0.4576.

Run `python SCT-16.py` with [_elegant.py](../helpers/_elegant.py) and [_matter_model.py](../helpers/_matter_model.py) in the same directory.

??? example "Reproducer: SCT-16.py"

    [Download SCT-16.py](SCT-16.py)

    ```python
    """SCT-16: ENERGY_STRAGGLE shifts the mean energy loss and narrows its stated spread

    With ENERGY_DECAY=1 and ENERGY_STRAGGLE=1, MATTER multiplies each electron's exponential energy loss
    by 1 + 0.5·N(0, 1), with N(0, 1) a standard normal deviate, and redraws whenever the loss would be
    negative. The manual states that the loss then has a standard deviation of half its mean, the
    exponential loss; the redraw instead gives a truncated normal distribution whose mean is 2.762%
    larger and whose standard deviation is 0.4581 of that mean. For 200000 electrons at 1000 mₑc
    crossing hydrogen of 10⁻⁴ radiation lengths, the mean loss is 2.750 ± 0.1051% above the exponential
    law, and the ratio is 0.4576.

    The script tracks 200000 on-axis electrons at 1000 mₑc through thin hydrogen of 10⁻⁴ radiation
    lengths with ENERGY_DECAY=1 and FORCE_MCS=1, once with ENERGY_STRAGGLE=1 and once without as the
    control. It compares the sample mean and standard deviation of the energy loss with the manual's
    moments (mean equal to the exponential loss, standard deviation half of it) and with the moments
    of the truncated normal distribution that rejecting negative losses produces.
    """

    import numpy as np
    from _elegant import modes, verdict
    from _matter_model import gamma_loss, radiation_length, track
    from scipy.stats import norm

    SEED, P0, COUNT = 86420, 1000.0, 200000  # P0 in m_e c
    RHO = float(1e-4 * radiation_length(1, 1.0, 1.0))  # hydrogen, L = 1 m, L/X0 = 1e-4
    ELEMENT = f"L=1.0,Z=1,A=1.0,RHO={RHO!r},ENERGY_DECAY=1,FORCE_MCS=1,ENERGY_STRAGGLE={{s}}"
    # Mean loss of ENERGY_DECAY: gamma (1 - exp(-L/X0)) (matter.c:113, 265).
    MU = np.sqrt(1 + P0**2) * -np.expm1(-1.0 / radiation_length(1, 1.0, RHO))
    # The source draws X = MU (1 + Z/2), Z ~ N(0, 1), and redraws while X < 0 (matter.c:269-270),
    # i.e. Z conditioned on Z > -2. With h = phi(2)/Phi(2): E[Z | Z > -2] = h and
    # Var[Z | Z > -2] = 1 - 2h - h^2, so E[X] = MU (1 + h/2) and SD[X] = (MU/2) sqrt(1 - 2h - h^2).
    H = norm.pdf(2) / norm.cdf(2)
    TRUNC_MEAN, TRUNC_SD = MU * (1 + H / 2), MU / 2 * np.sqrt(1 - 2 * H - H * H)

    defect = False
    for mode in modes():
        plain = gamma_loss(P0, track(mode, ELEMENT.format(s=0), P0, SEED, COUNT)["p"])
        print(f"{mode} ENERGY_STRAGGLE=0 (control): mean gamma loss observed={plain.mean():.12g} "
              f"expected={MU:.12g}")  # fmt: skip
        if not np.allclose(plain, MU, rtol=1e-9, atol=0):
            raise SystemExit(f"{mode}: the control misses the exponential law")
        loss = gamma_loss(P0, track(mode, ELEMENT.format(s=1), P0, SEED, COUNT)["p"])
        mean, sd = loss.mean(), loss.std(ddof=1)
        se_mean = sd / np.sqrt(COUNT)
        # Standard error of the sample SD from the fourth central moment: Var(s^2) ~ (m4 - s^4)/N.
        m4 = np.mean((loss - mean) ** 4)
        se_sd = np.sqrt((m4 - sd**4) / COUNT) / (2 * sd)
        se_ratio = np.hypot(se_sd / mean, sd * se_mean / mean**2)  # conservative, ignores correlation
        print(f"{mode} ENERGY_STRAGGLE=1: mean gamma loss observed={mean:.8g}±{se_mean:.2g} "
              f"expected (manual)={MU:.8g}, truncated normal {TRUNC_MEAN:.8g}")  # fmt: skip
        print(f"{mode} ENERGY_STRAGGLE=1: SD/mean observed={sd / mean:.6f}±{se_ratio:.2g} "
              f"expected (manual)=0.5, truncated normal {TRUNC_SD / TRUNC_MEAN:.6f}")  # fmt: skip
        print(f"  mean shift over the exponential loss: observed={mean / MU - 1:+.4%}±{se_mean / MU:.4%}, "
              f"truncated normal {H / 2:+.4%}")  # fmt: skip
        if abs(mean - TRUNC_MEAN) > 5 * se_mean or abs(sd - TRUNC_SD) > 5 * se_sd:
            raise SystemExit(f"{mode}: the moments do not follow the redraw rule either")
        defect |= abs(mean - MU) > 5 * se_mean and abs(sd / mean - 0.5) > 5 * se_ratio
    print("comparison: 5 standard errors; control rtol=1e-9")
    verdict(defect)

    ```

## Provenance

| Role | Model | Effort | Date | Code examined |
|---|---|---|---|---|
| Found | `gpt-6-astra` | xhigh | 2026-11-04 | master (`0355dd6aca12`), source with SDDS a3f730a14c49, CPU, serial |
| Confirmed | `gpt-6.1-sol` | xhigh | 2026-11-04 | master (`0355dd6aca12`), source with SDDS a3f730a14c49, CPU, serial |

