---
title: "COL-36: CSRCSBEND computes CSR with the nominal bend radius, ignoring the field error FSE"
hide:
  - toc
---

# COL-36 — CSRCSBEND computes CSR with the nominal bend radius, ignoring the field error FSE

| | |
|---|---|
| Code | [Elegant](../index.md) · Collective effects |
| Priority | P3 |
| Severity | Wrong physics |
| Evidence | Reproduced, independently rechecked |
| Last reproduced | 2026-11-04 on Elegant master (`7514db3`) |
| Tags | <span class="bb-tag">partly-source-only</span> [?](../../methodology.md#tags) |
| Source | [actual radius, csbend.c:2846–2852](https://github.com/rtsoliday/elegant/blob/0355dd6aca12c861971fbf8f3ae78a0ded991734/src/csbend.c#L2846-L2852)<br>[CSR constant, csbend.c:3132–3133](https://github.com/rtsoliday/elegant/blob/0355dd6aca12c861971fbf8f3ae78a0ded991734/src/csbend.c#L3132-L3133)<br>[slippage, csbend.c:3437–3438](https://github.com/rtsoliday/elegant/blob/0355dd6aca12c861971fbf8f3ae78a0ded991734/src/csbend.c#L3437-L3438)<br>[IGF argument, csbend.c:3456–3458](https://github.com/rtsoliday/elegant/blob/0355dd6aca12c861971fbf8f3ae78a0ded991734/src/csbend.c#L3456-L3458)<br>[drift hand-off, csbend.c:3961](https://github.com/rtsoliday/elegant/blob/0355dd6aca12c861971fbf8f3ae78a0ded991734/src/csbend.c#L3961)<br>[CSBEND manual, errors, csbend.tex:124–133](https://github.com/rtsoliday/elegant/blob/0355dd6aca12c861971fbf8f3ae78a0ded991734/doc/csbend.tex#L124-L133)<br>[CSRCSBEND manual, csrcsbend.tex:25–28](https://github.com/rtsoliday/elegant/blob/0355dd6aca12c861971fbf8f3ae78a0ded991734/doc/csrcsbend.tex#L25-L28) |

CSRCSBEND computes the CSR strength, the transient slippage length and the argument of the integrated Green function (IGF=1) from the nominal radius ρ = L/ANGLE, although FSE, the documented field-strength error, bends the orbit on the radius ρ/(1 + FSE). Steady-state CSR scales as the field curvature to the power 2/3, so the wake should grow by (1 + FSE)^(2/3); in a 0.6 m, 0.3 rad bend whose FSE = 0 wake matches the analytic Gaussian wake within 0.1470%, the wake is instead identical for FSE = 0, 0.02 and 0.1. At FSE = 0.02 the wake should be 1.329% stronger and the slippage length 2.339 mm instead of 2.250 mm; the IGF wake is also unchanged.

Download these shared helpers into the same directory as the reproducer scripts:

- [Download _csr.py](../helpers/_csr.py) · [View source](../helpers/index.md#_csr)
- [Download _elegant.py](../helpers/_elegant.py) · [View source](../helpers/index.md#_elegant)

??? example "Reproducer: COL-36.py"

    [Download COL-36.py](COL-36.py)

    ```python
    """COL-36: CSRCSBEND computes CSR with the nominal bend radius, ignoring the field error FSE

    CSRCSBEND computes the CSR strength, the transient slippage length and the argument of the integrated Green function (IGF=1) from the nominal radius ρ = L/ANGLE, although FSE, the documented field-strength error, bends the orbit on the radius ρ/(1 + FSE). Steady-state CSR scales as the field curvature to the power 2/3, so the wake should grow by (1 + FSE)^(2/3); in a 0.6 m, 0.3 rad bend whose FSE = 0 wake matches the analytic Gaussian wake within 0.1470%, the wake is instead identical for FSE = 0, 0.02 and 0.1. At FSE = 0.02 the wake should be 1.329% stronger and the slippage length 2.339 mm instead of 2.250 mm; the IGF wake is also unchanged.

    The script tracks 240001 electrons (Gaussian quantiles, σ_ct = 35 μm, p = 4000 mₑc, 10⁻¹⁶ C)
    through a one-slice 0.6 m, 0.3 rad CSRCSBEND (400 bins) with FSE = 0, 0.02 and 0.1 and reads the
    exported wake. With STEADY_STATE=1 the wake must scale as the field curvature to the power 2/3,
    (1 + FSE)^(2/3), and the FSE = 0 control must match the analytic Gaussian wake. With
    STEADY_STATE=0 the reported slippage must follow the circle of the actual field radius to the
    fixed exit plane. With IGF=1 at p = 50 mₑc the script reports whether the wake changes with FSE.
    """

    import numpy as np
    from _csr import E_CHARGE, R_E, gaussian_beam, page, steady_state_wake, track
    from _elegant import verdict
    from scipy.optimize import brentq

    P0, SIGMA, CHARGE = 4000.0, 35e-6, 1e-16
    L, ANGLE = 0.6, 0.3
    RHO = L / ANGLE  # nominal radius; FSE changes the field, not the survey
    BEND = f"L={L},ANGLE={ANGLE},N_SLICES=1,INTEGRATION_ORDER=6,BINS=400,SG_HALFWIDTH=1,SGDERIV_HALFWIDTH=3,SGDERIV_ORDER=3"
    FSES = (0.0, 0.02, 0.1)
    K = 2 * (CHARGE / E_CHARGE) * R_E * L / (3 * RHO**2) ** (1 / 3)  # see _csr.steady_state_wake


    def wake(fse: float, mode: str, p0: float = P0, count: int = 240001):
        lattice = f'Q: CHARGE,TOTAL={CHARGE!r}\nB: CSRCSBEND,{BEND},{mode},FSE={fse!r},OUTPUT_FILE="wake.sdds"'
        beam = gaussian_beam(count, SIGMA, p0)
        return page(track(lattice, "Q,B", beam, p0, outputs=("wake.sdds",))["wake.sdds"])


    def slippage(fse: float) -> float:
        """Small-angle slippage rho_B psi^3/24 on the orbit of field radius rho_B = rho/(1 + FSE).

        The orbit starts on the reference axis and ends where the circle of radius rho_B crosses the
        design exit plane (through rho (1 - cos A, sin A) with normal (sin A, cos A)); psi is the
        angle it has turned through there. This is the slippage law the code applies to the
        nominal orbit, evaluated on the orbit the field produces.
        """
        rho_b = RHO / (1 + fse)
        point, normal = (
            RHO * np.array([1 - np.cos(ANGLE), np.sin(ANGLE)]),
            np.array([np.sin(ANGLE), np.cos(ANGLE)]),
        )
        psi = brentq(
            lambda a: np.dot(rho_b * np.array([1 - np.cos(a), np.sin(a)]) - point, normal),
            1e-9,
            1.0,
            xtol=1e-16,
        )
        return rho_b * psi**3 / 24


    defect = False
    steady = {fse: wake(fse, "STEADY_STATE=1") for fse in FSES}
    reference = steady[0.0][0]["DeltaGamma"]
    analytic = steady_state_wake(steady[0.0][0]["s"], SIGMA, K)
    control = np.max(np.abs(reference - analytic)) / np.max(np.abs(analytic))
    print(f"serial FSE=0 steady-state wake: max |wake - analytic| / peak = {control:.4%} (control)")
    if control > 0.003:
        raise SystemExit("The FSE=0 control disagrees with the analytic wake.")
    for fse in FSES[1:]:
        columns, _ = steady[fse]
        # Steady-state CSR scales as rho^(-2/3) (k above); the field radius is rho/(1 + FSE).
        ratio = np.sum(columns["DeltaGamma"] * reference) / np.sum(reference**2)  # least-squares scale
        expected = (1 + fse) ** (2 / 3)
        print(
            f"serial FSE={fse} steady-state wake / FSE=0 wake: observed={ratio:.10f} expected={expected:.10f}"
        )
        defect |= not np.isclose(ratio, expected, rtol=1e-4)
    for fse in FSES:
        _, info = wake(fse, "STEADY_STATE=0")
        expected = slippage(fse)
        print(
            f"serial FSE={fse} transient slippage length (m): observed={info['SlippageLength']:.10g} expected={expected:.10g}"
        )
        if fse == 0 and not np.isclose(info["SlippageLength"], expected, rtol=1e-6):
            raise SystemExit("The FSE=0 slippage disagrees with rho angle^3/24.")
        defect |= fse != 0 and not np.isclose(info["SlippageLength"], expected, rtol=1e-4)
    # IGF: the integrated Green function depends on the curvature through mu = 3 gamma^3 z/(2 rho);
    # the script only reports whether its wake responds to FSE.
    igf = {
        fse: wake(fse, "STEADY_STATE=1,IGF=1", p0=50.0, count=40001)[0]["DeltaGamma"] for fse in FSES
    }
    for fse in FSES[1:]:
        same = np.array_equal(igf[fse], igf[0.0])
        print(f"serial IGF=1, p=50: FSE={fse} wake identical to FSE=0: {same}")
    verdict(defect)

    ```

## Provenance

| Role | Model | Effort | Date | Code examined |
|---|---|---|---|---|
| Found | `gemini-4-argon-medium` |  | 2026-11-04 | master (`7514db36f764`), source reading, not executed for this claim |
| Found (executed verification) | `claude-opus-5-5` | high | 2026-11-04 | master (`7514db36f764`), source with SDDS a3f730a14c49, CPU, serial |
| Confirmed (nominal radius in the CSR strength, the bend slippage and the IGF argument; not the proposed ANGLE/FSE equivalence) | `gpt-6.1-sol` | xhigh | 2026-11-04 | master (`7514db36f764`), source with SDDS a3f730a14c49, CPU, serial |

