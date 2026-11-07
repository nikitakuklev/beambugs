---
title: "COL-42: CSRCSBEND with zero or negligible ANGLE tracks a finite K1 as a drift"
hide:
  - toc
---

# COL-42 — CSRCSBEND with zero or negligible ANGLE tracks a finite K1 as a drift

| | |
|---|---|
| Code | [Elegant](../index.md) · Collective effects |
| Priority | P3 |
| Severity | Wrong physics |
| Evidence | Reproduced, independently rechecked |
| Last reproduced | 2026-11-04 on Elegant master (`7514db3`, head build) |
| Source | [zero angle, csbend.c:2743–2757](https://github.com/rtsoliday/elegant/blob/0355dd6aca12c861971fbf8f3ae78a0ded991734/src/csbend.c#L2743-L2757)<br>[large radius, csbend.c:2839–2844](https://github.com/rtsoliday/elegant/blob/0355dd6aca12c861971fbf8f3ae78a0ded991734/src/csbend.c#L2839-L2844)<br>[CSBEND quadrupole branch, csbend.c:805–826](https://github.com/rtsoliday/elegant/blob/0355dd6aca12c861971fbf8f3ae78a0ded991734/src/csbend.c#L805-L826)<br>[matrix, compute_matrices.c:1883–1887](https://github.com/rtsoliday/elegant/blob/0355dd6aca12c861971fbf8f3ae78a0ded991734/src/compute_matrices.c#L1883-L1887)<br>[CSRCSBEND manual, csrcsbend.tex:25–28](https://github.com/rtsoliday/elegant/blob/0355dd6aca12c861971fbf8f3ae78a0ded991734/doc/csrcsbend.tex#L25-L28) |

A CSRCSBEND with ANGLE = 0, or with a bend radius above 10⁶ m, tracks as a drift and drops a finite quadrupole gradient K1, although its manual says it behaves like CSBEND outside CSR, and CSBEND tracks the same element as a quadrupole. For a 0.8 m element with K1 = 1.7 m⁻² and an electron at x = 0.2 mm, linear quadrupole transport, which CSBEND and KQUAD match within 5×10⁻¹² m, gives x = 0.1007 mm and x′ = −0.2253 mrad; CSRCSBEND returns x = 0.2 mm and x′ = 0. At ANGLE = 0 this happens silently; at ANGLE = 10⁻⁷ rad it prints "Treated as EDRIFT".

Run `python COL-42.py` with [_csr.py](../helpers/_csr.py) and [_elegant.py](../helpers/_elegant.py) in the same directory.

??? example "Reproducer: COL-42.py"

    [Download COL-42.py](COL-42.py)

    ```python
    """COL-42: CSRCSBEND with zero or negligible ANGLE tracks a finite K1 as a drift

    A CSRCSBEND with ANGLE = 0, or with a bend radius above 10⁶ m, tracks as a drift and drops a finite quadrupole gradient K1, although its manual says it behaves like CSBEND outside CSR, and CSBEND tracks the same element as a quadrupole. For a 0.8 m element with K1 = 1.7 m⁻² and an electron at x = 0.2 mm, linear quadrupole transport, which CSBEND and KQUAD match within 5×10⁻¹² m, gives x = 0.1007 mm and x′ = −0.2253 mrad; CSRCSBEND returns x = 0.2 mm and x′ = 0. At ANGLE = 0 this happens silently; at ANGLE = 10⁻⁷ rad it prints "Treated as EDRIFT".

    The script tracks one electron (p = 1200 mₑc, x = 0.2 mm, x′ = 0) through 0.8 m elements with
    K1 = 1.7 m⁻² and 64 slices: CSRCSBEND (CSR=0) with ANGLE = 0 and 10⁻⁷ rad, and as controls
    CSBEND with the same angles and a KQUAD. Expected is linear quadrupole transport,
    x = x₀ cos(√K1 L) and x′ = −x₀ √K1 sin(√K1 L), which the controls reproduce; the script also
    reports the warning each run prints.
    """

    import numpy as np
    from _csr import final, gaussian_beam, track
    from _elegant import verdict

    L, K1, X0, P0 = 0.8, 1.7, 2e-4, 1200.0
    ATOL = 1e-11  # m and rad; CSBEND and KQUAD agree with linear transport within 5e-12
    # Linear transport through a focusing quadrupole of strength K1 and length L.
    K = np.sqrt(K1)
    X_EXPECTED, XP_EXPECTED = X0 * np.cos(K * L), -X0 * K * np.sin(K * L)
    BEAM = gaussian_beam(1, 0.0, P0, x0=X0)

    defect = False
    cases = [
        ("KQUAD", None),
        ("CSBEND", 0.0),
        ("CSBEND", 1e-7),
        ("CSRCSBEND", 0.0),
        ("CSRCSBEND", 1e-7),
    ]
    for kind, angle in cases:
        params = f"L={L},K1={K1},N_SLICES=64"
        if angle is not None:
            params += f",ANGLE={angle!r}"
        if kind == "CSRCSBEND":
            params += ",CSR=0,BINS=32"
        result = track(f"B: {kind},{params}", "B", BEAM, P0)
        out = final(result)
        x, xp = float(out["x"][0]), float(out["xp"][0])
        warnings = [
            ln.split("--- ", 1)[-1] for ln in result["log"].splitlines() if ln.startswith("*** Warning")
        ]
        warned = "; ".join(warnings) or "none"
        good = abs(x - X_EXPECTED) < ATOL and abs(xp - XP_EXPECTED) < ATOL
        label = kind if angle is None else f"{kind} ANGLE={angle!r}"
        print(
            f"serial {label}: (x, x') observed=({x:.12g}, {xp:.12g}) "
            f"expected=({X_EXPECTED:.12g}, {XP_EXPECTED:.12g}); warning: {warned}"
        )
        if kind != "CSRCSBEND" and not good:
            raise SystemExit(f"{label}: the control disagrees with quadrupole transport")
        defect |= kind == "CSRCSBEND" and not good
    print(f"comparison atol={ATOL:g}")
    verdict(defect)

    ```

## Provenance

| Role | Model | Effort | Date | Code examined |
|---|---|---|---|---|
| Found | `gemini-4-argon-medium` |  | 2026-11-04 | master (`7514db36f764`), source reading, not executed for this claim |
| Found<br><small>quadrupole omission at small nonzero bend angle</small> | `claude-opus-5-5` | high | 2026-11-04 | master (`7514db36f764`), source with SDDS a3f730a14c49, CPU, serial |
| Confirmed<br><small>also exact zero angle, where no warning is printed</small> | `gpt-6.1-sol` | xhigh | 2026-11-04 | master (`7514db36f764`), source with SDDS a3f730a14c49, CPU, serial |

