---
title: "TRK-49: Native backtracking repeats integrated magnetic steering"
hide:
  - toc
---

# TRK-49 — Native backtracking repeats integrated magnetic steering

| | |
|---|---|
| Code | [Elegant](../index.md) · Tracking |
| Priority | P2 |
| Failure | Wrong physics |
| Evidence | Reproduced, independently rechecked |
| Last reproduced | 2026-11-06 on Elegant commit [`0355dd6`](https://github.com/rtsoliday/elegant/commit/0355dd6aca12), committed 2026-11-03 |
| Source | [src/get_beamline.c:2120–2211](https://github.com/rtsoliday/elegant/blob/0355dd6aca12c861971fbf8f3ae78a0ded991734/src/get_beamline.c#L2120-L2211)<br>[src/multipole.c:804–805](https://github.com/rtsoliday/elegant/blob/0355dd6aca12c861971fbf8f3ae78a0ded991734/src/multipole.c#L804-L805)<br>[src/multipole.c:871–872](https://github.com/rtsoliday/elegant/blob/0355dd6aca12c861971fbf8f3ae78a0ded991734/src/multipole.c#L871-L872)<br>[src/multipole.c:1026–1033](https://github.com/rtsoliday/elegant/blob/0355dd6aca12c861971fbf8f3ae78a0ded991734/src/multipole.c#L1026-L1033)<br>[src/multipole.c:1441–1444](https://github.com/rtsoliday/elegant/blob/0355dd6aca12c861971fbf8f3ae78a0ded991734/src/multipole.c#L1441-L1444)<br>[src/quad_matrix.c:571–579](https://github.com/rtsoliday/elegant/blob/0355dd6aca12c861971fbf8f3ae78a0ded991734/src/quad_matrix.c#L571-L579)<br>[src/csbend.c:1083–1084](https://github.com/rtsoliday/elegant/blob/0355dd6aca12c861971fbf8f3ae78a0ded991734/src/csbend.c#L1083-L1084)<br>[doc/elegant.tex:7283–7287](https://github.com/rtsoliday/elegant/blob/0355dd6aca12c861971fbf8f3ae78a0ded991734/doc/elegant.tex#L7283-L7287)<br>[src/track_data.c:4313–4313](https://github.com/rtsoliday/elegant/blob/0355dd6aca12c861971fbf8f3ae78a0ded991734/src/track_data.c#L4313-L4313) |

Native backtracking through QUAD, KQUAD, KSEXT, DQCOR or CSBEND keeps the forward sign of integrated steering. A conservative forward traversal followed by its inverse must restore the input particle. Instead, a 0.6000 m zero-gradient KQUAD with normalized steering impulses of 1.700×10⁻⁴ and −1.100×10⁻⁴ adds another equal impulse on the backward traversal, giving round-trip momentum errors of 3.400×10⁻⁴ and −2.200×10⁻⁴. Negating the backward steering restores the input, while zero-steering and dedicated-corrector controls pass. The affine action of the analytic maps has the same sign error.

Run `python TRK-49.py` with [_elegant.py](../helpers/_elegant.py) and [_slicing.py](../helpers/_slicing.py) in the same directory.

??? example "Reproducer: TRK-49.py"

    [Download TRK-49.py](TRK-49.py)

    ```python
    """TRK-49: Native backtracking repeats integrated magnetic steering

    Native backtracking through QUAD, KQUAD, KSEXT, DQCOR or CSBEND keeps the forward sign of integrated steering. A conservative forward traversal followed by its inverse must restore the input particle. Instead, a 0.6000 m zero-gradient KQUAD with normalized steering impulses of 1.700×10⁻⁴ and −1.100×10⁻⁴ adds another equal impulse on the backward traversal, giving round-trip momentum errors of 3.400×10⁻⁴ and −2.200×10⁻⁴. Negating the backward steering restores the input, while zero-steering and dedicated-corrector controls pass. The affine action of the analytic maps has the same sign error.

    The script composes forward and native backward tracking, comparing with identity and opposite-steering inverse controls. Independent quadrature checks the forward constant-field trajectory, and analytic affine maps and a dedicated corrector provide additional controls.
    """

    import numpy as np
    from _elegant import verdict
    from _slicing import bend, canonical, close, discrepancy, straight, track
    from scipy.integrate import quad


    def main():
        failures = []
        q = np.array([[0.0011, 0.0004, -0.0008, -0.0007, 0.0, 0.08]])
        for kind in ("KQUAD", "KSEXT", "QUAD", "DQCOR"):
            kicks = np.array([0.00017, 0 if kind == "QUAD" else -0.00011])
            initial = np.zeros((1, 6)) if kind == "QUAD" else q
            for slices in (16, 64):
                lat = straight(kind, vk=kicks[1], slices=slices)
                fwd = track(kind, lat, initial, matrix=True)
                back = track(kind, lat, fwd["q"], reverse=True)
                value = (
                    back["q"][0, [1, 3]]
                    if kind == "QUAD"
                    else (canonical(back["q"]) - canonical(initial))[0]
                )
                failed = discrepancy(value, np.zeros(2), 2 * kicks, 3e-14)
                failures.append(failed)
                if kind != "QUAD":
                    qi = canonical(initial)[0]
                    total = 1 + initial[0, 5]
                    # A constant force makes q(u)=q_in+u*K; integrate exact slope
                    # q/sqrt((1+delta)^2-|q|^2), rather than a paraxial trajectory.
                    displacement = np.array(
                        [
                            quad(
                                lambda u, j=j, qi=qi, kicks=kicks, total=total: (
                                    0.6
                                    * (qi[j] + u * kicks[j])
                                    / np.sqrt(total**2 - np.dot(qi + u * kicks, qi + u * kicks))
                                ),
                                0,
                                1,
                                epsabs=1e-16,
                                epsrel=1e-13,
                            )[0]
                            for j in (0, 1)
                        ]
                    )
                    close(
                        fwd["q"][0, [0, 2]] - initial[0, [0, 2]],
                        displacement,
                        2e-15,
                        "Continuous field quadrature",
                    )
                fixed = track(
                    kind,
                    straight(kind, hk=-kicks[0], vk=-kicks[1], slices=slices),
                    fwd["q"],
                    reverse=True,
                )
                error = close(fixed["q"] if failed else back["q"], initial, 5e-13, "Inverse control")
                print(
                    f"{kind} slices={slices}: roundtrip impulse={value.tolist()} expected=[0, 0] "
                    f"(inverse transport); inverse control max error={error:.4g}"
                )
            lat = straight(kind, hk=0, vk=0, strength=8 if kind == "KSEXT" else 1.3, slices=64)
            fwd = track(kind, lat, q)
            back = track(kind, lat, fwd["q"], reverse=True)
            close(back["q"], q, 5e-13, "Nonzero body, zero steering")
        for expand in (0, 1):
            for steering in (False, True):
                hk, vk = (0.00017, -0.00011) if steering else (0, 0)
                lat = bend(hk=hk, vk=vk, expand=expand)
                initial = np.zeros((1, 6))
                fwd = track("bend", lat, initial)
                back = track("bend", lat, fwd["q"], reverse=True)
                fixed = track("bend", bend(hk=-hk, vk=-vk, expand=expand), fwd["q"], reverse=True)
                failed = not np.allclose(back["q"], initial, rtol=0, atol=2e-12)
                error = close(
                    fixed["q"] if failed else back["q"], initial, 2e-12, "Bend inverse control"
                )
                if steering:
                    failures.append(failed)
                else:
                    close(back["q"], initial, 2e-12, "Zero-steering bend inverse")
                print(
                    f"CSBEND expand={expand} steering={int(steering)}: roundtrip slopes="
                    f"{back['q'][0, [1, 3]].tolist()} expected=[0, 0] (inverse transport); inverse control max error={error:.4g}"
                )
        for kind in ("QUAD", "KQUAD", "KSEXT", "DQCOR", "CSBEND"):
            lat = bend() if kind == "CSBEND" else straight(kind, vk=0)
            fwd = track(kind, lat, np.zeros(6), matrix=True)
            rev = track(kind, lat, np.zeros(6), reverse=True, matrix=True)
            residual = rev["C"] + rev["R"] @ fwd["C"]
            failures.append(not np.allclose(residual[:4], 0, rtol=0, atol=2e-12))
            print(
                f"{kind} analytic affine roundtrip={residual[:4].tolist()} expected=[0, 0, 0, 0] (Cback+Rback*Cforward)"
            )
        lat = "Q: EKICKER,L=0,HKICK=0.00017,VKICK=-0.00011\nBL: LINE=(Q)\n"
        fwd = track("corrector", lat, q)
        back = track("corrector", lat, fwd["q"], reverse=True)
        error = close(back["q"], q, 2e-13, "Dedicated corrector inverse")
        print(
            f"controls: continuous-field forward quadrature and zero-steering bodies pass; EKICKER roundtrip max error={error:.4g}"
        )
        verdict(any(failures))


    if __name__ == "__main__":
        main()

    ```

## Provenance

| Role | Model | Effort | Date | Code examined |
|---|---|---|---|---|
| Found | <span class="bb-nowrap">`gpt-6-astra`</span> | xhigh | <span class="bb-nowrap">2026-11-06</span> | master (`0355dd6aca12`), source with SDDS a3f730a14c49, CPU, double, serial, one thread |
| Confirmed | <span class="bb-nowrap">`gpt-6.1-sol`</span> | xhigh | <span class="bb-nowrap">2026-11-06</span> | master (`0355dd6aca12`), source with SDDS a3f730a14c49, CPU, double, serial, one thread |

