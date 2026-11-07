---
title: "TRK-48: Automatic element division repeats integrated magnetic steering"
hide:
  - toc
---

# TRK-48 — Automatic element division repeats integrated magnetic steering

| | |
|---|---|
| Code | [Elegant](../index.md) · Tracking |
| Priority | P2 |
| Severity | Wrong physics |
| Evidence | Reproduced, independently rechecked |
| Last reproduced | 2026-11-06 on Elegant master (`0355dd6`, committed 2026-11-03) |
| Source | [src/mad_parse.c:567–635](https://github.com/rtsoliday/elegant/blob/0355dd6aca12c861971fbf8f3ae78a0ded991734/src/mad_parse.c#L567-L635)<br>[src/track_data.c:377–378](https://github.com/rtsoliday/elegant/blob/0355dd6aca12c861971fbf8f3ae78a0ded991734/src/track_data.c#L377-L378)<br>[src/track_data.c:1165–1166](https://github.com/rtsoliday/elegant/blob/0355dd6aca12c861971fbf8f3ae78a0ded991734/src/track_data.c#L1165-L1166)<br>[src/track_data.c:1265–1266](https://github.com/rtsoliday/elegant/blob/0355dd6aca12c861971fbf8f3ae78a0ded991734/src/track_data.c#L1265-L1266)<br>[src/track_data.c:1959–1960](https://github.com/rtsoliday/elegant/blob/0355dd6aca12c861971fbf8f3ae78a0ded991734/src/track_data.c#L1959-L1960)<br>[src/track_data.c:4109–4112](https://github.com/rtsoliday/elegant/blob/0355dd6aca12c861971fbf8f3ae78a0ded991734/src/track_data.c#L4109-L4112)<br>[src/multipole.c:1331–1332](https://github.com/rtsoliday/elegant/blob/0355dd6aca12c861971fbf8f3ae78a0ded991734/src/multipole.c#L1331-L1332)<br>[src/csbend.c:1083–1084](https://github.com/rtsoliday/elegant/blob/0355dd6aca12c861971fbf8f3ae78a0ded991734/src/csbend.c#L1083-L1084)<br>[doc/elegant.tex:3005–3046](https://github.com/rtsoliday/elegant/blob/0355dd6aca12c861971fbf8f3ae78a0ded991734/doc/elegant.tex#L3005-L3046)<br>[doc/elegant.tex:7273–7275](https://github.com/rtsoliday/elegant/blob/0355dd6aca12c861971fbf8f3ae78a0ded991734/doc/elegant.tex#L7273-L7275)<br>[doc/kquad.tex:64–68](https://github.com/rtsoliday/elegant/blob/0355dd6aca12c861971fbf8f3ae78a0ded991734/doc/kquad.tex#L64-L68)<br>[src/track_data.c:4313–4313](https://github.com/rtsoliday/elegant/blob/0355dd6aca12c861971fbf8f3ae78a0ded991734/src/track_data.c#L4313-L4313) |

Automatically dividing a QUAD, KQUAD, KSEXT, DQCOR or CSBEND copies its full integrated steering into every piece. The parameters HKICK and VKICK, or XKICK and YKICK for CSBEND, must instead be allocated between the pieces. Dividing a 0.6000 m zero-gradient KQUAD into five pieces changes its requested normalized horizontal impulse from 1.700×10⁻⁴ to 8.500×10⁻⁴. Analytic affine maps also repeat the steering. Explicit pieces carrying one fifth of each kick recover the original total impulse; zero-steering controls pass.

Run `python TRK-48.py` with [_elegant.py](../helpers/_elegant.py) and [_slicing.py](../helpers/_slicing.py) in the same directory.

??? example "Reproducer: TRK-48.py"

    [Download TRK-48.py](TRK-48.py)

    ```python
    """TRK-48: Automatic element division repeats integrated magnetic steering

    Automatically dividing a QUAD, KQUAD, KSEXT, DQCOR or CSBEND copies its full integrated steering into every piece. The parameters HKICK and VKICK, or XKICK and YKICK for CSBEND, must instead be allocated between the pieces. Dividing a 0.6000 m zero-gradient KQUAD into five pieces changes its requested normalized horizontal impulse from 1.700×10⁻⁴ to 8.500×10⁻⁴. Analytic affine maps also repeat the steering. Explicit pieces carrying one fifth of each kick recover the original total impulse; zero-steering controls pass.

    The script compares native division with the sum of integrated straight-magnet impulses and explicitly allocated bend pieces. It checks both division commands, affine maps and zero-steering controls using fresh native processes.
    """

    import numpy as np
    from _elegant import verdict
    from _slicing import bend, canonical, close, discrepancy, straight, track


    def main():
        failures = []
        q = np.zeros((1, 6))
        for kind in ("KQUAD", "KSEXT", "QUAD", "DQCOR"):
            kicks = np.array([0.00017, 0 if kind == "QUAD" else -0.00011])
            for n in (1, 3, 5):
                lat = straight(kind, vk=kicks[1], slices=30 // n)
                obs = track(kind, lat, q, divisions=n, matrix=True)
                impulse = obs["q"][0, [1, 3]] if kind == "QUAD" else canonical(obs["q"])[0]
                # A uniform integrated field adds K once, independent of drift quadrature.
                if n == 1:
                    close(impulse, kicks, 3e-14, "Undivided impulse")
                    affine = obs["C"][[1, 3]]
                else:
                    failures.append(discrepancy(impulse, kicks, n * kicks, 3e-14))
                    failures.append(discrepancy(obs["C"][[1, 3]], affine, n * affine, 3e-14))
                    piece = straight(
                        kind, length=0.6 / n, hk=kicks[0] / n, vk=kicks[1] / n, slices=30 // n
                    ).splitlines()[0]
                    control = track(kind, piece + "\nBL: LINE=(" + ",".join(["Q"] * n) + ")\n", q)
                    control_impulse = (
                        control["q"][0, [1, 3]] if kind == "QUAD" else canonical(control["q"])[0]
                    )
                    close(control_impulse, kicks, 3e-14, "Allocated impulse")
                    command = track(
                        kind, lat, q, before=f"&divide_elements name=Q, divisions={n} &end\n"
                    )
                    close(command["q"], obs["q"], 2e-13, "Division command equivalence")
                print(
                    f"{kind} pieces={n}: impulse={impulse.tolist()} expected={kicks.tolist()} "
                    f"(sum of allocated kicks); affine C2={obs['C'][1]:.16g}"
                )
            zero = track(kind, straight(kind, hk=0, vk=0), q, divisions=5)
            close(zero["q"][:, :4], 0, 1e-15, "Zero-steering control")
        for expand in (0, 1):
            base = track("bend", bend(expand=expand), q, matrix=True)
            auto = track("bend", bend(expand=expand), q, divisions=3, matrix=True)
            # Equal field requires K_piece/theta_piece=K/theta. Preserve 66 total steps
            # for the physical control; match the copier's 23 per piece for the bad law.
            copies = []
            for allocated in (True, False):
                factor = 3 if allocated else 1
                piece = bend(
                    length=0.2,
                    angle=0.015,
                    hk=0.00017 / factor,
                    vk=-0.00011 / factor,
                    slices=22 if allocated else 23,
                    expand=expand,
                ).splitlines()[0]
                copies.append(track("bend", piece + "\nBL: LINE=(Q,Q,Q)\n", q, matrix=True))
            error = close(copies[0]["q"], base["q"], 2e-12, "Equal-field bend pieces")
            failures.append(discrepancy(auto["q"], base["q"], copies[1]["q"], 2e-12))
            failures.append(discrepancy(auto["C"], base["C"], copies[1]["C"], 2e-12))
            print(
                f"CSBEND expand={expand} pieces=3: xprime={auto['q'][0, 1]:.16g} "
                f"expected={base['q'][0, 1]:.16g} (equal-field pieces); allocated max error={error:.4g}"
            )
        by_length = track(
            "length selector",
            straight("KQUAD"),
            q,
            before="&divide_elements name=Q, maximum_length=0.21 &end\n",
        )
        failures.append(
            discrepancy(canonical(by_length["q"])[0], [0.00017, -0.00011], [0.00051, -0.00033], 3e-14)
        )
        print(
            "controls: explicit K/N pieces, zero steering and both division commands pass; maximum_length selects three pieces"
        )
        verdict(any(failures))


    if __name__ == "__main__":
        main()

    ```

## Provenance

| Role | Model | Effort | Date | Code examined |
|---|---|---|---|---|
| Found | `gpt-6-astra` | xhigh | 2026-11-06 | master (`0355dd6aca12`), source with SDDS a3f730a14c49, CPU, double, serial, one thread |
| Confirmed | `gpt-6.1-sol` | xhigh | 2026-11-06 | master (`0355dd6aca12`), source with SDDS a3f730a14c49, CPU, double, serial, one thread |

