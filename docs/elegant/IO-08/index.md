---
title: "IO-08: Serial WATCH divides custom-species kinetic energy by charge magnitude"
hide:
  - toc
---

# IO-08 — Serial WATCH divides custom-species kinetic energy by charge magnitude

| | |
|---|---|
| Code | [Elegant](../index.md) · Lattices, particles and I/O |
| Priority | P4 |
| Severity | Silent wrong result |
| Evidence | Reproduced, independently rechecked |
| Last reproduced | 2026-11-06 on Elegant master (`0355dd6`, committed 2026-11-03) |
| Tags | <span class="bb-tag">partly-source-only</span> [?](../../methodology.md#tags) |
| Source | [sdds_support.c:1374–1379](https://github.com/rtsoliday/elegant/blob/0355dd6aca12c861971fbf8f3ae78a0ded991734/src/sdds_support.c#L1374-L1379)<br>[elegant.c:3306–3308](https://github.com/rtsoliday/elegant/blob/0355dd6aca12c861971fbf8f3ae78a0ded991734/src/elegant.c#L3306-L3308)<br>[sdds_support.c:495](https://github.com/rtsoliday/elegant/blob/0355dd6aca12c861971fbf8f3ae78a0ded991734/src/sdds_support.c#L495)<br>[521](https://github.com/rtsoliday/elegant/blob/0355dd6aca12c861971fbf8f3ae78a0ded991734/src/sdds_support.c#L521)<br>[elegant.tex:1497–1517](https://github.com/rtsoliday/elegant/blob/0355dd6aca12c861971fbf8f3ae78a0ded991734/doc/elegant.tex#L1497-L1517)<br>[elegant.c:3292–3297](https://github.com/rtsoliday/elegant/blob/0355dd6aca12c861971fbf8f3ae78a0ded991734/src/elegant.c#L3292-L3297)<br>[elegant.tex:7422–7431](https://github.com/rtsoliday/elegant/blob/0355dd6aca12c861971fbf8f3ae78a0ded991734/doc/elegant.tex#L7422-L7431)<br>[constants.h:28–35](https://github.com/rtsoliday/SDDS/blob/a3f730a14c49ddd4f4d6347b42d60d34aa7cadc6/include/constants.h#L28-L35) |

For a custom species whose charge magnitude differs from one elementary charge, serial WATCH reports mean kinetic energy divided by that charge magnitude in elementary-charge units. A species with mass 100 electron masses and charge +2e reports 48.35 MeV instead of 96.70 MeV for the tested momentum distribution; charge −2e gives the same error, while +0.5e doubles the result. The expected kinetic energy follows from mass and momentum and is independent of charge. Both centroid and parameter modes fail; unit-charge output and particle drift times are correct.

Run `python IO-08.py` with [_elegant.py](../helpers/_elegant.py), [_watch_energy.py](../helpers/_watch_energy.py) and [_diagnostics.py](../helpers/_diagnostics.py) in the same directory.

??? example "Reproducer: IO-08.py"

    [Download IO-08.py](IO-08.py)

    ```python
    """IO-08: Serial WATCH divides custom-species kinetic energy by charge magnitude

    For a custom species whose charge magnitude differs from one elementary charge, serial WATCH reports mean kinetic energy divided by that charge magnitude in elementary-charge units. A species with mass 100 electron masses and charge +2e reports 48.35 MeV instead of 96.70 MeV for the tested momentum distribution; charge −2e gives the same error, while +0.5e doubles the result. The expected kinetic energy follows from mass and momentum and is independent of charge. Both centroid and parameter modes fail; unit-charge output and particle drift times are correct.

    Twenty fixed particles with mass 100 electron masses cross a 0.75 m exact drift for charges +e, +2e, −2e and +0.5e in separate serial runs. The script compares both WATCH energy modes and an ID-selected output with a charge-independent 70-digit relativistic oracle and checks the precise 1/|q/e| error. Momentum, IDs and flight time provide controls;
    """

    from _elegant import verdict
    from _watch_energy import ME, check


    def main():
        defects = []
        for label, ratio in (("q+1", -1), ("q+2", -2), ("q-2", 2), ("q+0.5", -0.5)):
            # change_particle charge_ratio is relative to the electron's negative charge.
            defects.extend(check("serial", label, f"custom,mass_ratio=100,charge_ratio={ratio}",
                                 100 * ME, charge=abs(ratio)))
        verdict(all(defects))


    if __name__ == "__main__":
        main()

    ```

## Provenance

| Role | Model | Effort | Date | Code examined |
|---|---|---|---|---|
| Found | `gpt-6-astra` | xhigh | 2026-11-06 | master (`0355dd6aca12`), source with SDDS a3f730a14c49, CPU, double, serial, one thread |
| Confirmed<br><small>either charge sign, fractional charge magnitude, selected IDs</small> | `gpt-6.1-sol` | xhigh | 2026-11-06 | master (`0355dd6aca12`), source with SDDS a3f730a14c49, CPU, double, serial, one thread |

