---
title: "COL-29: insert_sceffects clear leaves a negative specification count, crashing or silently dropping every SCMULT"
hide:
  - toc
---

# COL-29 — insert_sceffects clear leaves a negative specification count, crashing or silently dropping every SCMULT

| | |
|---|---|
| Code | [Elegant](../index.md) · Collective effects |
| Priority | P2 |
| Severity | Silent wrong result |
| Evidence | Reproduced, independently rechecked |
| Last reproduced | 2026-11-04 on Elegant master (`0355dd6`, head build) |
| Source | [clearSCSpecs, get_beamline.c:2364–2377](https://github.com/rtsoliday/elegant/blob/0355dd6aca12c861971fbf8f3ae78a0ded991734/src/get_beamline.c#L2364-L2377)<br>[clear and disable, insertSCeffects.cc:81–87](https://github.com/rtsoliday/elegant/blob/0355dd6aca12c861971fbf8f3ae78a0ded991734/src/insertSCeffects.cc#L81-L87)<br>[warning, insertSCeffects.cc:112–116](https://github.com/rtsoliday/elegant/blob/0355dd6aca12c861971fbf8f3ae78a0ded991734/src/insertSCeffects.cc#L112-L116)<br>[addSCSpec, get_beamline.c:2339–2357](https://github.com/rtsoliday/elegant/blob/0355dd6aca12c861971fbf8f3ae78a0ded991734/src/get_beamline.c#L2339-L2357)<br>[disable and clear, elegant.tex:4480–4481](https://github.com/rtsoliday/elegant/blob/0355dd6aca12c861971fbf8f3ae78a0ded991734/doc/elegant.tex#L4480-L4481) |

Setting clear = 1 on insert_sceffects decrements the count of insertion specifications to −1 instead of 0, so the next specification is written one entry before its array. Clearing and replacing an earlier command with the same selector then completes with no SCMULT and no warning, so the run has no space charge; clearing in a fresh run, clearing alone, and clearing with disable = 1 after an earlier command crash. The manual says clear deletes prior insertions and disable makes a command ignored, so each run must complete, and all but the lone clear must insert one SCMULT. In a one-segment ring with a 0.8000 nC bunch, the replacement's expected linear slope kick of 2.379×10⁻⁴ becomes exactly zero.

Run `python COL-29.py` with [_elegant.py](../helpers/_elegant.py) and [_scinsert.py](../helpers/_scinsert.py) in the same directory.

??? example "Reproducer: COL-29.py"

    [Download COL-29.py](COL-29.py)

    ```python
    """COL-29: insert_sceffects clear leaves a negative specification count, crashing or silently dropping every SCMULT

    Setting clear = 1 on insert_sceffects decrements the count of insertion specifications to −1 instead
    of 0, so the next specification is written one entry before its array. Clearing and replacing an
    earlier command with the same selector then completes with no SCMULT and no warning, so the run has
    no space charge; clearing in a fresh run, clearing alone, and clearing with disable = 1 after an
    earlier command crash. The manual says clear deletes prior insertions and disable makes a command
    ignored, so each run must complete, and all but the lone clear must insert one SCMULT. In a
    one-segment ring with a 0.8000 nC bunch, the replacement's expected linear slope kick of 2.379×10⁻⁴
    becomes exactly zero.

    The script runs a one-segment ring (a 0.8 nC bunch of 243 particles and a 1 m EMATRIX quarter
    turn) with insert_sceffects commands that use clear: clear and replace a prior command, clear on a
    fresh run, clear alone, and clear together with disable after a prior command. The expected outcome
    follows the manual's clear and disable descriptions, and the expected kick is the manual's linear
    formula for one SCMULT after the segment. A prior command followed by a disabled one is the control.
    """

    import numpy as np
    from _elegant import verdict
    from _scinsert import N, failure_line, linear_kick, spec, track

    EXPECTED = linear_kick()
    PROBE = int(np.argmax(EXPECTED[:, 0]))  # the reported particle: largest horizontal kick
    ATOL = 1e-18
    PRIOR = spec(name="R")
    # (label, commands, SCMULT insertions the manual requires, control?)
    CASES = (
        ("prior command, then disable=1 (control)", PRIOR + spec(name="R", disable=1), 1, True),
        (
            "prior command, then clear=1 with the same selector",
            PRIOR + spec(name="R", clear=1),
            1,
            False,
        ),
        ("clear=1 with a selector, fresh run", spec(name="R", clear=1), 1, False),
        ("clear=1 alone, fresh run", "&insert_sceffects clear=1 &end\n", 0, False),
        ("prior command, then clear=1,disable=1", PRIOR + spec(name="R", clear=1, disable=1), 1, False),
    )

    defect = False
    # Serial only: after the corrupting write, Pelegant can spin without finishing, so a 3-rank run
    # would wait for the native timeout.
    for mode in ("serial",):
        for label, commands, count, control in CASES:
            result = track(commands, mode)
            expected = f"exit 0, {count} SCMULT" + (
                f", kick={EXPECTED[PROBE, 0]:.16g}" if count else ", no kick"
            )
            if result["status"]:
                observed = f"exit {result['status']} ({failure_line(result['log'])})"
                ok = False
            else:
                kick = result["kick"]
                observed = f"exit 0, {len(result['insertions'])} SCMULT, kick={kick[PROBE, 0]:.16g}"
                want = EXPECTED if count else np.zeros((N, 2))
                ok = len(result["insertions"]) == count and bool(
                    np.allclose(kick, want, rtol=0, atol=ATOL)
                )
            print(f"{mode} {label}: observed={observed} expected={expected} (particle {PROBE + 1})")
            if control and not ok:
                raise SystemExit(f"{mode} {label}: the control fails")
            defect |= not ok
    print(f"comparison: kicks atol={ATOL:g} against the manual's linear kick for one SCMULT")
    verdict(defect)

    ```

## Provenance

| Role | Model | Effort | Date | Code examined |
|---|---|---|---|---|
| Found | `gpt-6-astra` | xhigh | 2026-11-04 | master (`0355dd6aca12`), source with SDDS a3f730a14c49, CPU, serial |
| Confirmed | `gpt-6.1-sol` | xhigh | 2026-11-04 | master (`0355dd6aca12`), source with SDDS a3f730a14c49, CPU, serial |

