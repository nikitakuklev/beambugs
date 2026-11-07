---
title: "COL-60: A table-support error marks only the final array particle lost"
hide:
  - toc
---

# COL-60 — A table-support error marks only the final array particle lost

| | |
|---|---|
| Code | [Bmad](../index.md) · Collective effects |
| Priority | P2 |
| Severity | Wrong physics |
| Evidence | Probe-pinned |
| Last reproduced | 2026-11-05 on Bmad 20261031-0 (`c4c629f`, head build) |
| Tags | <span class="bb-tag">partly-source-only</span> [?](../../methodology.md#tags) |
| Source | [`sr_z_long_wake`, bmad/multiparticle/wake_mod.f90:548–580](https://github.com/bmad-sim/bmad-ecosystem/blob/c4c629fc9071d3a56cb1ba5bc714baba65e9c5d3/bmad/multiparticle/wake_mod.f90#L548-L580)<br>[`track1_sr_wake`, bmad/multiparticle/wake_mod.f90:731–735](https://github.com/bmad-sim/bmad-ecosystem/blob/c4c629fc9071d3a56cb1ba5bc714baba65e9c5d3/bmad/multiparticle/wake_mod.f90#L731-L735)<br>[`Short-Range Longitudinal Wake With a Wake Table (s:sr.wake.z.long)`, bmad/doc/attributes.tex:3108–3118](https://github.com/bmad-sim/bmad-ecosystem/blob/c4c629fc9071d3a56cb1ba5bc714baba65e9c5d3/bmad/doc/attributes.tex#L3108-L3118) |

When a bunch exceeds a tabulated short-range wake’s support, the error path marks only the final stored particle lost. With the first particle outside the table and the last inside it, the outlier remains alive while the in-window particle is lost. Both kicks remain zero, although the documented unrecoverable support error requires tracking of the whole bunch to stop.

Run `python COL-60.py` with [_bmad.py](../helpers/_bmad.py) and [_wake_tables_and_bbu.py](../helpers/_wake_tables_and_bbu.py) in the same directory.

??? example "Reproducer: COL-60.py"

    [Download COL-60.py](COL-60.py)

    ```python
    """COL-60: A table-support error marks only the final array particle lost

    When a bunch exceeds a tabulated short-range wake’s support, the error path marks only the final stored particle lost. With the first particle outside the table and the last inside it, the outlier remains alive while the in-window particle is lost. Both kicks remain zero, although the documented unrecoverable support error requires tracking of the whole bunch to stop.

    The script puts the first particle outside the wake window and the final particle inside it. It requires the support-error diagnostic and zero kicks, then compares both loss flags with rejection of the whole bunch.
    """

    import re

    from _bmad import compare, fortran, verdict
    from _wake_tables_and_bbu import WAKE_SUPPORT_SOURCE, require, rows, table_lattice

    lattice, _ = table_lattice()
    result = fortran(
        WAKE_SUPPORT_SOURCE, ["2"], files={"lat.bmad": lattice}, outputs=["diagnostic.log"]
    )
    diagnostic = " ".join(result.outputs["diagnostic.log"].decode().split())
    if not re.fullmatch(
        r"\[ERROR(?: \| [^]]+)?\] sr_z_long_wake: The bunch is longer than the sr-z wake can handle for element: W",
        diagnostic,
    ):
        raise RuntimeError(diagnostic)
    print("diagnostic: " + diagnostic)
    actual = rows(result.log)
    require("support failure delta", actual["kicks"], [0, 0])
    verdict(compare("loss flags; whole bunch must stop", actual["lost"], [1, 1], atol=0))

    ```

## Provenance

| Role | Model | Effort | Date | Code examined |
|---|---|---|---|---|
| Found | `gemini-4-argon-high` |  | 2026-10-20 | 20261020-0 (`19b8afb46a40`), source, production, CPU, double, serial |
| Confirmed | `gpt-6-astra` | xhigh | 2026-10-20 | 20261020-0 (`19b8afb46a40`), source, production, CPU, double, serial |

