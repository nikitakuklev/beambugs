---
title: "COL-59: Tabulated-wake origin depends on particle storage order"
hide:
  - toc
---

# COL-59 — Tabulated-wake origin depends on particle storage order

| | |
|---|---|
| Code | [Bmad](../index.md) · Collective effects |
| Priority | P2 |
| Failure | Wrong physics |
| Evidence | Probe-pinned |
| Last reproduced | 2026-11-05 on Bmad release [20261031-0](https://github.com/bmad-sim/bmad-ecosystem/tree/20261031-0) (`c4c629f`) |
| Tags | <span class="bb-tag">partly-source-only</span> [?](../../methodology.md#tags) |
| Source | [`track1_sr_wake`, bmad/multiparticle/wake_mod.f90:728–754](https://github.com/bmad-sim/bmad-ecosystem/blob/c4c629fc9071d3a56cb1ba5bc714baba65e9c5d3/bmad/multiparticle/wake_mod.f90#L728-L754)<br>[`sr_z_long_wake`, bmad/multiparticle/wake_mod.f90:498–505](https://github.com/bmad-sim/bmad-ecosystem/blob/c4c629fc9071d3a56cb1ba5bc714baba65e9c5d3/bmad/multiparticle/wake_mod.f90#L498-L505) |

Reordering the stored particles of an otherwise identical bunch changes its tabulated short-range wake kicks. The dispatcher selects the table origin by averaging the storage indices of the head and tail, rather than their longitudinal coordinates. In a three-particle case, the leading particle’s normalized momentum kick changes from −1.194 × 10⁻⁶ to −1.000 × 10⁻⁶ under a storage permutation. A direct calculation with a fixed physical center agrees with an independent charge-deposition and interpolation sum.

Run `python COL-59.py` with [_bmad.py](../helpers/_bmad.py) and [_wake_tables_and_bbu.py](../helpers/_wake_tables_and_bbu.py) in the same directory.

??? example "Reproducer: COL-59.py"

    [Download COL-59.py](COL-59.py)

    ```python
    """COL-59: Tabulated-wake origin depends on particle storage order

    Reordering the stored particles of an otherwise identical bunch changes its tabulated short-range wake kicks. The dispatcher selects the table origin by averaging the storage indices of the head and tail, rather than their longitudinal coordinates. In a three-particle case, the leading particle’s normalized momentum kick changes from −1.194 × 10⁻⁶ to −1.000 × 10⁻⁶ under a storage permutation. A direct calculation with a fixed physical center agrees with an independent charge-deposition and interpolation sum.

    The script tracks a fixed three-particle set in two storage orders and remaps both results to physical order. A direct call with a fixed midpoint is checked against an independent cloud-in-cell sum.
    """

    import numpy as np
    from _bmad import compare, fortran, verdict
    from _wake_tables_and_bbu import WAKE_SUPPORT_SOURCE, require, rows, table_kicks, table_lattice

    lattice, table = table_lattice()
    actual = rows(fortran(WAKE_SUPPORT_SOURCE, ["1"], files={"lat.bmad": lattice}).log)
    z = np.array([0.013, 0.002, -0.014])
    charge = np.array([2e-12, 0, 0])
    require(
        "fixed midpoint direct delta; independent CIC sum",
        actual["permutation0"],
        table_kicks(z, charge, table, -0.0005),
        rtol=3e-14,
        atol=1e-21,
    )
    print(
        f"physical-order delta A={actual['permutation1'].tolist()} B={actual['permutation2'].tolist()}"
    )
    verdict(
        compare(
            "storage permutation delta difference; invariant physical bunch",
            actual["permutation1"] - actual["permutation2"],
            np.zeros(3),
            atol=2e-21,
        )
    )

    ```

## Provenance

| Role | Model | Effort | Date | Code examined |
|---|---|---|---|---|
| Found | <span class="bb-nowrap">`gemini-4-argon-high`</span> |  | <span class="bb-nowrap">2026-10-20</span> | 20261020-0 (`19b8afb46a40`), source, production, CPU, double, serial |
| Confirmed | <span class="bb-nowrap">`gpt-6-astra`</span> | xhigh | <span class="bb-nowrap">2026-10-20</span> | 20261020-0 (`19b8afb46a40`), source, production, CPU, double, serial |

