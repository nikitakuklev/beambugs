---
title: "TAO-23: Single-element show matrix syntax disagrees with its manual"
hide:
  - toc
---

# TAO-23 — Single-element show matrix syntax disagrees with its manual

| | |
|---|---|
| Code | [Bmad](../index.md) · Tao |
| Priority | P4 |
| Severity | Documentation or example mismatch |
| Evidence | Reproduced, independently rechecked |
| Last reproduced | 2026-11-05 on Bmad 20261031-0 (`c4c629f`) |
| Source | [single-endpoint selection, tao/code/tao_show_this.f90:5224–5230](https://github.com/bmad-sim/bmad-ecosystem/blob/c4c629fc9071d3a56cb1ba5bc714baba65e9c5d3/tao/code/tao_show_this.f90#L5224-L5230)<br>[show taylor_map, s:show.taylor, tao/doc/command-list.tex:3297–3300](https://github.com/bmad-sim/bmad-ecosystem/blob/c4c629fc9071d3a56cb1ba5bc714baba65e9c5d3/tao/doc/command-list.tex#L3297-L3300)<br>[show matrix alias, tao/doc/command-list.tex:2908–2911](https://github.com/bmad-sim/bmad-ecosystem/blob/c4c629fc9071d3a56cb1ba5bc714baba65e9c5d3/tao/doc/command-list.tex#L2908-L2911) |

Giving show matrix a single element name does not select that element’s map as the Tao manual promises. In an open line, it returns transport from the beginning to the element; in a closed ring, it returns a full turn at that element. A 1.7 m drift preceded by a 0.7 m drift therefore displays R₁₂ = 2.400 m instead of the element’s 1.700 m. Explicit endpoints and the -elements option return the correct element map.

Download these shared helpers into the same directory as the reproducer scripts:

- [Download _bmad.py](../helpers/_bmad.py) · [View source](../helpers/index.md#_bmad)
- [Download _tao_contracts_normal_form.py](../helpers/_tao_contracts_normal_form.py) · [View source](../helpers/index.md#_tao_contracts_normal_form)

??? example "Reproducer: TAO-23.py"

    [Download TAO-23.py](TAO-23.py)

    ```python
    """TAO-23: Single-element show matrix syntax disagrees with its manual

    Giving show matrix a single element name does not select that element’s map as the Tao manual promises. In an open line, it returns transport from the beginning to the element; in a closed ring, it returns a full turn at that element. A 1.7 m drift preceded by a 0.7 m drift therefore displays R₁₂ = 2.400 m instead of the element’s 1.700 m. Explicit endpoints and the -elements option return the correct element map.

    The script compares the shorthand, explicit endpoints, and -elements for a two-drift open line and a closed FODO ring. Independent drift and thick-quadrupole matrices identify the selected transport in each case.
    """

    import numpy as np
    from _bmad import compare, verdict
    from _tao_contracts_normal_form import FODO, LATTICE, block, fodo_matrix, tao


    def matrix(log, command):
        rows = [
            line.split(":")[0].split()
            for line in block(log, command).splitlines()
            if ":" in line and len(line.split(":")[0].split()) == 6
        ]
        values = np.array(rows, dtype=float)
        assert values.shape == (6, 6)
        return values


    open_lat = LATTICE.replace("e: drift, l=1.7", "d: drift, l=0.7\ne: drift, l=1.7").replace(
        "lat: line=(e)", "lat: line=(d,e)"
    )
    bad = False
    for lattice, name, previous, length, geometry in (
        (open_lat, "E", "D", 1.7, "open"),
        (FODO, "4", "3", 0.8, "closed"),
    ):
        commands = [
            f"show matrix {name}",
            f"show matrix -elements {name}",
            f"show matrix {previous} {name}",
        ]
        result = tao(lattice, commands)
        short, element, explicit = [matrix(result.log, c) for c in commands]
        p0c = 1e6 if geometry == "open" else 1e9
        # Bmad's z=-beta*c*(t-t_ref) gives R56=L/gamma0**2 on axis through
        # these straight sections at constant reference momentum.
        inverse_gamma2 = 510998.95069**2 / (p0c**2 + 510998.95069**2)
        expected = np.eye(6)
        expected[0, 1] = expected[2, 3] = length
        expected[4, 5] = length * inverse_gamma2
        assert not compare(f"{geometry}, -elements control", element, expected, atol=5e-9)
        assert not compare(f"{geometry}, explicit endpoints control", explicit, expected, atol=5e-9)
        actual_map = np.eye(6)
        if geometry == "open":
            actual_map[0, 1] = actual_map[2, 3] = 2.4
            actual_map[4, 5] = 2.4 * inverse_gamma2
        else:
            actual_map[:4, :4] = fodo_matrix()
            actual_map[4, 5] = 2 * inverse_gamma2
        assert not compare(f"{geometry}, shorthand endpoint selection", short, actual_map, atol=5e-9)
        bad |= compare(
            f"{geometry}, shorthand versus documented element map", short, expected, atol=5e-9
        )
    verdict(bad)

    ```

## Provenance

| Role | Model | Effort | Date | Code examined |
|---|---|---|---|---|
| Found<br><small>source check</small> | `gpt-6-astra` | xhigh | 2026-10-20 | 20261020-0 (`19b8afb46a40`) |
| Confirmed<br><small>source check</small> | `gpt-6-astra` | xhigh | 2026-10-21 | 20261020-0 (`19b8afb46a40`) |
| Confirmed | `gpt-6.1-sol` | xhigh | 2026-11-05 | 20261031-0 (`c4c629fc9071`), source, production, GNU Fortran 15.3.0, CPU, double, serial, OpenMP=1 |

