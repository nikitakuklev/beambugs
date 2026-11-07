---
title: "COL-20: The CSR method list omits the experimental three-dimensional selector"
hide:
  - toc
---

# COL-20 — The CSR method list omits the experimental three-dimensional selector

| | |
|---|---|
| Code | [Bmad](../index.md) · Collective effects |
| Priority | P4 |
| Failure | Documentation or example mismatch |
| Evidence | Reproduced, independently rechecked |
| Last reproduced | 2026-11-05 on Bmad release [20261031-0](https://github.com/bmad-sim/bmad-ecosystem/tree/20261031-0) (`c4c629f`) |
| Tags | <span class="bb-tag">experimental</span> <span class="bb-tag">known-limitation</span> <span class="bb-tag">partly-source-only</span> [?](../../methodology.md#tags) |
| Source | [Bmad manual, CSR and Space Charge Methods, s:csr.sc.meth, bmad/doc/methods.tex:634–639](https://github.com/bmad-sim/bmad-ecosystem/blob/c4c629fc9071d3a56cb1ba5bc714baba65e9c5d3/bmad/doc/methods.tex#L634-L639)<br>[Bmad manual, CSR3D mesh parameter, s:sc.com, bmad/doc/param-structs.tex:545–549](https://github.com/bmad-sim/bmad-ecosystem/blob/c4c629fc9071d3a56cb1ba5bc714baba65e9c5d3/bmad/doc/param-structs.tex#L545-L549)<br>[CSR method enum, bmad/modules/bmad_struct.f90:263–264](https://github.com/bmad-sim/bmad-ecosystem/blob/c4c629fc9071d3a56cb1ba5bc714baba65e9c5d3/bmad/modules/bmad_struct.f90#L263-L264)<br>[experimental wrapper and geometry guard, bmad/space_charge/csr_and_space_charge_mod.f90:1909–1922](https://github.com/bmad-sim/bmad-ecosystem/blob/c4c629fc9071d3a56cb1ba5bc714baba65e9c5d3/bmad/space_charge/csr_and_space_charge_mod.f90#L1909-L1922) |

When users consult the CSR method list, it presents off and 1_dim as the possible settings but omits steady_state_3d. The parser accepts steady_state_3d, and the mesh-parameter description refers to it. Its separate tracking wrapper is explicitly marked experimental and nonoperational and restricts collective kicks to nonzero-length sector bends. The method list should explain that selector and its status, or identify itself as a list of supported methods only.

Run `python COL-20.py` with [_bmad.py](../helpers/_bmad.py) and [_collective_contracts.py](../helpers/_collective_contracts.py) in the same directory.

??? example "Reproducer: COL-20.py"

    [Download COL-20.py](COL-20.py)

    ```python
    """COL-20: The CSR method list omits the experimental three-dimensional selector

    When users consult the CSR method list, it presents off and 1_dim as the possible settings but omits steady_state_3d. The parser accepts steady_state_3d, and the mesh-parameter description refers to it. Its separate tracking wrapper is explicitly marked experimental and nonoperational and restricts collective kicks to nonzero-length sector bends. The method list should explain that selector and its status, or identify itself as a list of supported methods only.

    The script parses an sbend using steady_state_3d and checks the resulting method enum. It compares
    that accepted selector with the manual method list and the wrapper's experimental declaration.
    """

    from _bmad import verdict
    from _collective_contracts import parse_lattice, rows, section, source_text

    accepted, log = parse_lattice("sbend, l=1, g=0.1, csr_method=steady_state_3d")
    if not accepted:
        raise RuntimeError(log)
    method = rows(log)["csr_method"][0]
    if method != 3:
        raise RuntimeError("Parser did not store the steady_state_3d enum")
    manual = section(
        source_text("bmad/doc/methods.tex"), "The possible settings for", "The possible settings of"
    )
    wrapper = source_text("bmad/space_charge/csr_and_space_charge_mod.f90")
    experimental = "EXPERIMENTAL. NOT CURRENTLY OPERATIONAL!" in wrapper
    listed = "steady_state_3d" in manual.lower()
    print(
        f"steady_state_3d: parser_accepts={accepted}, enum={method:g}, manual_lists={listed}; expected selector documented"
    )
    print(f"wrapper source declares experimental/nonoperational={experimental}")
    verdict(accepted and not listed)

    ```

## Provenance

| Role | Model | Effort | Date | Code examined |
|---|---|---|---|---|
| Found<br><small>source check</small> | `gpt-6-astra` | xhigh | 2026-10-20 | 20261020-0 (`19b8afb46a40`) |
| Confirmed<br><small>source check</small> | `gpt-6-astra` | xhigh | 2026-10-21 | 20261020-0 (`19b8afb46a40`) |
| Confirmed | `gpt-6.1-sol` | xhigh | 2026-11-05 | 20261031-0 (`c4c629fc9071`), source, production, GNU Fortran 15.3.0, CPU, double, serial, OpenMP=1 |

