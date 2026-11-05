---
title: "COL-18: CSR step-control documentation names two nonexistent attributes"
hide:
  - toc
---

# COL-18 — CSR step-control documentation names two nonexistent attributes

| | |
|---|---|
| Code | [Bmad](../index.md) · Collective effects |
| Priority | P4 |
| Severity | Documentation or example mismatch |
| Evidence | Reproduced, independently rechecked |
| Last reproduced | 2026-11-05 on Bmad 20261031-0 (`c4c629f`) |
| Tags | <span class="bb-tag">partly-source-only</span> [?](../../methodology.md#tags) |
| Source | [Bmad manual, Space_Charge_Common_Struct, s:sc.com, bmad/doc/param-structs.tex:487–494](https://github.com/bmad-sim/bmad-ecosystem/blob/c4c629fc9071d3a56cb1ba5bc714baba65e9c5d3/bmad/doc/param-structs.tex#L487-L494)<br>[track1_bunch_csr lookup and diagnostic, bmad/space_charge/csr_and_space_charge_mod.f90:220–225](https://github.com/bmad-sim/bmad-ecosystem/blob/c4c629fc9071d3a56cb1ba5bc714baba65e9c5d3/bmad/space_charge/csr_and_space_charge_mod.f90#L220-L225)<br>[track1_bunch_csr3d lookup and diagnostic, bmad/space_charge/csr_and_space_charge_mod.f90:1936–1942](https://github.com/bmad-sim/bmad-ecosystem/blob/c4c629fc9071d3a56cb1ba5bc714baba65e9c5d3/bmad/space_charge/csr_and_space_charge_mod.f90#L1936-L1942)<br>[registered CSR_DS_STEP, bmad/modules/attribute_mod.f90:723–729](https://github.com/bmad-sim/bmad-ecosystem/blob/c4c629fc9071d3a56cb1ba5bc714baba65e9c5d3/bmad/modules/attribute_mod.f90#L723-L729) |

When users set the element step size using either spelling in the CSR parameter description, the lattice parser rejects the input. The manual names csr_ds_track and csr_track_step, while the accepted attribute is csr_ds_step. Both CSR wrappers also print the stale name CSR_TRACK_STEP in their missing-step diagnostic. The documented names and diagnostics should identify csr_ds_step and retain the global ds_track_step fallback.

Download these shared helpers into the same directory as the reproducer scripts:

- [Download _bmad.py](../helpers/_bmad.py) · [View source](../helpers/index.md#_bmad)
- [Download _collective_contracts.py](../helpers/_collective_contracts.py) · [View source](../helpers/index.md#_collective_contracts)

??? example "Reproducer: COL-18.py"

    [Download COL-18.py](COL-18.py)

    ```python
    """COL-18: CSR step-control documentation names two nonexistent attributes

    When users set the element step size using either spelling in the CSR parameter description, the lattice parser rejects the input. The manual names csr_ds_track and csr_track_step, while the accepted attribute is csr_ds_step. Both CSR wrappers also print the stale name CSR_TRACK_STEP in their missing-step diagnostic. The documented names and diagnostics should identify csr_ds_step and retain the global ds_track_step fallback.

    The script parses an sbend with csr_ds_step as a successful control, then tries each documented
    spelling in a fresh parser process. It checks the explicit bad-attribute diagnostic and the
    manual text from BMAD_SOURCE. BMAD_DIST selects the production build.
    """

    from _bmad import verdict
    from _collective_contracts import parse_lattice, source_text

    manual = source_text("bmad/doc/param-structs.tex")
    if not parse_lattice("sbend, l=1, g=0.1, csr_ds_step=0.01")[0]:
        raise RuntimeError("Registered csr_ds_step control failed")
    defect = False
    for name in ["csr_ds_track", "csr_track_step"]:
        accepted, log = parse_lattice(f"sbend, l=1, g=0.1, {name}=0.01")
        if not accepted and "BAD ATTRIBUTE NAME:" not in log:
            raise RuntimeError(log)
        documented = "\\vn{" + name + "}" in manual
        print(
            f"{name}: documented={documented}, parser_accepts={accepted}; expected documented input accepted"
        )
        defect |= documented and not accepted
    verdict(defect)

    ```

## Provenance

| Role | Model | Effort | Date | Code examined |
|---|---|---|---|---|
| Found<br><small>source check</small> | `gpt-6-astra` | xhigh | 2026-10-20 | 20261020-0 (`19b8afb46a40`) |
| Confirmed<br><small>source check</small> | `gpt-6-astra` | xhigh | 2026-10-21 | 20261020-0 (`19b8afb46a40`) |
| Confirmed | `gpt-6.1-sol` | xhigh | 2026-11-05 | 20261031-0 (`c4c629fc9071`), source, production, GNU Fortran 15.3.0, CPU, double, serial, OpenMP=1 |

