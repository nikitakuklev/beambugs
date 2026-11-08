---
title: "XLT-24: MAD export without a reference orbit aborts for a full-fringe quadrupole"
hide:
  - toc
---

# XLT-24 — MAD export without a reference orbit aborts for a full-fringe quadrupole

| | |
|---|---|
| Code | [Bmad](../index.md) · Translators |
| Priority | P4 |
| Failure | Crash or API hazard |
| Evidence | Reproduced, independently rechecked |
| Last reproduced | 2026-11-05 on Bmad release [20261031-0](https://github.com/bmad-sim/bmad-ecosystem/tree/20261031-0) (`c4c629f`) |
| Tags | <span class="bb-tag">partly-source-only</span> [?](../../methodology.md#tags) |
| Source | [optional reference-orbit allocation, bmad/output/write_lattice_mad_format.f90:180–186](https://github.com/bmad-sim/bmad-ecosystem/blob/c4c629fc9071d3a56cb1ba5bc714baba65e9c5d3/bmad/output/write_lattice_mad_format.f90#L180-L186)<br>[quadrupole fringe-map construction, bmad/output/write_lattice_mad_format.f90:328–358](https://github.com/bmad-sim/bmad-ecosystem/blob/c4c629fc9071d3a56cb1ba5bc714baba65e9c5d3/bmad/output/write_lattice_mad_format.f90#L328-L358)<br>[reference charge lookup, bmad/ptc/ele_to_fibre.f90:172–176](https://github.com/bmad-sim/bmad-ecosystem/blob/c4c629fc9071d3a56cb1ba5bc714baba65e9c5d3/bmad/ptc/ele_to_fibre.f90#L172-L176)<br>[charge_of unset-species error, sim_utils/interfaces/particle_species_mod.f90:1512–1522](https://github.com/bmad-sim/bmad-ecosystem/blob/c4c629fc9071d3a56cb1ba5bc714baba65e9c5d3/sim_utils/interfaces/particle_species_mod.f90#L1512-L1522)<br>[writer argument contract, bmad/output/write_lattice_mad_format.f90:28–31](https://github.com/bmad-sim/bmad-ecosystem/blob/c4c629fc9071d3a56cb1ba5bc714baba65e9c5d3/bmad/output/write_lattice_mad_format.f90#L28-L31) |

Calling the MAD-X lattice writer without its optional reference orbit aborts for a quadrupole with full fringe fields. The writer allocates default coordinates without setting their particle species, then passes one into the fringe-map calculation. The resulting charge lookup encounters the unset species and stops. Supplying the tracked reference orbit succeeds, as does the no-orbit call for the same quadrupole without fringe fields.

Run `python XLT-24.py` with [_bmad.py](../helpers/_bmad.py) and [_lattice_import_export.py](../helpers/_lattice_import_export.py) in the same directory.

??? example "Reproducer: XLT-24.py"

    [Download XLT-24.py](XLT-24.py)

    ```python
    """XLT-24: MAD export without a reference orbit aborts for a full-fringe quadrupole

    Calling the MAD-X lattice writer without its optional reference orbit aborts for a quadrupole with full fringe fields. The writer allocates default coordinates without setting their particle species, then passes one into the fringe-map calculation. The resulting charge lookup encounters the unset species and stops. Supplying the tracked reference orbit succeeds, as does the no-orbit call for the same quadrupole without fringe fields.

    The script compiles an inline caller of the foreign-lattice writer and checks tracked-orbit and fringe-free controls. It then calls the full-fringe export without an orbit and recognizes only the unset-species charge failure.
    """

    from _bmad import RunFailure, verdict
    from _lattice_import_export import HEADER, inspect, valid

    lattice = (
        HEADER
        + """q:quadrupole,l=.5,k1=1.2,fringe_type=full
    lat:line=(q)
    use,lat
    """
    )
    valid(inspect(lattice, "MAD-X"))
    valid(inspect(lattice.replace("fringe_type=full", "fringe_type=none"), "MAD-X", noorbit=True))
    print("controls: tracked-orbit full fringe and no-orbit fringe-free exports complete")
    try:
        result = inspect(lattice, "MAD-X", noorbit=True)
    except RunFailure as exc:
        if (
            "PARSE_ERROR  F" not in exc.log
            or "charge_of:" not in exc.log
            or "CHARGE NOT KNOWN FOR SPECIES: -999" not in exc.log
        ):
            raise
        print(
            f"no-orbit full fringe: status={exc.status}; charge_of rejected unset species; expected completed export"
        )
        print("charge_of: CHARGE NOT KNOWN FOR SPECIES: -999")
        defect = True
    else:
        valid(result)
        print("no-orbit full fringe: completed export; expected completed export")
        defect = False
    verdict(defect)

    ```

## Provenance

| Role | Model | Effort | Date | Code examined |
|---|---|---|---|---|
| Found | <span class="bb-nowrap">`gpt-6.1-sol`</span> | xhigh | <span class="bb-nowrap">2026-10-21</span> | foreign-export-embedded-kicks (`5afc3c33a462`), source, production, CPU, double, serial |
| Confirmed | <span class="bb-nowrap">`claude-opus-5-5`</span> | xhigh | <span class="bb-nowrap">2026-10-22</span> | 20261021-0 (`d00a8f17e55a`), source, production, CPU, double, serial |

