---
title: "XLT-18: noaperture documentation contradicts the MAD-X output guard"
hide:
  - toc
---

# XLT-18 — noaperture documentation contradicts the MAD-X output guard

| | |
|---|---|
| Code | [Bmad](../index.md) · Translators |
| Priority | P4 |
| Severity | Documentation or example mismatch |
| Evidence | Reproduced, independently rechecked |
| Last reproduced | 2026-11-05 on Bmad 20261031-0 (`c4c629f`) |
| Source | [CLI aperture-option contract, util_programs/bmad_to_mad_sad_elegant/bmad_to_mad_sad_elegant.f90:23–27](https://github.com/bmad-sim/bmad-ecosystem/blob/c4c629fc9071d3a56cb1ba5bc714baba65e9c5d3/util_programs/bmad_to_mad_sad_elegant/bmad_to_mad_sad_elegant.f90#L23-L27)<br>[writer aperture-option contract, bmad/output/write_lattice_mad_format.f90:36–39](https://github.com/bmad-sim/bmad-ecosystem/blob/c4c629fc9071d3a56cb1ba5bc714baba65e9c5d3/bmad/output/write_lattice_mad_format.f90#L36-L39)<br>[MAD-X aperture output guard, bmad/output/write_lattice_mad_format.f90:987–997](https://github.com/bmad-sim/bmad-ecosystem/blob/c4c629fc9071d3a56cb1ba5bc714baba65e9c5d3/bmad/output/write_lattice_mad_format.f90#L987-L997)<br>[option parser, util_programs/bmad_to_mad_sad_elegant/bmad_to_mad_sad_elegant.f90:65–67](https://github.com/bmad-sim/bmad-ecosystem/blob/c4c629fc9071d3a56cb1ba5bc714baba65e9c5d3/util_programs/bmad_to_mad_sad_elegant/bmad_to_mad_sad_elegant.f90#L65-L67)<br>[MAD-X dispatch, util_programs/bmad_to_mad_sad_elegant/bmad_to_mad_sad_elegant.f90:126–129](https://github.com/bmad-sim/bmad-ecosystem/blob/c4c629fc9071d3a56cb1ba5bc714baba65e9c5d3/util_programs/bmad_to_mad_sad_elegant/bmad_to_mad_sad_elegant.f90#L126-L129) |

The translator documentation says the noaperture option is ignored for MAD-X non-drift elements that can carry aperture attributes. In fact, using the option suppresses those attributes. A quadrupole with 0.01 m and 0.02 m half-apertures exports both limits normally and neither with noaperture. The option documentation and the output guard therefore specify different behavior.

Download these shared helpers into the same directory as the reproducer scripts:

- [Download _bmad.py](../helpers/_bmad.py) · [View source](../helpers/index.md#_bmad)
- [Download _lattice_import_export.py](../helpers/_lattice_import_export.py) · [View source](../helpers/index.md#_lattice_import_export)

??? example "Reproducer: XLT-18.py"

    [Download XLT-18.py](XLT-18.py)

    ```python
    """XLT-18: noaperture documentation contradicts the MAD-X output guard

    The translator documentation says the noaperture option is ignored for MAD-X non-drift elements that can carry aperture attributes. In fact, using the option suppresses those attributes. A quadrupole with 0.01 m and 0.02 m half-apertures exports both limits normally and neither with noaperture. The option documentation and the output guard therefore specify different behavior.

    The script reads the CLI and writer contracts from BMAD_SOURCE and exports the same quadrupole with and without noaperture. It checks the default aperture values and reports whether the documented exception agrees with the option output.
    """

    import re

    from _bmad import compare, source, verdict
    from _lattice_import_export import HEADER, definition, export

    root = source()
    driver = (root / "util_programs/bmad_to_mad_sad_elegant/bmad_to_mad_sad_elegant.f90").read_text()
    writer = (root / "bmad/output/write_lattice_mad_format.f90").read_text()
    driver_contract = "in this case this argument is ignored." in driver
    writer_contract = "with an aperture so in this case this argument is ignored." in writer
    if not driver_contract or not writer_contract:
        raise RuntimeError("The aperture contract changed; inspect the new documentation")
    lattice = (
        HEADER
        + """q:quadrupole,l=.5,k1=.2,x_limit=.01,y_limit=.02
    lat:line=(q)
    use,lat
    """
    )
    normal = definition(export(lattice), "Q")
    option = definition(export(lattice, noaperture=True), "Q")
    match = re.search(r"aperture\s*=\s*\{([^}]+)\}", normal, re.IGNORECASE)
    if match is None:
        raise RuntimeError(normal)
    if compare(
        "default half-apertures (m)", [float(v) for v in match[1].split(",")], [0.01, 0.02], atol=0
    ):
        raise RuntimeError("Default apertures incorrect")
    missing = re.search(r"\baperture\s*=", option, re.IGNORECASE) is None
    print(f"noaperture: aperture attributes present={not missing}; documented expected=True")
    verdict(missing)

    ```

## Provenance

| Role | Model | Effort | Date | Code examined |
|---|---|---|---|---|
| Found<br><small>source check</small> | `gpt-6-astra` | xhigh | 2026-10-20 | 20261020-0 (`19b8afb46a40`) |
| Confirmed<br><small>source check</small> | `gpt-6-astra` | xhigh | 2026-10-21 | 20261020-0 (`19b8afb46a40`) |
| Confirmed<br><small>CLI/API noaperture documentation and output</small> | `gpt-6.1-sol` | xhigh | 2026-11-05 | 20261031-0 (`c4c629fc9071`), source, production, GNU Fortran 15.3.0, CPU, double, serial, OpenMP=1 |

