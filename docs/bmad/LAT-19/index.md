---
title: "LAT-19: The manual contradicts the additive initialization time offset"
hide:
  - toc
---

# LAT-19 — The manual contradicts the additive initialization time offset

| | |
|---|---|
| Code | [Bmad](../index.md) · Lattices, particles and I/O |
| Priority | P4 |
| Severity | Documentation or example mismatch |
| Evidence | Reproduced, independently rechecked |
| Last reproduced | 2026-11-05 on Bmad 20261031-0 (`c4c629f`, head build) |
| Tags | <span class="bb-tag">partly-source-only</span> [?](../../methodology.md#tags) |
| Source | [Bmad manual, Beam_Init_Struct Structure (`s:beam.init.struct`), bmad/doc/beam-init.tex:264–283](https://github.com/bmad-sim/bmad-ecosystem/blob/c4c629fc9071d3a56cb1ba5bc714baba65e9c5d3/bmad/doc/beam-init.tex#L264-L283)<br>[bunch_init_end_calc time offset and center, bmad/multiparticle/beam_utils.f90:1891–1914](https://github.com/bmad-sim/bmad-ecosystem/blob/c4c629fc9071d3a56cb1ba5bc714baba65e9c5d3/bmad/multiparticle/beam_utils.f90#L1891-L1914)<br>[coordinate-dependent finalization, bmad/multiparticle/beam_utils.f90:1938–1972](https://github.com/bmad-sim/bmad-ecosystem/blob/c4c629fc9071d3a56cb1ba5bc714baba65e9c5d3/bmad/multiparticle/beam_utils.f90#L1938-L1972) |

Initializing an ordinary moving-particle bunch with use_particle_start enabled adds t_offset to the time specified by particle_start. The manual first says the offset always applies, then says it is ignored in this mode. An electron at 1 MeV/c with starting time 2 ns and offset 3 ns is initialized at 5 ns, with longitudinal coordinate −0.5339 m determined by the initial 2 ns. The contradictory paragraph should describe the additive time convention.

Run `python LAT-19.py` with [_bmad.py](../helpers/_bmad.py) and [_s3_12_particle_contracts.py](../helpers/_s3_12_particle_contracts.py) in the same directory.

??? example "Reproducer: LAT-19.py"

    [Download LAT-19.py](LAT-19.py)

    ```python
    """LAT-19: The manual contradicts the additive initialization time offset

    Initializing an ordinary moving-particle bunch with use_particle_start enabled adds t_offset to the time specified by particle_start. The manual first says the offset always applies, then says it is ignored in this mode. An electron at 1 MeV/c with starting time 2 ns and offset 3 ns is initialized at 5 ns, with longitudinal coordinate −0.5339 m determined by the initial 2 ns. The contradictory paragraph should describe the additive time convention.

    The script initializes one particle from an inline lattice and compares its time, speed and longitudinal coordinate with the finite-beta expectation. BMAD_SOURCE supplies the contradictory manual statements.
    """

    import numpy as np
    from _bmad import compare, fortran, verdict
    from _s3_12_particle_contracts import TIME, document_claim, rows

    lat = "parameter[particle]=electron\nparameter[p0c]=1e6\nparticle_start[t]=2e-9\nd: drift,l=1\nl: line=(d)\nuse,l\n"
    row = rows(fortran(TIME, files={"lat.bmad": lat}).log)["time"]
    beta = 1e6 / np.hypot(1e6, row[3])
    if compare(
        "time (s), z (m), beta; additive offset and z=-beta*c*2ns",
        row[:3],
        [5e-9, -beta * 299792458 * 2e-9, beta],
        atol=0,
        rtol=3e-15,
    ):
        raise RuntimeError("Additive-time control failed")
    verdict(
        document_claim(
            "bmad/doc/beam-init.tex",
            r"the setting of \vn{beam_init%t_offset} will be ignored in favor of \vn{particle_start[t]}",
        )
    )

    ```

## Provenance

| Role | Model | Effort | Date | Code examined |
|---|---|---|---|---|
| Found | `gpt-6-astra` | xhigh | 2026-10-20 | 20261020-0 (`19b8afb46a40`), source, production, CPU, double, serial |
| Confirmed | `gpt-6-astra` | xhigh | 2026-10-21 | 20261020-0 (`19b8afb46a40`), source, production, CPU, double, serial |
| Confirmed | `gpt-6.1-sol` | xhigh | 2026-11-05 | 20261031-0 (`c4c629fc9071`), source, production, GNU Fortran 15.3.0, CPU, double, serial, OpenMP=1 |

