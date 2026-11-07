---
title: "LAT-18: The documented Gaussian default differs from both source defaults"
hide:
  - toc
---

# LAT-18 — The documented Gaussian default differs from both source defaults

| | |
|---|---|
| Code | [Bmad](../index.md) · Lattices, particles and I/O |
| Priority | P4 |
| Severity | Documentation or example mismatch |
| Evidence | Reproduced, independently rechecked |
| Last reproduced | 2026-11-05 on Bmad release [20261031-0](https://github.com/bmad-sim/bmad-ecosystem/tree/20261031-0) (`c4c629f`) |
| Source | [Bmad manual, Beam_Init_Struct Structure (`s:beam.init.struct`), bmad/doc/beam-init.tex:21–33](https://github.com/bmad-sim/bmad-ecosystem/blob/c4c629fc9071d3a56cb1ba5bc714baba65e9c5d3/bmad/doc/beam-init.tex#L21-L33)<br>[random_state_struct, sim_utils/math/random_mod.f90:22–41](https://github.com/bmad-sim/bmad-ecosystem/blob/c4c629fc9071d3a56cb1ba5bc714baba65e9c5d3/sim_utils/math/random_mod.f90#L22-L41)<br>[beam_init_struct, bmad/modules/bmad_struct.f90:1184–1190](https://github.com/bmad-sim/bmad-ecosystem/blob/c4c629fc9071d3a56cb1ba5bc714baba65e9c5d3/bmad/modules/bmad_struct.f90#L1184-L1190)<br>[init_bunch_distribution converter selection, bmad/multiparticle/beam_utils.f90:408–413](https://github.com/bmad-sim/bmad-ecosystem/blob/c4c629fc9071d3a56cb1ba5bc714baba65e9c5d3/bmad/multiparticle/beam_utils.f90#L408-L413) |

Omitting the Gaussian-converter setting selects Ziggurat in both the beam-initialization structure and the random-state structure. The manual instead lists exact as the default and gives only exact and quick as choices. Runtime initialization of both structures exposes the disagreement directly. The documented default and supported converter list should include Ziggurat.

Run `python LAT-18.py` with [_bmad.py](../helpers/_bmad.py) and [_s3_12_particle_contracts.py](../helpers/_s3_12_particle_contracts.py) in the same directory.

??? example "Reproducer: LAT-18.py"

    [Download LAT-18.py](LAT-18.py)

    ```python
    """LAT-18: The documented Gaussian default differs from both source defaults

    Omitting the Gaussian-converter setting selects Ziggurat in both the beam-initialization structure and the random-state structure. The manual instead lists exact as the default and gives only exact and quick as choices. Runtime initialization of both structures exposes the disagreement directly. The documented default and supported converter list should include Ziggurat.

    The script compiles a small program that reports the untouched beam and random-state defaults from the built modules. It also generates a finite Gaussian sample using the default converter. BMAD_SOURCE supplies the documented default.
    """

    import numpy as np
    from _bmad import fortran, verdict
    from _s3_12_particle_contracts import document_claim

    program = """
    program defaults
    use bmad
    use random_mod
    implicit none
    type(beam_init_struct) bi
    type(random_state_struct) rs
    real(rp) sample(4)
    print *, 'BEAM_DEFAULT ', trim(bi%random_gauss_converter)
    print *, 'STATE_DEFAULT ', trim(ran_gauss_converter_name(rs%gauss_converter))
    call ran_seed_put(314159)
    call ran_gauss(sample)
    write(*,'(a,4es26.17e3)') 'SAMPLE ', sample
    end program
    """
    log = fortran(program).log
    print(log.strip())
    assert "BEAM_DEFAULT ziggurat" in log and "STATE_DEFAULT ziggurat" in log
    sample = next(line.split()[1:] for line in log.splitlines() if line.startswith("SAMPLE "))
    sample = np.array(sample, dtype=float)
    assert sample.shape == (4,) and np.isfinite(sample).all()
    verdict(
        document_claim(
            "bmad/doc/beam-init.tex", 'character random_gauss_converter ! "exact" (default) or "quick".'
        )
    )

    ```

## Provenance

| Role | Model | Effort | Date | Code examined |
|---|---|---|---|---|
| Found<br><small>source check</small> | `gpt-6-astra` | xhigh | 2026-10-20 | 20261020-0 (`19b8afb46a40`) |
| Confirmed<br><small>source check</small> | `gpt-6-astra` | xhigh | 2026-10-21 | 20261020-0 (`19b8afb46a40`) |
| Confirmed | `gpt-6.1-sol` | xhigh | 2026-11-05 | 20261031-0 (`c4c629fc9071`), source, production, GNU Fortran 15.3.0, CPU, double, serial, OpenMP=1 |

