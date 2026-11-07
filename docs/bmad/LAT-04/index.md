---
title: "LAT-04: HDF5 momentum-centroid shifts leave beta inconsistent"
hide:
  - toc
---

# LAT-04 — HDF5 momentum-centroid shifts leave beta inconsistent

| | |
|---|---|
| Code | [Bmad](../index.md) · Lattices, particles and I/O |
| Priority | P2 |
| Severity | Wrong physics |
| Evidence | Reproduced, independently rechecked |
| Last reproduced | 2026-11-05 on Bmad release [20261031-0](https://github.com/bmad-sim/bmad-ecosystem/tree/20261031-0) (`c4c629f`) |
| Tags | <span class="bb-tag">partly-source-only</span> [?](../../methodology.md#tags) |
| Source | [`bunch_init_end_calc`, bmad/multiparticle/beam_utils.f90:1923–1968](https://github.com/bmad-sim/bmad-ecosystem/blob/c4c629fc9071d3a56cb1ba5bc714baba65e9c5d3/bmad/multiparticle/beam_utils.f90#L1923-L1968)<br>[beta-dependent bunch spacing, bmad/multiparticle/beam_utils.f90:1991–1998](https://github.com/bmad-sim/bmad-ecosystem/blob/c4c629fc9071d3a56cb1ba5bc714baba65e9c5d3/bmad/multiparticle/beam_utils.f90#L1991-L1998)<br>[centroid-offset contract, bmad/doc/beam-init.tex:318–327](https://github.com/bmad-sim/bmad-ecosystem/blob/c4c629fc9071d3a56cb1ba5bc714baba65e9c5d3/bmad/doc/beam-init.tex#L318-L327) |

Applying a momentum-centroid offset while initializing an HDF5 beam changes particle momentum without updating β, the speed divided by the speed of light. An electron starting at 0.5 MeV/c and shifted to 1 MeV/c retains β = 0.6994 instead of the mass-shell value 0.8905. The final momentum determines β and the subsequent bunch-spacing conversion. The zero-offset control has the correct β.

Run `python LAT-04.py` with [_bmad.py](../helpers/_bmad.py) and [_s3_11_particle_io.py](../helpers/_s3_11_particle_io.py) in the same directory.

??? example "Reproducer: LAT-04.py"

    [Download LAT-04.py](LAT-04.py)

    ```python
    """LAT-04: HDF5 momentum-centroid shifts leave beta inconsistent

    Applying a momentum-centroid offset while initializing an HDF5 beam changes particle momentum without updating β, the speed divided by the speed of light. An electron starting at 0.5 MeV/c and shifted to 1 MeV/c retains β = 0.6994 instead of the mass-shell value 0.8905. The final momentum determines β and the subsequent bunch-spacing conversion. The zero-offset control has the correct β.

    The script writes one 0.5 MeV/c electron to HDF5 and initializes it with zero or unit momentum offset at the same reference momentum. It compares the returned speed with β = P/√(P² + m²) using the final momentum, with momentum and rest energy in eV units.
    """

    import numpy as np
    from _bmad import compare, verdict
    from _s3_11_particle_io import execute


    def case(offset):
        return execute(f"""
    ele%value(p0c$) = 5e5_rp
    ele%value(e_tot$) = hypot(ele%value(p0c$), mass_of(electron$))
    call reallocate_beam(beam, 1, 1)
    beam%bunch(1)%particle%state = alive$
    beam%bunch(1)%particle%species = electron$
    beam%bunch(1)%particle%p0c = ele%value(p0c$)
    beam%bunch(1)%particle%beta = ele%value(p0c$)/ele%value(e_tot$)
    beam%bunch(1)%particle%charge = 1
    call hdf5_write_beam('beam.h5', beam%bunch, .false., err)
    if (err) error stop 'HDF5 write failed'
    bi%position_file = 'beam.h5'
    bi%center(6) = {offset}.0_rp
    call init_beam_distribution(ele, param, bi, back, err, conserve_momentum=.true.)
    if (err) error stop 'initialization failed'
    p => back%bunch(1)%particle(1)
    call emit('particle', [p%p0c, p%vec(6), p%beta, mass_of(electron$)])
    """)["particle"]


    defect = False
    for offset in (0, 1):
        p0c, delta, beta, mass = case(offset)
        if compare(f"offset={offset}: reference (eV/c), delta", [p0c, delta], [5e5, offset], atol=0):
            raise RuntimeError("Momentum control failed")
        pc = (1 + delta) * p0c
        # Mass shell: E^2 = (Pc)^2 + (mc^2)^2, beta = Pc/E.
        mismatch = compare(
            f"offset={offset}: beta from final mass shell", beta, pc / np.hypot(pc, mass), atol=2e-15
        )
        if offset == 0 and mismatch:
            raise RuntimeError("Zero-offset control failed")
        defect |= mismatch
    verdict(defect)

    ```

## Provenance

| Role | Model | Effort | Date | Code examined |
|---|---|---|---|---|
| Found | `gpt-6.1-sol` | xhigh | 2026-10-21 | 20261020-0 (`19b8afb46a40`), source, production, CPU, double, serial |
| Confirmed | `gpt-6-astra` | xhigh | 2026-10-21 | 20261020-0 (`19b8afb46a40`), source, production, CPU, double, serial |

