---
title: "LAT-05: Photon initialization leaves even particles dark"
hide:
  - toc
---

# LAT-05 — Photon initialization leaves even particles dark

| | |
|---|---|
| Code | [Bmad](../index.md) · Lattices, particles and I/O |
| Priority | P3 |
| Failure | Wrong physics |
| Evidence | Reproduced, independently rechecked |
| Last reproduced | 2026-11-05 on Bmad release [20261031-0](https://github.com/bmad-sim/bmad-ecosystem/tree/20261031-0) (`c4c629f`) |
| Source | [`init_bunch_distribution`, bmad/multiparticle/beam_utils.f90:521–534](https://github.com/bmad-sim/bmad-ecosystem/blob/c4c629fc9071d3a56cb1ba5bc714baba65e9c5d3/bmad/multiparticle/beam_utils.f90#L521-L534) |

Generating a photon bunch assigns a horizontal field only to odd-numbered particles and leaves every even-numbered particle dark. Four photons receive squared field norms 0.5, 0, 0.5, 0, with no vertical field. The initializer’s stated contract gives half the photons horizontal field 1 and half vertical field 1, so each squared field norm should be 1. Both the polarization assignment and its field amplitude differ from that contract.

Run `python LAT-05.py` with [_bmad.py](../helpers/_bmad.py) and [_s3_11_particle_io.py](../helpers/_s3_11_particle_io.py) in the same directory.

??? example "Reproducer: LAT-05.py"

    [Download LAT-05.py](LAT-05.py)

    ```python
    """LAT-05: Photon initialization leaves even particles dark

    Generating a photon bunch assigns a horizontal field only to odd-numbered particles and leaves every even-numbered particle dark. Four photons receive squared field norms 0.5, 0, 0.5, 0, with no vertical field. The initializer’s stated contract gives half the photons horizontal field 1 and half vertical field 1, so each squared field norm should be 1. Both the polarization assignment and its field amplitude differ from that contract.

    The script creates four photons with a fixed random seed and zero input beam widths. It checks the photon species and compares the unordered field vectors and their squared norms with the initializer’s half-horizontal, half-vertical unit-field contract. The application defines the conversion from field norm to physical intensity.
    """

    import numpy as np
    from _bmad import compare, verdict
    from _s3_11_particle_io import execute

    result = execute("""
    bi%species = 'photon'
    bi%n_particle = 4
    bi%bunch_charge = 1
    call init_beam_distribution(ele, param, bi, beam, err)
    if (err) error stop 'photon initialization failed'
    if (size(beam%bunch(1)%particle) /= 4) error stop 'wrong photon count'
    do i = 1, 4
      if (beam%bunch(1)%particle(i)%species /= photon$) error stop 'wrong species'
      write(label, '(a,i0)') 'p', i
      call emit(trim(label), beam%bunch(1)%particle(i)%field)
    enddo
    """)
    fields = np.array([result[f"p{i}"] for i in range(1, 5)])
    # The source contract specifies two unit horizontal and two unit vertical fields,
    # without prescribing particle order. Compare the multiset of field vectors.
    expected = np.array([[0, 1], [0, 1], [1, 0], [1, 0]])
    ordered = fields[np.lexsort((fields[:, 1], fields[:, 0]))]
    defect = compare(
        "photon fields sorted by components; half unit horizontal, half unit vertical",
        ordered,
        expected,
        atol=2e-15,
    )
    # The unit-field contract gives |Ex|^2 + |Ey|^2 = 1. The application sets the
    # conversion of this norm to physical intensity (Bmad manual, Photon Phase Space Coordinates).
    defect |= compare(
        "photon squared field norm; unit-field contract",
        np.sum(fields**2, axis=1),
        np.ones(4),
        atol=2e-15,
    )
    verdict(defect)

    ```

## Provenance

| Role | Model | Effort | Date | Code examined |
|---|---|---|---|---|
| Found | `gpt-6.1-sol` | xhigh | 2026-10-21 | 20261020-0 (`19b8afb46a40`), source, production, CPU, double, serial |
| Confirmed | `gpt-6-astra` | xhigh | 2026-10-21 | 20261020-0 (`19b8afb46a40`), source, production, CPU, double, serial |

