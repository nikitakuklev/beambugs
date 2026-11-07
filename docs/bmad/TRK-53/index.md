---
title: "TRK-53: PTC drops the thin reference-bend terms of a multipole"
hide:
  - toc
---

# TRK-53 — PTC drops the thin reference-bend terms of a multipole

| | |
|---|---|
| Code | [Bmad](../index.md) · Tracking, RF, fields and spin |
| Priority | P2 |
| Severity | Wrong physics |
| Evidence | Reproduced, independently rechecked |
| Last reproduced | 2026-11-05 on Bmad 20261031-0 (`c4c629f`, head build) |
| Source | [ele_to_an_bn multipole case](https://github.com/bmad-sim/bmad-ecosystem/blob/c4c629fc9071d3a56cb1ba5bc714baba65e9c5d3/bmad/ptc/ptc_interface_mod.f90#L3255-L3269)<br>[Bmad Manual, “Multipole” (s:mult)](https://github.com/bmad-sim/bmad-ecosystem/blob/c4c629fc9071d3a56cb1ba5bc714baba65e9c5d3/bmad/doc/elements.tex#L3860-L3878)<br>[native thin-bend map](https://github.com/bmad-sim/bmad-ecosystem/blob/c4c629fc9071d3a56cb1ba5bc714baba65e9c5d3/bmad/modules/multipole_mod.f90#L248-L254) |

A multipole with K0L_status set to bends_reference loses its off-momentum kick and longitudinal displacement in PTC tracking and PTC-derived Taylor maps. The paraxial thin reference bend requires Δpₓ = K₀L·δ and Δz = −K₀L·x, where K₀L is the integrated dipole strength, pₓ is normalized horizontal momentum and δ is relative momentum offset. At K₀L = 0.001, x = 1 mm and δ = 0.01, native tracking returns Δpₓ = 10⁻⁵ and Δz = −1 μm, while PTC and Taylor tracking return zero for both. The PTC interface removes the dipole coefficients without supplying the reference-bend terms.

Run `python TRK-53.py` with [_bmad.py](../helpers/_bmad.py) and [_tracking_contracts_materials.py](../helpers/_tracking_contracts_materials.py) in the same directory.

??? example "Reproducer: TRK-53.py"

    [Download TRK-53.py](TRK-53.py)

    ```python
    """TRK-53: PTC drops the thin reference-bend terms of a multipole

    A multipole with K0L_status set to bends_reference loses its off-momentum kick and longitudinal displacement in PTC tracking and PTC-derived Taylor maps. The paraxial thin reference bend requires Δpₓ = K₀L·δ and Δz = −K₀L·x, where K₀L is the integrated dipole strength, pₓ is normalized horizontal momentum and δ is relative momentum offset. At K₀L = 0.001, x = 1 mm and δ = 0.01, native tracking returns Δpₓ = 10⁻⁵ and Δz = −1 μm, while PTC and Taylor tracking return zero for both. The PTC interface removes the dipole coefficients without supplying the reference-bend terms.

    The script tracks a fixed displaced, off-momentum particle with native, PTC and Taylor methods.
    It compares the kick and longitudinal shift with the analytic thin-bend map, then checks an
    on-axis reference particle as an identity control.
    """

    import numpy as np
    from _bmad import compare, verdict
    from _tracking_contracts_materials import lattice, orbit, tao

    defects = []
    for control in (False, True):
        x, delta = (0.0, 0.0) if control else (0.001, 0.01)
        # Hamiltonian H=-K0L*x*delta gives dpx=K0L*delta and dz=-K0L*x.
        expected = np.array([0.001 * delta, -0.001 * x])
        for method in ("bmad_standard", "symp_lie_ptc", "taylor"):
            lat = lattice(
                f"e: multipole,k0l=.001,k0l_status=bends_reference,tracking_method={method}",
                start=f"particle_start[x]={x}\nparticle_start[pz]={delta}",
            )
            command = "pipe ele:orbit 1"
            actual = orbit(tao(lat, [command])[command])
            if compare(
                f"{method} unchanged x,y,py,delta control",
                actual[[0, 2, 3, 5]],
                [x, 0, 0, delta],
                atol=2e-15,
            ):
                raise AssertionError("Unchanged-coordinate control failed")
            bad = compare(
                f"{method} reference_control={control}: (px,z m)", actual[[1, 4]], expected, atol=2e-15
            )
            if control or method == "bmad_standard":
                if bad:
                    raise AssertionError("Thin-bend control failed")
            else:
                defects.append(bad)
    verdict(any(defects))

    ```

## Provenance

| Role | Model | Effort | Date | Code examined |
|---|---|---|---|---|
| Found | `claude-opus-5-5` | xhigh | 2026-10-24 | 20261021-0 (`d00a8f17e55a`), source, production, CPU, double, serial |
| Confirmed | `gpt-6-astra` | xhigh | 2026-11-05 | 20261031-0 (`c4c629fc9071`), source, production, GNU Fortran 15.3.0, CPU, double, serial, OpenMP=1 |

