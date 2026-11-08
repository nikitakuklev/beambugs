---
title: "LAT-11: SynchPartRedefinitionZdE.center_Z doubles the bunch's mean z instead of removing it"
hide:
  - toc
---

# LAT-11 — `SynchPartRedefinitionZdE.center_Z` doubles the bunch's mean z instead of removing it

| | |
|---|---|
| Code | [PyORBIT](../index.md) · Lattice API |
| Priority | P4 |
| Failure | Silent wrong result |
| Evidence | Reproduced, independently rechecked |
| Last reproduced | 2026-11-07 on PyORBIT commit [`28df660`](https://github.com/PyORBIT-Collaboration/PyORBIT3/commit/28df66018aeb), committed 2026-10-14 |
| Source | [SynchPartRedefinitionZdE.cc#L129-L155](https://github.com/PyORBIT-Collaboration/PyORBIT3/blob/28df66018aebe4e2b780b9747f4a2d7b8430a5c0/src/orbit/SynchPartRedefinition/SynchPartRedefinitionZdE.cc#L129-L155)<br>[SynchPartRedefinitionZdE.cc#L90-L127](https://github.com/PyORBIT-Collaboration/PyORBIT3/blob/28df66018aebe4e2b780b9747f4a2d7b8430a5c0/src/orbit/SynchPartRedefinition/SynchPartRedefinitionZdE.cc#L90-L127)<br>[wrap_synch_part_redefinition_z_de.cc#L149-L152](https://github.com/PyORBIT-Collaboration/PyORBIT3/blob/28df66018aebe4e2b780b9747f4a2d7b8430a5c0/src/orbit/SynchPartRedefinition/wrap_synch_part_redefinition_z_de.cc#L149-L152) |

`SynchPartRedefinitionZdE.center_Z(bunch)` is documented to transform the synchronous particle's z to the bunch average, after which the bunch's mean longitudinal position ⟨z⟩ is zero. It passes +⟨z⟩ to `shift_Z`, which adds its argument to every particle's z, so ⟨z⟩ doubles instead. Three particles at z = 1, 2 and 3 mm end at 3, 4 and 5 mm instead of −1, 0 and 1 mm. The energy counterpart `center_dE` centers dE correctly, and `shift_Z(bunch, −⟨z⟩)` gives the centered bunch.

Run `python LAT-11.py`.

??? example "Reproducer: LAT-11.py"

    [Download LAT-11.py](LAT-11.py)

    ```python
    """LAT-11: `SynchPartRedefinitionZdE.center_Z` doubles the bunch's mean z instead of removing it

    `SynchPartRedefinitionZdE.center_Z(bunch)` is documented to transform the synchronous particle's z to the bunch average, after which the bunch's mean longitudinal position ⟨z⟩ is zero. It passes +⟨z⟩ to `shift_Z`, which adds its argument to every particle's z, so ⟨z⟩ doubles instead. Three particles at z = 1, 2 and 3 mm end at 3, 4 and 5 mm instead of −1, 0 and 1 mm. The energy counterpart `center_dE` centers dE correctly, and `shift_Z(bunch, −⟨z⟩)` gives the centered bunch.

    The script builds a 2.5 MeV proton bunch of three particles with z = 1, 2, 3 mm and calls `center_Z`, then compares each z and the mean with the centered values z − ⟨z⟩. The controls are `center_dE` on the same bunch, whose mean dE must become 0, and `shift_Z(bunch, −⟨z⟩)` with ⟨z⟩ from `analyzeBunch`.
    """

    import os

    for _v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
        os.environ.setdefault(_v, "1")

    from orbit.core.bunch import Bunch, SynchPartRedefinitionZdE

    Z0 = (1e-3, 2e-3, 3e-3)  # m; mean 2 mm
    DE0 = (2e-5, 3e-5, 4e-5)  # GeV; mean 30 keV
    MEAN_Z = sum(Z0) / 3


    def bunch():
        b = Bunch()
        b.mass(0.938272013)
        b.charge(1.0)
        b.getSyncParticle().kinEnergy(2.5e-3)
        for z, de in zip(Z0, DE0):
            b.addParticle(0.0, 0.0, 0.0, 0.0, z, de)
        return b


    def mm(b):
        return ", ".join(f"{b.z(i) * 1e3:.6f}" for i in range(b.getSize()))


    def mean_z(b):
        return sum(b.z(i) for i in range(b.getSize())) / b.getSize()


    redef = SynchPartRedefinitionZdE()
    expected = ", ".join(f"{(z - MEAN_Z) * 1e3:.6f}" for z in Z0)  # centered: z_i - <z>

    b = bunch()
    redef.center_Z(b)
    print(
        f"[center_Z] z after: {mm(b)} mm, mean {mean_z(b) * 1e3:.6f} mm; expected {expected} mm, mean 0 (z - <z>)"
    )
    defect = abs(mean_z(b)) > 1e-9

    b = bunch()
    redef.analyzeBunch(b)
    redef.shift_Z(b, -redef.getAvg_Z())
    print(
        f"[control, shift_Z(-<z>)] z after: {mm(b)} mm, mean {mean_z(b) * 1e3:.3e} mm; expected {expected} mm, mean 0"
    )
    control = abs(mean_z(b)) < 1e-15

    b = bunch()
    redef.center_dE(b)
    mean_de = sum(b.dE(i) for i in range(3)) / 3
    print(f"[control, center_dE] mean dE after {mean_de:.3e} GeV, expected 0")
    control = control and abs(mean_de) < 1e-15

    print("DEFECT REPRODUCED" if defect and control else "NOT REPRODUCED")

    ```

## Provenance

| Role | Model | Effort | Date | Code examined |
|---|---|---|---|---|
| Found | `claude-opus-5-5` | high | 2026-11-07 | main (`28df66018aeb`), source, serial, CPU, double |
| Confirmed | `claude-sonnet-5-5` | high | 2026-11-07 | main (`28df66018aeb`), source, serial, CPU, double |

