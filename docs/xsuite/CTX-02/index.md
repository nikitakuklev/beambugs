---
title: "CTX-02: Copied nested fields disconnect from parent storage"
hide:
  - toc
---

# CTX-02 — Copied nested fields disconnect from parent storage

| | |
|---|---|
| Code | [Xsuite](../index.md) · Contexts and kernels |
| Priority | P1 |
| Severity | Wrong physics |
| Evidence | Reproduced, independently rechecked |
| Last reproduced | 2026-10-29 on xobjects main (`dfe34ae`) |
| Source | [HybridClass.copy, hybrid_class.py:344–360](https://github.com/xsuite/xobjects/blob/dfe34ae824bd8b202015f4350e655238b5bb9511/xobjects/hybrid_class.py#L344-L360)<br>[nested-object tests](https://github.com/xsuite/xobjects/blob/dfe34ae824bd8b202015f4350e655238b5bb9511/tests/test_hybrid_class.py#L95-L138) |

Copying a `SpaceChargeBiGaussian` and changing its public longitudinal-profile population leaves the tracking kernel using the old intensity, because the copied child fields are disconnected from the parent's storage. At fixed geometry the kick must scale linearly with population, as a freshly constructed control confirms. For a 0.7 m element with longitudinal RMS size 0.13 m and transverse RMS size 1.2 mm, increasing the population from 7.000×10⁹ to 1.750×10¹⁰ leaves a 1.3 GeV/c proton at the bunch center and x = 0.7 mm with normalized horizontal momentum `px` = 3.145×10⁻⁶ instead of 7.862×10⁻⁶. Both default-buffer and same-buffer copies fail. The original object remains unchanged.

Download these shared helpers into the same directory as the reproducer scripts:

- [Download _prebuilt.py](../helpers/_prebuilt.py) · [View source](../helpers/index.md#_prebuilt)
- [Download _common.py](../helpers/_common.py) · [View source](../helpers/index.md#_common)

??? example "Reproducer: CTX-02.py"

    [Download CTX-02.py](CTX-02.py)

    ```python
    """CTX-02: Copied nested fields disconnect from parent storage

    Copying a `SpaceChargeBiGaussian` and changing its public longitudinal-profile population leaves the
    tracking kernel using the old intensity, because the copied child fields are disconnected from the
    parent's storage. At fixed geometry the kick must scale linearly with population, as a freshly
    constructed control confirms. For a 0.7 m element with longitudinal RMS size 0.13 m and transverse
    RMS size 1.2 mm, increasing the population from 7.000×10⁹ to 1.750×10¹⁰ leaves a 1.3 GeV/c proton at
    the bunch center and x = 0.7 mm with normalized horizontal momentum `px` = 3.145×10⁻⁶ instead of
    7.862×10⁻⁶. Both default-buffer and same-buffer copies fail. The original object remains unchanged.

    The script first shows on a minimal nested struct that setting a copied child field (gain 11) leaves
    the copy's raw storage at the old value (3). It then copies a frozen round `SpaceChargeBiGaussian`
    (population 7×10⁹, longitudinal RMS 0.13 m, transverse RMS 1.2 mm, length 0.7 m) into a new and into
    the same buffer, sets the copy's population to 1.75×10¹⁰ and tracks two 1.3 GeV/c protons at x = 0.7
    and 1.9 mm, `zeta` = 0 and 0.08 m. The kicks must be 2.5 times the original, as a freshly built
    element with the higher population gives; the copies give exactly the original kick.
    """

    from typing import ClassVar

    import xfields as xf
    import xobjects as xo
    from _prebuilt import CTX, identity, kernels, np, xt

    identity()
    print("ORACLE population scaling")


    class Cell(xo.HybridClass):
        gain: float
        _xofields: ClassVar[dict] = {"gain": xo.Float64}


    class Pair(xo.HybridClass):
        cell: Cell
        _xofields: ClassVar[dict] = {"cell": Cell}


    p = Pair(cell=Cell(gain=3), _context=CTX)
    q = p.copy()
    q.cell.gain = 11
    print("STRUCT copied Python/raw/original", q.cell.gain, q._xobject.cell.gain, p.cell.gain)
    assert q.cell.gain == 11 and q._xobject.cell.gain == 3 and p.cell.gain == 3


    def make(pop):
        return xf.SpaceChargeBiGaussian(
            longitudinal_profile=xf.LongitudinalProfileQGaussian(
                number_of_particles=pop, sigma_z=0.13, _context=CTX
            ),
            length=0.7,
            sigma_x=0.0012,
            sigma_y=0.0012,
            _context=CTX,
        )


    def kick(element):
        particles = xt.Particles(p0c=1.3e9, x=[0.0007, 0.0019], zeta=[0, 0.08], _context=CTX)
        element.track(particles)
        return np.array(particles.px)


    original = make(7e9)
    base = kick(original)
    control = kick(make(17.5e9))
    np.testing.assert_allclose(control, 2.5 * base, rtol=3e-15, atol=0)
    for mode in ("new_buffer", "same_buffer"):
        copied = original.copy(**({"_buffer": original._buffer} if mode == "same_buffer" else {}))
        copied.longitudinal_profile.number_of_particles = 17.5e9
        got = kick(copied)
        print(
            mode,
            "population Python/raw",
            copied.longitudinal_profile.number_of_particles,
            copied._xobject.longitudinal_profile.number_of_particles,
        )
        print(mode, "kick observed", got, "expected", control, "ratio_to_original", got / base)
        np.testing.assert_array_equal(got, base)
        assert (
            copied.longitudinal_profile.number_of_particles
            != copied._xobject.longitudinal_profile.number_of_particles
        )
        assert original.longitudinal_profile.number_of_particles == 7e9
        copied._xobject.longitudinal_profile.number_of_particles = 17.5e9
        np.testing.assert_allclose(kick(copied), control, rtol=3e-15, atol=0)

    kernels()
    print("DEFECT REPRODUCED")

    ```

## Provenance

| Role | Model | Effort | Date | Code examined |
|---|---|---|---|---|
| Found | `gpt-6-astra` | xhigh | 2026-10-28 | xobjects 0.7.2 (`dfe34ae824bd`), installed packages, prebuilt kernels, CPU, serial |
| Confirmed | `gpt-6.1-sol` | xhigh | 2026-10-28 | xobjects 0.7.2 (`dfe34ae824bd`), installed packages, prebuilt kernels, CPU, serial |

