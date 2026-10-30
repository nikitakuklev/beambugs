---
title: "API-06: PALS import discards quadrupole tilt and skew"
hide:
  - toc
---

# API-06 — PALS import discards quadrupole tilt and skew

| | |
|---|---|
| Code | [ImpactX](../index.md) · Python API |
| Priority | P2 |
| Severity | Wrong physics |
| Evidence | Reproduced, independently rechecked |
| Last reproduced | 2026-10-29 on development (`fa0a43b9bb07`), source, CPU, double, serial |
| Source | [`pals_to_impactx.py#L112-L130`](https://github.com/BLAST-ImpactX/impactx/blob/b701a6e9d1f9e30cefb9a3e18e82ae7e02b9d943/src/python/impactx/pals_to_impactx.py#L112-L130) |

Importing a PALS lattice ignores a quadrupole's tilt `tilt1` and skew coefficient `Ks1`: the converter builds a normal quadrupole from `Kn1` (or `Bn1`) alone, with no warning about the dropped properties. PALS defines the normalized coefficient as `(Kn1 + i·Ks1)·exp(−2i·tilt1)`, and native rotated `ChrQuad` elements agree with an independent matrix calculation. For a γ = 5 proton at `x = 1 mm` through `L = 0.2 m`, `k = 2 m⁻²`, a quadrupole tilted by 90° gives `px = −3.947e-4` instead of +4.054e-4, and a skew quadrupole given as `Kn1=0, Ks1=2` gives `py = 0` instead of 4.000e-4. A skew quadrupole without `Kn1` is rejected with an error.

## Reproducer

Requires pals_schema, scipy. Set `IMPACTX_SOURCE` to a clone of the ImpactX repository.

??? example "API-06.py"

    [Download API-06.py](API-06.py)

    ```python
    """API-06: PALS quadrupole tilt and skew discarded."""

    import os

    for key in (
        "OMP_NUM_THREADS",
        "OPENBLAS_NUM_THREADS",
        "MKL_NUM_THREADS",
        "NUMEXPR_NUM_THREADS",
        "NPROC",
        "PJRT_NPROC",
    ):
        os.environ[key] = "1"
    os.sched_setaffinity(0, {min(os.sched_getaffinity(0))})

    import inspect
    import subprocess
    import warnings
    from importlib.metadata import version

    import impactx.pals_to_impactx as reader
    import numpy as np
    import pals
    from impactx import ImpactX, elements
    from scipy.linalg import expm

    REPO = os.environ["IMPACTX_SOURCE"]
    path = "src/python/impactx/pals_to_impactx.py"
    pinned = subprocess.check_output(["git", "-C", REPO, "show", f"615921c6867c:{path}"], text=True)
    assert inspect.getsource(reader) == pinned
    assert pinned == subprocess.check_output(
        ["git", "-C", REPO, "show", f"b701a6e9d1f9e30cefb9a3e18e82ae7e02b9d943:{path}"], text=True
    )
    print("impactx-noacc", version("impactx-noacc"), "pals-schema", version("pals-schema"))
    print("installed PALS converter equals pinned and development sources")


    # PALS: (By+i Bx)/(P0/q) = (Kn+i Ks) exp(-2i tilt) (x+i y).
    # Lorentz force for forward motion: px'=-q By/P0; py'=q Bx/P0.
    # At pt=0 the paraxial transverse system is exactly linear.
    def transfer(length, kn, ks=0.0, tilt=0.0):
        coeff = complex(kn, ks) * np.exp(-2j * tilt)
        a, b = coeff.real, coeff.imag
        generator = np.array([[0, 1, 0, 0], [-a, 0, b, 0], [0, 0, 0, 1], [b, 0, a, 0]])
        return expm(length * generator)


    sim = ImpactX()
    sim.particle_shape = 1
    sim.space_charge = False
    sim.diagnostics = False
    sim.slice_step_diagnostics = False
    sim.verbose = 0
    sim.init_grids()
    columns = ["position_x", "momentum_x", "position_y", "momentum_y"]


    def track(lattice, transverse):
        u = np.asarray(transverse, dtype=float)
        sim.beam.clear_particles()
        ref = sim.beam.ref
        ref.reset()
        ref.set_species("proton").set_kin_energy_MeV(4 * ref.mass_MeV)
        sim.beam.add_n_particles(
            x=u[:, 0].copy(),
            px=u[:, 1].copy(),
            y=u[:, 2].copy(),
            py=u[:, 3].copy(),
            t=np.zeros(len(u)),
            pt=np.zeros(len(u)),
            qm=ref.qm_ratio_SI,
            w=np.ones(len(u)),
        )
        sim.lattice.clear()
        sim.lattice.extend(lattice)
        sim.track_particles()
        data = sim.beam.to_df(local=True)
        if data is None:
            raise AssertionError("Test unexpectedly lost every particle")
        return data[columns].to_numpy(copy=True)


    def convert(item):
        sim.lattice.clear()
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            sim.lattice.from_pals(pals.BeamLine(name="independent", line=[item]))
        assert not caught, [str(w.message) for w in caught]
        return list(sim.lattice)


    def quad(length, kn, **kwargs):
        return pals.Quadrupole(
            name="q",
            length=length,
            MagneticMultipoleP=pals.MagneticMultipoleParameters(Kn1=kn, **kwargs),
        )


    try:
        # Check two independent field-strength/length pairs.
        for length, k in [(0.2, 2.0), (0.35, 3.2)]:
            u = 1e-4 * np.vstack([np.eye(4), -np.eye(4)])
            plain = track(convert(quad(length, k)), u)
            normal_expected = u @ transfer(length, k).T
            np.testing.assert_allclose(plain, normal_expected, rtol=2e-14, atol=2e-19)
            for label, ks, tilt in [
                ("tilt90", 0.0, np.pi / 2),
                ("skew", k, 0.0),
                ("combined", -0.8, 0.31),
            ]:
                kn = 0.0 if label == "skew" else k
                actual = track(convert(quad(length, kn, Ks1=ks, tilt1=tilt)), u)
                expected = u @ transfer(length, kn, ks, tilt).T
                # An independently constructed native element implements the same complex field.
                coeff = complex(kn, ks) * np.exp(-2j * tilt)
                rotation = -0.5 * np.angle(coeff)
                native = track(
                    [elements.ChrQuad(ds=length, k=abs(coeff), rotation=np.degrees(rotation))], u
                )
                np.testing.assert_allclose(native, expected, rtol=2e-14, atol=2e-19)
                np.testing.assert_allclose(actual, u @ transfer(length, kn).T, rtol=2e-14, atol=2e-19)
                assert np.max(np.abs(actual - expected)) > 1e-5
                print(
                    f"L={length:g} k={k:g} {label}: map discrepancy={np.max(np.abs(actual - expected)):.12g}; native oracle error={np.max(np.abs(native - expected)):.3g}"
                )
            probe = np.array([[1e-3, 0, 0, 0]])
            tilted = track(convert(quad(length, k, tilt1=np.pi / 2)), probe)[0, 1]
            skew = track(convert(quad(length, 0.0, Ks1=k)), probe)[0, 3]
            print(
                f"L={length:g}: tilt px={tilted:.15g} expected={(probe @ transfer(length, k, tilt=np.pi / 2).T)[0, 1]:.15g}; skew py={skew:.15g} expected={(probe @ transfer(length, 0, k).T)[0, 3]:.15g}"
            )

        length, k = 0.2, 2.0
        try:
            convert(
                pals.Quadrupole(
                    name="integrated",
                    length=length,
                    MagneticMultipoleP=pals.MagneticMultipoleParameters(Kn1L=length * k),
                )
            )
        except RuntimeError as exc:
            print("Unsupported integrated-only control explicitly rejects:", exc)
        else:
            raise AssertionError("Integrated-only control should reject")
    finally:
        sim.finalize()
    print("DEFECT REPRODUCED")

    ```

## Provenance

| Role | Model | Effort | Date | Code examined |
|---|---|---|---|---|
| Found | `gpt-6-astra` | xhigh | 2026-10-28 | 26.10 (`615921c6867c`), impactx-noacc wheel, CPU, double, serial |
| Confirmed | `gpt-6.1-sol` | xhigh | 2026-10-28 | 26.10 (`615921c6867c`), impactx-noacc wheel, CPU, double, serial |

