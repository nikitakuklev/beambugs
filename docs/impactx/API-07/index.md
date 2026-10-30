---
title: "API-07: PALS import discards body offsets and active apertures"
hide:
  - toc
---

# API-07 — PALS import discards body offsets and active apertures

| | |
|---|---|
| Code | [ImpactX](../index.md) · Python API |
| Priority | P2 |
| Severity | Wrong physics |
| Evidence | Reproduced, independently rechecked |
| Last reproduced | 2026-10-29 on development (`fa0a43b9bb07`), source, CPU, double, serial |
| Source | [`pals_to_impactx.py#L100-L130`](https://github.com/BLAST-ImpactX/impactx/blob/b701a6e9d1f9e30cefb9a3e18e82ae7e02b9d943/src/python/impactx/pals_to_impactx.py#L100-L130) |

Importing a PALS lattice ignores body offsets (`BodyShiftP`) and apertures (`ApertureP`) on drifts and quadrupoles, without a warning. A body shift displaces the element and its field, so transport becomes `M·(u − d) + d` for map `M` and offset `d`, and an active aperture removes particles outside its limits; native `ChrQuad(dx=d)` and `Drift` followed by `Aperture` give these results. For a γ = 5 proton, a quadrupole (`L = 0.2 m`, `Kn1 = 2 m⁻²`) offset by 1 mm gives an on-axis particle `px = 0` instead of 3.947e-4, and a drift with an active ±1 mm rectangular exit aperture passes all four test particles where two should be removed. Misalignment and aperture models read from PALS files are lost.

## Reproducer

Requires pals_schema, scipy. Set `IMPACTX_SOURCE` to a clone of the ImpactX repository.

??? example "API-07.py"

    [Download API-07.py](API-07.py)

    ```python
    """API-07: PALS body offsets and active apertures discarded."""

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
        length, k = 0.2, 2.0
        for displacement in [1e-3, -0.6e-3]:
            item = quad(length, k)
            item.BodyShiftP = pals.BodyShiftParameters(x_offset=displacement)
            input_u = np.array([[0.0, 0.0, 0.0, 0.0], [0.4e-3, -0.2e-3, 0.3e-3, 0.1e-3]])
            shift = np.array([displacement, 0, 0, 0])
            expected = (input_u - shift) @ transfer(length, k).T + shift
            actual = track(convert(item), input_u)
            native = track([elements.ChrQuad(ds=length, k=k, dx=displacement)], input_u)
            np.testing.assert_allclose(native, expected, rtol=2e-14, atol=2e-19)
            np.testing.assert_allclose(actual, input_u @ transfer(length, k).T, rtol=2e-14, atol=2e-19)
            assert abs(actual[0, 1] - expected[0, 1]) > 1e-4
            print(
                f"offset={displacement:g}: on-axis px={actual[0, 1]:.15g} expected={expected[0, 1]:.15g}"
            )

        # Zero transverse momenta keep x and y fixed, independent of aperture placement or slicing.
        aperture_u = np.array(
            [[-2e-3, 0, 0, 0], [0, 0, 0, 0], [0.5e-3, 0, 0.2e-3, 0], [0, 0, 1.2e-3, 0]]
        )
        for active in [False, True]:
            item = pals.Drift(
                name="d",
                length=length,
                ApertureP=pals.ApertureParameters(
                    x_limits=[-1e-3, 1e-3],
                    y_limits=[-1e-3, 1e-3],
                    shape="RECTANGULAR",
                    location="EXIT_END",
                    aperture_active=active,
                ),
            )
            actual = track(convert(item), aperture_u)
            expected = (
                aperture_u[np.all(np.abs(aperture_u[:, [0, 2]]) <= 1e-3, axis=1)]
                if active
                else aperture_u
            )
            lattice = [elements.Drift(ds=length)]
            if active:
                lattice.append(elements.Aperture(aperture_x=1e-3, aperture_y=1e-3, shape="rectangular"))
            native = track(lattice, aperture_u)
            # Loss collection can reorder surviving particles.
            np.testing.assert_allclose(
                native[np.argsort(native[:, 0])],
                expected[np.argsort(expected[:, 0])],
                rtol=0,
                atol=1e-19,
            )
            assert len(actual) == 4
            print(
                f"active={active}: imported survivors={len(actual)}; expected/native survivors={len(expected)}/{len(native)}"
            )

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

