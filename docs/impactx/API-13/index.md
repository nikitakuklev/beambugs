---
title: "API-13: MAD-X BV=-1 retains forward straight-multipole polarity"
hide:
  - toc
---

# API-13 — MAD-X `BV=-1` retains forward straight-multipole polarity

| | |
|---|---|
| Code | [ImpactX](../index.md) · Python API |
| Priority | P2 |
| Failure | Wrong physics |
| Evidence | Reproduced, independently rechecked |
| Last reproduced | 2026-11-05 on ImpactX commit [`f9c2a82`](https://github.com/BLAST-ImpactX/impactx/commit/f9c2a82e24b7), committed 2026-11-05 |
| Source | [`madx_to_impactx.py#L766-L798`](https://github.com/BLAST-ImpactX/impactx/blob/b701a6e9d1f9e30cefb9a3e18e82ae7e02b9d943/src/python/impactx/madx_to_impactx.py#L766-L798)<br>[sextupole/octupole branches](https://github.com/BLAST-ImpactX/impactx/blob/b701a6e9d1f9e30cefb9a3e18e82ae7e02b9d943/src/python/impactx/madx_to_impactx.py#L1327-L1370)<br>[thin-multipole branch](https://github.com/BLAST-ImpactX/impactx/blob/b701a6e9d1f9e30cefb9a3e18e82ae7e02b9d943/src/python/impactx/madx_to_impactx.py#L1375-L1435)<br>[`read_lattice()`](https://github.com/BLAST-ImpactX/impactx/blob/b701a6e9d1f9e30cefb9a3e18e82ae7e02b9d943/src/python/impactx/madx_to_impactx.py#L1604-L1632)<br>[`bv` contract](https://github.com/BLAST-ImpactX/impactx/blob/b701a6e9d1f9e30cefb9a3e18e82ae7e02b9d943/src/python/impactx/madx_to_impactx.py#L80-L83) |

When ImpactX imports a MAD-X lattice for a reversed beam (`BV=-1`), straight normal and skew multipole strengths keep their forward-beam polarity, because the importer omits the `bv` factor. MAD-X defines `BV=-1` as reversing the magnetic force, native MAD-X applies it, and the importer's own documentation specifies the reversal; analytic maps and an independent integration of the equations of motion give the reversed result. For a 1.4 GeV (total energy) proton in a 0.23 m quadrupole with `k1` = 1.7 m⁻² and tilt 0.21 rad, a test particle exits with normalized momentum pₓ = −7.699×10⁻⁵ instead of +2.189×10⁻⁵. The reversed polarity applies to body quadrupoles, sextupoles and octupoles and to thin quadrupole through dodecapole terms; forward beams and kickers import correctly.

Run `python API-13.py`.

??? example "Reproducer: API-13.py"

    [Download API-13.py](API-13.py)

    ```python
    """API-13: MAD-X `BV=-1` retains forward straight-multipole polarity

    When ImpactX imports a MAD-X lattice for a reversed beam (`BV=-1`), straight normal and skew
    multipole strengths keep their forward-beam polarity, because the importer omits the `bv` factor.
    MAD-X defines `BV=-1` as reversing the magnetic force, native MAD-X applies it, and the importer's
    own documentation specifies the reversal; analytic maps and an independent integration of the
    equations of motion give the reversed result. For a 1.4 GeV (total energy) proton in a 0.23 m
    quadrupole with `k1` = 1.7 m⁻² and tilt 0.21 rad, a test particle exits with normalized momentum pₓ
    = −7.699×10⁻⁵ instead of +2.189×10⁻⁵. The reversed polarity applies to body quadrupoles, sextupoles
    and octupoles and to thin quadrupole through dodecapole terms; forward beams and kickers import
    correctly.

    The script writes its own MAD-X decks and imports them with `lattice.load_file()` for both `BV`
    signs: tilted normal quadrupoles at 1.4 and 4.7 GeV with `k1` = ±1.7 m⁻² in the linear, paraxial and
    exact models, tilted mixed normal/skew thin multipoles of orders 1 to 5, and tilted mixed thick
    quadrupoles, sextupoles and octupoles. Sixteen particles are compared with analytic quadrupole
    matrices, thin kicks and a DOP853 integration of the unsplit Hamilton equations, all with strengths
    multiplied by `BV`; native MAD-X, run through cpymad, must agree with the same oracle. `BV=1`, a
    directly built negative-strength `Quad` and a `BV=-1` horizontal kicker are the controls.
    """

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

    import importlib.metadata
    import json
    import math
    import sys
    import warnings
    from pathlib import Path
    from tempfile import TemporaryDirectory

    import numpy as np
    from impactx import Config, ImpactX, RefPart, elements, read_beam
    from scipy.integrate import solve_ivp

    ROOT = Path(__file__).resolve().parent
    C = 299792458.0
    N = 16


    def emit(**row):
        print(json.dumps(row), flush=True)


    def deck(name, text):
        path = ROOT / (name + ".madx")
        path.write_text(text)
        return str(path)


    def load(sim, path, floor="linear", slices=1):
        sim.lattice.clear()
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            ref = RefPart()
            read_beam(ref, path)
            sim.lattice.load_file(path, nslice=slices, min_model=floor)
        return ref, [str(w.message) for w in caught]


    def track(sim, ref, q):
        """Coordinates are (x, px, y, py, t, pt), with pt = -delta(E)/(P0*c)."""
        b = sim.beam
        b.clear_particles()
        b.set_ref_particle(ref)
        b.add_n_particles(*q[:, [0, 2, 4, 1, 3, 5]].T.copy(), qm=ref.qm_ratio_SI, w=np.ones(N))
        sim.track_particles()
        frame = b.to_df(local=True).reset_index(drop=True).sort_values("idcpu")
        assert len(frame) == N
        return frame[
            ["position_x", "momentum_x", "position_y", "momentum_y", "position_t", "momentum_t"]
        ].to_numpy(), b.ref.copy()


    def phase():
        u = np.linspace(-1, 1, N)
        return np.array(
            [
                0.0007 * (1 + 0.4 * u),
                3e-5 * u,
                -0.0005 * (1 - 0.25 * u),
                -2e-5 * u,
                0.001 * u,
                np.zeros(N),
            ]
        ).T


    def rotate(q, angle):
        r = q.copy()
        c, s = math.cos(angle), math.sin(angle)
        r[:, 0] = c * q[:, 0] + s * q[:, 2]
        r[:, 2] = -s * q[:, 0] + c * q[:, 2]
        r[:, 1] = c * q[:, 1] + s * q[:, 3]
        r[:, 3] = -s * q[:, 1] + c * q[:, 3]
        return r


    def linear_quad(q, strength, length, tilt):
        r = rotate(q, tilt)
        for i, k in ((0, strength), (2, -strength)):
            w = math.sqrt(abs(k))
            c = math.cos(w * length) if k > 0 else math.cosh(w * length)
            s = math.sin(w * length) if k > 0 else math.sinh(w * length)
            x, p = r[:, i].copy(), r[:, i + 1].copy()
            r[:, i] = c * x + s / w * p
            r[:, i + 1] = (-1 if k > 0 else 1) * w * s * x + c * p
        return rotate(r, -tilt)


    def exact_flow(q, order, kn, ks, length, beta, tilt):
        """Integrate the unsplit straight-field Hamilton equations with DOP853.

        B_y+iB_x is proportional to (kn+i*ks)*(x+i*y)**order/order!.
        The mass shell fixes the longitudinal normalized momentum pz.
        """
        initial = rotate(q, tilt)

        def rhs(_s, flat):
            p = flat.reshape(N, 6)
            x, px, y, py, _t, pt = p.T
            pz = np.sqrt(1 - 2 * pt / beta + pt**2 - px**2 - py**2)
            field = (kn + 1j * ks) * (x + 1j * y) ** order / math.factorial(order)
            return np.array(
                [
                    px / pz,
                    -field.real,
                    py / pz,
                    field.imag,
                    (1 / beta - pt) / pz - 1 / beta,
                    np.zeros(N),
                ]
            ).T.ravel()

        sol = solve_ivp(rhs, (0, length), initial.ravel(), method="DOP853", rtol=2e-13, atol=2e-16)
        assert sol.success
        return rotate(sol.y[:, -1].reshape(N, 6), -tilt)


    def check_bv(sim):
        q = phase()
        worst_linear = 0.0
        for energy in (1.4, 4.7):
            for k in (1.7, -1.7):
                for bv in (1, -1):
                    for floor in ("linear", "paraxial", "exact"):
                        text = f"""beam,particle=proton,energy={energy},bv={bv};
    q:quadrupole,l=0.23,k1={k},tilt=0.21;
    cell:line=(q); use,period=cell;
    """
                        path = deck(f"quad_E{energy}_k{k}_bv{bv}_{floor}", text)
                        ref, messages = load(sim, path, floor)
                        for e in sim.lattice:
                            if hasattr(e, "int_order"):
                                e.int_order, e.mapsteps = 6, 20
                        out, _ = track(sim, ref, q)
                        if floor == "exact":
                            oracle = exact_flow(q, 1, bv * k, 0, 0.23, ref.beta, 0.21)
                            forward = exact_flow(q, 1, k, 0, 0.23, ref.beta, 0.21)
                        else:
                            oracle = linear_quad(q, bv * k, 0.23, 0.21)
                            forward = linear_quad(q, k, 0.23, 0.21)
                        error = float(np.max(np.abs(out[:, :4] - oracle[:, :4])))
                        forward_error = float(np.max(np.abs(out[:, :4] - forward[:, :4])))
                        assert forward_error < 5e-14
                        assert error < 5e-14 if bv == 1 else error > 3e-4
                        if floor == "linear" and bv == 1:
                            worst_linear = max(worst_linear, error)
                        emit(
                            case="quad",
                            energy=energy,
                            k=k,
                            bv=bv,
                            floor=floor,
                            px_observed=float(out[0, 1]),
                            px_oracle=float(oracle[0, 1]),
                            max_transverse_error=error,
                            forward_error=forward_error,
                            warnings=messages,
                        )
        emit(case="linear_control_max", max_error=worst_linear)

        for order in (1, 2, 3, 4, 5):
            kn = {1: 0.3, 2: 700.0, 3: 8e5, 4: 3e9, 5: 2e13}[order]
            ks = -0.37 * kn
            for bv in (1, -1):
                for scale in (0.5, 1.0):
                    tilt = -0.17
                    normal = [0.0] * order + [scale * kn]
                    skew = [0.0] * order + [scale * ks]
                    fmt = lambda a: "{" + ",".join(map(str, a)) + "}"
                    text = f"""beam,particle=proton,energy=1.7,bv={bv};
    m:multipole,knl={fmt(normal)},ksl={fmt(skew)},tilt={tilt};
    cell:line=(m);use,period=cell;
    """
                    ref, _ = load(sim, deck(f"thin_n{order}_bv{bv}_s{scale}", text))
                    out, _ = track(sim, ref, q)
                    local = rotate(q, tilt)
                    field = (
                        (kn + 1j * ks)
                        * scale
                        * (local[:, 0] + 1j * local[:, 2]) ** order
                        / math.factorial(order)
                    )
                    local[:, 1] -= bv * field.real
                    local[:, 3] += bv * field.imag
                    oracle = rotate(local, -tilt)
                    err = float(np.max(np.abs(out - oracle)))
                    assert err < 2e-17 if bv == 1 else err > 1e-6
                    emit(
                        case="thin",
                        order=order,
                        bv=bv,
                        scale=scale,
                        max_error=err,
                        px_observed=float(out[0, 1]),
                        px_oracle=float(oracle[0, 1]),
                    )

        for order, kn in ((1, 2.3), (2, 1250.0), (3, 1.8e6)):
            ks = 0.4 * kn
            family = {1: "quadrupole", 2: "sextupole", 3: "octupole"}[order]
            for length in (0.08, 0.16):
                for bv in (1, -1):
                    text = f"""beam,particle=proton,energy=1.7,bv={bv};
    m:{family},l={length},k{order}={kn},k{order}s={ks},tilt=-0.13;
    cell:line=(m);use,period=cell;
    """
                    ref, _ = load(sim, deck(f"thick_n{order}_L{length}_bv{bv}", text))
                    e = sim.lattice[0]
                    e.int_order, e.mapsteps = 6, 12
                    out, _ = track(sim, ref, q)
                    oracle = exact_flow(q, order, bv * kn, bv * ks, length, ref.beta, -0.13)
                    forward = exact_flow(q, order, kn, ks, length, ref.beta, -0.13)
                    err = float(np.max(np.abs(out - oracle)))
                    ferr = float(np.max(np.abs(out - forward)))
                    assert ferr < 5e-14
                    assert err < 5e-14 if bv == 1 else err > 1e-5
                    emit(
                        case="thick_mixed",
                        order=order,
                        length=length,
                        bv=bv,
                        max_error=err,
                        forward_error=ferr,
                        px_observed=float(out[0, 1]),
                        px_oracle=float(oracle[0, 1]),
                    )

        # Direct negative-strength control from the discovery quadrupole check.
        ref = RefPart().set_species("proton")
        ref.set_kin_energy_MeV(2000 - ref.mass_MeV)
        sim.lattice.clear()
        sim.lattice.append(elements.Quad(ds=0.4, k=-2, nslice=32))
        out, _ = track(sim, ref, q)
        oracle = linear_quad(q, -2, 0.4, 0)
        err = float(np.max(np.abs(out[:, :4] - oracle[:, :4])))
        assert err < 2e-16
        emit(case="direct_negative_strength_control", max_transverse_error=err)

        path = deck(
            "kicker_control",
            """beam,particle=proton,energy=1.7,bv=-1;
    h:hkicker,kick=0.0004;cell:line=(h);use,period=cell;
    """,
        )
        ref, _ = load(sim, path)
        out, _ = track(sim, ref, q)
        assert np.max(np.abs(out[:, 1] - q[:, 1] + 0.0004)) < 2e-19
        emit(case="kicker_control", delta_px=float(out[0, 1] - q[0, 1]))


    def native():
        from cpymad.madx import Madx

        emit(case="native_version", cpymad=importlib.metadata.version("cpymad"))
        for bv in (1, -1):
            with Madx(stdout=False) as madx:
                path = ROOT / f"quad_E1.4_k1.7_bv{bv}_linear.madx"
                madx.call(str(path))
                tw = madx.twiss(betx=1, bety=1, rmatrix=True)
                q = np.zeros((N, 6))
                q[:, 0] = 1
                oracle = linear_quad(q, bv * 1.7, 0.23, 0.21)
                err = abs(float(tw.re21[-1]) - oracle[0, 1])
                assert err < 3e-15
                emit(
                    case="native_quad",
                    bv=bv,
                    R21=float(tw.re21[-1]),
                    oracle=float(oracle[0, 1]),
                    error=err,
                    version=str(madx.version),
                )
        q = phase()
        for bv in (1, -1):
            for order in (1, 2, 3, 4, 5):
                with Madx(stdout=False) as madx:
                    path = ROOT / f"thin_n{order}_bv{bv}_s1.0.madx"
                    madx.call(str(path))
                    tw = madx.twiss(betx=1, bety=1, x=q[0, 0], px=q[0, 1], y=q[0, 2], py=q[0, 3])
                    kn = {1: 0.3, 2: 700.0, 3: 8e5, 4: 3e9, 5: 2e13}[order]
                    local = rotate(q, -0.17)
                    f = (
                        kn
                        * (1 - 0.37j)
                        * (local[:, 0] + 1j * local[:, 2]) ** order
                        / math.factorial(order)
                    )
                    local[:, 1] -= bv * f.real
                    local[:, 3] += bv * f.imag
                    oracle = rotate(local, 0.17)
                    err = max(
                        abs(float(tw.px[-1]) - oracle[0, 1]), abs(float(tw.py[-1]) - oracle[0, 3])
                    )
                    assert err < 2e-18
                    emit(
                        case="native_thin",
                        order=order,
                        bv=bv,
                        px=float(tw.px[-1]),
                        py=float(tw.py[-1]),
                        error=err,
                    )


    def run():
        global ROOT
        emit(
            case="environment",
            python=sys.version.split()[0],
            impactx=importlib.metadata.version("impactx-noacc"),
            numpy=np.__version__,
            scipy=importlib.metadata.version("scipy"),
            build=Config.to_dict(),
        )
        print("Oracle: x''=-BV*k1*x; F=BV*(kn+i*ks)*(x+i*y)**n/n!, dpx=-Re(F), dpy=Im(F).", flush=True)
        config = Config.to_dict()
        assert not any(config[k] for k in ("have_gpu", "have_mpi", "have_omp"))
        assert config["precision"] == config["precision_particles"] == "DOUBLE"
        with TemporaryDirectory(prefix="api-13-", dir=ROOT) as directory:
            ROOT = Path(directory)
            sim = ImpactX()
            try:
                sim.tiny_profiler = sim.memory_profiler = False
                sim.particle_shape = 1
                sim.space_charge = "off"
                sim.csr = sim.isr = sim.spin = sim.diagnostics = sim.slice_step_diagnostics = False
                sim.verbose = 0
                sim.init_grids()
                check_bv(sim)
            finally:
                sim.finalize()
            native()
        print("DEFECT REPRODUCED", flush=True)


    if __name__ == "__main__":
        run()

    ```

## Provenance

| Role | Model | Effort | Date | Code examined |
|---|---|---|---|---|
| Found | <span class="bb-nowrap">`gpt-6-astra`</span> | xhigh | <span class="bb-nowrap">2026-10-29</span> | 26.10 (`615921c6867c`), impactx-noacc wheel, CPU, double, serial |
| Confirmed | <span class="bb-nowrap">`gpt-6.1-sol`</span> | xhigh | <span class="bb-nowrap">2026-10-29</span> | 26.10 (`615921c6867c`), impactx-noacc wheel, CPU, double, serial |

