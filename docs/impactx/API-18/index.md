---
title: "API-18: Marker insertion compresses field profiles and repeats linear maps in each piece"
hide:
  - toc
---

# API-18 — Marker insertion compresses field profiles and repeats linear maps in each piece

| | |
|---|---|
| Code | [ImpactX](../index.md) · Python API |
| Priority | P1 |
| Failure | Wrong physics |
| Evidence | Reproduced, independently rechecked |
| Last reproduced | 2026-11-05 on ImpactX commit [`f9c2a82`](https://github.com/BLAST-ImpactX/impactx/commit/f9c2a82e24b7), committed 2026-11-05 |
| Issue | [[gh:https://github.com/BLAST-ImpactX/impactx/issues/1728]], filed from this evidence |
| Source | [`Insert.cpp#L64-L74`](https://github.com/BLAST-ImpactX/impactx/blob/fa0a43b9bb07decba15daa38ed56ca947160ed1d/src/elements/transformation/Insert.cpp#L64-L74)<br>[`RFCavity.H#L294-L317`](https://github.com/BLAST-ImpactX/impactx/blob/fa0a43b9bb07decba15daa38ed56ca947160ed1d/src/elements/RFCavity.H#L294-L317) |

`insert_element_every_ds()`, which places a thin element such as a `Marker` at regular spacing, splits the thick elements it falls in by copying them with a shorter `ds`, so each piece of a split `RFCavity`, `SoftQuadrupole` or `SoftSolenoid` holds a compressed copy of the whole field profile. A midpoint marker in a 1 m cavity with profile ½(1 − cos 2πs/L) cuts a 10 GeV electron's energy gain by 20.00%, to 0.1735 MeV instead of 0.2169 MeV; the unsplit cavity and ImpactX's own `nslice=2` subdivision match the analytic gain integral. Constant profiles are unaffected. It also splits `LinearMap` and `SpinMap`, documented as unsliceable, without an error, and each piece applies the whole map: a 60° `SpinMap` rotation becomes 120°.

Run `python API-18.py`.

??? example "Reproducer: API-18.py"

    [Download API-18.py](API-18.py)

    ```python
    """API-18: Marker insertion compresses field profiles and repeats linear maps in each piece

    `insert_element_every_ds()`, which places a thin element such as a `Marker` at regular spacing,
    splits the thick elements it falls in by copying them with a shorter `ds`, so each piece of a split
    `RFCavity`, `SoftQuadrupole` or `SoftSolenoid` holds a compressed copy of the whole field profile. A
    midpoint marker in a 1 m cavity with profile ½(1 − cos 2πs/L) cuts a 10 GeV electron's energy gain
    by 20.00%, to 0.1735 MeV instead of 0.2169 MeV; the unsplit cavity and ImpactX's own `nslice=2`
    subdivision match the analytic gain integral. Constant profiles are unaffected. It also splits
    `LinearMap` and `SpinMap`, documented as unsliceable, without an error, and each piece applies the
    whole map: a 60° `SpinMap` rotation becomes 120°.

    The script splits each 1 m element at 0.5 m with a `Marker` and tracks a 10 GeV electron at x = 1
    mm, pₓ = 0.1 mrad through the unsplit element, `nslice=2` (profile elements) and the split copy,
    each in a fresh child process. The profile elements use f(s) = ½(1 − cos 2πs) over [0, 1] m
    (`cos_coefficients` [1, 0.5]): the cavity gain is compared with the ultrarelativistic integral
    mc²·∫f(s) sin(πs) ds, and the magnets with an RK4 integration of the paraxial quadrupole and
    canonical solenoid equations, for f(s) and for the compressed f(2s mod 1). `LinearMap` holds a 1 m
    drift matrix, `SpinMap` rotates the spin by 60° and `Programmable` drifts as in the documented
    how-to; constant profiles and an insertion at the element exit are the controls.
    """

    import os
    import sys

    for variable in (
        "NPROC",
        "PJRT_NPROC",
        "OMP_NUM_THREADS",
        "MKL_NUM_THREADS",
        "OPENBLAS_NUM_THREADS",
        "NUMEXPR_NUM_THREADS",
    ):
        os.environ[variable] = "1"
    os.environ["PYTHONDONTWRITEBYTECODE"] = "1"
    sys.dont_write_bytecode = True
    os.sched_setaffinity(0, {min(os.sched_getaffinity(0))})

    import json
    import math
    import subprocess
    from pathlib import Path

    import numpy as np

    E0 = 10e3  # electron kinetic energy [MeV]; ultrarelativistic (gamma = 1.96e4)
    MC2 = 0.51099895000  # electron rest energy [MeV]
    X0 = np.array([1e-3, 1e-4, 0.0, 0.0])  # particle (x, px, y, py); t = pt = 0
    STEPS = 2000  # mapsteps of the unsplit element
    GSCALE, BSCALE = 4.0, 4.0  # SoftQuadrupole [1/m^2], SoftSolenoid (unit=0) [1/m]


    def run(config):
        """One simulation: build the element, optionally split it, track, report."""
        from impactx import ImpactX, Map3x6, Map6x6, Vector3, elements

        kind, mode = config["kind"], config["mode"]
        sim = ImpactX()
        try:
            sim.verbose = 0
            sim.tiny_profiler = False
            sim.memory_profiler = False
            sim.diagnostics = False
            sim.slice_step_diagnostics = False
            sim.space_charge = False
            sim.csr = False
            sim.isr = False
            sim.spin = kind == "SpinMap"
            sim.particle_shape = 2
            sim.init_grids()
            ref = sim.beam.ref  # retained through tracking
            ref.set_species("electron").set_kin_energy_MeV(E0)

            coefficients = [2.0, 0.0] if config.get("constant") else [1.0, 0.5]
            nslice = 2 if mode == "nslice2" else 1
            steps = config.get("steps", STEPS)
            field = dict(
                ds=1.0,
                cos_coefficients=coefficients,
                sin_coefficients=[0.0, 0.0],
                mapsteps=steps,
                nslice=nslice,
            )
            if kind == "RFCavity":
                # f = c / (2 m), phase -90 deg: dgamma/ds = escale f(s) sin(pi ct / 1 m)
                el = elements.RFCavity(escale=1.0, freq=299792458.0 / 2, phase=-90.0, **field)
            elif kind == "SoftQuadrupole":
                el = elements.SoftQuadrupole(gscale=GSCALE, **field)
            elif kind == "SoftSolenoid":
                el = elements.SoftSolenoid(bscale=BSCALE, unit=0, **field)
            elif kind == "LinearMap":
                matrix = Map6x6.identity()  # transverse matrix of a 1 m drift
                matrix[1, 2] = matrix[3, 4] = 1.0
                el = elements.LinearMap(R=matrix, ds=1.0)
            elif kind == "SpinMap":
                v = Vector3()
                v[3] = math.pi / 3  # rotation by 60 degrees about z
                el = elements.SpinMap(v=v, A=Map3x6(), ds=1.0)
            elif kind == "Programmable":
                # Callback pattern of docs/source/usage/howto/python_extend.rst: the hooks
                # capture the element and drift by el.ds / el.nslice per slice.
                el = elements.Programmable(ds=1.0, nslice=nslice)

                def beam_drift(pti, refpart):
                    soa = pti.soa().to_xp().real
                    soa["position_x"][:] += el.ds / el.nslice * soa["momentum_x"][:]
                    soa["position_y"][:] += el.ds / el.nslice * soa["momentum_y"][:]

                def ref_drift(refpart):
                    refpart.z += el.ds / el.nslice
                    refpart.s += el.ds / el.nslice

                el.beam_particles = beam_drift
                el.ref_particle = ref_drift
            else:
                raise ValueError(kind)

            lattice = elements.KnownElementsList([el])
            if mode in ("split", "boundary"):
                lattice = elements.transformation.insert_element_every_ds(
                    lattice, ds=1.0 if mode == "boundary" else 0.5, element=elements.Marker("m")
                )
            pieces = [[type(e).__name__, e.ds] for e in lattice]

            one, zero = np.ones(1), np.zeros(1)
            spin = dict(sx=one, sy=zero, sz=zero) if sim.spin else {}
            sim.beam.add_n_particles(
                x=X0[0] * one,
                px=X0[1] * one,
                y=X0[2] * one,
                py=X0[3] * one,
                t=zero,
                pt=zero,
                qm=ref.qm_ratio_SI,
                w=one,
                **spin,
            )
            sim.lattice.extend(lattice)
            sim.track_particles()
            p = sim.beam.to_df(local=True).iloc[0]
            result = dict(
                pieces=pieces,
                original_ds=el.ds,
                gain=ref.kin_energy_MeV - E0,
                s=ref.s,
                u=[float(p[k]) for k in ("position_x", "momentum_x", "position_y", "momentum_y")],
            )
            if sim.spin:
                result["spin"] = [float(p[k]) for k in ("spin_x", "spin_y", "spin_z")]
        finally:
            sim.finalize()
        return result


    def child(config):
        completed = subprocess.run(
            [sys.executable, str(Path(__file__).resolve()), "--child"],
            input=json.dumps(config),
            capture_output=True,
            text=True,
            timeout=290,
            check=False,
        )
        if completed.returncode:
            raise RuntimeError(completed.stdout + completed.stderr)
        records = [
            line.removeprefix("RESULT ")
            for line in completed.stdout.splitlines()
            if line.startswith("RESULT ")
        ]
        assert len(records) == 1, completed.stdout + completed.stderr
        return json.loads(records[0])


    def f_profile(s, compressed):
        """On-axis profile at s in [0, 1] m; compressed: the whole profile in each half."""
        u = (2 * s) % 1.0 if compressed else s
        return 0.5 * (1 - math.cos(2 * math.pi * u))


    def oracle_transverse(kind, compressed, n=4000):
        """RK4 of the paraxial equations through the profile, from X0."""

        def rhs(s, w):
            x, px, y, py = w
            if kind == "SoftQuadrupole":
                k = GSCALE * f_profile(s, compressed)
                return np.array([px, -k * x, py, k * y])
            # Canonical solenoid Hamiltonian H = ((px + a y)^2 + (py - a x)^2) / 2, a = bscale f / 2
            a = 0.5 * BSCALE * f_profile(s, compressed)
            vx, vy = px + a * y, py - a * x
            return np.array([vx, a * vy, vy, -a * vx])

        h, w = 1.0 / n, X0.copy()
        for i in range(n):
            s = i * h
            k1 = rhs(s, w)
            k2 = rhs(s + h / 2, w + h / 2 * k1)
            k3 = rhs(s + h / 2, w + h / 2 * k2)
            k4 = rhs(s + h, w + h * k3)
            w = w + h / 6 * (k1 + 2 * k2 + 2 * k3 + k4)
        return w


    def fmt(values):
        return "[" + ", ".join(f"{v:.12g}" for v in values) + "]"


    def main():
        import amrex.space3d as amr
        import impactx

        print(f"ImpactX={impactx.__version__} pyAMReX={amr.__version__} NumPy={np.__version__}")
        print("Serial CPU/double; one CPU and numerical thread; one finalized simulation per child")
        print(f"electron {E0:g} MeV; particle (x, px, y, py) = {fmt(X0)}; split = Marker at s=0.5 m")
        defects = []

        # Field-profile elements.
        gain_f = 4 / (3 * math.pi) * MC2
        gain_c = 16 / (15 * math.pi) * MC2
        print("RFCavity: escale=1/m, f=c/(2 m), phase=-90 deg; reference energy gain [MeV]")
        print(
            "  oracle (ultrarelativistic): mc^2 int f(s) sin(pi s) ds = 4/(3 pi) mc^2 ="
            f" {gain_f:.12g}; with f(2s mod 1): 16/(15 pi) mc^2 = {gain_c:.12g}"
        )
        rf = {mode: child(dict(kind="RFCavity", mode=mode)) for mode in ("unsplit", "nslice2", "split")}
        print(f"  pieces={rf['split']['pieces']} original_ds={rf['split']['original_ds']:g}")
        for mode, result in rf.items():
            print(f"  {mode:8s} gain={result['gain']:.12g}")
        assert abs(rf["unsplit"]["gain"] - gain_f) < 5e-8
        assert abs(rf["nslice2"]["gain"] - gain_f) < 5e-8
        assert abs(rf["split"]["gain"] - gain_c) < 5e-8
        print(
            f"  split/unsplit={rf['split']['gain'] / rf['unsplit']['gain']:.6f}"
            f" (oracle 0.8); |unsplit-oracle|={abs(rf['unsplit']['gain'] - gain_f):.2g}"
            f" |split-compressed oracle|={abs(rf['split']['gain'] - gain_c):.2g}"
        )
        defects.append(abs(rf["split"]["gain"] - gain_f) > 1e-3)

        for kind in ("SoftQuadrupole", "SoftSolenoid"):
            scale = f"gscale={GSCALE:g}/m^2" if kind == "SoftQuadrupole" else f"bscale={BSCALE:g}/m"
            print(f"{kind}: {scale}; particle (x, px, y, py) at exit")
            ok, bad = oracle_transverse(kind, False), oracle_transverse(kind, True)
            print(f"  oracle RK4 f(s)        {fmt(ok)}")
            print(f"  oracle RK4 f(2s mod 1) {fmt(bad)}")
            runs = {mode: child(dict(kind=kind, mode=mode)) for mode in ("unsplit", "nslice2", "split")}
            for mode, result in runs.items():
                print(f"  {mode:8s} {fmt(result['u'])}")
            e_unsplit = np.max(np.abs(np.array(runs["unsplit"]["u"]) - ok))
            e_nslice = np.max(np.abs(np.array(runs["nslice2"]["u"]) - ok))
            e_split = np.max(np.abs(np.array(runs["split"]["u"]) - bad))
            assert max(e_unsplit, e_nslice, e_split) < 1e-9, (e_unsplit, e_nslice, e_split)
            print(
                f"  max|unsplit-oracle f(s)|={e_unsplit:.2g} max|nslice2-oracle f(s)|={e_nslice:.2g}"
                f" max|split-oracle f(2s)|={e_split:.2g}"
            )
            defects.append(np.max(np.abs(np.array(runs["split"]["u"]) - ok)) > 1e-5)

        # Controls: a constant profile at the same step length, and an insertion at the exit.
        print("Controls (expected: split equals unsplit)")
        for kind in ("RFCavity", "SoftQuadrupole", "SoftSolenoid"):
            whole = child(dict(kind=kind, mode="unsplit", constant=True))
            halves = child(dict(kind=kind, mode="split", constant=True, steps=STEPS // 2))
            d_gain = abs(halves["gain"] - whole["gain"])
            d_u = np.max(np.abs(np.array(halves["u"]) - whole["u"]))
            print(f"  {kind} constant profile: |d gain|={d_gain:.2g} max|d (x,px,y,py)|={d_u:.2g}")
            assert d_gain < 1e-12 and d_u < 1e-14
        boundary = child(dict(kind="RFCavity", mode="boundary"))
        print(
            f"  RFCavity Marker every 1 m: pieces={boundary['pieces']}"
            f" gain-unsplit={boundary['gain'] - rf['unsplit']['gain']:.2g}"
        )
        assert boundary["gain"] == rf["unsplit"]["gain"]

        # Elements documented as unable to be sliced: no error, whole map in each piece.
        print("Unsliceable elements (helper raises only for Thin elements)")
        x_once = X0[0] + 1.0 * X0[1]
        x_twice = X0[0] + 2.0 * X0[1]
        for kind in ("LinearMap", "Programmable"):
            runs = {mode: child(dict(kind=kind, mode=mode)) for mode in ("unsplit", "split")}
            print(f"  {kind}: pieces={runs['split']['pieces']}")
            for mode, result in runs.items():
                print(f"    {mode:8s} x={result['u'][0]:.12g} s={result['s']:.12g}")
            print(f"    oracle: 1 m drift x=x0+px0*1 m={x_once:.12g}; applied twice {x_twice:.12g}")
            assert abs(runs["unsplit"]["u"][0] - x_once) < 1e-18
            assert abs(runs["split"]["u"][0] - x_twice) < 1e-18
            defects.append(abs(runs["split"]["u"][0] - x_once) > 1e-6)
        runs = {mode: child(dict(kind="SpinMap", mode=mode)) for mode in ("unsplit", "split")}
        once = [math.cos(math.pi / 3), math.sin(math.pi / 3), 0.0]
        twice = [math.cos(2 * math.pi / 3), math.sin(2 * math.pi / 3), 0.0]
        print(f"  SpinMap: v=(0, 0, pi/3); pieces={runs['split']['pieces']}")
        for mode, result in runs.items():
            print(f"    {mode:8s} spin={fmt(result['spin'])}")
        print(f"    oracle: Rodrigues 60 deg about z {fmt(once)}; 120 deg {fmt(twice)}")
        assert np.max(np.abs(np.array(runs["unsplit"]["spin"]) - once)) < 1e-15
        assert np.max(np.abs(np.array(runs["split"]["spin"]) - twice)) < 1e-15
        defects.append(np.max(np.abs(np.array(runs["split"]["spin"]) - once)) > 1e-6)

        print("DEFECT REPRODUCED" if all(defects) else "NOT REPRODUCED")


    if __name__ == "__main__":
        if sys.argv[1:] == ["--child"]:
            print("RESULT " + json.dumps(run(json.load(sys.stdin))), flush=True)
        else:
            assert len(sys.argv) == 1
            main()

    ```

## Provenance

| Role | Model | Effort | Date | Code examined |
|---|---|---|---|---|
| Found | `claude-opus-5-5` | high | 2026-10-29 | development (`fa0a43b9bb07`), source, CPU, double, serial |
| Found<br><small>LinearMap duplication; source only</small> | `gpt-6-astra` | xhigh | 2026-10-28 | 26.10 (`615921c6867c`), impactx-noacc wheel, CPU, double, serial |
| Confirmed | `gpt-6.1-sol` | xhigh | 2026-10-29 | development (`fa0a43b9bb07`), source, CPU, double, serial |

