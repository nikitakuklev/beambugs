---
title: "API-15: Programmable reference callbacks lose mutations without a retained live wrapper"
hide:
  - toc
---

# API-15 — Programmable reference callbacks lose mutations without a retained live wrapper

| | |
|---|---|
| Code | [ImpactX](../index.md) · Python API |
| Priority | P3 |
| Severity | Wrong physics |
| Evidence | Reproduced, independently rechecked |
| Last reproduced | 2026-10-29 on development (`fa0a43b9bb07`), source, CPU, double, serial |
| Source | [`src/python/elements.cpp#L2239-L2251`](https://github.com/BLAST-ImpactX/impactx/blob/615921c6867c895cf3a8fcf1d12991079d0f7d75/src/python/elements.cpp#L2239-L2251)<br>[`src/elements/Programmable.cpp#L19-L57`](https://github.com/BLAST-ImpactX/impactx/blob/615921c6867c895cf3a8fcf1d12991079d0f7d75/src/elements/Programmable.cpp#L19-L57)<br>[`src/particles/PushAll.H#L105-L132`](https://github.com/BLAST-ImpactX/impactx/blob/615921c6867c895cf3a8fcf1d12991079d0f7d75/src/particles/PushAll.H#L105-L132)<br>[`docs/source/usage/python.rst#L2323-L2352`](https://github.com/BLAST-ImpactX/impactx/blob/615921c6867c895cf3a8fcf1d12991079d0f7d75/docs/source/usage/python.rst#L2323-L2352) |

A `Programmable` element's reference-particle callback loses its changes to the reference particle unless a Python object for `sim.beam.ref` is being held. The documented contract applies the reference update before beam transport with no such prerequisite, and analytic drift and energy-gain formulas give the required result. A 0.375 m programmable drift traversed over three periods at γ = 1.25 leaves the reference at `s = 0` instead of 1.125 m, and an electron energy update leaves γ at 1.050 instead of 1.250, after which beam coordinates are off by up to 1.376e-4. Holding `sim.beam.ref` gives the correct result; the bundled example and tests hold it.

## Reproducer

Checks reference drift and energy updates against analytic formulas. Reference momenta use mc normalization; beam momenta use the design momentum. Writes no files. Each native case runs in a fresh child process.

??? example "API-15.py"

    [Download API-15.py](API-15.py)

    ```python
    """API-15: Programmable reference mutations depend on a retained live wrapper.

    Checks reference drift and energy updates against analytic formulas.
    Reference momenta use mc normalization; beam momenta use the design momentum.
    Writes no files. Each native case runs in a fresh child process.
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

    import gc
    import json
    import math
    import subprocess
    from pathlib import Path

    import amrex.space3d as amr
    import impactx
    import numpy as np
    from impactx import ImpactX, elements

    FIELDS = ("s", "x", "y", "z", "t", "pt", "px", "py", "pz")
    COORDS = ("position_x", "momentum_x", "position_y", "momentum_y", "position_t", "momentum_t")


    def snapshot(ref):
        return {name: getattr(ref, name) for name in FIELDS}


    def initialize(config):
        sim = ImpactX()
        sim.tiny_profiler = False
        sim.memory_profiler = False
        sim.particle_shape = 2
        sim.space_charge = False
        sim.csr = False
        sim.isr = False
        sim.spin = False
        sim.diagnostics = False
        sim.slice_step_diagnostics = False
        sim.verbose = 0
        sim.init_grids()

        # This temporary wrapper goes out of scope before the caller tracks.
        ref = sim.beam.ref
        ref.set_species(config.get("species", "proton"))
        ref.set_kin_energy_MeV((config["gamma"] - 1) * ref.mass_MeV)
        if config.get("tilted"):
            bg = ref.beta_gamma
            ref.px, ref.py, ref.pz = 0.3 * bg, -0.4 * bg, math.sqrt(0.75) * bg
        samples = np.arange(12, dtype=float) - 5.5
        initial = np.column_stack(
            (
                samples * 2e-4,
                (samples + 0.5) * 4e-5,
                -samples * 1e-4,
                (samples - 0.25) * 3e-5,
                samples * 3e-4,
                (samples + 0.125) * 2e-5,
            )
        )
        sim.beam.add_n_particles(
            x=initial[:, 0],
            px=initial[:, 1],
            y=initial[:, 2],
            py=initial[:, 3],
            t=initial[:, 4],
            pt=initial[:, 5],
            qm=ref.qm_ratio_SI,
            w=np.ones(12),
        )
        return sim, initial, snapshot(ref)


    def run(config):
        sim, initial, start = initialize(config)
        initial_bg = sim.beam.ref.beta_gamma
        events = []
        held = None
        mode = config.get("handle", "none")
        if mode == "ref":
            held = sim.beam.ref
        elif mode == "beam":
            held = sim.beam
        elif mode == "copy":
            held = sim.beam.ref.copy()
        elif mode == "released":
            held = sim.beam.ref
            held = None
        elif mode != "none":
            raise ValueError(mode)
        gc.collect()
        try:
            if config["kind"] in ("drift", "energy"):
                length = config["length"]
                slices = config["slices"]
                h = length / slices
                element = elements.Programmable(ds=length, nslice=slices)

                def push_reference(ref):
                    before = snapshot(ref)
                    if config["kind"] == "energy":
                        gamma_new = ref.gamma + config["gain"]
                        ref.set_kin_energy_MeV((gamma_new - 1) * ref.mass_MeV)
                    else:
                        scale = h / math.sqrt(ref.gamma**2 - 1)
                        for position, momentum in (("x", "px"), ("y", "py"), ("z", "pz")):
                            setattr(
                                ref, position, getattr(ref, position) + scale * getattr(ref, momentum)
                            )
                        ref.t += scale * ref.gamma
                        ref.s += h
                    events.append({"ref_before": before, "ref_after": snapshot(ref)})

                def push_beam(pti, ref):
                    data = pti.soa().to_xp().real
                    if config["kind"] == "drift":
                        data["position_x"][:] += h * data["momentum_x"][:]
                        data["position_y"][:] += h * data["momentum_y"][:]
                        data["position_t"][:] += h / (ref.gamma**2 - 1) * data["momentum_t"][:]
                    else:
                        # Same absolute energy gain for every particle; Delta E stays
                        # fixed and all canonical momenta renormalize by P_old/P_new.
                        ratio = initial_bg / ref.beta_gamma
                        for name in ("momentum_x", "momentum_y", "momentum_t"):
                            data[name][:] *= ratio
                    events.append({"tile_ref": snapshot(ref)})

                element.ref_particle = push_reference
                element.beam_particles = push_beam
                if config.get("whole"):
                    from impactx import ImpactXParIter

                    def whole(beam, step, period):
                        ref = beam.ref
                        push_reference(ref)
                        for pti in ImpactXParIter(beam, level=0):
                            push_beam(pti, ref)

                    element.push = whole
                sim.lattice.append(element)
            elif config["kind"] == "native_drift":
                sim.lattice.append(elements.Drift(ds=config["length"], nslice=config["slices"]))
            else:
                raise ValueError(config["kind"])
            sim.periods = config.get("periods", 1)
            sim.track_particles()
            final_ref = snapshot(sim.beam.ref)
            frame = sim.beam.to_df(local=True).reset_index(drop=True).sort_values("idcpu")
            final = frame[list(COORDS)].to_numpy()
            assert final.shape == (12, 6) and np.isfinite(final).all()
            result = {
                "config": config,
                "start": start,
                "reference": final_ref,
                "initial": initial.tolist(),
                "final": final.tolist(),
                "events": events,
                "handle_retained": held is not None,
            }
        finally:
            # Keep wrapper controls alive through tracking, then release before teardown.
            held = None
            sim.finalize()
        return result


    def child(mode, payload):
        completed = subprocess.run(
            [sys.executable, str(Path(__file__).resolve()), mode],
            input=json.dumps(payload),
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


    def provenance():
        print(
            f"Python {sys.version.split()[0]} ImpactX={impactx.__version__} pyAMReX={amr.__version__} NumPy={np.__version__}",
            flush=True,
        )
        print(
            "Serial CPU/double; one CPU and numerical thread; one finalized simulation per child",
            flush=True,
        )


    def verify_reference():
        rows = []
        # Vary gamma, reference direction, length, slice count, period count, and
        # wrapper ownership independently of the discovery worker's fixtures.
        for gamma, length, slices, periods, tilted in (
            (1.25, 0.375, 5, 3, True),
            (7.3, 0.14, 2, 2, False),
        ):
            base = dict(
                kind="drift", gamma=gamma, length=length, slices=slices, periods=periods, tilted=tilted
            )
            for handle in ("none", "beam", "ref", "released", "copy", "whole", "native"):
                config = dict(base, handle="none" if handle in ("whole", "native") else handle)
                if handle == "whole":
                    config["whole"] = True
                if handle == "native":
                    config["kind"] = "native_drift"
                name = f"reference_g{gamma}_{handle}"
                result = child("--native", config)
                distance = length * periods
                start = result["start"]
                bg = math.sqrt(gamma**2 - 1)
                expected = dict(start, s=start["s"] + distance, t=start["t"] + distance * gamma / bg)
                for p, v in (("x", "px"), ("y", "py"), ("z", "pz")):
                    expected[p] = start[p] + distance * start[v] / bg
                failed = handle in ("none", "beam", "released", "copy")
                if failed:
                    assert result["events"][0]["ref_after"]["s"] > 0
                    assert result["events"][1]["tile_ref"]["s"] == 0
                target = start if failed else expected
                np.testing.assert_allclose(
                    list(result["reference"].values()), list(target.values()), rtol=3e-15, atol=3e-16
                )
                a = np.array(result["initial"])
                prediction = a.copy()
                prediction[:, 0] += distance * a[:, 1]
                prediction[:, 2] += distance * a[:, 3]
                prediction[:, 4] += distance / bg**2 * a[:, 5]
                particle_error = float(np.max(abs(np.array(result["final"]) - prediction)))
                assert particle_error < 2e-18, particle_error
                rows.append(
                    dict(
                        case=name,
                        s=result["reference"]["s"],
                        expected_s=distance,
                        t=result["reference"]["t"],
                        expected_t=expected["t"],
                        x=result["reference"]["x"],
                        expected_x=expected["x"],
                        particle_error=particle_error,
                    )
                )
                print("REFERENCE", json.dumps(rows[-1]), flush=True)

        for species, gamma, gain in (("electron", 1.05, 0.2), ("proton", 20.0, 2.5)):
            for handle in ("none", "ref", "whole"):
                config = dict(
                    kind="energy",
                    species=species,
                    gamma=gamma,
                    gain=gain,
                    length=0.0,
                    slices=1,
                    handle="none" if handle == "whole" else handle,
                    whole=handle == "whole",
                )
                name = f"energy_{species}_{handle}"
                result = child("--native", config)
                expected_gamma = gamma + gain
                ratio = math.sqrt((gamma**2 - 1) / (expected_gamma**2 - 1))
                expected = np.array(result["initial"])
                expected[:, [1, 3, 5]] *= ratio
                actual = np.array(result["final"])
                error = float(np.max(abs(actual - expected)))
                observed_gamma = -result["reference"]["pt"]
                if handle == "none":
                    assert abs(-result["events"][0]["ref_after"]["pt"] - expected_gamma) < 1e-14
                    assert abs(-result["events"][1]["tile_ref"]["pt"] - gamma) < 1e-14
                    assert abs(observed_gamma - gamma) < 1e-14
                    np.testing.assert_array_equal(actual, result["initial"])
                    assert error > 1e-6
                else:
                    assert abs(observed_gamma - expected_gamma) < 1e-14
                    assert error < 2e-18
                rows.append(
                    dict(
                        case=name,
                        observed_gamma=observed_gamma,
                        expected_gamma=expected_gamma,
                        expected_momentum_ratio=ratio,
                        max_coordinate_error=error,
                        callback_gammas=[
                            -item["tile_ref"]["pt"] for item in result["events"] if "tile_ref" in item
                        ],
                    )
                )
                print("ENERGY", json.dumps(rows[-1]), flush=True)
        return rows


    if __name__ == "__main__":
        if sys.argv[1:] == ["--native"]:
            print("RESULT " + json.dumps(run(json.load(sys.stdin))))
        else:
            assert len(sys.argv) == 1
            provenance()
            print(
                "Oracle: delta(s)=D, delta(x,y,z)=D*(px,py,pz)/sqrt(gamma^2-1), delta(t)=D*gamma/sqrt(gamma^2-1)"
            )
            print("Energy oracle: (px,py,pt)_f = sqrt((gamma_i^2-1)/(gamma_f^2-1))*(px,py,pt)_i")
            verify_reference()
            print("DEFECT REPRODUCED", flush=True)

    ```

## Provenance

| Role | Model | Effort | Date | Code examined |
|---|---|---|---|---|
| Found | `gpt-6-astra` | xhigh | 2026-10-29 | 26.10 (`615921c6867c`), impactx-noacc wheel, CPU, double, serial |
| Confirmed | `gpt-6.1-sol` | xhigh | 2026-10-29 | 26.10 (`615921c6867c`), impactx-noacc wheel, CPU, double, serial |

