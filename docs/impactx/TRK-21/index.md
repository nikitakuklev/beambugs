---
title: "TRK-21: Field-free zero-frequency RFCavity returns nonfinite transport"
hide:
  - toc
---

# TRK-21 — Field-free zero-frequency RFCavity returns nonfinite transport

| | |
|---|---|
| Code | [ImpactX](../index.md) · Tracking |
| Priority | P3 |
| Severity | Crash or API hazard |
| Evidence | Reproduced, independently rechecked |
| Last reproduced | 2026-11-06 on ImpactX development (`f9c2a82`, head build) |
| Tags | <span class="bb-tag">partly-source-only</span> [?](../../methodology.md#tags) |
| Source | [`RFCavity.H:414–428`](https://github.com/BLAST-ImpactX/impactx/blob/615921c6867c895cf3a8fcf1d12991079d0f7d75/src/elements/RFCavity.H#L414-L428)<br>[`464–481`](https://github.com/BLAST-ImpactX/impactx/blob/615921c6867c895cf3a8fcf1d12991079d0f7d75/src/elements/RFCavity.H#L464-L481)<br>[`193–204`](https://github.com/BLAST-ImpactX/impactx/blob/615921c6867c895cf3a8fcf1d12991079d0f7d75/src/elements/RFCavity.H#L193-L204)<br>[`145–150`](https://github.com/BLAST-ImpactX/impactx/blob/615921c6867c895cf3a8fcf1d12991079d0f7d75/src/elements/RFCavity.H#L145-L150)<br>[`559–563`](https://github.com/BLAST-ImpactX/impactx/blob/615921c6867c895cf3a8fcf1d12991079d0f7d75/src/elements/RFCavity.H#L559-L563)<br>[`elements.cpp:2395–2398`](https://github.com/BLAST-ImpactX/impactx/blob/615921c6867c895cf3a8fcf1d12991079d0f7d75/src/python/elements.cpp#L2395-L2398)<br>[parameter contract](https://github.com/BLAST-ImpactX/impactx/blob/615921c6867c895cf3a8fcf1d12991079d0f7d75/docs/source/usage/parameters.rst#L1748-L1774)<br>[`lineartransport.H:59–77`](https://github.com/BLAST-ImpactX/impactx/blob/615921c6867c895cf3a8fcf1d12991079d0f7d75/src/elements/mixin/lineartransport.H#L59-L77) |

Setting a field-free finite-length RFCavity to zero frequency produces NaN transverse particle coordinates instead of linear drift transport. The constructor and frequency setter accept this value without an error, and the parameter documentation gives no nonzero-frequency restriction. For a 1.200 m element at reference Lorentz factor γ₀ = 3, an initial horizontal position of 0.004000 m and normalized horizontal momentum of 0.003000 should give 0.007600 m; the result is NaN. The transfer map and covariance transport are also nonfinite, and multiple slices contaminate all six coordinates. Positive-frequency field-free cavities and zero-frequency ShortRF controls remain finite.

Run `python TRK-21.py`.

??? example "Reproducer: TRK-21.py"

    [Download TRK-21.py](TRK-21.py)

    ```python
    """TRK-21: Field-free zero-frequency RFCavity returns nonfinite transport

    Setting a field-free finite-length RFCavity to zero frequency produces NaN transverse particle coordinates instead of linear drift transport. The constructor and frequency setter accept this value without an error, and the parameter documentation gives no nonzero-frequency restriction. For a 1.200 m element at reference Lorentz factor γ₀ = 3, an initial horizontal position of 0.004000 m and normalized horizontal momentum of 0.003000 should give 0.007600 m; the result is NaN. The transfer map and covariance transport are also nonfinite, and multiple slices contaminate all six coordinates. Positive-frequency field-free cavities and zero-frequency ShortRF controls remain finite.

    Setup: `taskset` on `PATH` and CPU 0 among the allowed CPUs; the script runs each case in a child process pinned to CPU 0.

    The script compares ten serial native cases with a 90-digit Decimal solution of the free quadratic Hamiltonian. It checks particles, per-slice transfer maps, covariance transport and the reference clock, including constructor and setter inputs and multiple slices. Positive-frequency cavities, Drift and zero-frequency ShortRF provide controls; each child owns and finalizes one simulation.
    """

    import json
    import os
    import subprocess
    import sys
    from decimal import Decimal, localcontext
    from importlib.metadata import version
    from pathlib import Path
    from typing import Any

    for _key in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS",
                 "NUMEXPR_NUM_THREADS", "NPROC", "PJRT_NPROC"):
        os.environ[_key] = "1"
    os.sched_setaffinity(0, {0})

    import numpy as np

    INITIAL = np.array(
        [
            [0.004, 0.003, -0.005, -0.002, 0.006, -0.004],
            [-0.003, 0.001, 0.002, 0.004, -0.001, 0.002],
            [0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
        ]
    )  # x,px,y,py,t,pt
    COLS = ["position_x", "momentum_x", "position_y", "momentum_y", "position_t", "momentum_t"]
    LENGTH = 1.2


    def child(spec):
        from impactx import Config, ImpactX, Map6x6, elements

        sim = ImpactX()
        try:
            for k, v in dict(
                particle_shape=1,
                space_charge="off",
                spin=False,
                isr=False,
                csr=False,
                diagnostics=False,
                slice_step_diagnostics=False,
                tiny_profiler=False,
                memory_profiler=False,
                verbose=0,
            ).items():
                setattr(sim, k, v)
            sim.init_grids()
            ref = sim.beam.ref
            ref.set_charge_qe(1).set_mass_MeV(938.27208816).set_kin_energy_MeV(2 * 938.27208816)
            ref.t = -0.47
            start = {k: getattr(ref, k) for k in ("t", "s", "z", "pt", "pz", "gamma")}
            if spec["kind"] == "Drift":
                el = elements.Drift(ds=LENGTH)
            elif spec["kind"] == "ShortRF":
                el = elements.ShortRF(V=spec["field"], freq=spec["f"], phase=spec["phase"])
            else:
                el = elements.RFCavity(
                    ds=LENGTH,
                    escale=spec["field"],
                    freq=spec["f"] if not spec.get("setter") else 1.3e9,
                    phase=spec["phase"],
                    cos_coefficients=spec["profile"],
                    sin_coefficients=[0.0] * len(spec["profile"]),
                    mapsteps=spec["steps"],
                    nslice=spec["nslice"],
                )
                if spec.get("setter"):
                    el.freq = spec["f"]
            matrix = el.transfer_map(ref).to_numpy().copy()
            cov = Map6x6.identity()
            el.push(cov, ref)
            cov = cov.to_numpy().copy()
            # The public particle insertion order is x,y,t,px,py,pt.
            p = INITIAL[:, [0, 2, 4, 1, 3, 5]]
            sim.beam.add_n_particles(*[a.copy() for a in p.T], qm=ref.qm_ratio_SI, w=np.ones(len(p)))
            sim.lattice.append(el)
            sim.track_particles()
            return dict(
                particles=sim.beam.to_df(local=True).sort_index()[COLS].to_numpy().tolist(),
                map=matrix.tolist(),
                covariance=cov.tolist(),
                before=start,
                after={k: getattr(ref, k) for k in start},
                config=Config.to_dict(),
            )
        finally:
            sim.finalize()


    def oracle(before, length):
        with localcontext() as ctx:
            ctx.prec = 90
            # H=(px²+py²)/2+pt²/[2(γ₀²−1)] gives x′=px, y′=py, t′=pt/(γ₀²−1).
            g, ll = Decimal(before["gamma"]), Decimal(length)
            bg = (g * g - 1).sqrt()
            expected = []
            for p in INITIAL:
                x, px, y, py, t, pt = map(Decimal, p)
                expected.append(
                    list(map(float, [x + ll * px, px, y + ll * py, py, t + ll * pt / (g * g - 1), pt]))
                )
            clock = float(Decimal(before["t"]) + ll * g / bg)
        return np.array(expected), clock


    def main():
        base: dict[str, Any] = dict(
            kind="RFCavity", f=0.0, field=0.0, profile=[1.0, -1.0], phase=45.0, steps=10, nslice=1
        )
        specs: list[dict[str, Any]] = [
            dict(base, name="zero_smooth"),
            dict(base, name="zero_constant", profile=[2.0]),
            dict(base, name="zero_profile", field=0.3, profile=[0.0]),
            dict(base, name="zero_phase", phase=0.0),
            dict(base, name="zero_sliced", nslice=4, steps=32),
            dict(base, name="zero_setter", setter=True),
            dict(base, name="positive_tiny", f=1e-6),
            dict(base, name="positive_rf", f=1.3e9),
            dict(base, name="drift", kind="Drift"),
            dict(base, name="short_zero", kind="ShortRF"),
        ]
        print(f"Python {sys.version.split()[0]} impactx-noacc={version('impactx-noacc')} "
              f"numpy={np.__version__}; CPU 0, double, serial; "
              "three fixed protons per case, no random seed", flush=True)
        failures = []
        for spec in specs:
            r = subprocess.run(
                ["taskset", "-c", "0", sys.executable, str(Path(__file__).resolve()),
                 "--child", json.dumps(spec)],
                capture_output=True, text=True, timeout=290, check=True,
            )
            result = json.loads(
                next(line[7:] for line in r.stdout.splitlines() if line.startswith("RESULT "))
            )
            length = 0.0 if spec["kind"] == "ShortRF" else LENGTH
            expected, clock = oracle(result["before"], length)
            observed = np.array(result["particles"])
            matrix = np.array(result["map"])
            covariance = np.array(result["covariance"])
            bad = int(np.count_nonzero(~np.isfinite(observed)))
            map_bad = int(np.count_nonzero(~np.isfinite(matrix)))
            cov_bad = int(np.count_nonzero(~np.isfinite(covariance)))
            # transfer_map and push each act on one slice; tracking uses all slices.
            expected_map = np.eye(6)
            expected_map[0, 1] = expected_map[2, 3] = length / spec["nslice"]
            expected_map[4, 5] = length / spec["nslice"] / (result["before"]["gamma"] ** 2 - 1)
            expected_cov = expected_map @ expected_map.T
            error = float(np.max(abs(observed - expected))) if bad == 0 else None
            clock_error = abs(result["after"]["t"] - clock)
            assert clock_error < 3e-14, (spec, clock_error)
            assert result["after"]["gamma"] == result["before"]["gamma"]
            failed = (bad != 0 or map_bad != 0 or cov_bad != 0 or
                      error is None or error > 3e-16 or
                      not np.allclose(matrix, expected_map, rtol=0, atol=3e-15) or
                      not np.allclose(covariance, expected_cov, rtol=0, atol=3e-15))
            if spec["kind"] == "RFCavity" and spec["f"] == 0:
                failures.append(failed)
            else:
                assert not failed, (spec, error)
            print(f"{spec['name']}: observed={observed[0].tolist()} "
                  f"expected={expected[0].tolist()} particle_nonfinite={bad}/18 "
                  f"map_nonfinite={map_bad}/36 covariance_nonfinite={cov_bad}/36 "
                  f"expected_nonfinite=0 max_coordinate_error={error} "
                  f"clock_error={clock_error:.4g} m; expected from 90-digit free Hamiltonian flow, "
                  "R=I+drift shears and covariance=R Rᵀ", flush=True)
        print("DEFECT REPRODUCED" if any(failures) else "NOT REPRODUCED")


    if __name__ == "__main__":
        if len(sys.argv) > 1:
            print("RESULT " + json.dumps(child(json.loads(sys.argv[2]))))
        else:
            main()

    ```

## Provenance

| Role | Model | Effort | Date | Code examined |
|---|---|---|---|---|
| Found | `gpt-6-astra` | xhigh | 2026-11-06 | 26.10 (`615921c6867c`), impactx-noacc wheel, CPU NOACC, double, serial, SIMD 4, FFT |
| Confirmed | `gpt-6.1-sol` | xhigh | 2026-11-06 | 26.10 (`615921c6867c`), impactx-noacc wheel, CPU NOACC, double, serial, SIMD 4, FFT |

