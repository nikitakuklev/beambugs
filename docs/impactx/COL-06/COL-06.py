"""COL-06: 3D FFT integrated Green function loses far-cell accuracy

With the 3D FFT space-charge solver, cancellation in the integrated Green function can reverse the field on highly anisotropic meshes. On a 128³-cell mesh covering 1.280 mm × 1.280 µm × 1.280 mm, a symmetric eight-charge source carrying 1.000 nC at reference Lorentz factor γ₀ = 2 gives −23.23 V/m vertically where independent cell-volume quadrature and a 90-digit prism integral give +64.30 V/m, a 136.1% vector error. Both references include the implemented centered field stencil. An ordinary cubic-mesh control agrees within 3.562×10⁻¹⁰ relative.

Setup: `taskset` on `PATH` and CPU 0 among the allowed CPUs; the script runs each case in a child process pinned to CPU 0.

The script runs four public FFT solves with fixed positive sources and reads every nonzero deposited source cell. Independent 12- and 20-point Gauss–Legendre volume rules integrate a rationalized centered field stencil; a 90-digit logarithmic prism integral checks the wrong-sign case. A cubic mesh provides a control, and each serial child owns and finalizes one simulation.
"""

import json
import math
import os
import subprocess
import sys
from importlib.metadata import version
from itertools import product
from pathlib import Path

for _key in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS",
             "NUMEXPR_NUM_THREADS", "NPROC", "PJRT_NPROC"):
    os.environ[_key] = "1"
os.sched_setaffinity(0, {0})

import mpmath as mp
import numpy as np

KE = 1 / (4 * math.pi * 8.8541878188e-12)
Q = 1e-9
mp.mp.dps = 90
MP_KE = 1 / (4 * mp.mpf(math.pi) * mp.mpf(8.8541878188e-12))


def child(spec):
    from amrex import space3d as amr
    from impactx import Config, ImpactX, elements

    sim = ImpactX()
    try:
        n = spec["n"]
        cell = np.array(spec["cell"])
        for k, v in dict(
            n_cell=[n] * 3,
            max_grid_size_x=[n],
            max_grid_size_y=[n],
            max_grid_size_z=[n],
            particle_shape=1,
            space_charge="3D",
            poisson_solver="fft",
            strang_split=False,
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
        sim.domain = amr.RealBox((-n * cell / 2).tolist(), (n * cell / 2).tolist())
        sim.init_grids()
        ref = sim.beam.ref
        ref.set_charge_qe(1).set_mass_MeV(938.27208816).set_kin_energy_MeV(938.27208816)
        xyz = np.array([[-n / 2 + 1] * 3, [-n / 2 + 2] * 3]) * cell
        if spec.get("symmetric"):
            xyz = np.array(list(product(*[(-n / 2 + 1, n / 2 - 1)] * 3))) * cell
        phase = np.zeros((len(xyz), 6))
        phase[:, :2] = xyz[:, :2]
        phase[:, 2] = -xyz[:, 2] / ref.beta
        sim.beam.add_n_particles(
            *[v.copy() for v in phase.T],
            qm=ref.qm_ratio_SI,
            w=np.full(len(xyz), Q / len(xyz) / ref.charge),
        )
        sim.lattice.append(elements.Drift(ds=1e-12))
        sim.track_particles()

        def array(mf):
            blocks = mf.to_numpy()
            assert len(blocks) == 1, len(blocks)
            g = tuple(mf.n_grow_vect)
            return blocks[0][tuple(slice(a, -a if a else None) for a in g) + (0,)]

        rho = array(sim.rho(0))
        sources = np.argwhere(rho != 0)
        targets = [[n - 3, n - 5, n - 1], [n // 2, n // 2, n // 2], [n // 4, n // 3, n // 2]]
        if spec.get("symmetric"):
            targets = [
                [n // 2, n // 2 + 1, n // 2],
                [n // 2, n // 2 + 8, n // 2],
                [n // 2, n // 2 + 36, n // 2],
            ]
        phi = array(sim.phi(0))
        f = [array(sim.space_charge_field(0, c)) for c in ["x", "y", "z"]]
        return dict(
            spec=spec,
            config=Config.to_dict(),
            sources=[[idx.tolist(), float(rho[tuple(idx)])] for idx in sources],
            targets=[
                dict(
                    index=idx, phi=float(phi[tuple(idx)]), field=[float(ff[tuple(idx)]) for ff in f]
                )
                for idx in targets
            ],
            gamma=ref.gamma,
            beta=ref.beta,
            integrated_charge=float(np.sum(rho) * math.prod(cell)),
        )
    finally:
        sim.finalize()


def quadrature(center, delta, n):
    nodes, weights = np.polynomial.legendre.leggauss(n)
    offsets = np.array(list(product(nodes, repeat=3))) * np.array(delta) / 2
    w = np.prod(np.array(list(product(weights, repeat=3))), axis=1) / 8
    d = np.array(center) - offsets
    r = np.linalg.norm(d, axis=1)
    phi = KE * Q * math.fsum((w / r).tolist())
    fields = []
    for axis, step in enumerate(delta):
        lo, hi = d.copy(), d.copy()
        lo[:, axis] -= step
        hi[:, axis] += step
        rl, rh = np.linalg.norm(lo, axis=1), np.linalg.norm(hi, axis=1)
        # (1/rl - 1/rh)/(2h) = 2*x / (rl*rh*(rl+rh)).
        f = 2 * d[:, axis] / (rl * rh * (rl + rh))
        fields.append(KE * Q * math.fsum((w * f).tolist()))
    return phi, np.array(fields)


def primitive(v):
    x, y, z = v
    r = mp.sqrt(x * x + y * y + z * z)
    # An alternate logarithmic antiderivative of 1/r; the extra gauge terms
    # relative to the asinh representation cancel in the triple difference.
    return (
        x * y * mp.log(z + r)
        + y * z * mp.log(x + r)
        + z * x * mp.log(y + r)
        - (
            x * x * mp.atan(y * z / (x * r))
            + y * y * mp.atan(x * z / (y * r))
            + z * z * mp.atan(x * y / (z * r))
        )
        / 2
    )


def potential(center, delta):
    center = [abs(c) for c in center]
    return MP_KE * mp.fsum(
        math.prod(sign)
        * primitive([c + s * d / 2 for c, s, d in zip(center, sign, delta, strict=True)])
        for sign in product([-1, 1], repeat=3)
    )


def main():
    print(f"Python {sys.version.split()[0]} impactx-noacc={version('impactx-noacc')} "
          f"numpy={np.__version__} mpmath={mp.__version__}; "
          "CPU 0, double, serial; fixed positive sources, no random seed", flush=True)
    failures = []
    for name, spec in [
        ("cubic", dict(n=32, cell=[0.001] * 3)),
        ("thin_y", dict(n=128, cell=[0.001, 0.000001, 0.001])),
        ("thin_z", dict(n=128, cell=[0.001, 0.001, 0.000001])),
        ("symmetric_flat", dict(n=128, cell=[1e-5, 1e-8, 1e-5], symmetric=True)),
    ]:
        r = subprocess.run(
            ["taskset", "-c", "0", sys.executable, str(Path(__file__).resolve()),
             "--child", json.dumps(spec)],
            check=True, capture_output=True, text=True, timeout=290,
        )
        d = json.loads(next(line[7:] for line in r.stdout.splitlines() if line.startswith("RESULT ")))
        cell = np.array(spec["cell"])
        # ABLASTR stretches the longitudinal cell; ImpactX differentiates in lab spacing.
        stretch = np.array([1.0, 1.0, 1 / math.sqrt(1 - d["beta"] ** 2)])
        delta = cell * stretch
        assert abs(d["integrated_charge"] / Q - 1) < 1e-13
        assert abs(d["gamma"] - 2) < 1e-15
        for target_number, target in enumerate(d["targets"]):
            contributions = []
            mp_field = [mp.mpf(0)] * 3
            mp_absolute = [mp.mpf(0)] * 3
            check_prism = name == "symmetric_flat" and target_number == 0
            for idx, rho in d["sources"]:
                center = (np.array(target["index"]) - idx) * delta
                p20, f20 = quadrature(center, delta, 20)
                p12, f12 = quadrature(center, delta, 12)
                assert abs(p20 / p12 - 1) < 3e-15
                assert np.max(abs((f20 - f12) / f20)) < 4e-15
                # Quadrature returns the cell-averaged field of Q; actual rho is authoritative.
                factor = rho * math.prod(delta) / Q
                contributions.append(f20 * factor * stretch)
                if check_prism:
                    c, h = list(map(mp.mpf, center)), list(map(mp.mpf, delta))
                    for axis in range(3):
                        lo, hi = c.copy(), c.copy()
                        lo[axis] -= h[axis]
                        hi[axis] += h[axis]
                        value = mp.mpf(rho) * (potential(lo, h) - potential(hi, h)) / (2 * mp.mpf(cell[axis]))
                        mp_field[axis] += value
                        mp_absolute[axis] += abs(value)
            expected = np.array([math.fsum(v[k] for v in contributions) for k in range(3)])
            observed = np.array(target["field"])
            assert np.all(np.isfinite(observed))
            vector_error = float(np.linalg.norm(observed - expected) / np.linalg.norm(expected))
            extra = ""
            if check_prism:
                discrepancy = max(abs(float(mp_field[k]) - expected[k]) / float(mp_absolute[k])
                                  for k in range(3))
                assert discrepancy < 4e-15, discrepancy
                condition = float(mp_absolute[1] / abs(mp_field[1]))
                extra = (f" prism90_Ey={float(mp_field[1]):.12g} V/m "
                         f"oracle_difference_over_sum_abs={discrepancy:.4g} "
                         f"physical_Ey_condition={condition:.8g}")
            if name == "cubic":
                assert vector_error < 1e-8, (target, vector_error)
            else:
                failures.append(vector_error > 1e-6)
            print(f"{name} node={target['index']}: observed_field={observed.tolist()} V/m "
                  f"expected_field={expected.tolist()} V/m vector_relative_error={vector_error:.12g} "
                  f"deposited_nodes={len(d['sources'])}; expected from uniform-cell volume "
                  f"quadrature of the implemented centered stencil{extra}", flush=True)
    print("DEFECT REPRODUCED" if any(failures) else "NOT REPRODUCED")


if __name__ == "__main__":
    if len(sys.argv) > 1:
        print("RESULT " + json.dumps(child(json.loads(sys.argv[2]))))
    else:
        main()
