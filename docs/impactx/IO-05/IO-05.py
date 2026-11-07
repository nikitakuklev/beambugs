"""IO-05: BeamMonitor additive position records omit the laboratory projection

With a BeamMonitor at a nonzero reference heading, reading the standard openPMD
position-plus-positionOffset records combines local transverse deviations with
laboratory reference coordinates. The sum omits the rotation required to obtain
laboratory particle positions. In a 0.2000 rad bend example, the horizontal
separation is 12.51 mm from the records instead of the geometrically correct
12.26 mm, a 2.034% error. Raw local coordinates, reference attributes, tracking
maps, and aligned straight-line output are correct.

Setup: run pinned to one CPU (`taskset -c <n>`), with `taskset` on `PATH`; the script refuses to run
on more and pins its child processes to the same CPU.

Eight deterministic native cases run in fresh processes on one pinned CPU. A sector
Hamiltonian oscillator and independently integrated reference tangent provide
expected local coordinates and laboratory geometry. Straight, opposite-bend,
tilted-reference, closed-heading, encoding and metadata-only controls isolate
the geometry-output discrepancy. Temporary HDF5 files and child results live
beside the script and are removed on exit.
"""

import json
import math
import os
import subprocess
import sys
from dataclasses import asdict, dataclass
from pathlib import Path
from tempfile import TemporaryDirectory

HERE = Path(__file__).resolve().parent
THREADS = (
    "OMP_NUM_THREADS",
    "MKL_NUM_THREADS",
    "OPENBLAS_NUM_THREADS",
    "NUMEXPR_NUM_THREADS",
    "NPROC",
    "PJRT_NPROC",
)
for name in THREADS:
    os.environ[name] = "1"
os.environ["PYTHONDONTWRITEBYTECODE"] = "1"
if len(os.sched_getaffinity(0)) != 1:
    raise RuntimeError("Launch pinned to one CPU, for example taskset -c 5")


@dataclass(frozen=True)
class Case:
    angles: tuple[float, ...]
    heading: float = 0.0
    encoding: str = "g"
    particles: bool = True


CASES = {
    "straight": Case((0.0,)),
    "bend_positive": Case((0.6,)),
    "bend_negative": Case((-0.6,)),
    "bend_small": Case((0.2,)),
    "tilted_straight": Case((0.0,), heading=0.6),
    "closed_heading": Case((0.6, -0.6)),
    "file_encoding": Case((0.6,), encoding="f"),
    "metadata_only": Case((0.6,), particles=False),
}


def native(case, directory):
    import importlib.metadata

    import h5py
    import impactx
    import numpy as np
    from impactx import ImpactX, RefPart, elements

    spec = CASES[case]
    config = impactx.Config.to_dict()
    if any(config[k] for k in ("have_mpi", "have_gpu", "have_omp")):
        raise RuntimeError("This reproducer requires a serial CPU build")
    if config["precision"] != "DOUBLE" or config["precision_particles"] != "DOUBLE":
        raise RuntimeError("This reproducer requires double precision")
    if not config["have_openpmd"] or not config["openpmd_backends"]["hdf5"]:
        raise RuntimeError("This reproducer requires openPMD with HDF5")
    sim = ImpactX()
    monitor = None
    try:
        for name, value in dict(
            tiny_profiler=False,
            memory_profiler=False,
            verbose=0,
            particle_shape=1,
            space_charge="off",
            csr=False,
            isr=False,
            spin=False,
            diagnostics=False,
            slice_step_diagnostics=False,
            diag_file_prefix=str(directory),
        ).items():
            setattr(sim, name, value)
        sim.init_grids()
        gamma = 4.0
        beta = math.sqrt(1.0 - 1.0 / gamma**2)
        ref = RefPart().set_species("proton")
        ref.set_kin_energy_MeV((gamma - 1.0) * ref.mass_MeV)
        heading = spec.heading
        bg = math.sqrt(gamma**2 - 1.0)
        ref.x, ref.y, ref.z, ref.t = 0.12, -0.04, 0.25, 0.08
        ref.px = -bg * math.sin(heading)
        ref.pz = bg * math.cos(heading)
        sim.beam.set_ref_particle(ref)
        u = np.linspace(-1.0, 1.0, 12)
        # Deterministic paired coordinates; no longitudinal energy spread.
        phase = np.column_stack(
            (0.006 * u, 0.0003 * u, 0.002 * u, -0.0001 * u, 0.0007 * u, np.zeros_like(u))
        )
        sim.beam.add_n_particles(
            **dict(zip(("x", "px", "y", "py", "t", "pt"), phase.T, strict=True)),
            qm=ref.qm_ratio_SI,
            w=np.ones_like(u),
        )
        lattice = [
            elements.Drift(ds=1.25)
            if angle == 0.0
            else elements.Sbend(ds=1.25, rc=1.25 / angle, nslice=3)
            for angle in spec.angles
        ]
        monitor = elements.BeamMonitor(
            "contract",
            backend="h5",
            encoding=spec.encoding,
            particles=spec.particles,
            beam_moments=False,
        )
        sim.lattice.extend([*lattice, monitor])
        sim.track_particles()
        columns = [
            "position_x",
            "momentum_x",
            "position_y",
            "momentum_y",
            "position_t",
            "momentum_t",
        ]
        frame = sim.beam.to_df(local=True).reset_index(drop=True)
        memory = frame.sort_values("idcpu")[columns].to_numpy()
        monitor.finalize()
        files = list((directory / "openPMD").glob("*.h5"))
        if len(files) != 1:
            raise RuntimeError(f"Expected one output file: {files}")
        result = {
            "case": case,
            "spec": asdict(spec),
            "gamma": gamma,
            "beta": beta,
            "initial": phase.tolist(),
            "memory": memory.tolist(),
            "affinity": sorted(os.sched_getaffinity(0)),
            "threads": {name: os.environ[name] for name in THREADS},
            "python": sys.version,
            "numpy": np.__version__,
            "h5py": h5py.__version__,
            "impactx_noacc": importlib.metadata.version("impactx-noacc"),
            "impactx_module": impactx.__file__,
            "config": impactx.Config.to_dict(),
        }
        with h5py.File(files[0], "r") as file:
            (iteration,) = file["data"].values()
            beam = iteration["particles/beam"]
            result["openPMD"] = file.attrs["openPMD"].decode()
            result["openPMDextension"] = int(file.attrs["openPMDextension"])
            assert result["openPMD"] == "1.1.0" and result["openPMDextension"] == 0
            result["reference"] = {
                name: float(beam.attrs[name + "_ref"])
                for name in ("x", "y", "z", "t", "px", "py", "pz", "pt", "s")
            }
            result["records"] = sorted(beam.keys())
            if spec.particles:
                indices = np.argsort(beam["id"][:])
                records = {}
                for record in ("position", "positionOffset", "momentum"):
                    records[record] = {}
                    for axis, node in beam[record].items():
                        values = (
                            np.full(12, node.attrs["value"])
                            if "value" in node.attrs
                            else node[:][indices]
                        )
                        records[record][axis] = (values * node.attrs["unitSI"]).tolist()
                result["data"] = records
                for record in ("position", "positionOffset"):
                    assert np.array_equal(
                        beam[record].attrs["unitDimension"], [1, 0, 0, 0, 0, 0, 0]
                    )
                    assert all(node.attrs["unitSI"] == 1 for node in beam[record].values())
                result["units"] = {
                    record: beam[record].attrs["unitDimension"].tolist() for record in records
                }
        (directory / "result.json").write_text(json.dumps(result, indent=2) + "\n")
    finally:
        if monitor is not None:
            monitor.finalize()
        sim.finalize()


def check(case, directory):
    import numpy as np
    from scipy.integrate import solve_ivp

    result = json.loads((directory / "result.json").read_text())
    spec = result["spec"]
    phase = np.array(result["initial"])
    reference = np.array([0.12, -0.04, 0.25, 0.08])
    heading = spec.get("heading", 0.0)
    beta = result["beta"]
    # At pt=0, the quadratic sector Hamiltonian is
    # H=(px²+py²+h²*x²)/2 + h*x*pt/beta + O(pt²).
    # Its flow is an oscillator in x and a drift in y. The time coordinate
    # follows t'=h*x/beta. No native map or returned transport matrix is used.
    for angle in spec["angles"]:
        length = 1.25
        h = angle / length
        if h == 0.0:
            phase[:, 0] += length * phase[:, 1]
            phase[:, 2] += length * phase[:, 3]
        else:
            x, px = phase[:, 0].copy(), phase[:, 1].copy()
            c, s = math.cos(angle), math.sin(angle)
            phase[:, 0] = c * x + s / h * px
            phase[:, 1] = -h * s * x + c * px
            phase[:, 2] += length * phase[:, 3]
            phase[:, 4] += s / beta * x + (1 - c) / (h * beta) * px

        # Integrate r_ref'=(-sin psi, 0, cos psi), (ct_ref)'=1/beta
        # and psi'=h independently of the reference-particle implementation.
        def rhs(_s, state, curvature=h):
            return [-math.sin(state[4]), 0.0, math.cos(state[4]), 1 / beta, curvature]

        solution = solve_ivp(
            rhs,
            (0.0, length),
            [*reference, heading],
            method="DOP853",
            rtol=2.3e-14,
            atol=2e-15,
        )
        if not solution.success:
            raise RuntimeError(solution.message)
        reference, heading = solution.y[:4, -1], float(solution.y[4, -1])
    stored_ref = result["reference"]
    ref_observed = np.array([stored_ref[a] for a in ("x", "y", "z", "t")])
    ref_error = float(np.max(np.abs(ref_observed - reference)))
    bg = math.sqrt(result["gamma"] ** 2 - 1)
    expected_p = np.array([-bg * math.sin(heading), 0, bg * math.cos(heading), -result["gamma"]])
    momentum_error = float(
        np.max(np.abs(np.array([stored_ref[a] for a in ("px", "py", "pz", "pt")]) - expected_p))
    )
    assert ref_error < 4e-15, (case, ref_error)
    assert momentum_error < 4e-15, (case, momentum_error)
    memory = np.array(result["memory"])
    local_error = float(np.max(np.abs(memory - phase)))
    assert local_error < 4e-17, (case, local_error)
    metrics = dict(
        case=case,
        heading=heading,
        local_error=local_error,
        reference_error=ref_error,
        reference_momentum_error=momentum_error,
    )
    if spec.get("particles", True):
        data = result["data"]
        raw = np.column_stack(
            [
                data[r][a]
                for r, a in (
                    ("position", "x"),
                    ("momentum", "x"),
                    ("position", "y"),
                    ("momentum", "y"),
                    ("position", "t"),
                    ("momentum", "t"),
                )
            ]
        )
        assert np.array_equal(raw, memory), case
        offset = np.column_stack([data["positionOffset"][a] for a in ("x", "y", "t")])
        assert np.array_equal(offset, np.tile([stored_ref[a] for a in ("x", "y", "t")], (12, 1))), (
            case
        )
        summed_x = np.array(data["position"]["x"]) + offset[:, 0]
        # The local horizontal basis at the monitor is (cos psi,0,sin psi).
        # Adding the reference origin changes the origin, not these axes.
        # Reference geometry was independently checked above; use its stored
        # origin here to isolate the rotation from numerical quadrature error.
        laboratory_x = stored_ref["x"] + math.cos(heading) * phase[:, 0]
        residual = float(np.max(np.abs(summed_x - laboratory_x)))
        predicted = float(np.max(np.abs((1 - math.cos(heading)) * phase[:, 0])))
        assert abs(residual - predicted) < 5e-16, (case, residual, predicted)
        gap_sum = float(summed_x[-1] - summed_x[0])
        gap_lab = float(laboratory_x[-1] - laboratory_x[0])
        # The raw local coordinates and reference attributes suffice for a
        # correct laboratory reconstruction when the basis is explicitly
        # rotated; the standard component-wise sum does not perform it.
        recovered_lab = stored_ref["x"] + math.cos(heading) * (summed_x - offset[:, 0])
        assert np.max(np.abs(recovered_lab - laboratory_x)) < 5e-16
        metrics.update(
            raw_memory_error=0.0,
            offsets_reference_error=0.0,
            sum_gap=gap_sum,
            laboratory_gap=gap_lab,
            laboratory_interpretation_error=residual,
            laboratory_error_predicted=predicted,
            relative_gap_error=gap_sum / gap_lab - 1.0,
        )
    else:
        assert result["records"] == ["empty"], result["records"]
        metrics["position_records"] = False
    if spec["particles"]:
        print(
            f"{case}: heading={heading:.6g} observed_gap={gap_sum:.17g} m "
            f"expected_gap={gap_lab:.17g} m relative_error={gap_sum / gap_lab - 1:.17g} "
            f"max_X_error={residual:.17g} m local_error={local_error:.3g} "
            f"reference_error={ref_error:.3g} m raw_memory_error=0 offsets_reference_error=0; "
            "expected X=X_ref+cos(heading)*x from the exit transverse basis",
            flush=True,
        )
    else:
        print(
            f"{case}: observed_position_records=False expected=False "
            f"local_error={local_error:.3g} reference_error={ref_error:.3g} m; "
            "particles=False writes reference metadata only",
            flush=True,
        )
    return metrics


def main():
    metrics = []
    with TemporaryDirectory(prefix="io-05-", dir=HERE) as temporary:
        for case in CASES:
            directory = Path(temporary) / case
            directory.mkdir()
            child = subprocess.run(
                [
                    "taskset",
                    "-c",
                    str(next(iter(os.sched_getaffinity(0)))),
                    sys.executable,
                    str(Path(__file__).resolve()),
                    "--native",
                    case,
                    str(directory),
                ],
                cwd=directory,
                env=os.environ.copy(),
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                check=False,
                timeout=290,
            )
            if child.returncode:
                raise RuntimeError(f"Native case {case} failed:\n{child.stdout.decode()}")
            if not metrics:
                result = json.loads((directory / "result.json").read_text())
                import scipy

                print(
                    f"Python {sys.version.split()[0]} impactx-noacc={result['impactx_noacc']} "
                    f"numpy={result['numpy']} scipy={scipy.__version__} h5py={result['h5py']}; "
                    "CPU, double, serial, SIMD width 4, HDF5, openPMD 1.1.0; "
                    "12 fixed particles per case, no random seed",
                    flush=True,
                )
            metrics.append(check(case, directory))
    by_case = {item["case"]: item for item in metrics}
    assert by_case["straight"]["laboratory_interpretation_error"] < 5e-16
    assert by_case["closed_heading"]["laboratory_interpretation_error"] < 5e-16
    failures = ("bend_positive", "bend_negative", "bend_small", "tilted_straight", "file_encoding")
    reproduced = any(by_case[name]["laboratory_interpretation_error"] > 5e-16 for name in failures)
    print("DEFECT REPRODUCED" if reproduced else "NOT REPRODUCED", flush=True)


if __name__ == "__main__":
    if len(sys.argv) == 4 and sys.argv[1] == "--native":
        native(sys.argv[2], Path(sys.argv[3]))
    elif len(sys.argv) == 1:
        main()
    else:
        raise ValueError("Use no arguments, or --native CASE DIRECTORY")
