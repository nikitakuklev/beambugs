"""Shared serial ImpactX/MAD-X harness for geometry and radiation reproducers."""

import hashlib
import importlib.metadata
import json
import math
import os
import subprocess
import sys
import tempfile
import warnings
from pathlib import Path

ROOT = Path(__file__).resolve().parent
COLS = ["position_x", "momentum_x", "position_y", "momentum_y", "position_t", "momentum_t"]
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
    raise RuntimeError("Launch on a single CPU, for example with taskset -c 3")
# Child processes inherit this single-CPU affinity.
(CPU,) = os.sched_getaffinity(0)

import numpy as np


def run(config):
    result = subprocess.run(
        [sys.executable, str(Path(__file__).resolve()), "child"],
        input=json.dumps(config),
        text=True,
        capture_output=True,
        timeout=290,
        check=False,
        cwd=ROOT,
    )
    if result.returncode:
        raise RuntimeError(result.stdout + result.stderr)
    rows = [line[7:] for line in result.stdout.splitlines() if line.startswith("RESULT ")]
    if len(rows) != 1:
        raise RuntimeError(result.stdout + result.stderr)
    return json.loads(rows[0])


def madx_check(deck, rays):
    from cpymad.madx import Madx

    # MAD-X T,PT are both the negatives of ImpactX t,pt.
    with (
        tempfile.TemporaryDirectory(dir=ROOT, prefix="madx-") as folder,
        Madx(stdout=False, cwd=folder) as mad,
    ):
        mad.input(deck)
        tw = mad.twiss(betx=2, bety=3, rmatrix=True)
        matrix = np.array([[tw[f"re{i + 1}{j + 1}"][-1] for j in range(6)] for i in range(6)])
        sign = np.diag([1, 1, 1, 1, -1, -1])
        survey = mad.survey()
        out = {
            # Element names without occurrence suffix and their exit positions.
            "names": [str(n).split(":")[0] for n in tw.name],
            "positions": [float(v) for v in tw.s],
            "s": float(tw.s[-1]),
            "matrix": (sign @ matrix @ sign).tolist(),
            "survey_theta": float(survey.theta[-1]),
        }
        mad.input("track,onepass=true,onetable=true;")
        for q in rays:
            mad.command.start(x=q[0], px=q[1], y=q[2], py=q[3], t=-q[4], pt=-q[5])
        mad.input("run,turns=1;endtrack;")
        table = mad.table.trackone
        selection = np.asarray(table.turn) == 1
        values = (
            np.column_stack(
                [np.asarray(table[k])[selection] for k in ("x", "px", "y", "py", "t", "pt")]
            )
            @ sign
        )
        out["rays"] = values[np.argsort(np.asarray(table.number)[selection])].tolist()
        return out


def native(config):
    from impactx import ImpactX, elements

    sim = ImpactX()
    try:
        settings = dict(
            verbose=0,
            diagnostics=False,
            slice_step_diagnostics=False,
            tiny_profiler=False,
            memory_profiler=False,
            particle_shape=2,
            space_charge=False,
            csr=False,
            isr=False,
            spin=False,
            strang_split=False,
        )
        if config["kind"] == "radiation":
            settings.update(
                isr=config.get("enabled", True),
                isr_order=config.get("order", 1),
                isr_on_ref_part=config.get("ref_loss", False),
                strang_split=config.get("strang", False),
            )
        for key, value in settings.items():
            setattr(sim, key, value)
        sim.init_grids()
        ref = sim.beam.ref
        gamma = config.get("gamma", 3.0)
        ref.set_species("electron").set_kin_energy_MeV((gamma - 1) * ref.mass_MeV)
        beta, bg, mass = ref.beta, ref.beta_gamma, ref.mass
        if config["kind"] != "radiation":
            q = np.zeros((3, 6))
            if config["kind"] == "sequence":
                q[1, :2] = [0.0002, 0.0007]
                q[2, 3] = -0.0003
            else:
                q[1, 0], q[2, 5] = 0.0008, 0.0002
            deck = config["deck"]
            native_result = madx_check(deck, q)
            with tempfile.TemporaryDirectory(dir=ROOT, prefix="deck-") as folder:
                path = Path(folder) / "case.madx"
                path.write_text(deck)
                with warnings.catch_warnings(record=True) as caught:
                    warnings.simplefilter("always")
                    sim.lattice.load_file(str(path), min_model=config.get("model", "linear"))
            if config.get("native_bend"):
                sim.lattice.clear()
                sim.lattice.append(elements.ThinDipole(theta=math.degrees(config["angle"]), rc=10))
            matrix = sim.lattice.transfer_map(ref).to_numpy().copy()
            records = sim.lattice.to_dicts()
            info = dict(
                native=native_result,
                matrix=matrix.tolist(),
                records=records,
                warnings=[str(w.message) for w in caught],
                initial=q.tolist(),
            )
        else:
            q = np.zeros((8192, 6))
            q[:, 0] = config["x"]
            radius, length = 10.0, 1.0
            edge = config.get("edge", 0.0)
            sim.lattice.append(elements.DipEdge(psi=edge, rc=radius, g=0, location="entry"))
            for _ in range(config.get("pieces", 1)):
                size = length / config.get("pieces", 1)
                if config.get("body", "Sbend") == "ExactSbend":
                    bend = elements.ExactSbend(
                        ds=size, phi=math.degrees(size / radius), nslice=config["n"]
                    )
                elif config.get("body") == "CFbend":
                    bend = elements.CFbend(ds=size, rc=radius, k=0, nslice=config["n"])
                else:
                    bend = elements.Sbend(ds=size, rc=radius, nslice=config["n"])
                sim.lattice.append(bend)
            sim.lattice.append(elements.DipEdge(psi=edge, rc=radius, g=0, location="exit"))
            info = {}
        sim.beam.add_n_particles(
            *q[:, [0, 2, 4, 1, 3, 5]].T.copy(), qm=ref.qm_ratio_SI, w=np.ones(len(q))
        )
        sim.track_particles()
        frame = sim.beam.to_df(local=True).reset_index(drop=True).sort_values("idcpu")
        final = frame[COLS].to_numpy()
        if config["kind"] == "radiation":
            pt = final[:, 5]
            info.update(
                mean_pt=float(pt.mean()),
                var_pt=float(pt.var(ddof=1)),
                pt_sha256=hashlib.sha256(pt.tobytes()).hexdigest(),
                max_pt=float(np.max(np.abs(pt))),
                mean_t=float(final[:, 4].mean()),
                mean_x=float(final[:, 0].mean()),
            )
        else:
            info["rays"] = final.tolist()
        info.update(
            beta=beta,
            bg=bg,
            mass=mass,
            s=ref.s,
            ct=ref.t,
            heading=math.atan2(ref.px, ref.pz),
            ref_gamma=ref.gamma,
        )
        return info
    finally:
        sim.finalize()


def environment():
    print(
        "environment",
        {
            name: importlib.metadata.version(name)
            for name in ("impactx-noacc", "cpymad", "numpy", "scipy")
        },
        f"CPU={CPU} numerical_threads=1 AMReX_seed=1",
    )


if __name__ == "__main__":
    if sys.argv[1:] != ["child"]:
        raise RuntimeError("Use an entry reproducer")
    print("RESULT " + json.dumps(native(json.load(sys.stdin))))
