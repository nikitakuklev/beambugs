"""API-17: CSR-only dashboard exports omit required particle_shape.

Runs the unchanged dashboard export modules with substituted UI construction. Export text runs in fresh children;
tracing disables profiling/diagnostics and records results before finalization.
Only explicit-shape controls edit the export. Writes no files.
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

import copy
import importlib.util
import json
import subprocess
import types
from collections import defaultdict
from pathlib import Path
from typing import Any

import amrex.space3d as amr
import impactx
import numpy as np

ROOT = Path(os.environ["IMPACTX_SOURCE"], "src/python/impactx/dashboard")
INSTALLED = Path(impactx.__file__).parent / "dashboard"


class State:
    def __init__(self):
        object.__setattr__(self, "data", {})
        object.__setattr__(self, "pending", set())
        object.__setattr__(self, "handlers", defaultdict(list))
        object.__setattr__(self, "events", [])

    def __getattr__(self, key):
        try:
            return self.data[key]
        except KeyError as exc:
            raise AttributeError(key) from exc

    def __setattr__(self, key, value):
        if key not in self.data or self.data[key] != value:
            self.pending.add(key)
        self.data[key] = value

    def update(self, values):
        for key, value in copy.deepcopy(values).items():
            setattr(self, key, value)

    def dirty(self, key):
        self.pending.add(key)

    def change(self, *keys):
        def register(func):
            for key in keys:
                self.handlers[key].append(func)
            return func

        return register

    def flush(self):
        # Deliver all registered server-side state handlers until no work remains.
        for _ in range(50):
            if not self.pending:
                return
            keys = sorted(self.pending)
            self.pending.clear()
            for key in keys:
                for func in self.handlers[key]:
                    self.events.append(func.__qualname__)
                    func(**{key: self.data[key]})
        raise RuntimeError("event harness did not reach quiescence")


class Controller:
    def add(self, key):
        def register(func):
            if hasattr(self, key):
                raise RuntimeError(f"duplicate controller {key}")
            setattr(self, key, func)
            return func

        return register


def package(name, **attributes):
    result = types.ModuleType(name)
    result.__path__ = []
    result.__dict__.update(attributes)
    sys.modules[name] = result
    return result


def load(relative, loaded):
    path = ROOT / relative
    # An installed source mismatch is an error; never silently switch versions.
    assert path.read_text() == (INSTALLED / relative).read_text(), relative
    name = "impactx.dashboard." + relative.removesuffix(".py").replace("/", ".")
    spec = importlib.util.spec_from_file_location(name, path)
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    loaded.append(relative)
    return module


def open_backend():
    state = State()
    ctrl = Controller()
    loaded = []
    package(
        "impactx.dashboard",
        state=state,
        ctrl=ctrl,
        html=types.SimpleNamespace(),
        vuetify=types.SimpleNamespace(VBtn=object),
    )
    input_pkg = package("impactx.dashboard.Input")
    package(
        "impactx.dashboard.Input.components",
        CardBase=object,
        CardComponents=object,
        InputComponents=object,
        NavigationComponents=object,
    )
    for name in (
        "Toolbar",
        "Toolbar.file_imports",
        "Toolbar.file_imports.python",
        "Input.lattice",
        "Input.distribution",
        "Input.space_charge",
        "Input.simulation_parameters",
        "Input.validation",
        "Input.visualization",
        "Input.visualization.lattice",
        "Run",
    ):
        package("impactx.dashboard." + name)
    package("impactx.dashboard.Toolbar.file_imports.python.parser", DashboardParser=object)
    load("Input/defaults_helper.py", loaded)
    defaults = load("Input/defaults.py", loaded)
    input_pkg.DashboardDefaults = defaults.DashboardDefaults
    state.update(defaults.DashboardDefaults.DEFAULT_VALUES)
    general = load("Input/utils.py", loaded)
    input_pkg.GeneralFunctions = general.GeneralFunctions
    validation = load("Input/validation/inputs.py", loaded)
    errors = load("Input/validation/errors_tracker.py", loaded)
    validation_pkg = sys.modules["impactx.dashboard.Input.validation"]
    validation_pkg.__dict__.update(
        DashboardValidation=validation.DashboardValidation,
        errors_tracker=errors.errors_tracker,
    )
    statistics = load("Input/visualization/lattice/statistics.py", loaded)
    sys.modules["impactx.dashboard.Input.visualization.lattice"].__dict__.update(
        update_lattice_statistics=statistics.update_lattice_statistics,
        StatComponents=statistics.LatticeVisualizerStatisticComponents,
    )
    load("Input/visualization/lattice/ui.py", loaded)
    load("Input/lattice/utils.py", loaded)
    variables = load("Input/lattice/variable_handler.py", loaded)
    ui = load("Input/lattice/ui.py", loaded)
    load("Input/distribution/utils.py", loaded)
    generator = load("Run/simulation.py", loaded)
    state.selected_distribution_parameters = {}
    for key in (
        "npart_error_message",
        "kin_energy_error_message",
        "bunch_charge_C_error_message",
        "charge_qe_error_message",
        "mass_MeV_error_message",
        "periods_error_message",
        "csr_bins_error_message",
        "error_message_n_cell_x",
        "error_message_n_cell_y",
        "error_message_n_cell_z",
        "error_message_blocking_factor_x",
        "error_message_blocking_factor_y",
        "error_message_blocking_factor_z",
    ):
        setattr(state, key, "")
    state.prob_relative_fields = [{"error_message": ""}]
    load("Input/space_charge/utils.py", loaded)
    load("Input/space_charge/ui.py", loaded)
    load("Input/simulation_parameters/ui.py", loaded)
    state.n_cell = [16, 16, 16]
    state.prob_relative = [1.1]
    state.flush()
    return types.SimpleNamespace(
        state=state,
        ctrl=ctrl,
        ui=ui,
        variables=variables.LatticeVariableHandler,
        generator=generator,
        loaded=loaded,
    )


def lengths(state):
    return [
        float(next(p["sim_input"] for p in el["parameters"] if p["parameter_name"] == "ds"))
        for el in state.selected_lattice_list
    ]


def prepare_drifts(b, count, distinct=False):
    s = b.state
    s.selected_lattice_list = []
    s.lattice_elements_using_variables = {}
    s.variables = [{"name": "L", "value": 0.125, "error_message": ""}]
    s.selected_lattice = "Drift"
    for index in range(count):
        b.ctrl.add_latticeElement()
        for key, value in (("name", f"d{index}" if distinct else "d"), ("ds", "0.125")):
            b.ctrl.updateLatticeElementParameters(
                index, key, value, "str" if key == "name" else "float"
            )
    s.flush()


COORDS = ("position_x", "momentum_x", "position_y", "momentum_y", "position_t", "momentum_t")


def run_export(script):
    namespace: dict[str, Any] = {"__name__": "__main__"}
    result: dict[str, Any] = {"error_type": None, "error": None}
    lines = script.splitlines()
    finalize_line = next(i for i, line in enumerate(lines, 1) if line.strip() == "sim.finalize()")
    init_line = next(i for i, line in enumerate(lines, 1) if line.strip() == "sim.init_grids()")
    configured_output = False
    finalized = False

    def observe(frame, event, arg):
        nonlocal configured_output
        if frame.f_code.co_filename == "<dashboard-export>" and event == "line":
            sim = frame.f_locals.get("sim")
            if sim is not None:
                if not configured_output:
                    sim.tiny_profiler = False
                    sim.memory_profiler = False
                    sim.diagnostics = False
                    sim.spin = False
                    sim.isr = False
                    configured_output = True
                if frame.f_lineno == init_line:
                    # The generator enables slice diagnostics; suppress file output
                    # through instrumentation without changing its exported source.
                    sim.slice_step_diagnostics = False
                if frame.f_lineno == finalize_line:
                    frame_data = sim.beam.to_df(local=True)
                    assert frame_data is not None and len(frame_data) == 24
                    assert np.isfinite(frame_data[list(COORDS)].to_numpy()).all()
                    result.update(
                        shape=sim.particle_shape, s=sim.beam.ref.s, particles=len(frame_data)
                    )
        return observe

    try:
        sys.settrace(observe)
        exec(compile(script, "<dashboard-export>", "exec"), namespace)  # noqa: S102 -- upstream export under review
        finalized = True
    except RuntimeError as exc:
        result.update(error_type=type(exc).__name__, error=str(exc))
    finally:
        sys.settrace(None)
        if "sim" in namespace and not finalized:
            namespace["sim"].finalize()
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


def verify_csr(b):
    prepare_drifts(b, 1, distinct=True)
    s = b.state
    s.variables = [{"name": "", "value": "", "error_message": ""}]
    s.lattice_elements_using_variables = {}
    s.distribution = "Gaussian"
    s.distribution_type = "Quadratic"
    s.selected_distribution_parameters = {
        name: {"value": value, "error_message": []}
        for name, value in dict(
            lambdaX=2e-4, lambdaY=3e-4, lambdaT=1e-3, lambdaPx=1e-5, lambdaPy=2e-5, lambdaPt=3e-5
        ).items()
    }
    s.npart = 24
    s.kin_energy_MeV = 75.0
    s.bunch_charge_C = 1e-12
    s.csr_bins = 32
    rows = []
    for csr, shape, explicit_shape, charge_mode in (
        (True, 1, False, "false"),
        (True, 3, False, "false"),
        (True, 1, True, "false"),
        (True, 3, True, "false"),
        (False, 2, False, "false"),
        (True, 2, False, "3D"),
    ):
        s.space_charge, s.csr, s.particle_shape = charge_mode, csr, shape
        s.flush()
        script = b.generator.dashboard_sim_inputs(is_exporting=True)
        emitted_shape = "sim.particle_shape =" in script
        if explicit_shape:
            script = script.replace(
                "sim.init_grids()", f"sim.particle_shape = {shape}\nsim.init_grids()"
            )
        name = f"csr_{int(csr)}_shape{shape}_explicit{int(explicit_shape)}_sc{charge_mode}"
        result = child("--export", script)
        if csr and charge_mode == "false" and not explicit_shape:
            assert emitted_shape is False
            assert result["error_type"] == "RuntimeError"
            assert (
                result["error"]
                == "particle_shape is not set, cannot initialize grids with guard cells for collective effects."
            )
        else:
            assert result["error_type"] is None
            assert result["shape"] == shape
            assert abs(result["s"] - 0.125) < 1e-15
        rows.append(dict(case=name, emitted_shape=emitted_shape, **result))
        print("CSR", json.dumps(rows[-1]), flush=True)
    return rows


if __name__ == "__main__":
    if sys.argv[1:] == ["--export"]:
        print("RESULT " + json.dumps(run_export(json.load(sys.stdin))))
    else:
        assert len(sys.argv) == 1
        provenance()
        print(
            "Contract: CSR requires selected particle_shape before init_grids; successful drift has s=0.125 m"
        )
        verify_csr(open_backend())
        print("DEFECT REPRODUCED", flush=True)
