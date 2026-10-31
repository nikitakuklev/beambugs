---
title: "API-16: Dashboard variable refresh leaves equal elements at stale lengths"
hide:
  - toc
---

# API-16 — Dashboard variable refresh leaves equal elements at stale lengths

| | |
|---|---|
| Code | [ImpactX](../index.md) · Python API |
| Priority | P3 |
| Severity | Silent wrong result |
| Evidence | Reproduced, independently rechecked |
| Last reproduced | 2026-10-29 on ImpactX development (`fa0a43b`) |
| Tags | <span class="bb-tag">partly-source-only</span> [?](../../methodology.md#tags) |
| Source | [`src/python/impactx/dashboard/Input/lattice/variable_handler.py#L128-L147`](https://github.com/BLAST-ImpactX/impactx/blob/615921c6867c895cf3a8fcf1d12991079d0f7d75/src/python/impactx/dashboard/Input/lattice/variable_handler.py#L128-L147)<br>[`src/python/impactx/dashboard/Input/lattice/ui.py#L137-L178`](https://github.com/BLAST-ImpactX/impactx/blob/615921c6867c895cf3a8fcf1d12991079d0f7d75/src/python/impactx/dashboard/Input/lattice/ui.py#L137-L178) |

In the ImpactX dashboard, when several equal elements, such as identical drifts, have their lengths bound to one lattice variable, changing the variable can leave some of them at the old length, because the update finds elements by value equality instead of identity. Every bound parameter must take the variable's current value, so N drifts bound to a length L total N·L. For eight equal drifts bound in reverse order and changed from 0.125 m to 0.375 m, the exported lattice is 2.5 m long instead of 3 m, and a test particle at γ = 3.5 exits at x = −1.600 mm instead of −1.700 mm, with no error shown. Binding in index order or giving the drifts distinct names avoids it, and two further explicit refreshes restore the lengths.

??? example "Reproducer: API-16.py"

    [Download API-16.py](API-16.py)

    ```python
    """API-16: Dashboard variable refresh leaves equal elements at stale lengths

    In the ImpactX dashboard, when several equal elements, such as identical drifts, have their lengths
    bound to one lattice variable, changing the variable can leave some of them at the old length,
    because the update finds elements by value equality instead of identity. Every bound parameter must
    take the variable's current value, so N drifts bound to a length L total N·L. For eight equal drifts
    bound in reverse order and changed from 0.125 m to 0.375 m, the exported lattice is 2.5 m long
    instead of 3 m, and a test particle at γ = 3.5 exits at x = −1.600 mm instead of −1.700 mm, with no
    error shown. Binding in index order or giving the drifts distinct names avoids it, and two further
    explicit refreshes restore the lengths.

    Set IMPACTX_SOURCE to a clone of the ImpactX repository. The script loads fifteen unmodified
    dashboard modules from that clone, each checked equal to the installed copy, substitutes only the
    Trame UI construction and a state object that delivers every registered server-side handler, and
    drives the dashboard controls to create equal drifts, bind their lengths to a variable L and change
    L. It exports each lattice with the dashboard's generator, tracks twelve particles at γ = 3.5
    through it in a fresh child process and compares them with N drifts of the new length. Eight drifts
    bound in reverse order, and five and four with L decreased and increased, show the bug; forward
    binding order and distinct names are the controls.
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
    import math
    import subprocess
    import types
    from collections import defaultdict
    from pathlib import Path

    import amrex.space3d as amr
    import impactx
    import numpy as np
    from impactx import ImpactX, elements

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
        try:
            namespace = {"elements": elements}
            exec(config["lattice_source"], namespace)  # noqa: S102 -- unchanged upstream generator output
            sim.lattice.extend(namespace["lattice_configuration"])
            sim.track_particles()
            reference = snapshot(sim.beam.ref)
            frame = sim.beam.to_df(local=True).reset_index(drop=True).sort_values("idcpu")
            final = frame[list(COORDS)].to_numpy()
            assert final.shape == (12, 6) and np.isfinite(final).all()
            return {
                "start": start,
                "reference": reference,
                "initial": initial.tolist(),
                "final": final.tolist(),
            }
        finally:
            sim.finalize()


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


    def verify_variables(b):
        rows = []
        for count, order_kind, distinct, new_value in (
            (8, "reverse", False, 0.375),
            (8, "forward", False, 0.375),
            (8, "reverse", True, 0.375),
            (5, "reverse", False, 0.0625),
            (4, "reverse", False, 0.375),
        ):
            prepare_drifts(b, count, distinct)
            order = list(range(count))
            if order_kind == "reverse":
                order.reverse()
            for index in order:
                b.ctrl.updateLatticeElementParameters(index, "ds", "L", "float")
                b.state.flush()
            assert lengths(b.state) == [0.125] * count
            assert (
                sum(
                    item["parameter_name"] == "ds"
                    for item in b.state.lattice_elements_using_variables.values()
                )
                == count
            )
            identities = [id(el) for el in b.state.selected_lattice_list]
            assert len(set(identities)) == count
            if not distinct:
                assert all(
                    el == b.state.selected_lattice_list[0] for el in b.state.selected_lattice_list
                )
            b.state.events.clear()
            b.ctrl.update_variable("value", 0, str(new_value))
            after_controller = lengths(b.state)
            b.state.flush()
            after_flush = lengths(b.state)
            assert b.state.disableRunSimulationButton is False
            events = b.state.events.copy()
            # A native transport check consumes the exact upstream generator output.
            name = f"variables_n{count}_{order_kind}_{'distinct' if distinct else 'equal'}_{new_value}"
            generated = b.generator.build_lattice_list()
            native_result = child("--native", dict(gamma=3.5, lattice_source=generated))
            initial = np.array(native_result["initial"])
            expected_length = count * new_value
            expected = initial.copy()
            expected[:, 0] += expected_length * initial[:, 1]
            expected[:, 2] += expected_length * initial[:, 3]
            expected[:, 4] += expected_length / (3.5**2 - 1) * initial[:, 5]
            error = float(np.max(abs(np.array(native_result["final"]) - expected)))
            generated_length = sum(after_flush)
            generated_oracle = initial.copy()
            generated_oracle[:, 0] += generated_length * initial[:, 1]
            generated_oracle[:, 2] += generated_length * initial[:, 3]
            generated_oracle[:, 4] += generated_length / (3.5**2 - 1) * initial[:, 5]
            generated_error = float(np.max(abs(np.array(native_result["final"]) - generated_oracle)))
            assert generated_error < 3e-18
            assert abs(native_result["reference"]["s"] - generated_length) < 1e-14
            if order_kind == "reverse" and not distinct:
                assert after_flush != [new_value] * count
                assert error > 1e-6
            else:
                assert after_flush == [new_value] * count
                assert error < 3e-18
            repaired = []
            for _ in range(4):
                b.variables.update_lattice_parameter_bindings()
                b.state.flush()
                repaired.append(lengths(b.state))
            assert repaired[-1] == [new_value] * count
            row = dict(
                case=name,
                after_controller=after_controller,
                after_event_flush=after_flush,
                expected_length=expected_length,
                actual_length=native_result["reference"]["s"],
                expected_x0=expected[0, 0],
                actual_x0=native_result["final"][0][0],
                max_coordinate_error=error,
                generated_lattice_oracle_error=generated_error,
                subsequent_refreshes=repaired,
                events=events,
                registry_count=len(b.state.lattice_elements_using_variables),
            )
            rows.append(row)
            print("VARIABLES", json.dumps(row), flush=True)
        return rows


    if __name__ == "__main__":
        if sys.argv[1:] == ["--native"]:
            print("RESULT " + json.dumps(run(json.load(sys.stdin))))
        else:
            assert len(sys.argv) == 1
            provenance()
            print(
                "Oracle: shared L requires total length N*L; x_f=x_i+N*L*px_i, y_f=y_i+N*L*py_i, t_f=t_i+N*L*pt_i/(gamma^2-1)"
            )
            verify_variables(open_backend())
            print("DEFECT REPRODUCED", flush=True)

    ```

## Provenance

| Role | Model | Effort | Date | Code examined |
|---|---|---|---|---|
| Found | `gpt-6-astra` | xhigh | 2026-10-29 | 26.10 (`615921c6867c`), impactx-noacc wheel, CPU, double, serial |
| Confirmed | `gpt-6.1-sol` | xhigh | 2026-10-29 | 26.10 (`615921c6867c`), impactx-noacc wheel, CPU, double, serial |

