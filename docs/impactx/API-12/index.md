---
title: "API-12: Dashboard reimports envelope and reference exports as particle tracking"
hide:
  - toc
---

# API-12 — Dashboard reimports envelope and reference exports as particle tracking

| | |
|---|---|
| Code | [ImpactX](../index.md) · Python API |
| Priority | P4 |
| Severity | Crash or API hazard |
| Evidence | Reproduced, independently rechecked |
| Last reproduced | 2026-10-29 on ImpactX development (`fa0a43b`, head build) |
| Tags | <span class="bb-tag">partly-source-only</span> [?](../../methodology.md#tags) |
| Source | [`helper.py#L35-L63`](https://github.com/BLAST-ImpactX/impactx/blob/b701a6e9d1f9e30cefb9a3e18e82ae7e02b9d943/src/python/impactx/dashboard/Toolbar/file_imports/python/helper.py#L35-L63)<br>[`simulation.py#L172-L213`](https://github.com/BLAST-ImpactX/impactx/blob/b701a6e9d1f9e30cefb9a3e18e82ae7e02b9d943/src/python/impactx/dashboard/Run/simulation.py#L172-L213) |

A simulation exported from the ImpactX dashboard in envelope or reference tracking mode re-imports as particle tracking: the export writes only the tracking call (`track_envelope()` or `track_reference(ref)`), and the import does not read it and keeps its particle-tracking default. The import is meant to restore the configuration the file describes, and that call identifies the tracking model directly. An envelope export, `sim.init_envelope(ref, distr)` followed by `sim.track_envelope()`, regenerates as `sim.add_particles(bunch_charge_C, distr, npart)` and `sim.track_particles()`, and the two bundled envelope examples the dashboard offers also import as particle tracking. The changed mode is visible in the interface; particle-mode exports and scripts with an explicit `tracking_mode` assignment keep their mode.

Run `python API-12.py`.

??? example "Reproducer: API-12.py"

    [Download API-12.py](API-12.py)

    ```python
    """API-12: Dashboard reimports envelope and reference exports as particle tracking

    A simulation exported from the ImpactX dashboard in envelope or reference tracking mode re-imports
    as particle tracking: the export writes only the tracking call (`track_envelope()` or
    `track_reference(ref)`), and the import does not read it and keeps its particle-tracking default.
    The import is meant to restore the configuration the file describes, and that call identifies the
    tracking model directly. An envelope export, `sim.init_envelope(ref, distr)` followed by
    `sim.track_envelope()`, regenerates as `sim.add_particles(bunch_charge_C, distr, npart)` and
    `sim.track_particles()`, and the two bundled envelope examples the dashboard offers also import as
    particle tracking. The changed mode is visible in the interface; particle-mode exports and scripts
    with an explicit `tracking_mode` assignment keep their mode.

    Setup: `IMPACTX_SOURCE` set to a clone of BLAST-ImpactX/impactx at tag `26.10`; the script loads the
    dashboard modules from it.

    The script loads the dashboard's unmodified parser and export modules from that clone behind package
    shells, replacing only the reactive state object so that no Trame web UI or server is needed. For
    each of the three tracking modes it exports a simulation, parses the export, applies the returned
    state and compares the regenerated tracking commands with the original ones; it also imports the two
    bundled envelope examples. Particle mode and an envelope export prefixed with an explicit
    `tracking_mode` assignment are the controls; quadratic distribution parameters keep the test clear
    of the dashboard's Twiss-parsing defect.
    """

    import os

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
    os.sched_setaffinity(0, {min(os.sched_getaffinity(0))})

    import copy
    import importlib
    import sys
    from pathlib import Path
    from types import ModuleType, SimpleNamespace

    import impactx
    import numpy as np

    ROOT = Path(os.environ["IMPACTX_SOURCE"], "src/python/impactx/dashboard")
    sys.dont_write_bytecode = True


    class State(SimpleNamespace):
        def change(self, *names):
            return lambda function: function

        def update(self, values):
            vars(self).update(values)


    state = State()
    for suffix in (
        "",
        "Input",
        "Input.distribution",
        "Toolbar",
        "Toolbar.file_imports",
        "Toolbar.file_imports.python",
        "Run",
    ):
        name = "impactx.dashboard" + ("." + suffix if suffix else "")
        package = ModuleType(name)
        package.__path__ = [str(ROOT.joinpath(*suffix.split(".")) if suffix else ROOT)]
        package.__package__ = name
        sys.modules[name] = package
    vars(sys.modules["impactx.dashboard"])["state"] = state


    def source_module(suffix):
        result = importlib.import_module("impactx.dashboard." + suffix)
        assert result.__file__ is not None
        assert Path(result.__file__).is_relative_to(ROOT)
        return result


    defaults = source_module("Input.defaults").DashboardDefaults
    state.update(copy.deepcopy(defaults.DEFAULT_VALUES))
    parser = source_module("Toolbar.file_imports.python.parser").DashboardParser
    general = source_module("Input.utils").GeneralFunctions
    vars(sys.modules["impactx.dashboard.Input"])["GeneralFunctions"] = general
    generation = source_module("Run.simulation")


    def parse(content):
        return parser.parse_impactx_simulation_file({"content": content.encode()})


    def configure_export(mode="Envelope Tracking", twiss_input=True):
        state.update(copy.deepcopy(defaults.DEFAULT_VALUES))
        state.tracking_mode = mode
        state.space_charge = "3D"
        state.mass_MeV = 938.2720894282573
        state.charge_qe = 1
        state.kin_energy_MeV = 10.0
        state.kin_energy_on_ui = 10.0
        state.bunch_charge_C = 1e-9
        state.n_cell = [16, 16, 16]
        state.prob_relative = [3.1]
        state.distribution = "Gaussian"
        state.distribution_type = "Twiss" if twiss_input else "Quadratic"
        state.selected_lattice_list = [
            {
                "name": "Drift",
                "parameters": [{"parameter_name": "ds", "sim_input": 0.2, "parameter_type": "float"}],
            }
        ]
        parameters = (
            {
                "beta_x": 1.0,
                "beta_y": 2.0,
                "beta_t": 3.0,
                "emitt_x": 1e-6,
                "emitt_y": 2e-6,
                "emitt_t": 3e-6,
                "alpha_x": -0.5,
                "alpha_y": 0.25,
                "alpha_t": 0.0,
            }
            if twiss_input
            else {
                "lambdaX": 1e-3,
                "lambdaY": 1e-3,
                "lambdaT": 1e-2,
                "lambdaPx": 1e-4,
                "lambdaPy": 1e-4,
                "lambdaPt": 1e-4,
                "muxpx": 0.0,
                "muypy": 0.0,
                "mutpt": 0.0,
            }
        )
        state.selected_distribution_parameters = {
            key: {"value": value, "error_message": []} for key, value in parameters.items()
        }
        return parameters


    def case5():
        for selected in ("Envelope Tracking", "Reference Tracking", "Particle Tracking"):
            configure_export(selected, twiss_input=False)
            exported = generation.dashboard_sim_inputs(is_exporting=True)
            restored = parse(exported)
            original_commands = generation.build_tracking_commands()
            state.update(restored)
            restored_commands = generation.build_tracking_commands()
            assert restored["tracking_mode"] == "Particle Tracking"
            assert "sim.track_particles()" in restored_commands
            assert "tracking_mode =" not in exported
            print(
                f"C5 selected={selected!r} imported={restored['tracking_mode']!r} original={original_commands!r} regenerated={restored_commands!r}"
            )
            if selected == "Envelope Tracking":
                explicit = parse("tracking_mode = 'Envelope Tracking'\n" + exported)
                assert explicit["tracking_mode"] == "Envelope Tracking"
                print("C5 explicit tracking_mode assignment control preserves Envelope Tracking")
        for example in (
            "fodo_space_charge/run_fodo_envelope_sc.py",
            "expanding_beam/run_expanding_envelope.py",
        ):
            text = (ROOT.parents[3] / "examples" / example).read_text()
            assert "sim.track_envelope()" in text
            print(f"C5 bundled {example}: imported={parse(text)['tracking_mode']!r}")


    if __name__ == "__main__":
        print(
            f"Python {sys.version.split()[0]} impactx={impactx.__version__} numpy={np.__version__}",
            flush=True,
        )
        case5()
        print("DEFECT REPRODUCED", flush=True)

    ```

## Provenance

| Role | Model | Effort | Date | Code examined |
|---|---|---|---|---|
| Found | `gpt-6-astra` | xhigh | 2026-10-28 | 26.10 (`615921c6867c`), impactx-noacc wheel, CPU, double, serial |
| Confirmed | `gpt-6.1-sol` | xhigh | 2026-10-28 | 26.10 (`615921c6867c`), impactx-noacc wheel, CPU, double, serial |

