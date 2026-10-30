---
title: "API-09: Dashboard Twiss import corrupts scientific notation and drops negative alpha"
hide:
  - toc
---

# API-09 — Dashboard Twiss import corrupts scientific notation and drops negative alpha

| | |
|---|---|
| Code | [ImpactX](../index.md) · Python API |
| Priority | P3 |
| Severity | Wrong physics |
| Evidence | Reproduced, independently rechecked |
| Last reproduced | 2026-10-29 on development (`fa0a43b9bb07`), source, CPU, double, serial |
| Tags | <span class="bb-tag">partly-source-only</span> [?](../../methodology.md#tags) |
| Source | [`helper.py#L122-L162`](https://github.com/BLAST-ImpactX/impactx/blob/b701a6e9d1f9e30cefb9a3e18e82ae7e02b9d943/src/python/impactx/dashboard/Toolbar/file_imports/python/helper.py#L122-L162) |

The dashboard's import of Python input files reads Twiss parameters with a pattern that keeps only the mantissa of numbers in scientific notation and skips negative values. Re-importing the dashboard's own export must restore the same beam. A Gaussian beam exported with `beta_x=1`, `emitt_x=1e-6` and `alpha_x=-0.5` re-imports with `emitt_x=1` and no `alpha_x`, so the initial `x` variance is 1 m² instead of 1e-6 m², and `alpha_x` keeps the value the interface held before (zero by default, which removes the 5e-7 m `x`–`px` correlation). The `y` and `t` emittances lose their exponents as well.

## Reproducer

Set `IMPACTX_SOURCE` to a clone of the ImpactX repository.

Load complete, unmodified source modules using package shells to avoid the unavailable Trame UI. Only the reactive state object is replaced. Parser, defaults, parameter conversion and code generation are the real modules. Run one case per process. No output files are created.

??? example "API-09.py"

    [Download API-09.py](API-09.py)

    ```python
    """API-09: Dashboard Twiss numeric corruption; no web server is started.

    Load complete, unmodified source modules using package shells to avoid the
    unavailable Trame UI. Only the reactive state object is replaced. Parser,
    defaults, parameter conversion and code generation are the real modules.
    Run one case per process. No output files are created.
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
    from impactx import ImpactX, RefPart, distribution, twiss

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


    def simulation():
        sim = ImpactX()
        sim.verbose = 0
        sim.diagnostics = False
        sim.particle_shape = 1
        sim.space_charge = "3D"
        sim.init_grids()
        return sim


    def case2():
        intended = configure_export()
        exported = generation.dashboard_sim_inputs(is_exporting=True)
        restored = {
            key: float(value) for key, value in parse(exported)["distribution"]["parameters"].items()
        }
        print(
            "C2 exported",
            next(line.strip() for line in exported.splitlines() if "emitt_x=" in line),
        )
        print("C2 restored", restored)
        assert restored["emitt_x"] == 1.0 and "alpha_x" not in restored
        sim = simulation()
        try:
            ref = RefPart().set_species("proton").set_kin_energy_MeV(10)
            for label, params in (("intended", intended), ("restored", restored)):
                sim.init_envelope(
                    ref,
                    distribution.Gaussian(
                        **twiss(**{key: np.float64(value) for key, value in params.items()})
                    ),
                )
                covariance = sim.envelope.envelope.to_numpy().copy()
                # Independent Courant-Snyder covariance, in normalized canonical px.
                beta, emit, alpha = (
                    params["beta_x"],
                    params["emitt_x"],
                    params.get("alpha_x", 0.0),
                )
                expected = emit * np.array([[beta, -alpha], [-alpha, (1 + alpha**2) / beta]])
                np.testing.assert_allclose(covariance[:2, :2], expected, rtol=8e-16, atol=1e-21)
                print(f"C2 {label} Cxx={covariance[0, 0]:.16g} Cxpx={covariance[0, 1]:.16g}")
        finally:
            sim.finalize()
        decimal_export = exported.replace("emitt_x=1e-06", "emitt_x=0.000001")
        assert float(parse(decimal_export)["distribution"]["parameters"]["emitt_x"]) == 1e-6
        print("C2 fixed-decimal control preserves emittance")


    if __name__ == "__main__":
        print(
            f"Python {sys.version.split()[0]} impactx={impactx.__version__} numpy={np.__version__}",
            flush=True,
        )
        case2()
        print("DEFECT REPRODUCED", flush=True)

    ```

## Provenance

| Role | Model | Effort | Date | Code examined |
|---|---|---|---|---|
| Found | `gpt-6-astra` | xhigh | 2026-10-28 | 26.10 (`615921c6867c`), impactx-noacc wheel, CPU, double, serial |
| Confirmed | `gpt-6.1-sol` | xhigh | 2026-10-28 | 26.10 (`615921c6867c`), impactx-noacc wheel, CPU, double, serial |

