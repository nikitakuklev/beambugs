---
title: "API-10: Dashboard envelope commands omit the selected beam intensity"
hide:
  - toc
---

# API-10 — Dashboard envelope commands omit the selected beam intensity

| | |
|---|---|
| Code | [ImpactX](../index.md) · Python API |
| Priority | P3 |
| Severity | Wrong physics |
| Evidence | Reproduced, independently rechecked |
| Last reproduced | 2026-10-29 on ImpactX development (`fa0a43b`, head build) |
| Tags | <span class="bb-tag">partly-source-only</span> [?](../../methodology.md#tags) |
| Source | [`simulation.py#L12-L20`](https://github.com/BLAST-ImpactX/impactx/blob/b701a6e9d1f9e30cefb9a3e18e82ae7e02b9d943/src/python/impactx/dashboard/Run/simulation.py#L12-L20)<br>[`init_envelope` contract](https://github.com/BLAST-ImpactX/impactx/blob/b701a6e9d1f9e30cefb9a3e18e82ae7e02b9d943/docs/source/usage/python.rst#L340-L348) |

In envelope tracking mode, the ImpactX dashboard generates `sim.init_envelope(ref, distr)` without the selected beam intensity, so the envelope starts with zero bunch charge and its space-charge kicks vanish. `init_envelope` defines the intensity as bunch charge in C for 3D space charge and current in A for 2D, and the bundled envelope example passes it. Running the generated code for a 10 MeV proton envelope with 1 nC selected, through a 0.2 m drift with 3D space charge, leaves the normalized-momentum variance ⟨pₓ²⟩ at its vacuum value 1.000×10⁻⁸ instead of 3.889×10⁻⁵, the result of a closed-form uniform-ellipsoid field calculation. ImpactX logs a zero-charge warning; according to the source, the dashboard's Run and Export both use this generator.

Run `python API-10.py`.

??? example "Reproducer: API-10.py"

    [Download API-10.py](API-10.py)

    ```python
    """API-10: Dashboard envelope commands omit the selected beam intensity

    In envelope tracking mode, the ImpactX dashboard generates `sim.init_envelope(ref, distr)` without
    the selected beam intensity, so the envelope starts with zero bunch charge and its space-charge
    kicks vanish. `init_envelope` defines the intensity as bunch charge in C for 3D space charge and
    current in A for 2D, and the bundled envelope example passes it. Running the generated code for a 10
    MeV proton envelope with 1 nC selected, through a 0.2 m drift with 3D space charge, leaves the
    normalized-momentum variance ⟨pₓ²⟩ at its vacuum value 1.000×10⁻⁸ instead of 3.889×10⁻⁵, the result
    of a closed-form uniform-ellipsoid field calculation. ImpactX logs a zero-charge warning; according
    to the source, the dashboard's Run and Export both use this generator.

    Setup: `IMPACTX_SOURCE` set to a clone of BLAST-ImpactX/impactx at tag `26.10`; the script loads the
    dashboard modules from it.

    The script loads the dashboard's unmodified generator modules from that clone behind package shells,
    replacing only the reactive state object so that no Trame web UI or server is needed, and selects
    envelope tracking, 3D space charge and 1 nC for a 10 MeV proton Gaussian beam. It executes the
    generated tracking commands through a 0.2 m drift and compares the full covariance matrix with
    vacuum drift transport. As controls, `init_envelope` called with 1 nC and 0.5 nC must match a
    drift–kick–drift calculation using the closed-form field of a prolate uniform ellipsoid with the
    same rms sizes.
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
    import math
    import sys
    from pathlib import Path
    from types import ModuleType, SimpleNamespace

    import impactx
    import numpy as np
    from impactx import ImpactX, RefPart, distribution, elements

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


    def drift_matrix(length, bg2):
        matrix = np.eye(6)
        matrix[0, 1] = matrix[2, 3] = length
        matrix[4, 5] = length / bg2
        return matrix


    def analytic_envelope(ref, covariance, length, charge):
        """Prolate-spheroid electric field, with M(L/2) K(L) M(L/2).

        The rms-equivalent uniform ellipsoid has axes sqrt(5)*sx, sqrt(5)*sy,
        sqrt(5)*beta*gamma*st in its rest frame. Its depolarization factors sum
        to one. This uses the closed spheroid field, not ImpactX's Carlson RD.
        """
        bg2 = ref.gamma**2 - 1
        half = drift_matrix(length / 2, bg2)
        mid = half @ covariance @ half.T
        radial = math.sqrt(mid[0, 0])
        long = math.sqrt(bg2 * mid[4, 4])
        assert mid[0, 0] == mid[2, 2] and long > radial
        eccentricity = math.sqrt(1 - (radial / long) ** 2)
        nz = (
            (1 - eccentricity**2)
            / (2 * eccentricity**3)
            * (math.log((1 + eccentricity) / (1 - eccentricity)) - 2 * eccentricity)
        )
        nx = (1 - nz) / 2
        rc_n = abs(ref.charge * charge) / (4 * math.pi * 8.8541878188e-12 * ref.mass * 299792458**2)
        common = 3 * length * rc_n / (5 * math.sqrt(5) * bg2 * radial**2 * long)
        kick = np.eye(6)
        kick[1, 0] = kick[3, 2] = common * nx
        kick[5, 4] = common * bg2 * nz
        total = half @ kick @ half
        return total @ covariance @ total.T


    def case3():
        parameters = configure_export(twiss_input=False)
        commands = generation.build_tracking_commands()
        print("C3 generated", repr(commands))
        print("C3 selected charge_C", state.bunch_charge_C)
        sim = simulation()
        try:
            ref = RefPart().set_species("proton").set_kin_energy_MeV(10)
            distr = distribution.Gaussian(**parameters)
            sim.lattice.append(elements.Drift(ds=0.2))
            initial = np.diag([1e-6, 1e-8, 1e-6, 1e-8, 1e-4, 1e-8])
            d = drift_matrix(0.2, ref.gamma**2 - 1)
            vacuum = d @ initial @ d.T
            exec(  # noqa: S102 -- execute the original generated simulation commands
                commands,
                {
                    "sim": sim,
                    "ref": ref,
                    "distr": distr,
                    "bunch_charge_C": state.bunch_charge_C,
                },
            )
            actual = sim.envelope.envelope.to_numpy().copy()
            assert sim.envelope.beam_intensity == 0.0
            np.testing.assert_allclose(actual, vacuum, rtol=8e-16, atol=1e-22)
            print(
                f"C3 dashboard intensity={sim.envelope.beam_intensity:g} Cpxpx={actual[1, 1]:.16g} vacuum={vacuum[1, 1]:.16g}"
            )
            for charge in (1e-9, 0.5e-9):
                sim.init_envelope(ref, distr, charge)
                sim.track_envelope()
                actual = sim.envelope.envelope.to_numpy().copy()
                expected = analytic_envelope(ref, initial, 0.2, charge)
                np.testing.assert_allclose(actual, expected, rtol=3e-14, atol=1e-22)
                print(
                    f"C3 explicit charge_C={charge:g} Cpxpx={actual[1, 1]:.16g} spheroid_oracle={expected[1, 1]:.16g} max_abs_error={np.max(np.abs(actual - expected)):.3g}"
                )
        finally:
            sim.finalize()


    if __name__ == "__main__":
        print(
            f"Python {sys.version.split()[0]} impactx={impactx.__version__} numpy={np.__version__}",
            flush=True,
        )
        case3()
        print("DEFECT REPRODUCED", flush=True)

    ```

## Provenance

| Role | Model | Effort | Date | Code examined |
|---|---|---|---|---|
| Found | `gpt-6-astra` | xhigh | 2026-10-28 | 26.10 (`615921c6867c`), impactx-noacc wheel, CPU, double, serial |
| Confirmed | `gpt-6.1-sol` | xhigh | 2026-10-28 | 26.10 (`615921c6867c`), impactx-noacc wheel, CPU, double, serial |

