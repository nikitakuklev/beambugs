---
title: "API-08: Dashboard imports the bundled proton cyclotron with electron mass and charge"
hide:
  - toc
---

# API-08 — Dashboard imports the bundled proton cyclotron with electron mass and charge

| | |
|---|---|
| Code | [ImpactX](../index.md) · Python API |
| Priority | P3 |
| Severity | Wrong physics |
| Evidence | Reproduced, independently rechecked |
| Last reproduced | 2026-10-29 on ImpactX development (`fa0a43b`, head build) |
| Tags | <span class="bb-tag">partly-source-only</span> [?](../../methodology.md#tags) |
| Source | [`helper.py#L35-L63`](https://github.com/BLAST-ImpactX/impactx/blob/b701a6e9d1f9e30cefb9a3e18e82ae7e02b9d943/src/python/impactx/dashboard/Toolbar/file_imports/python/helper.py#L35-L63)<br>[`cyclotron/run_cyclotron.py` example](https://github.com/BLAST-ImpactX/impactx/blob/b701a6e9d1f9e30cefb9a3e18e82ae7e02b9d943/src/python/impactx/dashboard/Toolbar/examples_loader/loader.py#L18-L31) |

The ImpactX dashboard's import of Python input files ignores `set_species()` and keeps its electron mass and charge defaults, so the bundled `cyclotron/run_cyclotron.py` example, which sets a proton reference, imports as an electron. Importing a script must preserve the species it defines; replacing `set_species("proton")` with explicit mass and charge setters does preserve it. At the example's 4 keV kinetic energy K the imported magnetic rigidity is −2.137×10⁻⁴ T·m instead of +9.139×10⁻³ T·m, the proton value of Bρ = √(K(K + 2mc²))/(qc).

Run `python API-08.py`.

??? example "Reproducer: API-08.py"

    [Download API-08.py](API-08.py)

    ```python
    """API-08: Dashboard imports the bundled proton cyclotron with electron mass and charge

    The ImpactX dashboard's import of Python input files ignores `set_species()` and keeps its electron
    mass and charge defaults, so the bundled `cyclotron/run_cyclotron.py` example, which sets a proton
    reference, imports as an electron. Importing a script must preserve the species it defines;
    replacing `set_species("proton")` with explicit mass and charge setters does preserve it. At the
    example's 4 keV kinetic energy K the imported magnetic rigidity is −2.137×10⁻⁴ T·m instead of
    +9.139×10⁻³ T·m, the proton value of Bρ = √(K(K + 2mc²))/(qc).

    Setup: `IMPACTX_SOURCE` set to a clone of BLAST-ImpactX/impactx at tag `26.10`; the script loads the
    dashboard modules from it.

    The script loads the dashboard's unmodified parser modules from that clone behind package shells,
    replacing only the reactive state object so that no Trame web UI or server is needed, and parses the
    bundled cyclotron example. It builds the original and the imported reference particle in the
    installed core and compares each rigidity with Bρ = √(K(K + 2mc²))/(qc) evaluated in 60-digit
    decimal arithmetic. The same example with `set_species("proton")` replaced by explicit mass and
    charge setters is the control.
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
    from decimal import Decimal, localcontext
    from pathlib import Path
    from types import ModuleType, SimpleNamespace

    import impactx
    import numpy as np
    from impactx import RefPart

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


    def case1():
        content = Path(os.environ["IMPACTX_SOURCE"], "examples/cyclotron/run_cyclotron.py").read_text()
        values = parse(content)
        print(
            "C1 parsed",
            {key: values[key] for key in ("mass_MeV", "charge_qe", "kin_energy_MeV")},
        )
        assert values["kin_energy_MeV"] == 0.004
        assert values["mass_MeV"] == defaults.DEFAULT_VALUES["mass_MeV"]
        assert values["charge_qe"] == -1
        # Establish proton rest energy independently from the constants used by the core.
        with localcontext() as ctx:
            ctx.prec = 60
            c = Decimal(299792458)
            proton_mass = Decimal("1.67262192595e-27") * c * c / (Decimal("1.602176634e-19") * 1000000)
            expected_proton = float(
                (Decimal("0.004") * (Decimal("0.004") + 2 * proton_mass)).sqrt() * 1000000 / c
            )
            expected_electron = float(
                -(Decimal("0.004") * (Decimal("0.004") + 2 * Decimal(str(values["mass_MeV"])))).sqrt()
                * 1000000
                / c
            )
        original = RefPart().set_species("proton").set_kin_energy_MeV(0.004)
        imported = (
            RefPart()
            .set_mass_MeV(values["mass_MeV"])
            .set_charge_qe(values["charge_qe"])
            .set_kin_energy_MeV(values["kin_energy_MeV"])
        )
        np.testing.assert_allclose(original.mass_MeV, float(proton_mass), rtol=3e-16, atol=0)
        # The core forms gamma^2-1 at K/m=4.3e-6. Bound the resulting cancellation
        # by four double-precision ulps in gamma^2-1, rather than an arbitrary tolerance.
        for label, ref, expected in (
            ("proton", original, expected_proton),
            ("imported", imported, expected_electron),
        ):
            bg2 = float(
                (Decimal(str(ref.kin_energy_MeV)) / Decimal(str(ref.mass_MeV)))
                * (Decimal(2) + Decimal(str(ref.kin_energy_MeV)) / Decimal(str(ref.mass_MeV)))
            )
            relative_bound = 4 * np.finfo(float).eps / bg2
            assert abs(ref.rigidity_Tm / expected - 1) < relative_bound
            print(
                f"C1 {label} mass_MeV={ref.mass_MeV:.15g} q={ref.charge_qe:g} Brho={ref.rigidity_Tm:.15g} analytic={expected:.15g} relerr={ref.rigidity_Tm / expected - 1:.3g}"
            )
        control = parse(
            content.replace('set_species("proton")', "set_charge_qe(1).set_mass_MeV(938.2720894282573)")
        )
        assert control["charge_qe"] == 1 and control["mass_MeV"] == 938.2720894282573
        print("C1 explicit-setter control preserves proton mass and charge")


    if __name__ == "__main__":
        print(
            f"Python {sys.version.split()[0]} impactx={impactx.__version__} numpy={np.__version__}",
            flush=True,
        )
        case1()
        print("DEFECT REPRODUCED", flush=True)

    ```

## Provenance

| Role | Model | Effort | Date | Code examined |
|---|---|---|---|---|
| Found | `gpt-6-astra` | xhigh | 2026-10-28 | 26.10 (`615921c6867c`), impactx-noacc wheel, CPU, double, serial |
| Confirmed | `gpt-6.1-sol` | xhigh | 2026-10-28 | 26.10 (`615921c6867c`), impactx-noacc wheel, CPU, double, serial |

