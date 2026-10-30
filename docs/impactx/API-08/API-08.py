"""API-08: Dashboard proton species import loss; no web server is started.

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
