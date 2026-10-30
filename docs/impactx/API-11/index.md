---
title: "API-11: Dashboard silently accepts a partial lattice when repeated cells are unresolved"
hide:
  - toc
---

# API-11 — Dashboard silently accepts a partial lattice when repeated cells are unresolved

| | |
|---|---|
| Code | [ImpactX](../index.md) · Python API |
| Priority | P3 |
| Severity | Silent wrong result |
| Evidence | Reproduced, independently rechecked |
| Last reproduced | 2026-10-29 on development (`fa0a43b9bb07`), source, CPU, double, serial |
| Tags | <span class="bb-tag">partly-source-only</span> [?](../../methodology.md#tags) |
| Source | [`lattice_helper.py#L217-L219`](https://github.com/BLAST-ImpactX/impactx/blob/b701a6e9d1f9e30cefb9a3e18e82ae7e02b9d943/src/python/impactx/dashboard/Toolbar/file_imports/python/lattice_helper.py#L217-L219)<br>[`parse()` and `parse_cleaned_lattice()`](https://github.com/BLAST-ImpactX/impactx/blob/b701a6e9d1f9e30cefb9a3e18e82ae7e02b9d943/src/python/impactx/dashboard/Toolbar/file_imports/python/lattice_helper.py#L30-L80)<br>[`sim.lattice.extend(cell * 3)`](https://github.com/BLAST-ImpactX/impactx/blob/b701a6e9d1f9e30cefb9a3e18e82ae7e02b9d943/docs/source/usage/howto/lattice_manipulation.rst#L51-L65) |

When the dashboard imports a Python input file whose lattice includes a repeated cell, such as `sim.lattice.extend(cell * 3)`, it drops the repetition without a message and accepts the preceding elements as the complete lattice. The ImpactX manual documents this form, and an importer that cannot evaluate it must report an incomplete import. A 0.4 m drift followed by three drift–quadrupole cells, 1.3 m in total, imports as the single 0.4 m drift, so the horizontal focusing term `R21` is 0 instead of −0.5510 m⁻¹; the same lattice written out element by element imports all seven elements.

## Reproducer

Set `IMPACTX_SOURCE` to a clone of the ImpactX repository.

Load complete, unmodified source modules using package shells to avoid the unavailable Trame UI. Only the reactive state object is replaced. Parser, defaults, parameter conversion and code generation are the real modules. Run one case per process. No output files are created.

??? example "API-11.py"

    [Download API-11.py](API-11.py)

    ```python
    """API-11: Silent partial repeated-cell import; no web server is started.

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

    import contextlib
    import copy
    import importlib
    import io
    import math
    import sys
    from pathlib import Path
    from types import ModuleType, SimpleNamespace

    import impactx
    import numpy as np
    from impactx import RefPart, elements

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


    def drift_matrix(length, bg2):
        matrix = np.eye(6)
        matrix[0, 1] = matrix[2, 3] = length
        matrix[4, 5] = length / bg2
        return matrix


    def quad_matrix(length, strength, bg2):
        matrix = drift_matrix(length, bg2)
        w = math.sqrt(strength)
        angle = w * length
        matrix[:2, :2] = [
            [math.cos(angle), math.sin(angle) / w],
            [-w * math.sin(angle), math.cos(angle)],
        ]
        matrix[2:4, 2:4] = [
            [math.cosh(angle), math.sinh(angle) / w],
            [w * math.sinh(angle), math.cosh(angle)],
        ]
        return matrix


    def case4():
        declarations = """from impactx import elements
    pre = elements.Drift(ds=0.4)
    d = elements.Drift(ds=0.2)
    q = elements.Quad(ds=0.1, k=2.0)
    cell = [d, q]
    """
        prefix = declarations + "sim.lattice.append(pre)\n"
        intended = prefix + "sim.lattice.extend(cell * 3)\n"
        messages = io.StringIO()
        with contextlib.redirect_stdout(messages):
            imported = parse(intended)["lattice_elements"]
        print("C4 imported", imported, "messages=", repr(messages.getvalue()))
        assert len(imported) == 1 and imported[0]["name"] == "Drift"
        assert messages.getvalue() == ""
        ref = RefPart().set_species("proton").set_kin_energy_MeV(10)
        bg2 = ref.gamma**2 - 1
        cell_matrix = quad_matrix(0.1, 2.0, bg2) @ drift_matrix(0.2, bg2)
        expected = np.linalg.matrix_power(cell_matrix, 3) @ drift_matrix(0.4, bg2)
        for label, content in (
            ("original", intended),
            ("imported", intended),
            ("unrolled_control", prefix + "sim.lattice.extend([d, q, d, q, d, q])\n"),
            ("single_cell_control", prefix + "sim.lattice.extend(cell)\n"),
        ):
            lattice = elements.KnownElementsList()
            if label == "original":
                exec(content, {"sim": SimpleNamespace(lattice=lattice)})  # noqa: S102 -- trusted inline test deck
            else:
                records = parse(content)["lattice_elements"]
                lattice.extend(
                    [
                        getattr(elements, record["name"])(
                            **{k: float(v) for k, v in record["parameters"].items()}
                        )
                        for record in records
                    ]
                )
            matrix = lattice.transfer_map(ref).to_numpy().copy()
            length = sum(element.ds for element in lattice)
            oracle = (
                drift_matrix(0.4, bg2)
                if label == "imported"
                else cell_matrix @ drift_matrix(0.4, bg2)
                if label == "single_cell_control"
                else expected
            )
            np.testing.assert_allclose(matrix, oracle, rtol=2e-15, atol=2e-16)
            np.testing.assert_allclose(
                length,
                0.4 if label == "imported" else 0.7 if label == "single_cell_control" else 1.3,
                rtol=0,
                atol=3e-16,
            )
            print(f"C4 {label} count={len(lattice)} length={length:.16g} R21={matrix[1, 0]:.16g}")


    if __name__ == "__main__":
        print(
            f"Python {sys.version.split()[0]} impactx={impactx.__version__} numpy={np.__version__}",
            flush=True,
        )
        case4()
        print("DEFECT REPRODUCED", flush=True)

    ```

## Provenance

| Role | Model | Effort | Date | Code examined |
|---|---|---|---|---|
| Found | `gpt-6-astra` | xhigh | 2026-10-28 | 26.10 (`615921c6867c`), impactx-noacc wheel, CPU, double, serial |
| Confirmed | `gpt-6.1-sol` | xhigh | 2026-10-28 | 26.10 (`615921c6867c`), impactx-noacc wheel, CPU, double, serial |

