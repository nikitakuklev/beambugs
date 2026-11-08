---
title: "API-12: The USE_MPI build option advertises openmpi, which the build treats as autodetection"
hide:
  - toc
---

# API-12 — The `USE_MPI` build option advertises `openmpi`, which the build treats as autodetection

| | |
|---|---|
| Code | [PyORBIT](../index.md) · Bindings and build |
| Priority | P4 |
| Failure | Documentation or example mismatch |
| Evidence | Reproduced, independently rechecked |
| Last reproduced | 2026-11-07 on PyORBIT commit [`28df660`](https://github.com/PyORBIT-Collaboration/PyORBIT3/commit/28df66018aeb), committed 2026-10-14 |
| Tags | <span class="bb-tag">partly-source-only</span> [?](../../methodology.md#tags) |
| Source | [meson_options.txt#L1](https://github.com/PyORBIT-Collaboration/PyORBIT3/blob/28df66018aebe4e2b780b9747f4a2d7b8430a5c0/meson_options.txt#L1)<br>[src/meson.build#L19-L51](https://github.com/PyORBIT-Collaboration/PyORBIT3/blob/28df66018aebe4e2b780b9747f4a2d7b8430a5c0/src/meson.build#L19-L51)<br>[README.md#L115-L125](https://github.com/PyORBIT-Collaboration/PyORBIT3/blob/28df66018aebe4e2b780b9747f4a2d7b8430a5c0/README.md#L115-L125) |

The description of the `USE_MPI` Meson option lists `mpich`, `openmpi`, `none` and `auto`, but the build recognizes `ompi` for OpenMPI and sends every other unlisted value to autodetection without a warning. Configuring with `-DUSE_MPI=openmpi` therefore takes the same branch as `-DUSE_MPI=bogus`: it uses MPICH when both libraries are installed and builds without MPI when neither is found, where an explicit request must select OpenMPI or fail. The README lists the correct value `ompi`.

Run `python API-12.py`.

??? example "Reproducer: API-12.py"

    [Download API-12.py](API-12.py)

    ```python
    """API-12: The `USE_MPI` build option advertises `openmpi`, which the build treats as autodetection

    The description of the `USE_MPI` Meson option lists `mpich`, `openmpi`, `none` and `auto`, but the build recognizes `ompi` for OpenMPI and sends every other unlisted value to autodetection without a warning. Configuring with `-DUSE_MPI=openmpi` therefore takes the same branch as `-DUSE_MPI=bogus`: it uses MPICH when both libraries are installed and builds without MPI when neither is found, where an explicit request must select OpenMPI or fail. The README lists the correct value `ompi`.

    Setup: the script reads `meson_options.txt` and `src/meson.build` of the source checkout named by `PYORBIT_SOURCE`.

    The script extracts the values that the option description advertises and the values that `src/meson.build` tests explicitly, and reports advertised values without a branch of their own. It also checks whether the final `else` branch, which every unrecognized value reaches, raises a configure error.
    """

    import os
    import re
    import sys

    src = os.environ.get("PYORBIT_SOURCE", "")
    if not src:
        sys.exit("PYORBIT_SOURCE must point to a PyORBIT source checkout")

    with open(os.path.join(src, "meson_options.txt")) as f:
        m = re.search(r"option\('USE_MPI'.*?description:\s*'[^(]*\(([^)]*)\)'", f.read(), re.DOTALL)
    assert m, "USE_MPI option not found"
    advertised = [v.strip() for v in m.group(1).split(",")]

    with open(os.path.join(src, "src", "meson.build")) as f:
        build = f.read().split("mpi_use = get_option('USE_MPI')", 1)[1]
    branches = re.findall(r"^(?:if|elif)\s+mpi_use\s*==\s*'([^']+)'", build, re.MULTILINE)
    # The first unindented `else` of the chain is the autodetection branch; it ends at the first unindented `endif`.
    fallback = build.split("\nelse", 1)[1].split("\nendif", 1)[0]

    # 'auto' is documented as autodetection, so it needs no branch of its own; every other named
    # choice must select itself.
    missing = [v for v in advertised if v != "auto" and v not in branches]
    print(f"advertised values {advertised}; explicitly tested values {branches}")
    print(f"advertised values that reach autodetection: observed {missing}, expected []")
    print(
        f"unrecognized value raises a configure error: observed {'error(' in fallback}, expected True"
    )
    print("DEFECT REPRODUCED" if missing or "error(" not in fallback else "NOT REPRODUCED")

    ```

## Provenance

| Role | Model | Effort | Date | Code examined |
|---|---|---|---|---|
| Found | <span class="bb-nowrap">`claude-opus-5-5`</span> | high | <span class="bb-nowrap">2026-11-07</span> | main (`28df66018aeb`), source, serial, CPU, double |
| Confirmed | <span class="bb-nowrap">`claude-sonnet-5-5`</span> | high | <span class="bb-nowrap">2026-11-07</span> | main (`28df66018aeb`), source, serial, CPU, double |

