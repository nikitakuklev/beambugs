---
title: "API-07: MPI_Group_translate_ranks returns a list whose first half is NULL"
hide:
  - toc
---

# API-07 — `MPI_Group_translate_ranks` returns a list whose first half is `NULL`

| | |
|---|---|
| Code | [PyORBIT](../index.md) · Bindings and build |
| Priority | P4 |
| Failure | Crash or API hazard |
| Evidence | Reproduced, independently rechecked |
| Last reproduced | 2026-11-07 on PyORBIT commit [`28df660`](https://github.com/PyORBIT-Collaboration/PyORBIT3/commit/28df66018aeb), committed 2026-10-14 |
| Source | [wrap_orbit_mpi_group_functions.hh#L126-L129](https://github.com/PyORBIT-Collaboration/PyORBIT3/blob/28df66018aebe4e2b780b9747f4a2d7b8430a5c0/src/mpi/wrap_orbit_mpi_group_functions.hh#L126-L129) |

`orbit_mpi.MPI_Group_translate_ranks(group_a, ranks, group_b)` creates a list of length n with `PyList_New(n)` and then appends the n translated ranks, so it returns 2n entries whose first n slots are `NULL`. Translating `[0]` within one group prints as `[<NULL>, 0]` instead of `[0]`, and indexing, `list()` or `sorted()` on the result crash the interpreter with a segmentation fault. The wrapper logic is the same with and without MPI.

Run `python API-07.py`.

??? example "Reproducer: API-07.py"

    [Download API-07.py](API-07.py)

    ```python
    """API-07: `MPI_Group_translate_ranks` returns a list whose first half is `NULL`

    `orbit_mpi.MPI_Group_translate_ranks(group_a, ranks, group_b)` creates a list of length n with `PyList_New(n)` and then appends the n translated ranks, so it returns 2n entries whose first n slots are `NULL`. Translating `[0]` within one group prints as `[<NULL>, 0]` instead of `[0]`, and indexing, `list()` or `sorted()` on the result crash the interpreter with a segmentation fault. The wrapper logic is the same with and without MPI.

    The script translates rank lists of length 1 and 3 from the world group into itself, which is the identity, and prints the length and representation of the result. It then indexes the result, and calls `list()` and `sorted()` on it, each in a child interpreter, and reports the exit status.
    """

    import os
    import subprocess
    import sys

    for _v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
        os.environ.setdefault(_v, "1")

    from orbit.core import orbit_mpi as mpi

    g = mpi.MPI_Comm_group(mpi.mpi_comm.MPI_COMM_WORLD)
    reproduced = False
    # Translating ranks of a group into the same group returns the input ranks.
    for ranks in ([0], [0, 0, 0]):
        r = mpi.MPI_Group_translate_ranks(g, ranks, g)
        print(
            f"translate_ranks(g, {ranks}, g): observed length {len(r)}, {r!r}; expected length {len(ranks)}, {ranks!r}"
        )
        reproduced |= len(r) != len(ranks)

    PRE = "from orbit.core import orbit_mpi as m\ng = m.MPI_Comm_group(m.mpi_comm.MPI_COMM_WORLD)\nr = m.MPI_Group_translate_ranks(g, [0], g)\n"
    for expr, exp in (("r[0]", "0"), ("list(r)", "[0]"), ("sorted(r)", "[0]")):
        p = subprocess.run(
            [sys.executable, "-I", "-c", PRE + f"print({expr})"],
            capture_output=True,
            check=False,
            text=True,
            timeout=60,
        )
        print(
            f"{expr} in a child: observed exit {p.returncode}, output {p.stdout.strip() or '<none>'}; expected exit 0, output {exp}"
        )
        reproduced |= p.returncode != 0

    print("DEFECT REPRODUCED" if reproduced else "NOT REPRODUCED")

    ```

## Provenance

| Role | Model | Effort | Date | Code examined |
|---|---|---|---|---|
| Found | `claude-opus-5-5` | high | 2026-11-07 | main (`28df66018aeb`), source, serial, CPU, double |
| Confirmed | `claude-sonnet-5-5` | high | 2026-11-07 | main (`28df66018aeb`), source, serial, CPU, double |

