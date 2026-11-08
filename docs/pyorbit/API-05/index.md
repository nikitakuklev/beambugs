---
title: "API-05: Integer-list MPI_Bcast ends the process on any negative element"
hide:
  - toc
---

# API-05 — Integer-list `MPI_Bcast` ends the process on any negative element

| | |
|---|---|
| Code | [PyORBIT](../index.md) · Bindings and build |
| Priority | P4 |
| Failure | Crash or API hazard |
| Evidence | Reproduced, independently rechecked |
| Last reproduced | 2026-11-07 on PyORBIT commit [`28df660`](https://github.com/PyORBIT-Collaboration/PyORBIT3/commit/28df66018aeb), committed 2026-10-14 |
| Source | [wrap_orbit_mpi_send_receive_functions.hh#L212-L223](https://github.com/PyORBIT-Collaboration/PyORBIT3/blob/28df66018aebe4e2b780b9747f4a2d7b8430a5c0/src/mpi/wrap_orbit_mpi_send_receive_functions.hh#L212-L223) |

Broadcasting a list of integers with `orbit_mpi.MPI_Bcast(..., MPI_INT, ...)` ends the whole process with exit status 1 if any element is negative, printing that the data "is not good". Negative values are valid `MPI_INT` data: a negative scalar broadcasts correctly, as do negative `MPI_DOUBLE` lists and an `MPI_Allreduce` of `[1, -2]`, so broadcasting `[1, -2]` must return `(1, -2)`. The `< 0` test rejects every negative value; a failed conversion, whose error value is −1, is caught only as a side effect.

Run `python API-05.py`.

??? example "Reproducer: API-05.py"

    [Download API-05.py](API-05.py)

    ```python
    """API-05: Integer-list `MPI_Bcast` ends the process on any negative element

    Broadcasting a list of integers with `orbit_mpi.MPI_Bcast(..., MPI_INT, ...)` ends the whole process with exit status 1 if any element is negative, printing that the data "is not good". Negative values are valid `MPI_INT` data: a negative scalar broadcasts correctly, as do negative `MPI_DOUBLE` lists and an `MPI_Allreduce` of `[1, -2]`, so broadcasting `[1, -2]` must return `(1, -2)`. The `< 0` test rejects every negative value; a failed conversion, whose error value is −1, is caught only as a side effect.

    The script broadcasts `[1, -2]` and `[-2147483648]` from rank 0 in child interpreters and reports the exit status and the last line of error output. Controls broadcast the scalar `-2`, the list `[1, 2]` and the `MPI_DOUBLE` list `[-2.5, 1.0]`, and all-reduce `[1, -2]`; each must return its input.
    """

    import os
    import subprocess
    import sys

    for _v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
        os.environ.setdefault(_v, "1")

    PRE = (
        "from orbit.core import orbit_mpi as m\n"
        "W = m.mpi_comm.MPI_COMM_WORLD\n"
        "INT, DOUBLE, SUM = m.mpi_datatype.MPI_INT, m.mpi_datatype.MPI_DOUBLE, m.mpi_op.MPI_SUM\n"
    )


    def run(code):
        """Exit status, stdout and last stderr line of `code` in a child interpreter."""
        p = subprocess.run(
            [sys.executable, "-I", "-c", PRE + code],
            capture_output=True,
            text=True,
            timeout=60,
            check=False,
        )
        err = [ln.strip() for ln in p.stderr.splitlines() if ln.strip()]
        return p.returncode, p.stdout.strip() or "<none>", err[-1] if err else "<none>"


    # On one rank a broadcast from the root and a SUM all-reduce return the input (sequences as tuples).
    cases = [
        ("Bcast([1, -2], MPI_INT)", "print(m.MPI_Bcast([1, -2], INT, 0, W))", "(1, -2)", False),
        (
            "Bcast([-2147483648], MPI_INT)",
            "print(m.MPI_Bcast([-2147483648], INT, 0, W))",
            "(-2147483648,)",
            False,
        ),
        ("control Bcast(-2, MPI_INT)", "print(m.MPI_Bcast(-2, INT, 0, W))", "-2", True),
        ("control Bcast([1, 2], MPI_INT)", "print(m.MPI_Bcast([1, 2], INT, 0, W))", "(1, 2)", True),
        (
            "control Bcast([-2.5, 1.0], MPI_DOUBLE)",
            "print(m.MPI_Bcast([-2.5, 1.0], DOUBLE, 0, W))",
            "(-2.5, 1.0)",
            True,
        ),
        (
            "control Allreduce([1, -2], MPI_INT, SUM)",
            "print(m.MPI_Allreduce([1, -2], INT, SUM, W))",
            "(1, -2)",
            True,
        ),
    ]
    reproduced = False
    for label, code, exp, control in cases:
        rc, out, err = run(code)
        print(
            f"{label}: observed exit {rc}, output {out}, stderr tail {err!r}; expected exit 0, output {exp}"
        )
        reproduced |= not control and (rc != 0 or out != exp)

    print("DEFECT REPRODUCED" if reproduced else "NOT REPRODUCED")

    ```

## Provenance

| Role | Model | Effort | Date | Code examined |
|---|---|---|---|---|
| Found | <span class="bb-nowrap">`claude-opus-5-5`</span> | high | <span class="bb-nowrap">2026-11-07</span> | main (`28df66018aeb`), source, serial, CPU, double |
| Confirmed | <span class="bb-nowrap">`claude-sonnet-5-5`</span> | high | <span class="bb-nowrap">2026-11-07</span> | main (`28df66018aeb`), source, serial, CPU, double |

