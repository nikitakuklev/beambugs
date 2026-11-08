---
title: "API-01: MPI_Bcast, MPI_Send and the impedance setters misread or crash on sequences other than lists and tuples"
hide:
  - toc
---

# API-01 — `MPI_Bcast`, `MPI_Send` and the impedance setters misread or crash on sequences other than lists and tuples

| | |
|---|---|
| Code | [PyORBIT](../index.md) · Bindings and build |
| Priority | P3 |
| Failure | Silent wrong result |
| Evidence | Reproduced, independently rechecked |
| Last reproduced | 2026-11-07 on PyORBIT commit [`28df660`](https://github.com/PyORBIT-Collaboration/PyORBIT3/commit/28df66018aeb), committed 2026-10-14 |
| Source | [wrap_orbit_mpi_send_receive_functions.hh#L205-L249](https://github.com/PyORBIT-Collaboration/PyORBIT3/blob/28df66018aebe4e2b780b9747f4a2d7b8430a5c0/src/mpi/wrap_orbit_mpi_send_receive_functions.hh#L205-L249)<br>[#L307-L328](https://github.com/PyORBIT-Collaboration/PyORBIT3/blob/28df66018aebe4e2b780b9747f4a2d7b8430a5c0/src/mpi/wrap_orbit_mpi_send_receive_functions.hh#L307-L328)<br>[wrap_orbit_mpi_group_functions.hh#L11-L19](https://github.com/PyORBIT-Collaboration/PyORBIT3/blob/28df66018aebe4e2b780b9747f4a2d7b8430a5c0/src/mpi/wrap_orbit_mpi_group_functions.hh#L11-L19)<br>[#L34-L42](https://github.com/PyORBIT-Collaboration/PyORBIT3/blob/28df66018aebe4e2b780b9747f4a2d7b8430a5c0/src/mpi/wrap_orbit_mpi_group_functions.hh#L34-L42)<br>[#L113-L124](https://github.com/PyORBIT-Collaboration/PyORBIT3/blob/28df66018aebe4e2b780b9747f4a2d7b8430a5c0/src/mpi/wrap_orbit_mpi_group_functions.hh#L113-L124)<br>[wrap_orbit_mpi_graph_functions.hh#L12-L32](https://github.com/PyORBIT-Collaboration/PyORBIT3/blob/28df66018aebe4e2b780b9747f4a2d7b8430a5c0/src/mpi/wrap_orbit_mpi_graph_functions.hh#L12-L32)<br>[#L89-L109](https://github.com/PyORBIT-Collaboration/PyORBIT3/blob/28df66018aebe4e2b780b9747f4a2d7b8430a5c0/src/mpi/wrap_orbit_mpi_graph_functions.hh#L89-L109)<br>[wrap_LImpedance.cc#L86-L106](https://github.com/PyORBIT-Collaboration/PyORBIT3/blob/28df66018aebe4e2b780b9747f4a2d7b8430a5c0/src/orbit/Impedances/wrap_LImpedance.cc#L86-L106)<br>[wrap_TImpedance.cc#L125-L165](https://github.com/PyORBIT-Collaboration/PyORBIT3/blob/28df66018aebe4e2b780b9747f4a2d7b8430a5c0/src/orbit/Impedances/wrap_TImpedance.cc#L125-L165)<br>[wrap_lspacechargecalc.cc#L74-L90](https://github.com/PyORBIT-Collaboration/PyORBIT3/blob/28df66018aebe4e2b780b9747f4a2d7b8430a5c0/src/spacecharge/wrap_lspacechargecalc.cc#L74-L90)<br>[`allreduce_sequence_int`#L48-L72](https://github.com/PyORBIT-Collaboration/PyORBIT3/blob/28df66018aebe4e2b780b9747f4a2d7b8430a5c0/src/mpi/wrap_orbit_mpi_send_receive_functions.hh#L48-L72) |

`orbit_mpi.MPI_Bcast`, `MPI_Send`, the group and graph-topology wrappers, and the impedance setters `LImpedance.assignImpedance`, `TImpedance.assignImpedance` and `LSpaceChargeCalc.assignImpedance` accept any Python sequence but read its elements with a macro that is valid only for lists and tuples. Other sequences are read through a tuple's memory layout: broadcasting `range(3)` as `MPI_INT` returns `(3, 1, 3)` instead of `(0, 1, 2)`, and passing a NumPy array to a broadcast, a send or an impedance setter ends the process with a segmentation fault. No Python exception is raised. Lists and tuples work, and `MPI_Allreduce` converts the sequence first and handles NumPy arrays correctly.

Run `python API-01.py`.

??? example "Reproducer: API-01.py"

    [Download API-01.py](API-01.py)

    ```python
    """API-01: `MPI_Bcast`, `MPI_Send` and the impedance setters misread or crash on sequences other than lists and tuples

    `orbit_mpi.MPI_Bcast`, `MPI_Send`, the group and graph-topology wrappers, and the impedance setters `LImpedance.assignImpedance`, `TImpedance.assignImpedance` and `LSpaceChargeCalc.assignImpedance` accept any Python sequence but read its elements with a macro that is valid only for lists and tuples. Other sequences are read through a tuple's memory layout: broadcasting `range(3)` as `MPI_INT` returns `(3, 1, 3)` instead of `(0, 1, 2)`, and passing a NumPy array to a broadcast, a send or an impedance setter ends the process with a segmentation fault. No Python exception is raised. Lists and tuples work, and `MPI_Allreduce` converts the sequence first and handles NumPy arrays correctly.

    The script broadcasts `range(3)` from rank 0 to itself, where the result must equal the root's data. It then passes a NumPy array to `MPI_Bcast`, `MPI_Send`, `MPI_Group_incl`, `MPI_Group_excl`, `MPI_Graph_create` and the three impedance setters, each in a child interpreter, and reports the exit status. Controls pass the same values as lists, and reduce the NumPy array with `MPI_Allreduce`.
    """

    import os
    import subprocess
    import sys

    for _v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
        os.environ.setdefault(_v, "1")

    from orbit.core import orbit_mpi as mpi

    MPI = (
        "from orbit.core import orbit_mpi as m\n"
        "W = m.mpi_comm.MPI_COMM_WORLD\n"
        "DOUBLE, SUM = m.mpi_datatype.MPI_DOUBLE, m.mpi_op.MPI_SUM\n"
    )


    def run(code):
        """Exit status and stdout of `code` in a child interpreter (a segmentation fault is -11)."""
        p = subprocess.run(
            [sys.executable, "-I", "-c", code], capture_output=True, text=True, timeout=60, check=False
        )
        return p.returncode, p.stdout.strip() or "<none>"


    W = mpi.mpi_comm.MPI_COMM_WORLD
    INT = mpi.mpi_datatype.MPI_INT

    reproduced = False

    # A broadcast from the root returns the root's data as a tuple: tuple(range(3)) = (0, 1, 2).
    obs = mpi.MPI_Bcast(range(3), INT, 0, W)
    print(f"Bcast(range(3), MPI_INT): observed {obs}, expected (0, 1, 2) (root data)")
    reproduced |= obs != (0, 1, 2)
    obs = mpi.MPI_Bcast([0, 1, 2], INT, 0, W)
    print(f"control Bcast([0, 1, 2], MPI_INT): observed {obs}, expected (0, 1, 2)")

    NP = "import numpy as np\n"
    Z8 = "np.ones(8, dtype=complex)"  # 8 harmonics of a 16-bin impedance table
    L8 = "[1 + 0j] * 8"
    LIMP = "from orbit.core.impedances import LImpedance\nZ = LImpedance(10.0, 1, 16)\n"
    TIMP = "from orbit.core.impedances import TImpedance\nZ = TImpedance(10.0, 1, 16, 1, 1)\n"
    LSC = "from orbit.core.spacecharge import LSpaceChargeCalc\nZ = LSpaceChargeCalc(3.0, 10.0, 1, 1, 16)\n"
    # (label, code with {a} for the argument, NumPy argument, list argument, expected output).
    # The serial send is a no-op returning MPI_SUCCESS (0); the setters print ok after storing.
    cases = [
        (
            "MPI_Bcast(a, MPI_DOUBLE)",
            MPI + "print(m.MPI_Bcast({a}, DOUBLE, 0, W))",
            "np.array([1.5, 2.5])",
            "[1.5, 2.5]",
            "(1.5, 2.5)",
        ),
        (
            "MPI_Send(a, MPI_DOUBLE)",
            MPI + "print(m.MPI_Send({a}, DOUBLE, 0, 0, W))",
            "np.array([1.5, 2.5])",
            "[1.5, 2.5]",
            "0",
        ),
        (
            "MPI_Group_incl(g, a)",
            MPI + "g = m.MPI_Comm_group(W)\nprint(m.MPI_Group_size(m.MPI_Group_incl(g, {a})))",
            "np.array([0])",
            "[0]",
            "1",
        ),
        (
            "MPI_Group_excl(g, a)",
            MPI + "g = m.MPI_Comm_group(W)\nprint(type(m.MPI_Group_excl(g, {a})).__name__)",
            "np.array([0])",
            "[0]",
            "MPI_Group",
        ),
        (
            "MPI_Graph_create(W, a, a, 0, W)",
            MPI + "print(type(m.MPI_Graph_create(W, {a}, {a}, 0, W)).__name__)",
            "np.array([1])",
            "[1]",
            "int",
        ),
        ("LImpedance.assignImpedance(a)", LIMP + "Z.assignImpedance({a})\nprint('ok')", Z8, L8, "ok"),
        (
            "TImpedance.assignImpedance('X', a, a)",
            TIMP + "Z.assignImpedance('X', {a}, {a})\nprint('ok')",
            Z8,
            L8,
            "ok",
        ),
        (
            "LSpaceChargeCalc.assignImpedance(a)",
            LSC + "Z.assignImpedance({a})\nprint('ok')",
            Z8,
            L8,
            "ok",
        ),
    ]
    for label, code, arr, lst, exp in cases:
        rc, out = run(NP + code.format(a=arr))
        print(
            f"{label} with a NumPy array: observed exit {rc}, output {out}; expected exit 0, output {exp}"
        )
        reproduced |= rc != 0 or out != exp
        rc, out = run(NP + code.format(a=lst))
        print(
            f"control {label} with a list: observed exit {rc}, output {out}; expected exit 0, output {exp}"
        )

    rc, out = run(NP + MPI + "print(m.MPI_Allreduce(np.array([1.5, 2.5]), DOUBLE, SUM, W))")
    print(
        f"control MPI_Allreduce(NumPy array, MPI_SUM): observed exit {rc}, output {out}; expected exit 0, output (1.5, 2.5)"
    )

    print("DEFECT REPRODUCED" if reproduced else "NOT REPRODUCED")

    ```

## Provenance

| Role | Model | Effort | Date | Code examined |
|---|---|---|---|---|
| Found | <span class="bb-nowrap">`claude-opus-5-5`</span> | high | <span class="bb-nowrap">2026-11-07</span> | main (`28df66018aeb`), source, serial, CPU, double |
| Confirmed | <span class="bb-nowrap">`claude-sonnet-5-5`</span> | high | <span class="bb-nowrap">2026-11-07</span> | main (`28df66018aeb`), source, serial, CPU, double |

