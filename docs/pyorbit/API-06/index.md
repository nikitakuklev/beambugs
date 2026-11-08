---
title: "API-06: Serial MPI_Comm_get_name returns six bytes of an unwritten buffer"
hide:
  - toc
---

# API-06 — Serial `MPI_Comm_get_name` returns six bytes of an unwritten buffer

| | |
|---|---|
| Code | [PyORBIT](../index.md) · Bindings and build |
| Priority | P4 |
| Failure | Crash or API hazard |
| Evidence | Reproduced, independently rechecked |
| Last reproduced | 2026-11-07 on PyORBIT commit [`28df660`](https://github.com/PyORBIT-Collaboration/PyORBIT3/commit/28df66018aeb), committed 2026-10-14 |
| Source | [orbit_mpi.cc#L268-L279](https://github.com/PyORBIT-Collaboration/PyORBIT3/blob/28df66018aebe4e2b780b9747f4a2d7b8430a5c0/src/mpi/orbit_mpi.cc#L268-L279)<br>[wrap_orbit_mpi_comm_functions.hh#L118-L130](https://github.com/PyORBIT-Collaboration/PyORBIT3/blob/28df66018aebe4e2b780b9747f4a2d7b8430a5c0/src/mpi/wrap_orbit_mpi_comm_functions.hh#L118-L130)<br>[orbit_mpi.cc#L126-L138](https://github.com/PyORBIT-Collaboration/PyORBIT3/blob/28df66018aebe4e2b780b9747f4a2d7b8430a5c0/src/mpi/orbit_mpi.cc#L126-L138)<br>[sc_1D_test.py#L29](https://github.com/PyORBIT-Collaboration/PyORBIT3/blob/28df66018aebe4e2b780b9747f4a2d7b8430a5c0/examples/SpaceCharge/sc2p5D/sc_1D_test.py#L29) |

In builds without MPI, `orbit_mpi.MPI_Comm_get_name(comm)` returns six bytes of a freshly allocated buffer that nothing writes: the serial stub assigns the text "no mpi" to its local copy of the pointer instead of copying it into the buffer, but reports length 6. The result is arbitrary heap content, so the call usually raises `UnicodeDecodeError` and otherwise returns garbage, where "no mpi" is expected. A shipped space-charge example prints this name. `MPI_Get_processor_name` has the same pointer assignment but reports length 0, so it returns an empty string.

Run `python API-06.py`.

??? example "Reproducer: API-06.py"

    [Download API-06.py](API-06.py)

    ```python
    """API-06: Serial `MPI_Comm_get_name` returns six bytes of an unwritten buffer

    In builds without MPI, `orbit_mpi.MPI_Comm_get_name(comm)` returns six bytes of a freshly allocated buffer that nothing writes: the serial stub assigns the text "no mpi" to its local copy of the pointer instead of copying it into the buffer, but reports length 6. The result is arbitrary heap content, so the call usually raises `UnicodeDecodeError` and otherwise returns garbage, where "no mpi" is expected. A shipped space-charge example prints this name. `MPI_Get_processor_name` has the same pointer assignment but reports length 0, so it returns an empty string.

    The script calls `MPI_Comm_get_name(MPI_COMM_WORLD)` in ten fresh child interpreters and prints each result or exception against the text the serial stub assigns. It also prints `MPI_Get_processor_name()`. It applies to a build without MPI (`MPI_Initialized() == 0`) and reports NOT REPRODUCED on an MPI build.
    """

    import os
    import subprocess
    import sys

    for _v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
        os.environ.setdefault(_v, "1")

    from orbit.core import orbit_mpi as mpi

    serial = mpi.MPI_Initialized() == 0
    CODE = (
        "from orbit.core import orbit_mpi as m\n"
        "try:\n"
        "    print(repr(m.MPI_Comm_get_name(m.mpi_comm.MPI_COMM_WORLD)))\n"
        "except Exception as e:\n"
        "    print(type(e).__name__)\n"
    )
    # The serial stub means to return "no mpi"; 6 is the length it reports.
    expected = repr("no mpi")
    bad = 0
    for i in range(10):
        p = subprocess.run(
            [sys.executable, "-I", "-c", CODE], capture_output=True, text=True, timeout=60, check=False
        )
        obs = p.stdout.strip() or f"exit {p.returncode}"
        print(f"MPI_Comm_get_name run {i}: observed {obs}, expected {expected}")
        bad += obs != expected
    print(
        f"MPI_Get_processor_name(): observed {mpi.MPI_Get_processor_name()!r} (serial stub reports length 0)"
    )
    print(f"serial build: {serial}")
    print("DEFECT REPRODUCED" if serial and bad else "NOT REPRODUCED")

    ```

## Provenance

| Role | Model | Effort | Date | Code examined |
|---|---|---|---|---|
| Found | `claude-opus-5-5` | high | 2026-11-07 | main (`28df66018aeb`), source, serial, CPU, double |
| Confirmed | `claude-sonnet-5-5` | high | 2026-11-07 | main (`28df66018aeb`), source, serial, CPU, double |

