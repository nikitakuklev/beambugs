---
title: "API-04: Serial MPI_Wtime returns process CPU time instead of wall-clock time"
hide:
  - toc
---

# API-04 — Serial `MPI_Wtime` returns process CPU time instead of wall-clock time

| | |
|---|---|
| Code | [PyORBIT](../index.md) · Bindings and build |
| Priority | P4 |
| Failure | Silent wrong result |
| Evidence | Reproduced, independently rechecked |
| Last reproduced | 2026-11-07 on PyORBIT commit [`28df660`](https://github.com/PyORBIT-Collaboration/PyORBIT3/commit/28df66018aeb), committed 2026-10-14 |
| Source | [orbit_mpi.cc#L140-L149](https://github.com/PyORBIT-Collaboration/PyORBIT3/blob/28df66018aebe4e2b780b9747f4a2d7b8430a5c0/src/mpi/orbit_mpi.cc#L140-L149) |

In builds without MPI, `orbit_mpi.MPI_Wtime()` returns `clock()/CLOCKS_PER_SEC`, the processor time used by the process, whereas MPI defines `MPI_Wtime` as elapsed wall-clock time. Time spent sleeping, waiting or in I/O is missing: across a 0.5 s `time.sleep`, `MPI_Wtime` advances by less than 10⁻⁴ s. Serial timings made with it understate elapsed time by the waiting time, and the same script measures a different quantity on an MPI build. Its intervals track the process CPU time, so they match elapsed time only while the process computes without being descheduled.

Run `python API-04.py`.

??? example "Reproducer: API-04.py"

    [Download API-04.py](API-04.py)

    ```python
    """API-04: Serial `MPI_Wtime` returns process CPU time instead of wall-clock time

    In builds without MPI, `orbit_mpi.MPI_Wtime()` returns `clock()/CLOCKS_PER_SEC`, the processor time used by the process, whereas MPI defines `MPI_Wtime` as elapsed wall-clock time. Time spent sleeping, waiting or in I/O is missing: across a 0.5 s `time.sleep`, `MPI_Wtime` advances by less than 10⁻⁴ s. Serial timings made with it understate elapsed time by the waiting time, and the same script measures a different quantity on an MPI build. Its intervals track the process CPU time, so they match elapsed time only while the process computes without being descheduled.

    The script reads `MPI_Wtime`, `time.perf_counter` (wall clock) and `time.process_time` (CPU time) around a 0.5 s sleep and compares the `MPI_Wtime` interval with the wall-clock one. As the control, a 0.3 s busy loop shows that the clock does advance, by the CPU time the process used. It applies to a build without MPI (`MPI_Initialized() == 0`) and reports NOT REPRODUCED on an MPI build.
    """

    import os
    import time

    for _v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
        os.environ.setdefault(_v, "1")

    from orbit.core import orbit_mpi as mpi

    serial = mpi.MPI_Initialized() == 0


    def intervals(wait):
        t0, w0, c0 = mpi.MPI_Wtime(), time.perf_counter(), time.process_time()
        wait()
        return mpi.MPI_Wtime() - t0, time.perf_counter() - w0, time.process_time() - c0


    def busy(seconds=0.3):
        w0 = time.perf_counter()
        while time.perf_counter() - w0 < seconds:
            pass


    # MPI_Wtime is elapsed wall-clock time (MPI standard, "Timers and Synchronization"), so its
    # interval must match perf_counter's to within the 1e-6 s tick plus scheduling noise.
    dt, dw, dc = intervals(lambda: time.sleep(0.5))
    print(f"serial build: {serial}")
    print(
        f"sleep(0.5): MPI_Wtime interval observed {dt:.3g} s, expected {dw:.4f} s (wall clock); CPU time {dc:.3g} s"
    )
    bt, bw, bc = intervals(busy)
    print(
        f"control 0.3 s busy loop: MPI_Wtime interval observed {bt:.4f} s, CPU time {bc:.4f} s; wall clock {bw:.4f} s"
    )
    print("DEFECT REPRODUCED" if serial and abs(dt - dw) > 0.1 else "NOT REPRODUCED")

    ```

## Provenance

| Role | Model | Effort | Date | Code examined |
|---|---|---|---|---|
| Found | `claude-opus-5-5` | high | 2026-11-07 | main (`28df66018aeb`), source, serial, CPU, double |
| Confirmed | `claude-sonnet-5-5` | high | 2026-11-07 | main (`28df66018aeb`), source, serial, CPU, double |

