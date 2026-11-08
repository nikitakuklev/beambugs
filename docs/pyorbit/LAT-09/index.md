---
title: "LAT-09: Linac Quad accepts odd and negative part counts and divides by zero for none"
hide:
  - toc
---

# LAT-09 — Linac `Quad` accepts odd and negative part counts and divides by zero for none

| | |
|---|---|
| Code | [PyORBIT](../index.md) · Lattice API |
| Priority | P4 |
| Failure | Crash or API hazard |
| Evidence | Reproduced, independently rechecked |
| Last reproduced | 2026-11-07 on PyORBIT commit [`28df660`](https://github.com/PyORBIT-Collaboration/PyORBIT3/commit/28df66018aeb), committed 2026-10-14 |
| Source | [LinacAccNodes.py#L392-L415](https://github.com/PyORBIT-Collaboration/PyORBIT3/blob/28df66018aebe4e2b780b9747f4a2d7b8430a5c0/py/orbit/py_linac/lattice/LinacAccNodes.py#L392-L415) |

`Quad.initialize` states that a linac quadrupole needs an even number of parts, at least two, so that correctors can sit at its center. Its guard `nParts < 2 and nParts % 2 != 0` rejects only a single part: three, five or any other odd count is accepted, a negative count is accepted with an empty part list, and zero parts raise `ZeroDivisionError` instead of the intended message. The rejection message also loses its first sentence because a later assignment overwrites it.

Run `python LAT-09.py`.

??? example "Reproducer: LAT-09.py"

    [Download LAT-09.py](LAT-09.py)

    ```python
    """LAT-09: Linac `Quad` accepts odd and negative part counts and divides by zero for none

    `Quad.initialize` states that a linac quadrupole needs an even number of parts, at least two, so that correctors can sit at its center. Its guard `nParts < 2 and nParts % 2 != 0` rejects only a single part: three, five or any other odd count is accepted, a negative count is accepted with an empty part list, and zero parts raise `ZeroDivisionError` instead of the intended message. The rejection message also loses its first sentence because a later assignment overwrites it.

    The script sets 0 to 5 and −2 parts on a 1.2 m linac quadrupole, each in its own subprocess because the rejection path exits the interpreter, and reports whether each count is accepted and the part lengths. The expected result is acceptance exactly for even counts of at least two; the counts 2 and 4 are the control.
    """

    import os
    import subprocess
    import sys

    for _v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
        os.environ.setdefault(_v, "1")

    CODE = """
    from orbit.py_linac.lattice import Quad
    q = Quad("q")
    q.setLength(1.2)
    try:
        q.setnParts({n})
        print("accepted, part lengths [m]", [round(q.getLength(i), 4) for i in range(q.getnParts())])
    except ZeroDivisionError as err:
        print("ZeroDivisionError:", err)
    """

    reproduced = False
    for n in (0, 1, 2, 3, 4, 5, -2):
        run = subprocess.run(
            [sys.executable, "-I", "-c", CODE.format(n=n)], capture_output=True, text=True, check=False
        )
        out = run.stdout.strip().splitlines()
        msg = [ln for ln in out if ln.startswith("ORBIT message")]
        result = (
            out[-1]
            if run.returncode == 0 and out
            else f"rejected (exit {run.returncode}): {msg[0] if msg else ''}"
        )
        should_accept = n >= 2 and n % 2 == 0
        accepted = run.returncode == 0 and bool(out) and out[-1].startswith("accepted")
        expectation = (
            "accepted (even count >= 2)" if should_accept else "rejected with the guard message"
        )
        print(f"nParts={n}: {result}; expected {expectation}")
        if should_accept and not accepted:
            raise RuntimeError(f"control case failed: an even part count {n} is not accepted")
        reproduced |= not should_accept and run.returncode == 0
    print("DEFECT REPRODUCED" if reproduced else "NOT REPRODUCED")

    ```

## Provenance

| Role | Model | Effort | Date | Code examined |
|---|---|---|---|---|
| Found | <span class="bb-nowrap">`claude-opus-5-5`</span> | high | <span class="bb-nowrap">2026-11-07</span> | main (`28df66018aeb`), source, serial, CPU, double |
| Confirmed | <span class="bb-nowrap">`claude-sonnet-5-5`</span> | high | <span class="bb-nowrap">2026-11-07</span> | main (`28df66018aeb`), source, serial, CPU, double |

