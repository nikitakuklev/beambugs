---
title: "IO-02: A second collect_bunch call overwrites the coordinates returned by the first"
hide:
  - toc
---

# IO-02 — A second `collect_bunch` call overwrites the coordinates returned by the first

| | |
|---|---|
| Code | [PyORBIT](../index.md) · Particles and I/O |
| Priority | P3 |
| Failure | Silent wrong result |
| Evidence | Reproduced, independently rechecked |
| Last reproduced | 2026-11-07 on PyORBIT commit [`28df660`](https://github.com/PyORBIT-Collaboration/PyORBIT3/commit/28df66018aeb), committed 2026-10-14 |
| Source | [serialize.py#L128-L130](https://github.com/PyORBIT-Collaboration/PyORBIT3/blob/28df66018aebe4e2b780b9747f4a2d7b8430a5c0/py/orbit/bunch_utils/serialize.py#L128-L130)<br>[serialize.py#L171-L178](https://github.com/PyORBIT-Collaboration/PyORBIT3/blob/28df66018aebe4e2b780b9747f4a2d7b8430a5c0/py/orbit/bunch_utils/serialize.py#L171-L178)<br>[serialize.py#L217-L217](https://github.com/PyORBIT-Collaboration/PyORBIT3/blob/28df66018aebe4e2b780b9747f4a2d7b8430a5c0/py/orbit/bunch_utils/serialize.py#L217-L217)<br>[serialize.py#L241-L243](https://github.com/PyORBIT-Collaboration/PyORBIT3/blob/28df66018aebe4e2b780b9747f4a2d7b8430a5c0/py/orbit/bunch_utils/serialize.py#L241-L243)<br>[serialize.py#L272-L274](https://github.com/PyORBIT-Collaboration/PyORBIT3/blob/28df66018aebe4e2b780b9747f4a2d7b8430a5c0/py/orbit/bunch_utils/serialize.py#L272-L274)<br>[test_lsc_gauss.py#L73-L75](https://github.com/PyORBIT-Collaboration/PyORBIT3/blob/28df66018aebe4e2b780b9747f4a2d7b8430a5c0/examples/SpaceCharge/sc1d/test_lsc_gauss.py#L73-L75) |

`collect_bunch(bunch, output_dir)` returns, by default, a NumPy memory map backed by a
file whose name depends only on the MPI rank (`collect_bunch_tmpfile_0.dat` on the root). The next
collection into the same directory, by the same process or by another job sharing it (the default is
`/tmp`), reopens that file in write mode, so every array returned earlier silently changes to the
new bunch's coordinates, or to zeros past the new length. Collecting a three-particle bunch with
x = 1, 2, 3 and then one with x = −5, −6, −7 changes the first result to −5, −6, −7; with a
one-particle second bunch it becomes −5, 0, 0. Copies made with `return_memmap=False` are
unaffected.

Run `python IO-02.py`.

??? example "Reproducer: IO-02.py"

    [Download IO-02.py](IO-02.py)

    ```python
    """IO-02: A second `collect_bunch` call overwrites the coordinates returned by the first

    `collect_bunch(bunch, output_dir)` returns, by default, a NumPy memory map backed by a file whose
    name depends only on the MPI rank (`collect_bunch_tmpfile_0.dat` on the root). The next collection
    into the same directory, by the same process or by another job sharing it (the default is `/tmp`),
    reopens that file in write mode, so every array returned earlier silently changes to the new bunch's
    coordinates, or to zeros past the new length. Collecting a three-particle bunch with x = 1, 2, 3 and
    then one with x = −5, −6, −7 changes the first result to −5, −6, −7; with a one-particle second
    bunch it becomes −5, 0, 0. Copies made with `return_memmap=False` are unaffected.

    The script collects a serial bunch with x = 1, 2, 3 into a fresh directory with the default
    `return_memmap=True`, then collects a second bunch into the same directory, from the same process or
    from a child interpreter, and prints the first result before and after. Controls repeat the sequence
    with `return_memmap=False` and with the second collection in another directory.
    """

    import subprocess
    import sys
    import tempfile

    import numpy as np
    from orbit.bunch_utils import collect_bunch
    from orbit.core.bunch import Bunch

    FIRST_X = [1.0, 2.0, 3.0]
    CHILD = (
        "import sys\n"
        "from orbit.core.bunch import Bunch\n"
        "from orbit.bunch_utils import collect_bunch\n"
        "b = Bunch()\n"
        "for x in (100.0, 200.0, 300.0):\n"
        "    b.addParticle(x, 0.0, 0.0, 0.0, 0.0, 0.0)\n"
        "collect_bunch(b, sys.argv[1])\n"
    )


    def bunch_with(xs):
        b = Bunch()
        for x in xs:
            b.addParticle(x, 0.0, 0.0, 0.0, 0.0, 0.0)
        return b


    def x_of(result):
        return np.array(result["coords"][:, 0]).tolist()


    def case(label, second, memmap=True, same_dir=True):
        """Collect FIRST_X, run `second(directory)`, and report the first result's x before and after."""
        d = tempfile.mkdtemp()
        r1 = collect_bunch(bunch_with(FIRST_X), d, return_memmap=memmap)
        before = x_of(r1)
        second(d if same_dir else tempfile.mkdtemp())
        after = x_of(r1)
        # A returned collection must keep the coordinates of the bunch it collected.
        print(
            f"{label}: first result x before {before}, after second collection {after}; expected {FIRST_X}"
        )
        return after != FIRST_X


    def same_process(xs, memmap=True):
        return lambda d: collect_bunch(bunch_with(xs), d, return_memmap=memmap)


    def other_process(d):
        subprocess.run([sys.executable, "-I", "-c", CHILD, d], check=True)


    changed = [
        case("equal-size second bunch, same process", same_process([-5.0, -6.0, -7.0])),
        case("one-particle second bunch, same process", same_process([-5.0])),
        case("second collection by another process", other_process),
    ]
    unchanged = [
        not case(
            "return_memmap=False (control)", same_process([-5.0, -6.0, -7.0], False), memmap=False
        ),
        not case(
            "second collection in another directory (control)",
            same_process([-5.0, -6.0, -7.0]),
            same_dir=False,
        ),
    ]
    print("DEFECT REPRODUCED" if all(changed) and all(unchanged) else "NOT REPRODUCED")

    ```

## Provenance

| Role | Model | Effort | Date | Code examined |
|---|---|---|---|---|
| Found | `claude-opus-5-5` | high | 2026-11-07 | main (`28df66018aeb`), source, serial, CPU, double |
| Confirmed | `claude-sonnet-5-5` | high | 2026-11-07 | main (`28df66018aeb`), source, serial, CPU, double |

