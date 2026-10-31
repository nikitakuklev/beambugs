---
title: "IO-07: Manager.copy disconnects containers from dependency tasks"
hide:
  - toc
---

# IO-07 — Manager.copy disconnects containers from dependency tasks

| | |
|---|---|
| Code | [Xsuite](../index.md) · Lattices, particles and I/O |
| Priority | P4 |
| Severity | Silent wrong result |
| Evidence | Reproduced, independently rechecked |
| Last reproduced | 2026-10-29 on xdeps main (`6484795`) |
| Source | [Manager.copy, tasks.py:518–527](https://github.com/xsuite/xdeps/blob/73147cf7d9f1fc7b0f49ddb55c0ce037b70b7353/xdeps/tasks.py#L518-L527)<br>[Manager.set_value](https://github.com/xsuite/xdeps/blob/73147cf7d9f1fc7b0f49ddb55c0ce037b70b7353/xdeps/tasks.py#L239-L253) |

In Xdeps, the dependency-graph library behind Xtrack's deferred expressions, `Manager.copy` deep-copies the containers, the tasks (expressions computing dependent values) and the dependency maps separately, so the returned containers and tasks refer to different copies and references into the copy stay bound to a hidden manager. `Manager.set_value` promises to update the target and run its dependent tasks, and `copy.deepcopy` of the whole manager keeps the copied graph connected. With `a.output = a.input * b.gain`, `a.input` = 4 and `b.gain` = 2.5, setting the copied gain to 7 leaves the copied output at 10 instead of 28; redefining it as `9 * a.input` and setting the input to 6 then gives 42 instead of 54. Users who copy a dependency manager directly are affected; no Xtrack element, matching or tracking path was tested.

??? example "Reproducer: IO-07.py"

    [Download IO-07.py](IO-07.py)

    ```python
    """IO-07: Manager.copy disconnects containers from dependency tasks

    In Xdeps, the dependency-graph library behind Xtrack's deferred expressions, `Manager.copy`
    deep-copies the containers, the tasks (expressions computing dependent values) and the dependency
    maps separately, so the returned containers and tasks refer to different copies and references into
    the copy stay bound to a hidden manager. `Manager.set_value` promises to update the target and run
    its dependent tasks, and `copy.deepcopy` of the whole manager keeps the copied graph connected. With
    `a.output = a.input * b.gain`, `a.input` = 4 and `b.gain` = 2.5, setting the copied gain to 7 leaves
    the copied output at 10 instead of 28; redefining it as `9 * a.input` and setting the input to 6
    then gives 42 instead of 54. Users who copy a dependency manager directly are affected; no Xtrack
    element, matching or tracking path was tested.

    The script builds a manager with containers `a = {input: 4}` and `b = {gain: 2.5}` linked by
    `a.output = a.input * b.gain` and copies it with `Manager.copy`. In the copy it sets the gain to 7
    (expected output 28), then redefines the output as `9 * a.input` and sets the input to 6 through its
    reference (expected 54). Copying with `copy.deepcopy` is the control, and the original graph must
    stay unchanged in both cases.
    """

    from copy import deepcopy

    import xdeps as xd

    print("PACKAGE xdeps", xd.__version__, xd.__file__)
    print("CONTEXT Python dependency graph; no tracking kernels or random seed")


    def make():
        manager = xd.Manager()
        a = manager.ref({"input": 4.0}, "a")
        b = manager.ref({"gain": 2.5}, "b")
        a["output"] = a["input"] * b["gain"]
        return manager


    for method in ("copy", "deepcopy"):
        original = make()
        duplicate = original.copy() if method == "copy" else deepcopy(original)
        da, db = duplicate.containers["a"], duplicate.containers["b"]
        duplicate.set_value(db["gain"], 7.0)
        observed = da._owner["output"]
        print(
            method,
            "gain=7 output=",
            observed,
            "expected=28",
            "refs_bound_to_result=",
            da._manager is duplicate,
        )
        print(method, "original=", original.containers["a"]._owner)
        if method == "copy":
            assert observed == 10.0
            assert da._manager is not duplicate
            target = next(iter(duplicate.tasks.values())).taskid
            assert target._owner._owner is not da._owner
            # Updating through a copied reference reaches another hidden manager.
            duplicate.set_value(da["output"], 9.0 * da["input"])
            da["input"] = 6.0
            print("copy redefine gain=9 then input=6 output=", da._owner["output"], "expected=54")
            assert da._owner["output"] == 42.0  # Hidden manager retained gain=7.
        else:
            assert observed == 28.0
            assert da._manager is duplicate
        assert original.containers["a"]._owner == {"input": 4.0, "output": 10.0}
    print("DEFECT REPRODUCED")

    ```

## Provenance

| Role | Model | Effort | Date | Code examined |
|---|---|---|---|---|
| Found | `gpt-6-astra` | xhigh | 2026-10-28 | xdeps 0.10.21 (`73147cf7d9f1`), installed packages, Python only, serial |
| Confirmed | `gpt-6.1-sol` | xhigh | 2026-10-28 | xdeps 0.10.21 (`73147cf7d9f1`), installed packages, Python only, serial |

