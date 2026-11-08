---
title: "API-16: Legacy launcher and build files reference components absent from the repository"
hide:
  - toc
---

# API-16 — Legacy launcher and build files reference components absent from the repository

| | |
|---|---|
| Code | [PyORBIT](../index.md) · Bindings and build |
| Priority | P5 |
| Failure | Documentation or example mismatch |
| Evidence | Reproduced, independently rechecked |
| Last reproduced | 2026-11-07 on PyORBIT commit [`28df660`](https://github.com/PyORBIT-Collaboration/PyORBIT3/commit/28df66018aeb), committed 2026-10-14 |
| Source | [bin/pyORBIT#L1-L10](https://github.com/PyORBIT-Collaboration/PyORBIT3/blob/28df66018aebe4e2b780b9747f4a2d7b8430a5c0/bin/pyORBIT#L1-L10)<br>[Dockerfile#L1-L7](https://github.com/PyORBIT-Collaboration/PyORBIT3/blob/28df66018aebe4e2b780b9747f4a2d7b8430a5c0/Dockerfile#L1-L7)<br>[MesonBuild.md#L36-L40](https://github.com/PyORBIT-Collaboration/PyORBIT3/blob/28df66018aebe4e2b780b9747f4a2d7b8430a5c0/MesonBuild.md#L36-L40) |

Three files in the repository describe build or launch paths that no longer exist. `bin/pyORBIT` runs `python -m orbit.core.pyORBIT`, a module the package does not contain, so it fails at once with an import error. The `Dockerfile` installs Python 2 and runs `make` after sourcing `setupEnvironment.sh`, but the repository has neither that script nor a `Makefile`. `MesonBuild.md` tells users to clone the `meson` branch of a personal fork, which no longer has that branch. The README and the Meson files describe the working build.

Run `python API-16.py`.

??? example "Reproducer: API-16.py"

    [Download API-16.py](API-16.py)

    ```python
    """API-16: Legacy launcher and build files reference components absent from the repository

    Three files in the repository describe build or launch paths that no longer exist. `bin/pyORBIT` runs `python -m orbit.core.pyORBIT`, a module the package does not contain, so it fails at once with an import error. The `Dockerfile` installs Python 2 and runs `make` after sourcing `setupEnvironment.sh`, but the repository has neither that script nor a `Makefile`. `MesonBuild.md` tells users to clone the `meson` branch of a personal fork, which no longer has that branch. The README and the Meson files describe the working build.

    Setup: the script reads the source checkout named by `PYORBIT_SOURCE` and calls `git ls-remote`, which needs network access to GitHub.

    The script runs the module that `bin/pyORBIT` launches in a child interpreter, checks that the files the `Dockerfile` uses exist in the checkout, and asks the remote of the `MesonBuild.md` clone command whether it has the requested branch.
    """

    import os
    import re
    import subprocess
    import sys

    src = os.environ.get("PYORBIT_SOURCE", "")
    if not src:
        sys.exit("PYORBIT_SOURCE must point to a PyORBIT source checkout")


    def read(*parts):
        with open(os.path.join(src, *parts)) as f:
            return f.read()


    bad = False
    launch = re.search(r"python -m (\S+)", read("bin", "pyORBIT"))
    assert launch, "no python -m line in bin/pyORBIT"
    mod = launch.group(1)
    p = subprocess.run(
        [sys.executable, "-I", "-c", f"import runpy; runpy.run_module({mod!r})"],
        capture_output=True,
        check=False,
        text=True,
    )
    tail = ([ln for ln in p.stderr.splitlines() if ln.strip()] or ["<none>"])[-1]
    print(
        f"bin/pyORBIT runs module {mod}: observed exit {p.returncode}, {tail!r}; expected an importable launcher module"
    )
    bad |= p.returncode != 0

    docker = read("Dockerfile")
    for name in ("setupEnvironment.sh", "Makefile"):
        used = name in docker or (name == "Makefile" and re.search(r"\bmake\b", docker) is not None)
        present = os.path.exists(os.path.join(src, name))
        print(f"Dockerfile uses {name}: {used}; observed present {present}, expected True")
        bad |= used and not present
    print(f"Dockerfile installs python2: {'python2' in docker}")

    clone = re.search(r"git clone -b (\S+) (\S+)", read("MesonBuild.md"))
    assert clone, "no git clone -b line in MesonBuild.md"
    branch, url = clone.group(1), clone.group(2)
    r = subprocess.run(
        ["git", "ls-remote", "--heads", url, branch],
        capture_output=True,
        text=True,
        timeout=60,
        check=False,
    )
    if r.returncode != 0:
        sys.exit(f"git ls-remote {url} failed: {r.stderr.strip()}")
    has = bool(r.stdout.strip())
    print(
        f"MesonBuild.md clones branch {branch!r} of {url}: observed branch exists {has}, expected True"
    )
    bad |= not has

    print("DEFECT REPRODUCED" if bad else "NOT REPRODUCED")

    ```

## Provenance

| Role | Model | Effort | Date | Code examined |
|---|---|---|---|---|
| Found | <span class="bb-nowrap">`claude-opus-5-5`</span> | high | <span class="bb-nowrap">2026-11-07</span> | main (`28df66018aeb`), source, serial, CPU, double |
| Confirmed | <span class="bb-nowrap">`claude-sonnet-5-5`</span> | high | <span class="bb-nowrap">2026-11-07</span> | main (`28df66018aeb`), source, serial, CPU, double |

