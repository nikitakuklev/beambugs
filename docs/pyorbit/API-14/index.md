---
title: "API-14: orbit.__version__ reads another distribution's version"
hide:
  - toc
---

# API-14 — `orbit.__version__` reads another distribution's version

| | |
|---|---|
| Code | [PyORBIT](../index.md) · Bindings and build |
| Priority | P5 |
| Failure | Crash or API hazard |
| Evidence | Reproduced, independently rechecked |
| Last reproduced | 2026-11-07 on PyORBIT commit [`28df660`](https://github.com/PyORBIT-Collaboration/PyORBIT3/commit/28df66018aeb), committed 2026-10-14 |
| Source | [py/orbit/__init__.py#L12-L22](https://github.com/PyORBIT-Collaboration/PyORBIT3/blob/28df66018aebe4e2b780b9747f4a2d7b8430a5c0/py/orbit/__init__.py#L12-L22)<br>[doc/source/conf.py#L18](https://github.com/PyORBIT-Collaboration/PyORBIT3/blob/28df66018aebe4e2b780b9747f4a2d7b8430a5c0/doc/source/conf.py#L18) |

`orbit/__init__.py` sets `__version__` from the installed metadata of `ess-python-tools`, an unrelated distribution, instead of `PyORBIT`. Without that package `orbit.__version__` is unset and code that reads it raises `AttributeError`; with it installed, PyORBIT reports that package's version. On an installed source build, `importlib.metadata.version("PyORBIT")` returns the development version while the attribute is unset, and with a synthetic `ess-python-tools` 9.9.9 metadata entry on the path, `orbit.__version__` is "9.9.9". The Sphinx configuration also labels the documentation with an older release, v3.0.1.

Run `python API-14.py`.

??? example "Reproducer: API-14.py"

    [Download API-14.py](API-14.py)

    ```python
    """API-14: `orbit.__version__` reads another distribution's version

    `orbit/__init__.py` sets `__version__` from the installed metadata of `ess-python-tools`, an unrelated distribution, instead of `PyORBIT`. Without that package `orbit.__version__` is unset and code that reads it raises `AttributeError`; with it installed, PyORBIT reports that package's version. On an installed source build, `importlib.metadata.version("PyORBIT")` returns the development version while the attribute is unset, and with a synthetic `ess-python-tools` 9.9.9 metadata entry on the path, `orbit.__version__` is "9.9.9". The Sphinx configuration also labels the documentation with an older release, v3.0.1.

    Setup: the script reads `doc/source/conf.py` of the source checkout named by `PYORBIT_SOURCE`.

    The script imports `orbit` in a child interpreter and compares `orbit.__version__` with the installed `PyORBIT` metadata version, then repeats this with a temporary directory holding `ess-python-tools` 9.9.9 metadata on `sys.path`. It prints the Sphinx `release` value and the latest release tag named in the checkout's git metadata.
    """

    import os
    import re
    import subprocess
    import sys
    import tempfile

    src = os.environ.get("PYORBIT_SOURCE", "")
    if not src:
        sys.exit("PYORBIT_SOURCE must point to a PyORBIT source checkout")

    CODE = (
        "import sys\n"
        "if len(sys.argv) > 1: sys.path.insert(0, sys.argv[1])\n"
        "import orbit\n"
        "from importlib.metadata import version\n"
        "print(getattr(orbit, '__version__', '<unset>'), version('PyORBIT'))\n"
    )


    def versions(*extra):
        p = subprocess.run(
            [sys.executable, "-I", "-c", CODE, *extra], capture_output=True, text=True, check=True
        )
        return p.stdout.split()


    # The package attribute must equal the installed distribution's version.
    reproduced = False
    attr, dist = versions()
    print(f"orbit.__version__ as installed: observed {attr}, expected {dist} (version('PyORBIT'))")
    reproduced |= attr != dist
    with tempfile.TemporaryDirectory() as d:
        meta = os.path.join(d, "ess_python_tools-9.9.9.dist-info")
        os.mkdir(meta)
        with open(os.path.join(meta, "METADATA"), "w") as f:
            f.write("Metadata-Version: 2.1\nName: ess-python-tools\nVersion: 9.9.9\n")
        attr, dist = versions(d)
    print(
        f"orbit.__version__ with ess-python-tools 9.9.9 metadata on the path: observed {attr}, expected {dist}"
    )
    reproduced |= attr != dist

    with open(os.path.join(src, "doc", "source", "conf.py")) as f:
        m = re.search(r'^release\s*=\s*"([^"]*)"', f.read(), re.MULTILINE)
    assert m, "release not found in conf.py"
    release = m.group(1)
    tag = subprocess.run(
        ["git", "-C", src, "describe", "--tags", "--abbrev=0"],
        capture_output=True,
        text=True,
        check=False,
    ).stdout.strip()
    print(
        f"Sphinx release label: observed {release}, expected the latest release tag {tag or '<no git metadata>'}"
    )

    print("DEFECT REPRODUCED" if reproduced else "NOT REPRODUCED")

    ```

## Provenance

| Role | Model | Effort | Date | Code examined |
|---|---|---|---|---|
| Found | <span class="bb-nowrap">`claude-opus-5-5`</span> | high | <span class="bb-nowrap">2026-11-07</span> | main (`28df66018aeb`), source, serial, CPU, double |
| Confirmed | <span class="bb-nowrap">`claude-sonnet-5-5`</span> | high | <span class="bb-nowrap">2026-11-07</span> | main (`28df66018aeb`), source, serial, CPU, double |

