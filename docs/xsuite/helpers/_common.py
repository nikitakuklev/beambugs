"""Shared setup and output for the defect reproducers.

`prepare()` moves to a fresh temporary working directory, where CFFI compiles the kernels, and
prints the package versions and context; `report()` prints the observed and expected rows and the
verdict line.
"""

import atexit
import importlib
import importlib.metadata
import os
import platform
import shutil
import tempfile
from pathlib import Path

import numpy as np
import xobjects as xo


def prepare(*, radiation="off", random_seed=None):
    # CFFI compiles into cwd; keep generated files outside the source trees.
    scratch = Path(tempfile.mkdtemp(prefix="xsuite-repro-"))
    atexit.register(shutil.rmtree, scratch, ignore_errors=True)
    os.chdir(scratch)
    xo.settings.allow_kernel_compilation = True
    print(f"Python {platform.python_version()}; {platform.platform()}")
    for name in ("xsuite", "xtrack", "xobjects", "xdeps", "xpart", "xfields", "xcoll"):
        module = importlib.import_module(name)
        print(f"{name} {module.__version__}: {module.__file__}")
    for name in ("numpy", "scipy", "madng-tpsa", "cffi"):
        print(f"{name} {importlib.metadata.version(name)}")
    randomness = (
        "deterministic, no random seed"
        if random_seed is None
        else f"fixed RNG seed(s)={random_seed}"
    )
    print(f"ContextCpu, serial; radiation={radiation}; {randomness}")
    print(
        f"allow_kernel_compilation={xo.settings.allow_kernel_compilation}; "
        f"force_kernel_compilation={xo.settings.force_kernel_compilation}"
    )


def report(cases, *, rtol, atol):
    np.set_printoptions(precision=15, linewidth=140)
    mismatch = False
    for label, (observed, expected) in cases.items():
        print(f"{label}: observed={np.asarray(observed)} expected={np.asarray(expected)}")
        mismatch |= not np.allclose(observed, expected, rtol=rtol, atol=atol)
    print(f"comparison: rtol={rtol:g}, atol={atol:g}")
    print("DEFECT REPRODUCED" if mismatch else "NOT REPRODUCED")
