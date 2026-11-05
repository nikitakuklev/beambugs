"""Serial CPU setup that runs Xsuite's prebuilt kernels with kernel compilation forbidden.

`identity()` prints the package versions and context; `kernels()` asserts that a prebuilt kernel
library was loaded and prints it.
"""

import importlib
import importlib.metadata
import os
import platform
from pathlib import Path

# Configure thread pools before importing NumPy or Xsuite.
for name in (
    "OMP_NUM_THREADS",
    "OPENBLAS_NUM_THREADS",
    "MKL_NUM_THREADS",
    "NUMEXPR_NUM_THREADS",
    "NPROC",
    "PJRT_NPROC",
):
    os.environ[name] = "1"
os.environ["XSUITE_ALLOW_KERNEL_COMPILATION"] = "0"
os.environ["XSUITE_FORCE_KERNEL_COMPILATION"] = "0"
os.environ["XSUITE_CFFI_FORBID_COMPILE"] = "1"

import numpy as np
import xcoll as xc
import xobjects as xo
import xtrack as xt
from _common import report

__all__ = ["CTX", "identity", "kernels", "np", "particles", "report", "xc", "xt"]

np.set_printoptions(precision=17)
xo.settings.allow_kernel_compilation = False
xo.settings.force_kernel_compilation = False
CTX = xo.ContextCpu(omp_num_threads=0)


def identity(*, radiation="off", random_seed=None):
    print(f"Python {platform.python_version()}; {platform.platform()}")
    for name in ("numpy", "scipy", "cffi"):
        print(name, importlib.metadata.version(name))
    for name in ("xsuite", "xtrack", "xobjects", "xpart", "xfields", "xdeps", "xcoll"):
        mod = importlib.import_module(name)
        print("PACKAGE", name, mod.__version__, mod.__file__)
    randomness = (
        "deterministic, no random seed"
        if random_seed is None
        else f"fixed RNG seed(s)={random_seed}"
    )
    print(
        f"CONTEXT CPU serial; radiation {radiation}; {randomness}; "
        "kernel compilation forbidden; prebuilt kernels"
    )


def kernels():
    libraries = sorted(
        {
            line.split()[-1]
            for line in Path("/proc/self/maps").read_text().splitlines()
            if "/xsuite/lib/" in line and ".so" in line
        }
    )
    assert libraries, "No prebuilt kernel library loaded"
    print("KERNEL_LIBRARIES", libraries)


def particles(**kwargs):
    return xt.Particles(_context=CTX, p0c=1e9, **kwargs)
