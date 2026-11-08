"""API-15: `from orbit.bunch_utils import *` raises `AttributeError` without NumPy

`orbit.bunch_utils` imports its serialization helpers inside `try/except ImportError`, so that the package loads without NumPy, but always lists them in `__all__`. Without NumPy, `from orbit.bunch_utils import *` therefore raises `AttributeError: module 'orbit.bunch_utils' has no attribute 'collect_bunch'` instead of importing `ParticleIdNumber`; seven of the eight `__all__` names are missing. Named imports such as `from orbit.bunch_utils import ParticleIdNumber` work, and with NumPy installed the star import works.

The script hides NumPy in child interpreters by setting `sys.modules["numpy"] = None`, which makes `import numpy` raise the `ImportError` the guard catches, and runs the star import and a listing of the `__all__` names the package lacks. Controls run the named import without NumPy and the star import with NumPy.
"""

import subprocess
import sys

NO_NUMPY = "import sys\nsys.modules['numpy'] = None\n"
cases = [
    (
        "star import without NumPy",
        NO_NUMPY + "from orbit.bunch_utils import *\nprint('ok', 'ParticleIdNumber' in dir())",
        "ok True",
        False,
    ),
    (
        "__all__ names missing without NumPy",
        NO_NUMPY
        + "import orbit.bunch_utils as b\nprint([n for n in b.__all__ if not hasattr(b, n)])",
        "[]",
        False,
    ),
    (
        "control named import without NumPy",
        NO_NUMPY + "from orbit.bunch_utils import ParticleIdNumber\nprint('ok')",
        "ok",
        True,
    ),
    (
        "control star import with NumPy",
        "from orbit.bunch_utils import *\nprint('ok', 'ParticleIdNumber' in dir())",
        "ok True",
        True,
    ),
]
reproduced = False
for label, code, exp, control in cases:
    p = subprocess.run(
        [sys.executable, "-I", "-c", code], capture_output=True, text=True, timeout=60, check=False
    )
    out = p.stdout.strip() or "<none>"
    err = ([ln for ln in p.stderr.splitlines() if ln.strip()] or ["<none>"])[-1]
    print(
        f"{label}: observed exit {p.returncode}, output {out}, stderr tail {err!r}; expected exit 0, output {exp}"
    )
    reproduced |= not control and (p.returncode != 0 or out != exp)

print("DEFECT REPRODUCED" if reproduced else "NOT REPRODUCED")
