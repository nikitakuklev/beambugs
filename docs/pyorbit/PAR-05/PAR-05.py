"""PAR-05: MAD-X attributes using asin, acos, atan, a name containing "pi" or a D exponent are set to zero

`readMADX` rewrites `sin(` to `math.sin(` before it looks for `asin(`, so `asin(x)` becomes the invalid `amath.sin(x)`, and likewise for `acos` and `atan`; it also replaces every substring `pi`, so a variable named `kpip` becomes `kmath.pip`. Numbers with the exponent letter D, such as `1.5d-3` (0.0015 in MAD-X), are not translated at all. An element attribute written with such an expression cannot be evaluated, and the reader prints a warning, sets the attribute to 0 and builds the element. `k1=asin(0.5)` gives a quadrupole with k₁ = 0 instead of 0.5236 m⁻², and `k1=kpip` with `kpip = 0.2` gives 0 instead of 0.2000 m⁻². Expressions with `sin` and with `pi` itself evaluate correctly.

The script writes one MAD-X sequence per expression, with a quadrupole whose k1 uses it, reads it with `readMADX` and prints the stored strength kq and whether the reader printed its warning. The expected values are the expressions evaluated with Python's `math`, and for the D-exponent literals the same number written with `e`; the controls are `sin(0.5)` and `2*pi/10`.
"""

import contextlib
import io
import math
import os

for _v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ.setdefault(_v, "1")
import tempfile

from orbit.teapot import TEAPOT_Lattice

CASES = (  # (variable definitions, k1 expression, expected value, control)
    ("", "asin(0.5)", math.asin(0.5), False),
    ("", "acos(0.5)", math.acos(0.5), False),
    ("", "atan(0.5)", math.atan(0.5), False),
    ("kpip = 0.2;\n", "kpip", 0.2, False),
    ("", "1.5d-3", 1.5e-3, False),  # MAD-X reads d and D as exponent letters
    ("", "1d-3", 1e-3, False),
    ("", "sin(0.5)", math.sin(0.5), True),
    ("", "2*pi/10", 2 * math.pi / 10, True),
)
bad = False
for defs, expr, expected, is_control in CASES:
    with tempfile.TemporaryDirectory() as d:
        path = os.path.join(d, "seq.madx")
        with open(path, "w") as f:
            f.write(
                defs
                + f"qa: quadrupole, l=1, k1={expr};\nring: sequence, l=10;\nqa, at=2;\nendsequence;\n"
            )
        lat = TEAPOT_Lattice()
        out = io.StringIO()
        with contextlib.redirect_stdout(out):
            lat.readMADX(path, "ring")
    kq = next(n for n in lat.getNodes() if n.getName() == "qa").getParam("kq")
    warned = "MADX File problem" in out.getvalue()
    label = "control, " if is_control else ""
    print(
        f"[{label}k1={expr}] kq observed {kq:.6f}, expected {expected:.6f}; reader warning printed: {warned}"
    )
    if not is_control:
        bad = bad or abs(kq - expected) > 1e-12
print("DEFECT REPRODUCED" if bad else "NOT REPRODUCED")
