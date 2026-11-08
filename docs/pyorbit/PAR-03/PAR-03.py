"""PAR-03: HKICKER and VKICKER lose their KICK strength on MAD and MAD-X import

In MAD-X and MAD-8 the strength of an HKICKER or VKICKER is its KICK attribute, the change of px or py; only the two-plane KICKER uses HKICK and VKICK. The TEAPOT element factory reads only `hkick` and `vkick`, so through `readMADX` and `readMAD` every single-plane corrector defined with KICK is built with zero kick and no message. For `hk: hkicker, kick=0.001` and `vk: vkicker, kick=0.002` an on-axis particle leaves with px = py = 0 instead of 1.000×10⁻³ and 2.000×10⁻³. Correctors written with `hkick=` or `vkick=` import correctly.

The script writes a MAD-X sequence and an equivalent MAD-8 line holding the two correctors to a temporary directory, reads each with its reader and tracks one on-axis proton. The expected px and py are the KICK values, the momentum change these correctors define. The control repeats the MAD-X sequence with the correctors written as `hkick=` and `vkick=`.
"""

import os

for _v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ.setdefault(_v, "1")
import tempfile

from orbit.core.bunch import Bunch
from orbit.teapot import TEAPOT_Lattice

MADX_TAIL = "ring: sequence, l=10;\nhk, at=4;\nvk, at=5;\nendsequence;\n"
CASES = (
    (
        "MAD-X, kick=",
        "readMADX",
        "seq.madx",
        "hk: hkicker, l=0, kick=0.001;\nvk: vkicker, l=0, kick=0.002;\n" + MADX_TAIL,
        False,
    ),
    (
        "MAD-8, kick=",
        "readMAD",
        "line.mad",
        "hk: hkicker, l=0.0, kick=0.001\nvk: vkicker, l=0.0, kick=0.002\nd: drift, l=1.0\nring: line=(d, hk, vk, d)\n",
        False,
    ),
    (
        "control, MAD-X, hkick= and vkick=",
        "readMADX",
        "seq.madx",
        "hk: hkicker, l=0, hkick=0.001;\nvk: vkicker, l=0, vkick=0.002;\n" + MADX_TAIL,
        True,
    ),
)
bad = False
for tag, reader, name, text, is_control in CASES:
    with tempfile.TemporaryDirectory() as d:
        path = os.path.join(d, name)
        with open(path, "w") as f:
            f.write(text)
        lat = TEAPOT_Lattice()
        getattr(lat, reader)(path, "ring")
    b = Bunch()
    b.mass(0.938272)
    b.charge(1)
    b.getSyncParticle().kinEnergy(1.0)
    b.addParticle(0.0, 0.0, 0.0, 0.0, 0.0, 0.0)
    lat.trackBunch(b)
    # HKICKER: dpx = KICK = 1e-3; VKICKER: dpy = KICK = 2e-3; drifts keep px and py.
    print(
        f"[{tag}] px observed {b.xp(0):.6e}, expected 1.000000e-03; py observed {b.yp(0):.6e}, expected 2.000000e-03"
    )
    if not is_control:
        bad = bad or abs(b.xp(0) - 1e-3) > 1e-12 or abs(b.yp(0) - 2e-3) > 1e-12
print("DEFECT REPRODUCED" if bad else "NOT REPRODUCED")
