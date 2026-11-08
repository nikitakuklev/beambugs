"""ELE-03: Field sources with a rotated coordinate transform return E and B in the inner frame

`ShiftedFieldSource`, the base of `DipoleFieldSource`, `QuadFieldSource`, `LoopFieldSource` and
`MagnetFieldSourceGrid3D`, maps the evaluation point into the source's inner frame with the 4×4 matrix set by
`transormfMatrix`. The loops meant to copy its 3×3 rotation block, whose inverse returns E and B to the external frame,
use uninitialized counters, and the release build drops them, so the field vector stays in the inner frame. A dipole
with inner field (0, 1, 0) T rolled by 0.3 rad returns (0, 1, 0) T instead of (−0.2955, 0.9553, 0) T, and a 1 T/m
quadrupole rolled by π/2 returns the field of a quadrupole rolled by π/4, a skew quadrupole, instead of the −1 T/m
normal quadrupole. Translations are correct.

The script gives a dipole, a quadrupole and a current-loop source a rotation, alone and with a translation, and
compares `getFields` with closed forms: B_ext(r) = R⁻¹ B_in(R r + t) for the dipole and the rolled quadrupole, and the
on-axis Biot–Savart field along the tilted loop axis. The control is a quadrupole with a translation only.
"""

import math
import os

for _v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ.setdefault(_v, "1")

import numpy as np
from orbit.core.field_sources import (  # ty: ignore[unresolved-import]
    DipoleFieldSource,
    LoopFieldSource,
    QuadFieldSource,
)
from orbit.core.orbit_utils import Matrix  # ty: ignore[unresolved-import]


def rotz(a):
    c, s = math.cos(a), math.sin(a)
    return np.array([[c, s, 0.0], [-s, c, 0.0], [0.0, 0.0, 1.0]])


def roty(b):
    c, s = math.cos(b), math.sin(b)
    return np.array([[c, 0.0, s], [0.0, 1.0, 0.0], [-s, 0.0, c]])


def matrix4(rot, shift=(0.0, 0.0, 0.0)):
    # Documented transform from external to inner coordinates: r_in = R r_ext + t (rotation block, last column t).
    m = Matrix(4, 4)
    m.unit()
    for i in range(3):
        for j in range(3):
            m.set(i, j, float(rot[i, j]))
        m.set(i, 3, float(shift[i]))
    return m


def field(src, p):
    return np.array(src.getFields(*p)[3:6])


def quad(rot, shift=(0.0, 0.0, 0.0)):
    q = QuadFieldSource()
    q.length(1.0)
    q.gradient(1.0)
    q.transormfMatrix(matrix4(rot, shift))
    return q


def report(label, obs, exp, tol):
    err = float(np.max(np.abs(obs - exp)))
    fmt = lambda v: "(" + ", ".join(f"{t:+.6e}" for t in v) + ")"
    print(f"[{label}] B observed {fmt(obs)} T, expected {fmt(exp)} T, max |diff| {err:.3e}")
    return err > tol


bad = []
# Uniform dipole B_in = (0, 1, 0) T rolled by Rz(0.3): B_ext = R^T B_in = (-sin 0.3, cos 0.3, 0).
d = DipoleFieldSource()
d.sizesXYZ(1.0, 1.0, 1.0)
d.fieldsXYZ(0.0, 1.0, 0.0)
d.transormfMatrix(matrix4(rotz(0.3)))
bad.append(
    report(
        "dipole (0, 1, 0) T, Rz(0.3)",
        field(d, (0.0, 0.0, 0.0)),
        rotz(0.3).T @ np.array([0.0, 1.0, 0.0]),
        1e-12,
    )
)

# Normal quadrupole B_in = G (y, x) under R = Rz(a): B_ext = G (-sin 2a x + cos 2a y, cos 2a x + sin 2a y).
# a = pi/2 is the -1 T/m normal quadrupole, B = -G (y, x).
x, y = 0.010, 0.002
for a in (0.3, math.pi / 2):
    exp = np.array(
        [-math.sin(2 * a) * x + math.cos(2 * a) * y, math.cos(2 * a) * x + math.sin(2 * a) * y, 0.0]
    )
    bad.append(
        report(
            f"quad G = 1 T/m, Rz({a:.4f}), (x, y) = (10, 2) mm",
            field(quad(rotz(a)), (x, y, 0.0)),
            exp,
            1e-12,
        )
    )

# Current loop (I = 100 A, radius 0.5 m) tilted by Ry(0.4): on its axis R^T e_z at distance 0.3 m the field is
# mu0 I R^2 / (2 (R^2 + d^2)^1.5) along that axis.
lp = LoopFieldSource()
lp.radius(0.5)
lp.current(100.0)
lp.transormfMatrix(matrix4(roty(0.4)))
axis = roty(0.4).T @ np.array([0.0, 0.0, 1.0])
b_axis = 4e-7 * math.pi * 100.0 * 0.5**2 / (2 * (0.5**2 + 0.3**2) ** 1.5)
bad.append(
    report(
        "loop 100 A, radius 0.5 m, Ry(0.4), 0.3 m on its axis",
        field(lp, tuple(0.3 * axis)),
        b_axis * axis,
        1e-10,
    )
)

# Rotation plus translation: B_ext = R^T B_in(R r + t).
t = np.array([-0.003, 0.001, 0.0])
p = np.array([x, y, 0.0])
p_in = rotz(0.3) @ p + t
exp = rotz(0.3).T @ np.array([p_in[1], p_in[0], 0.0])
bad.append(
    report("quad Rz(0.3) and shift (-3, 1, 0) mm", field(quad(rotz(0.3), t), tuple(p)), exp, 1e-12)
)

# Control, translation only: B = G (y + t_y, x + t_x) = (3, 7) mT.
ctrl = report(
    "control, quad shift (-3, 1, 0) mm only",
    field(quad(np.eye(3), t), tuple(p)),
    np.array([0.003, 0.007, 0.0]),
    1e-15,
)
print("DEFECT REPRODUCED" if all(bad) and not ctrl else "NOT REPRODUCED")
