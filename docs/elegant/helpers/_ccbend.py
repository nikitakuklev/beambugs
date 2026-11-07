"""CCBEND fixtures and an independent rigid-magnet ray tracer for forward/backward probes.

The native runner tracks fixed rays through one CCBEND in a fresh process and returns
(x, xprime, y, yprime, delta) with delta = p/p0 - 1, the optimizer results printed with
VERBOSE=1 and the log. The oracle places a straight-pole, rectangular-face magnet as a
rigid body in laboratory coordinates, intersects rays with its two faces and integrates
the exact Cartesian equations of motion between them with SciPy's DOP853, independently
of elegant's splitting integrator, coordinate rotations and alignment transforms.
"""

import os

for _name in (
    "OMP_NUM_THREADS",
    "MKL_NUM_THREADS",
    "OPENBLAS_NUM_THREADS",
    "NUMEXPR_NUM_THREADS",
    "NPROC",
    "PJRT_NPROC",
):
    os.environ[_name] = "1"

import re

import _elegant as native
import numpy as np
from scipy.integrate import solve_ivp

native.TIMEOUT = 290  # ty: ignore[invalid-assignment]  # Per-probe native deadline.

P0 = 1500.0  # reference momentum, units of m c
L = 0.6  # arc length
ANGLE = 0.04
K1 = 0.2
# On-axis, and off-axis rays at 3% above and 2% below reference momentum.
RAYS = np.array(
    [
        [0, 0, 0, 0, 0],
        [5e-4, 2e-4, -3e-4, 1e-4, 0.03],
        [-4e-4, -1e-4, 2e-4, -2e-4, -0.02],
    ],
    dtype=float,
)
DEFAULT = None  # parameter value that leaves the element default in place


def lattice(**params):
    """One CCBEND. Unless overridden, optimization, reference correction, edge kicks and
    radiation are off; FRINGEMODEL=0 with EDGE_ORDER=0 applies no edge kick
    (verticalRbendFringe in ccbend.c) but keeps model 0's centering objective."""
    values = dict(
        L=L,
        ANGLE=ANGLE,
        K1=K1,
        N_SLICES=64,
        INTEGRATION_ORDER=6,
        OPTIMIZE_FSE=0,
        OPTIMIZE_DX=0,
        REFERENCE_CORRECTION=0,
        FRINGEMODEL=0,
        EDGE_ORDER=0,
        SYNCH_RAD=0,
        ISR=0,
        VERBOSE=1,
    )
    for key, value in params.items():
        if value is DEFAULT:
            values.pop(key)
        else:
            values[key] = value
    body = ",".join(
        f"{k}={v}" if isinstance(v, int) else f"{k}={v:.17g}" for k, v in values.items()
    )
    return f"B: CCBEND,{body}\nBL: LINE=(B)\n"


def track(rays, *, reverse=False, mode="serial", **params):
    """Track rays through one CCBEND, natively backward when reverse is set.

    mode is "serial" or "mpi" (three-rank Pelegant, which needs at least three rays).
    Returns (coordinates, fits): fits lists the (FSE, dx) offsets that the element's
    optimizer reports."""
    q = np.array(rays, dtype=float, ndmin=2)
    beam = dict(zip(("x", "xp", "y", "yp"), q[:, :4].T, strict=True))
    beam.update(t=np.zeros(len(q)), p=P0 * (1 + q[:, 4]), particleID=np.arange(1, len(q) + 1))
    commands = (
        "&run_setup lattice=lattice.lte, use_beamline=BL, "
        f"p_central={P0:.17g}, back_tracking={int(reverse)}, default_order=1, output=out.sdds &end\n"
        "&run_control n_steps=1 &end\n&sdds_beam input=beam.sdds &end\n&track soft_failure=0 &end\n"
    )
    result = native.run(
        {"lattice.lte": lattice(**params), "run.ele": commands}, mode, {"beam.sdds": beam}
    )
    data = native.particles(result["out.sdds"])
    if not np.array_equal(data["particleID"], beam["particleID"]):
        raise RuntimeError("Particle IDs changed or particles were lost")
    out = np.column_stack([data[n] for n in ("x", "xp", "y", "yp", "p")])
    out[:, 4] = out[:, 4] / P0 - 1
    fits = [
        (float(f), float(d))
        for f, d in re.findall(r"optimized: FSE=\s*(\S+), dx=\s*(\S+), accuracy", result["log"])
    ]
    return out, fits


def roundtrip(forward_params, backward_params=None):
    """Forward pass, then native backtracking of its output; returns (forward, backward, fits)."""
    fwd, fits_f = track(RAYS, **forward_params)
    back, fits_b = track(
        fwd, reverse=True, **(forward_params if backward_params is None else backward_params)
    )
    return fwd, back, (fits_f, fits_b)


def error(a, b):
    """Largest transverse difference (positions in m, slopes in rad)."""
    return float(np.max(np.abs(np.asarray(a)[:, :4] - np.asarray(b)[:, :4])))


def close(a, b, atol, label):
    value = error(a, b)
    if not np.isfinite(value) or value > atol:
        raise RuntimeError(f"{label}: error={value:.17g}, tolerance={atol:g}")
    return value


def _rx(a):
    c, s = np.cos(a), np.sin(a)
    return np.array([[1, 0, 0], [0, c, -s], [0, s, c]])


def _ry(a):
    c, s = np.cos(a), np.sin(a)
    return np.array([[c, 0, s], [0, 1, 0], [-s, 0, c]])


def _rz(a):
    c, s = np.cos(a), np.sin(a)
    return np.array([[c, -s, 0], [s, c, 0], [0, 0, 1]])


class Magnet:
    """A rigid straight-pole magnet with rectangular faces, placed in laboratory coordinates.

    Laboratory frame: origin at the nominal entrance point, z along the incoming reference,
    x horizontal. A bend of signed angle theta deflects the reference toward -x for theta > 0.
    The magnet axis is the chord, rotated from z by -theta/2 about y, so the reference enters
    at theta/2 to it; the chord length is l = 2 (L/theta) sin(theta/2). The outgoing
    reference frame has its origin at the chord end and is rotated by -theta about y.

    Errors follow elegant's exact centre-referenced convention (MALIGN_METHOD=4): in the
    nominal magnet frame the magnet is rotated about its centre by R = Rx(pitch) Ry(yaw)
    Rz(roll), the rotation matrix of malign_mat.c, then displaced by d = (dx, dy, dz).
    A positive-angle bend fixes the signs; a negative-angle bend uses the same laboratory
    axes, as elegant does for DX and DY.

    Field, normalized by the reference rigidity, in magnet coordinates (x, y, z):
    By = (1 + fse) (theta/L + k1 x), Bx = (1 + fse) k1 y. Along the axis z a positive
    particle with momentum p (units of p0) obeys dpx/dz = -By, dpy/dz = Bx,
    dx/dz = px/pz, dy/dz = py/pz with pz = sqrt(|p|^2 - px^2 - py^2).
    """

    def __init__(
        self, angle=ANGLE, k1=K1, fse=0.0, d=(0.0, 0.0, 0.0), pitch=0.0, yaw=0.0, roll=0.0
    ):
        self.b0 = (1 + fse) * angle / L
        self.k1 = (1 + fse) * k1
        self.chord = 2 * (L / angle) * np.sin(angle / 2)
        nominal = _ry(-angle / 2)
        centre = np.array([0, 0, self.chord / 2])
        rot = _rx(pitch) @ _ry(yaw) @ _rz(roll)
        self.axes = nominal @ rot  # columns: magnet x, y, z in the laboratory
        self.origin = nominal @ (np.asarray(d, dtype=float) + centre - rot @ centre)
        self.exit_axes = _ry(-angle)
        self.exit_origin = nominal @ np.array([0, 0, self.chord])

    def _rhs(self, _z, s, momentum):
        x, y, px, py = s
        pz = np.sqrt(momentum**2 - px * px - py * py)
        return [px / pz, py / pz, -(self.b0 + self.k1 * x), self.k1 * y]

    def _body(self, r, p, z0, z1):
        """Carry a laboratory ray on the face z_magnet = z0 to the face z_magnet = z1."""
        local_r, local_p = self.axes.T @ (r - self.origin), self.axes.T @ p
        momentum = float(np.linalg.norm(p))
        sol = solve_ivp(
            self._rhs,
            (z0, z1),
            [local_r[0], local_r[1], local_p[0], local_p[1]],
            method="DOP853",
            rtol=2.3e-14,
            atol=1e-18,
            args=(momentum,),
        )
        if not sol.success:
            raise RuntimeError(sol.message)
        x, y, px, py = sol.y[:, -1]
        pz = np.sqrt(momentum**2 - px * px - py * py)
        return self.origin + self.axes @ [x, y, z1], self.axes @ [px, py, pz]

    @staticmethod
    def _to_plane(r, p, origin, normal):
        return r + p * (np.dot(origin - r, normal) / np.dot(p, normal))

    @staticmethod
    def _ray(q, origin, axes):
        x, xp, y, yp, delta = q
        direction = axes @ np.array([xp, yp, 1.0])
        return origin + axes @ np.array([x, y, 0.0]), (1 + delta) * direction / np.linalg.norm(
            direction
        )

    @staticmethod
    def _coords(r, p, origin, axes):
        local_r, local_p = axes.T @ (r - origin), axes.T @ p
        return np.array(
            [
                local_r[0],
                local_p[0] / local_p[2],
                local_r[1],
                local_p[1] / local_p[2],
                np.linalg.norm(p) - 1,
            ]
        )

    def forward(self, rays):
        out = []
        for q in np.atleast_2d(rays):
            r, p = self._ray(q, np.zeros(3), np.eye(3))
            r = self._to_plane(r, p, self.origin, self.axes[:, 2])
            r, p = self._body(r, p, 0.0, self.chord)
            r = self._to_plane(r, p, self.exit_origin, self.exit_axes[:, 2])
            out.append(self._coords(r, p, self.exit_origin, self.exit_axes))
        return np.array(out)

    def inverse(self, rays):
        """Exact inverse of forward: retrace from the outgoing frame to the incoming frame."""
        out = []
        for q in np.atleast_2d(rays):
            r, p = self._ray(q, self.exit_origin, self.exit_axes)
            r = self._to_plane(r, p, self.origin + self.chord * self.axes[:, 2], self.axes[:, 2])
            r, p = self._body(r, p, self.chord, 0.0)
            r = self._to_plane(r, p, np.zeros(3), np.array([0.0, 0.0, 1.0]))
            out.append(self._coords(r, p, np.zeros(3), np.eye(3)))
        return np.array(out)
