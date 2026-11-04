"""Shared setup for the KQUAD and KSEXT multipole-table reproducers: a 12-electron beam, inline
multipole tables, and an independent integrator for a straight magnet with steering fields.

The integrator follows the particles through a straight hard-edge magnet in canonical transverse
momenta q = p_perp/p0, solving the Lorentz equations with scipy, without any Elegant code.
"""

from __future__ import annotations

from math import factorial

import numpy as np
from scipy.integrate import solve_ivp

P0 = 45.0  # reference momentum, units of m_e c
N = 12  # particles, enough for both workers of a 3-rank run
RADIUS = 0.027  # m, reference radius of the multipole tables


def beam() -> dict[str, np.ndarray]:
    """Twelve electrons with transverse offsets and slopes and momenta 43 to 49 m_e c."""
    return {
        "x": np.linspace(-0.018, 0.021, N),
        "xp": np.linspace(0.003, -0.004, N),
        "y": np.linspace(0.008, -0.011, N),
        "yp": np.linspace(-0.002, 0.003, N),
        "t": np.zeros(N),
        "p": np.linspace(43.0, 49.0, N),
    }


def table_text(orders, normal, skew=None) -> str:
    """ASCII SDDS multipole table: parameter referenceRadius, columns order, normal, (skew)."""
    columns = "&column name=order, type=long &end\n&column name=normal, type=double &end\n"
    if skew is not None:
        columns += "&column name=skew, type=double &end\n"
    rows = "".join(
        f"{n} {a!r}" + (f" {b!r}" if skew is not None else "") + "\n"
        for n, a, b in zip(orders, normal, skew if skew is not None else normal, strict=True)
    )
    return (
        "SDDS1\n"
        '&parameter name=referenceRadius, type=double, units="m" &end\n'
        + columns
        + "&data mode=ascii &end\n"
        + f"{RADIUS!r}\n{len(orders)}\n"
        + rows
    )


def canonical(b: dict[str, np.ndarray]) -> np.ndarray:
    """Rows x, y, qx, qy with q = (p/p0) * slope / sqrt(1 + xp^2 + yp^2)."""
    ratio = b["p"] / P0
    qs = ratio / np.sqrt(1 + b["xp"] ** 2 + b["yp"] ** 2)
    return np.stack((b["x"], b["y"], qs * b["xp"], qs * b["yp"]))


def observable(state: np.ndarray, b: dict[str, np.ndarray]) -> np.ndarray:
    """Columns x, xp, y, yp from canonical rows x, y, qx, qy."""
    x, y, qx, qy = state
    qs = np.sqrt((b["p"] / P0) ** 2 - qx**2 - qy**2)
    return np.column_stack((x, qx / qs, y, qy / qs))


def straight_body(
    state: np.ndarray,
    b: dict[str, np.ndarray],
    main_order: int,
    strength: float,
    length: float,
    factor: float = 0.0,
    kick: tuple[float, float] = (0.0, 0.0),
    harmonics=((2, 0.07), (4, -0.025)),
) -> np.ndarray:
    """Canonical rows after a straight magnet of the given length (m).

    Complex field C = By + iBx divided by the reference rigidity, with z = x + iy:
    C = K_m z^m/m! - (H - iV)/L - factor/L sum_n a_n (H - i(-1)^(n/2) V) (z/R)^n,
    where K_m is `strength`, (H, V) = `kick` the steering angles spread over the length and a_n
    the normal steering fractions at order n (zero steering and zero factor leave the main field).
    The equations of motion are dx/ds = qx/qs, dy/ds = qy/qs, dqx/ds = -Re C, dqy/ds = Im C with
    qs = sqrt((p/p0)^2 - qx^2 - qy^2).
    """
    ratio = b["p"] / P0

    def rhs(_, flat):
        x, y, qx, qy = flat.reshape(4, N)
        z = x + 1j * y
        field = strength * z**main_order / factorial(main_order)
        field -= (kick[0] - 1j * kick[1]) / length
        for n, a in harmonics:
            coefficient = -factor * a * (kick[0] - 1j * (-1) ** (n // 2) * kick[1])
            field += coefficient / length * (z / RADIUS) ** n
        qs = np.sqrt(ratio**2 - qx**2 - qy**2)
        return np.stack((qx / qs, qy / qs, -field.real, field.imag)).ravel()

    sol = solve_ivp(rhs, (0, length), state.ravel(), method="DOP853", rtol=3e-13, atol=2e-15)
    if not sol.success:
        raise RuntimeError("straight-magnet integration failed")
    return sol.y[:, -1].reshape(4, N)
