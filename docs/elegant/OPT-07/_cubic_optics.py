"""Oracle for the higher-order chromaticities of a cell of cubic-order lens maps.

Elegant's third-order maps keep each element's 2x2 transverse matrix to second order in the
relative momentum offset delta (the cubic map). For such truncated maps the tune derivatives
d^2 nu/d delta^2 and d^3 nu/d delta^3 follow from Cauchy's formula on a circle in the complex delta
plane, independently of Elegant's trace-derivative formulas: if f is analytic, then
f^(n)(0)/n! is the mean of f(z)/z^n over the circle |z| = RADIUS.
"""

from __future__ import annotations

import math

import numpy as np

POINTS, RADIUS = 64, 0.05
Z = RADIUS * np.exp(2j * np.pi * np.arange(POINTS) / POINTS)


def taylor(function, order: int = 2) -> list[np.ndarray]:
    """Taylor coefficients c_0 ... c_order of a matrix function of delta, analytic near 0."""
    values = np.array([function(z) for z in Z])
    return [np.mean(values / Z[:, None, None] ** n, axis=0) for n in range(order + 1)]


def quad_matrix(k, length):
    """2x2 hard-edge quadrupole matrix, analytic in k (k < 0 defocuses); cos z and sin(z)/z are even."""
    z = np.sqrt(k + 0j) * length
    return np.array([[np.cos(z), np.sin(z) / z * length], [-z * np.sin(z) / length, np.cos(z)]])


def drift_matrix(length):
    return np.array([[1.0, length], [0.0, 1.0]])


def tune_derivatives(elements: list[list[np.ndarray]], periods: int = 1) -> tuple[float, float]:
    """(d^2 nu/d delta^2, d^3 nu/d delta^3) in cycles, for a cell repeated `periods` times.

    elements: Taylor coefficient lists in beam order. The tune is
    nu = periods * arccos(trace/2) / (2 pi), the trace taken of the cell polynomial map.
    """
    trace = np.empty(POINTS, dtype=complex)
    for i, z in enumerate(Z):
        cell = np.eye(2)
        for coefficients in elements:
            cell = sum(c * z**n for n, c in enumerate(coefficients)) @ cell
        trace[i] = np.trace(cell)
    tune = periods * np.arccos(trace / 2) / (2 * np.pi)
    second, third = (float(math.factorial(n) * np.mean(tune / Z**n).real) for n in (2, 3))
    return second, third


def fodo_tune_derivatives(plane: str, thin: bool, periods: int = 1) -> tuple[float, float]:
    """Cell D QD D QF (quadrupole strengths 0.6 and -0.54 m^-2, or integrated for thin lenses).

    Thick lenses are 0.5 m long; the focusing sign flips in y. Drifts are 1 m. The gradient scales
    as 1/(1 + delta) (rigidity).
    """
    sign = 1 if plane == "x" else -1
    drift = taylor(lambda z: drift_matrix(1.0))
    elements = []
    for strength in (0.6, -0.54):
        k = sign * strength
        if thin:
            lens = taylor(lambda z, k=k: np.array([[1.0, 0.0], [-k / (1 + z), 1.0]]))
        else:
            lens = taylor(lambda z, k=k: quad_matrix(k / (1 + z), 0.5))
        elements += [lens, drift]
    return tune_derivatives(elements, periods)
