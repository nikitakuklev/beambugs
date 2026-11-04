"""Prescribed-map helpers for Elegant's MATR element, which reads a map from a text file.

`matrix_text` writes the file in the layout MATR expects: the constant vector C, the rows of the
first-order matrix R, the second-order coefficients T and the third-order coefficients Q. Elegant
stores the symmetric coefficients once, with the indices ordered j >= k (>= l), and each entry
holds the total coefficient of the monomial x_j x_k (x_l) in coordinate i; coordinates are
(x, x', y, y', s, delta), indexed from zero here.
"""

from __future__ import annotations

import numpy as np

TUNES = (0.17, 0.23)  # horizontal and vertical betatron tunes of the rotation helper


def rotations(tunes: tuple[float, float] = TUNES) -> np.ndarray:
    """6x6 matrix: rotation by 2 pi nu in (x, x') and (y, y') with beta = 1 m and alpha = 0."""
    matrix = np.eye(6)
    for j, tune in zip((0, 2), tunes, strict=True):
        c, s = np.cos(2 * np.pi * tune), np.sin(2 * np.pi * tune)
        matrix[j : j + 2, j : j + 2] = [[c, s], [-s, c]]
    return matrix


def matrix_text(
    c: np.ndarray, r: np.ndarray, t: np.ndarray | None = None, q: np.ndarray | None = None
) -> str:
    """Contents of a MATR map file; t has shape (6, 6, 6) and q (6, 6, 6, 6), zero when omitted."""
    t = np.zeros((6, 6, 6)) if t is None else t
    q = np.zeros((6, 6, 6, 6)) if q is None else q
    lines = ["C: " + " ".join(f"{v:.17g}" for v in c)]
    lines += [f"R{i + 1}: " + " ".join(f"{v:.17g}" for v in r[i]) for i in range(6)]
    for i in range(6):
        for j in range(6):
            values = " ".join(f"{t[i, j, k]:.17g}" for k in range(j + 1))
            lines.append(f"T{i + 1}{j + 1}: {values}")
    for i in range(6):
        for j in range(6):
            for k in range(j + 1):
                values = " ".join(f"{q[i, j, k, l]:.17g}" for l in range(k + 1))
                lines.append(f"Q{i + 1}{j + 1}{k + 1}: {values}")
    return "\n".join(lines) + "\n"
