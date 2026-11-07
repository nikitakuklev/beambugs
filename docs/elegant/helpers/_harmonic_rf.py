"""Zero-time SHRFDF fixtures with physical transverse-momentum reconstruction."""

from _slicing import track

# isort: split
import numpy as np


def kick(mode, phase, rays):
    """Return Px/(mc) after one isolated harmonic; input longitudinal coordinates are zero."""
    lattice = (
        f"S:SHRFDF,PERIOD_LENGTH=.03,PERIOD_PHASE={np.pi / 2:.17g},"
        f"V{mode}=240000,PHASE{mode}={phase:.17g}\nBL:LINE=(S)\n"
    )
    raw = track("harmonic", lattice, rays, p0=3200)["raw"]
    return raw[:, 5] * raw[:, 1] / np.sqrt(1 + raw[:, 1] ** 2 + raw[:, 3] ** 2)
