"""Deterministic spin-Twiss rings and direct one-turn transport.

The default uses the serial prebuilt-kernel setup of the other reproducers. Explicitly setting
XSUITE_FORCE_KERNEL_COMPILATION=1 uses the common source-compilation setup,
so the same decks can check a complete Xtrack head source build.
"""

import os

for name in (
    "OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS",
    "NUMEXPR_NUM_THREADS", "NPROC", "PJRT_NPROC",
):
    os.environ[name] = "1"

import numpy as np
import xobjects as xo
import xtrack as xt
from _common import report

_MODE = os.environ.get("XSUITE_FORCE_KERNEL_COMPILATION", "0")
if _MODE == "1":
    from _common import prepare

    CTX = xo.ContextCpu(omp_num_threads=0)

    def setup():
        prepare()
        xo.settings.force_kernel_compilation = True
        print("KERNEL_MODE forced source compilation; CPU serial, binary64")

    def kernels():
        assert xo.settings.force_kernel_compilation
        print("KERNEL_MODE forced source compilation completed")

elif _MODE == "0":
    from _prebuilt import CTX, identity, kernels

    def setup():
        identity()

else:
    raise ValueError("XSUITE_FORCE_KERNEL_COMPILATION must be 0 or 1")

__all__ = ["G", "components", "kernels", "matrix", "np", "report", "ring", "setup", "track_spin"]
G = 0.00115965218128


def ring(tilt=0.0, *, rf=False):
    # Opposite frame rotations conjugate the map: the vertical invariant axis
    # becomes (-sin(tilt), cos(tilt), 0), without changing orbital or spin tunes.
    gamma, nbend = 200.0, 32
    elements = [xt.Rotation(rot_s_rad=tilt)]
    for index in range(nbend):
        elements.extend([
            xt.Multipole(knl=[0, (-1) ** index * 0.12]),
            xt.Bend(
                length=1.0, angle=2 * np.pi / nbend, k0_from_h=True,
                model="bend-kick-bend", num_multipole_kicks=1,
                edge_entry_active=False, edge_exit_active=False,
            ),
        ])
    if rf:
        from scipy.constants import c

        beta = np.sqrt(1 - 1 / gamma**2)
        elements.append(xt.Cavity(
            voltage=2e4, frequency=20 * beta * c / nbend, phase=np.pi,
        ))
    elements.append(xt.Rotation(rot_s_rad=-tilt))
    line = xt.Line(elements=elements)
    line.particle_ref = xt.Particles(
        _context=CTX, mass0=xt.ELECTRON_MASS_EV, q0=-1, gamma0=gamma,
        anomalous_magnetic_moment=G,
    )
    line.build_tracker(_context=CTX)
    return line


def components(particle):
    return np.array([particle.spin_x[0], particle.spin_y[0], particle.spin_z[0]])


def track_spin(line, spin):
    line.config.XTRACK_MULTIPOLE_NO_SYNRAD = False
    particle = line.particle_ref.copy()
    particle.spin_x, particle.spin_y, particle.spin_z = spin
    line.track(particle)
    assert particle.state[0] == 1
    return components(particle)


def matrix(line, amplitude=1.0):
    return np.column_stack([
        track_spin(line, basis * amplitude) / amplitude for basis in np.eye(3)
    ])
