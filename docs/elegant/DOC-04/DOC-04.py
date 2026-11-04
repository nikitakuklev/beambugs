"""DOC-04: SCMULT manual prints the averaging filter with the current turn in its history term

The manual's nonlinear SCMULT averaging equation uses σ̂ᵢ = fσᵢ + (1−f)σ̂ᵢ, where f is
`averaging_factor` and σᵢ and σ̂ᵢ are the actual and filtered rms beam sizes on turn i. For every
allowed f > 0 this reduces to σ̂ᵢ = σᵢ, while the surrounding text and measured kicks support the
preceding-turn recurrence σ̂ᵢ = fσᵢ + (1−f)σ̂ᵢ₋₁. For a 5.000 pC bunch with f = 0.2000, the first
particle's second-turn horizontal slope-kick magnitude is 3.972×10⁻⁸ instead of the printed
equation's 4.098×10⁻⁸, 3.085% smaller. Users interpreting averaged kicks from the printed equation
are affected.

The script tracks a 48-particle bunch for two turns around a closed four-sector ring with a
nonlinear SCMULT, for averaging factors f = 1, 0.6 and 0.2. The first turn is never filtered, so
the second-turn incoming particles are identical for every f, and the f = 1 kick is exactly what
the printed equation predicts for every f. A second prediction filters the sizes with the
preceding turn's size and evaluates the elliptical Gaussian field by a real Poisson integral,
without Elegant's complex error function. The comparison with f = 1 isolates the filter.
"""

import itertools
import sys

import numpy as np
from _elegant import modes, particles, run, verdict
from scipy.integrate import quad

C = 299792458.0  # SDDS constants.h
P = 23.0  # electron momentum, units of m_e c
SX, SY = 0.9e-3, 1.6e-3  # m
# A cold bunch: every combination of x = +-SX, y = +-SY and 12 arrival times, zero slopes.
X, Y, T = np.array(list(itertools.product([-SX, SX], [-SY, SY], np.arange(-11, 12, 2) * 5e-12))).T
N = len(X)
STRAIGHT, BEND_LENGTH = 1.0, 1.2  # m
H = np.pi / (2 * BEND_LENGTH)  # curvature of a 90-degree sector bend
CHARGES = (5e-12, 10e-12)  # C
FACTORS = (0.6, 0.2)
EQUATION_ATOL = 2e-18  # the second-turn incoming particles must agree between factors
RTOL, ATOL = 2e-9, 2e-18  # prediction by the Poisson integral (native erf agrees to 1.4e-9)


def turns(mode, factor, charge):
    """Coordinates (x, xp, y, yp, t, p) before and after SCMULT on each of two turns.

    Returns two arrays of shape (2, N, 6), ordered by particle ID.
    """
    lattice = (
        f"Q: CHARGE,TOTAL={charge!r}\n"
        f"DS: DRIF,L={STRAIGHT!r}\n"
        'W1: WATCH,FILENAME="before.sdds",MODE="coordinates"\n'
        "K: MARK\n"
        'W2: WATCH,FILENAME="after.sdds",MODE="coordinates"\n'
        f"DA: DRIF,L={STRAIGHT!r}\n"
        f"B: SBEND,L={BEND_LENGTH!r},ANGLE={np.pi / 2!r},K1={-(H**2) / 2!r}\n"
        "BL: LINE=(Q,DS,W1,K,W2,B,DA,B,DA,B,DA,B)\n"
    )
    # insert_sceffects places one SCMULT after the mark K.
    commands = (
        "&global_settings echo_namelists=0 &end\n"
        '&insert_sceffects name="K",skip=1,horizontal=1,vertical=1,nonlinear=1,'
        f"averaging_factor={factor!r} &end\n"
        f"&run_setup lattice=lattice.lte,use_beamline=BL,p_central={P!r},default_order=1,"
        "output=out.sdds &end\n"
        "&run_control n_steps=1,n_passes=2 &end\n"
        "&twiss_output matched=1 &end\n"
        "&sdds_beam input=beam.sdds &end\n"
        "&track &end\n"
    )
    beam = {"x": X, "y": Y, "t": T, "p": np.full(N, P), "particleID": np.arange(1, N + 1)}
    result = run(
        {"lattice.lte": lattice, "run.ele": commands},
        mode,
        {"beam.sdds": beam},
        outputs=("before.sdds", "after.sdds"),
    )
    names = ("x", "xp", "y", "yp", "t", "p")
    stacks = []
    for name in ("before.sdds", "after.sdds"):
        if result[name].n_pages != 2:
            sys.exit(f"{mode}: particles lost before the second turn")
        pages = [particles(result[name], page) for page in (0, 1)]
        if any(not np.array_equal(p["particleID"], np.arange(1, N + 1)) for p in pages):
            sys.exit(f"{mode}: particles lost")
        stacks.append(np.stack([np.column_stack([p[n] for n in names]) for p in pages]))
    return stacks[0], stacks[1]


def sizes(rows):
    """Rms sizes (x, y, s) of a bunch, with s = beta c t the longitudinal coordinate."""
    s = C * rows[:, 4] * rows[:, 5] / np.hypot(1, rows[:, 5])
    return np.array([rows[:, 0].std(), rows[:, 2].std(), s.std()])


def field_shape(rows, sigma):
    """Transverse field of the bunch at each particle, times the optical weight and density.

    The elliptical Gaussian field is evaluated by the real integral
      Gx = x/2 int_0^inf exp(-x^2/(2(sx^2+u)) - y^2/(2(sy^2+u)))/((sx^2+u)^(3/2) (sy^2+u)^(1/2)) du,
    and Gy likewise, whose central gradient is 1/[sx (sx + sy)]. Multiplying by sx (sx + sy) gives
    the correction to the accumulated linear optical strength, and the Gaussian longitudinal
    density exp(-s^2/(2 ss^2))/ss completes the shape. Rescaling u by max(sigma)^2 keeps the
    quadrature well conditioned. The strength SCMULT accumulates from the same incoming particles
    cancels in the ratio of two shapes, so only the sizes enter.
    """
    s = C * rows[:, 4] * rows[:, 5] / np.hypot(1, rows[:, 5])
    coordinates = np.column_stack([rows[:, 0], rows[:, 2], s])
    centered = coordinates - coordinates.mean(axis=0)
    sx, sy, ss = sigma
    scale = max(sx, sy) ** 2
    a, b = sx * sx / scale, sy * sy / scale
    shape = np.empty((len(rows), 2))
    for i, (x, y, z) in enumerate(centered):

        def integrand(v, plane, x=x, y=y):
            ax, by = a + v, b + v
            return np.exp(-0.5 * (x * x / (scale * ax) + y * y / (scale * by))) / (
                np.sqrt(ax * by) * (ax if plane == 0 else by)
            )

        gx = x / (2 * scale) * quad(integrand, 0, np.inf, args=(0,), epsabs=2e-13, epsrel=2e-13)[0]
        gy = y / (2 * scale) * quad(integrand, 0, np.inf, args=(1,), epsabs=2e-13, epsrel=2e-13)[0]
        density = np.exp(-0.5 * (z / ss) ** 2) / ss
        shape[i] = [gx * sx * (sx + sy) * density, gy * sy * (sx + sy) * density]
    return shape


def fmt(values):
    return "[" + ", ".join(f"{v:.16g}" for v in np.ravel(values)) + "]"


defect = False
for mode in modes():
    for charge in CHARGES:
        before1, after1 = turns(mode, 1.0, charge)
        # With f = 1 the effective size is the present size: the printed equation's prediction
        # for every f, since the incoming particles of the second turn do not depend on f.
        printed = after1[1, :, [1, 3]].T - before1[1, :, [1, 3]].T
        size0, size1 = sizes(before1[0]), sizes(before1[1])
        raw_shape = field_shape(before1[1], size1)
        for factor in FACTORS:
            before, after = turns(mode, factor, charge)
            label = f"{mode} {charge * 1e12:g} pC f={factor}"
            # The first turn is not filtered, so the first turn and the incoming second-turn
            # particles do not depend on f.
            if (
                np.max(np.abs(before - before1)) > EQUATION_ATOL
                or np.max(np.abs(after[0] - after1[0])) > EQUATION_ATOL
            ):
                sys.exit(f"{label}: f changes the first turn; the f = 1 kick is not the control")
            observed = after[1, :, [1, 3]].T - before[1, :, [1, 3]].T
            # Filter of the manual's text: sigma_hat_1 = f sigma_1 + (1 - f) sigma_hat_0, with
            # sigma_hat_0 = sigma_0 the first traversal's size. The accumulated optical strength
            # is the same as for f = 1, so the kick scales with the ratio of the field shapes.
            history = factor * size1 + (1 - factor) * size0
            preceding = printed * field_shape(before[1], history) / raw_shape
            print(
                f"{label}, second turn, particle 1: observed={fmt(observed[0])} "
                f"printed-equation expected={fmt(printed[0])}"
            )
            print(
                f"{label}, second turn, particle 1: observed={fmt(observed[0])} "
                f"preceding-turn expected={fmt(preceding[0])}"
            )
            matches_history = bool(np.allclose(observed, preceding, rtol=RTOL, atol=ATOL))
            matches_printed = bool(np.allclose(observed, printed, rtol=1e-12, atol=ATOL))
            if not matches_history:
                sys.exit(f"{label}: the kicks follow neither the printed equation nor the filter")
            print(
                f"{label}: first-particle kick differs from the printed equation by "
                f"{1 - observed[0, 0] / printed[0, 0]:.4%}; maximum over particles "
                f"{np.max(np.abs(observed - printed)):.6g} rad"
            )
            defect |= not matches_printed
print(f"comparison: rtol={RTOL:g}, atol={ATOL:g} against the preceding-turn filter")
verdict(defect)
