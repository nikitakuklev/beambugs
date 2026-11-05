"""DOC-04: SCMULT manual prints the averaging filter with the current turn in its history term

The manual's nonlinear SCMULT averaging equation reads σ̂ᵢ = fσᵢ + (1−f)σ̂ᵢ, with f =
`averaging_factor`, σᵢ the actual and σ̂ᵢ the filtered rms beam size on turn i. For every allowed f
> 0 this reduces to σ̂ᵢ = σᵢ, no filtering, whereas the text and the sizes shaping Elegant's field
follow the preceding-turn recurrence σ̂ᵢ = fσᵢ + (1−f)σ̂ᵢ₋₁. For a 5.000 pC bunch with kick-point
horizontal rms size 0.9000 mm on turn 1 and 0.9375 mm on turn 2, f = 0.2000 gives σ̂ = 0.9075 mm,
3.201% below the printed equation. With σ̂ in field shape and kick strength, the first particle's
second-turn slope kicks differ from the printed equation's by −0.1450% horizontally, +0.4543%
vertically.

The script tracks a 48-particle bunch for two turns around a closed four-sector ring with a
nonlinear SCMULT, for averaging factors f = 1, 0.6 and 0.2; the first turn is never filtered, so the
f = 1 kick is exactly what the printed equation predicts for every f. Elegant's kicks must match a
second prediction that filters the sizes with the preceding turn's size, keeps Elegant's strength
from the unfiltered sizes and evaluates the elliptical Gaussian field by a real Poisson integral,
without Elegant's complex error function. The script reports the filtered against the printed sizes,
and the kick with the filtered sizes also in the strength (Elegant's kick times the ratio of the two
accumulated strengths) against the printed equation's kick.
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

    Returns two arrays of shape (2, N, 6), ordered by particle ID, and the beta functions (x, y)
    at the start of the ring and at the exit of the drift DS that precedes the SCMULT.
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
        '&twiss_output matched=1,filename="twiss.sdds" &end\n'
        "&sdds_beam input=beam.sdds &end\n"
        "&track &end\n"
    )
    beam = {"x": X, "y": Y, "t": T, "p": np.full(N, P), "particleID": np.arange(1, N + 1)}
    result = run(
        {"lattice.lte": lattice, "run.ele": commands},
        mode,
        {"beam.sdds": beam},
        outputs=("before.sdds", "after.sdds", "twiss.sdds"),
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
    twiss = result["twiss.sdds"]
    elements = [str(n) for n in twiss[0, "ElementName"]]
    beta = np.column_stack([np.asarray(twiss[0, "betax"]), np.asarray(twiss[0, "betay"])])
    return stacks[0], stacks[1], (beta[elements.index("Q")], beta[elements.index("DS")])


def sizes(rows):
    """Rms sizes (x, y, s) of a bunch, with s = beta c t the longitudinal coordinate."""
    s = C * rows[:, 4] * rows[:, 5] / np.hypot(1, rows[:, 5])
    return np.array([rows[:, 0].std(), rows[:, 2].std(), s.std()])


def strength(betas, entrance, exit_sizes):
    """SCMULT's accumulated strength over the drift DS, the only element with length before it.

    accumulateSCMULT (insertSCeffects.cc:837-856) adds L/2 [beta_a/(a_i (a_x + a_y)) +
    beta_b/(b_i (b_x + b_y))] per plane i, with a the sizes at the start of the pass and b those at
    the exit of DS, where the SCMULT sits.
    """
    beta_a, beta_b = betas
    a, b = entrance[:2], exit_sizes[:2]
    return 0.5 * STRAIGHT * (beta_a / (a * a.sum()) + beta_b / (b * b.sum()))


def entrance_sizes(rows):
    """Rms sizes (x, y) at the start of the pass, from the coordinates after the drift DS."""
    return np.array(
        [(rows[:, 0] - STRAIGHT * rows[:, 1]).std(), (rows[:, 2] - STRAIGHT * rows[:, 3]).std()]
    )


def field_shape(rows, sigma):
    """Transverse field of the bunch at each particle, times the optical weight and density.

    The elliptical Gaussian field is evaluated by the real integral
      Gx = x/2 int_0^inf exp(-x^2/(2(sx^2+u)) - y^2/(2(sy^2+u)))/((sx^2+u)^(3/2) (sy^2+u)^(1/2)) du,
    and Gy likewise, whose central gradient is 1/[sx (sx + sy)]. Multiplying by sx (sx + sy) gives
    the correction to the accumulated linear optical strength, and the Gaussian longitudinal
    density exp(-s^2/(2 ss^2))/ss completes the shape. Rescaling u by max(sigma)^2 keeps the
    quadrature well conditioned. SCMULT accumulates the strength from the unfiltered sizes, which
    are the same for every f here, so it cancels in the ratio of two shapes and only the sizes enter.
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
        before1, after1, betas = turns(mode, 1.0, charge)
        # With f = 1 the effective size is the present size: the printed equation's prediction
        # for every f, since the incoming particles of the second turn do not depend on f.
        printed = after1[1, :, [1, 3]].T - before1[1, :, [1, 3]].T
        size0, size1 = sizes(before1[0]), sizes(before1[1])
        raw_shape = field_shape(before1[1], size1)
        for factor in FACTORS:
            before, after, _ = turns(mode, factor, charge)
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
            # sigma_hat_0 = sigma_0 the first traversal's size. Elegant accumulates the optical
            # strength from the unfiltered sizes, the same as for f = 1, so the kick scales with
            # the ratio of the field shapes; this tests the filter's history term only, not which
            # sizes the strength should use.
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
            # The documentation error itself: the printed equation gives sigma_hat = sigma_1, the
            # preceding-turn recurrence f sigma_1 + (1 - f) sigma_0.
            print(
                f"{label}: sizes (x, y) at SCMULT, mm: printed equation={fmt(size1[:2] * 1e3)} "
                f"preceding-turn={fmt(history[:2] * 1e3)} difference={fmt(history[:2] / size1[:2] - 1)}"
            )
            # The kick with the preceding-turn sizes in the strength as well as in the field shape,
            # the manual's kick with sigma_hat throughout: Elegant's kick rescaled by the ratio of
            # the strengths accumulated with the filtered and the measured sizes at the SCMULT.
            entrance = entrance_sizes(before[1])
            filtered = (
                observed * strength(betas, entrance, history) / strength(betas, entrance, size1)
            )
            print(
                f"{label}, second turn, particle 1: preceding-turn sizes throughout={fmt(filtered[0])} "
                f"printed-equation={fmt(printed[0])} difference={fmt(filtered[0] / printed[0] - 1)}"
            )
            print(
                f"{label}: Elegant's first-particle kick, whose strength keeps the measured sizes, "
                f"differs from the printed equation by {fmt(observed[0] / printed[0] - 1)}; "
                f"maximum over particles {np.max(np.abs(observed - printed)):.6g} rad"
            )
            defect |= not matches_printed
print(f"comparison: rtol={RTOL:g}, atol={ATOL:g} against the preceding-turn filter")
verdict(defect)
