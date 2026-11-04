"""COL-21: Sliced linear SCMULT normalizes every slice by the whole-bunch beam size

With constant matched optics, linear SCMULT space-charge forces, slicing on (`slice_duration` > 0),
interpolation off and slice populations meeting `slice_threshold`, Elegant normalizes each slice's
field gradient by the whole bunch's rms widths. The manual instead treats each slice independently;
for transverse rms widths σₓ and σᵧ, its horizontal gradient is proportional to 1/[σₓ(σₓ + σᵧ)], as
equal-width and isolated-slice controls confirm. For two equal-charge slices whose widths differ by a
factor of 2, the narrow slice's gradient is 9.742×10⁻⁷ m⁻¹ instead of 2.436×10⁻⁶ m⁻¹, 2.500 times
too weak, while the wide slice's is 1.600 times too strong. Ring simulations with unequal slice sizes
are affected.

The script tracks one turn around a ring of sixteen identical sector bends, with constant matched
beta functions, and a single SCMULT at its end; the beam is two clouds of 128 matched particles,
20.25 ns apart, so each cloud fills one slice. It fits each slice's kick gradient (the kick
divided by the offset from the slice centroid, from runs with and without charge) and compares
it with the manual's linear Gaussian field evaluated with that slice's own sizes. The controls
are two equal slices, each slice alone, and a slice threshold above the slice population, for
which the manual itself prescribes whole-bunch sizes.
"""

import sys

import numpy as np
from _elegant import modes, particles, run, verdict

# SDDS constants.h, as Elegant uses them.
C, QE, RE = 299792458.0, 1.60217733e-19, 2.81794092e-15
P0 = 100.0  # electron momentum, units of m_e c
RADIUS, K1 = 8.0, -0.0075  # m, 1/m^2
LENGTH = 2 * np.pi * RADIUS  # ring circumference, m
# A sector bend of radius R with gradient K1 focuses with k_x = 1/R^2 + K1 and k_y = -K1; a
# ring of identical bends has constant matched beta = 1/sqrt(k).
BETA = np.array([1 / np.sqrt(1 / RADIUS**2 + K1), 1 / np.sqrt(-K1)])
SIGMA = np.array([1.0e-3, 1.6e-3])  # rms x, y of the narrowest slice, m
N = 128  # particles per slice
Q = 1e-12  # C per slice
DT = 2.0**-30  # slice duration, s; a binary fraction keeps slice edges exact
SEPARATION = 20.25e-9  # s between the slice centers
RTOL, ATOL = 4e-10, 1e-16  # 1/m; floor from subtracting the transported slopes


def unit_cloud():
    """N points with zero mean and exactly unit covariance in four dimensions."""
    z = np.random.default_rng(88210).normal(size=(N, 4))
    z -= z.mean(axis=0)
    return np.linalg.qr(z)[0] * np.sqrt(N)


def beam(widths):
    """One matched cloud per slice, scaled by its width factor, at arrival times j*SEPARATION."""
    z = unit_cloud()
    clouds = [
        {
            "x": w * SIGMA[0] * z[:, 0],
            "xp": w * SIGMA[0] / BETA[0] * z[:, 1],
            "y": w * SIGMA[1] * z[:, 2],
            "yp": w * SIGMA[1] / BETA[1] * z[:, 3],
            "t": np.full(N, j * SEPARATION),
            "p": np.full(N, P0),
        }
        for j, w in enumerate(widths)
    ]
    return {key: np.concatenate([c[key] for c in clouds]) for key in clouds[0]}


def gradients(mode, widths, threshold):
    """Fitted kick gradients (x, y) of each slice, in 1/m."""
    columns = beam(widths)
    outputs = []
    for charge in (0.0, Q * len(widths)):
        lattice = (
            f"Q: CHARGE,TOTAL={charge!r}\n"
            f"B: SBEN,L={LENGTH / 16!r},ANGLE={2 * np.pi / 16!r},K1={K1!r}\n"
            "BL: LINE=(Q,16*B)\n"
        )
        # skip=0 inserts one SCMULT at the end of the beamline; it integrates the whole ring.
        commands = (
            "&insert_sceffects name=B,skip=0,horizontal=1,vertical=1,nonlinear=0,"
            f"slice_duration={DT!r},slice_threshold={threshold},slice_interpolation=0 &end\n"
            f"&run_setup lattice=lattice.lte,use_beamline=BL,p_central={P0!r},default_order=1,"
            "output=out.sdds &end\n"
            "&run_control n_steps=1,n_passes=1 &end\n"
            "&twiss_output matched=1 &end\n"
            "&sdds_beam input=beam.sdds &end\n"
            "&track &end\n"
        )
        result = run({"lattice.lte": lattice, "run.ele": commands}, mode, {"beam.sdds": columns})
        out = particles(result["out.sdds"])
        if len(out["p"]) != len(columns["p"]):
            sys.exit(f"{mode}: particles lost")
        outputs.append(out)
    before, after = outputs
    # NaN slopes would turn every comparison below into a spurious mismatch.
    for name in ("xp", "yp"):
        if not (np.all(np.isfinite(before[name])) and np.all(np.isfinite(after[name]))):
            sys.exit(f"{mode}: non-finite {name} in the uncharged or charged run")
    # Space charge changes only the slopes.
    for name in ("x", "y", "t", "p"):
        if not np.array_equal(after[name], before[name]):
            sys.exit(f"{mode}: space charge changed {name}")
    fitted = []
    for j in range(len(widths)):
        rows = slice(j * N, (j + 1) * N)
        offsets, kicks = [], []
        for plane, slope in (("x", "xp"), ("y", "yp")):
            offset = before[plane][rows] - before[plane][rows].mean()
            offsets.append(offset)
            kicks.append(after[slope][rows] - before[slope][rows])
        offsets, kicks = np.array(offsets), np.array(kicks)
        # Least-squares slope of kick against offset from the slice centroid.
        fitted.append(np.sum(offsets * kicks, axis=1) / np.sum(offsets**2, axis=1))
    fitted = np.array(fitted)
    if not np.all(np.isfinite(fitted)):
        sys.exit(f"{mode}: non-finite fitted gradient")
    return fitted


def field_gradient(sigma):
    """Manual's linear field for a slice of charge Q, uniform over DT, sizes sigma = (sx, sy).

    dx' = K L lambda x/[sx (sx + sy)], K = 2 N_e r_e/(gamma^3 beta^2), with the line density
    1/(c DT) of a uniform slice in place of the Gaussian factor, N_e = Q/e and
    gamma^3 beta^2 = p^2 sqrt(1 + p^2). With constant beta the optical weight SCMULT integrates
    over the ring, L beta/[sx (sx + sy)], divided by the beta at the kick, leaves exactly L.
    """
    k = 2 * (Q / QE) * RE / (P0**2 * np.sqrt(1 + P0**2))
    return k * LENGTH / (C * DT) / (sigma * sigma.sum())


def fmt(values):
    return "[" + ", ".join(f"{v:.15g}" for v in np.ravel(values)) + "]"


defect = False
print("kick gradients (x, y) in 1/m")
for mode in modes():
    cases = (
        ("slice widths 1 and 2", (1, 2), 100, False),
        ("control, slice widths 1 and 1", (1, 1), 100, True),
        ("control, slice width 1 alone", (1,), 100, True),
        ("control, slice width 2 alone", (2,), 100, True),
        ("control, threshold 1000, widths 1 and 2", (1, 2), 1000, True),
    )
    for label, widths, threshold, control in cases:
        observed = gradients(mode, widths, threshold)
        # Whole-bunch rms sizes of equal-population, co-centered clouds: sqrt(mean(w^2))*SIGMA.
        whole = field_gradient(np.sqrt(np.mean(np.square(widths))) * SIGMA)
        if threshold > N:
            # Below the threshold the manual replaces slice sizes with whole-bunch sizes.
            expected = np.array([whole for _ in widths])
        else:
            expected = np.array([field_gradient(w * SIGMA) for w in widths])
        bad = not np.allclose(observed, expected, rtol=RTOL, atol=ATOL)
        for j, w in enumerate(widths):
            print(
                f"{mode} {label}, slice {j + 1} (width {w}): "
                f"observed={fmt(observed[j])} expected={fmt(expected[j])}"
            )
        if control and bad:
            sys.exit(f"{mode} {label}: control disagrees with the field; the comparison is invalid")
        if not control:
            # Cause: every slice receives the gradient of the whole-bunch sizes.
            print(f"{mode} {label}: whole-bunch-size gradient={fmt(whole)}")
            if bad and not np.allclose(
                observed, np.broadcast_to(whole, observed.shape), rtol=RTOL, atol=ATOL
            ):
                sys.exit(
                    f"{mode} {label}: gradients follow neither the slice field nor the bunch's"
                )
            defect |= bad
print(f"comparison: rtol={RTOL:g}, atol={ATOL:g} 1/m")
verdict(defect)
