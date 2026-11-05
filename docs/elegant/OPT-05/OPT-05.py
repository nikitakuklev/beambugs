"""OPT-05: Zero-length MULT quadrupole cubic matrix contains infinities

The third-order transfer map of a zero-length quadrupole, MULT with L=0, ORDER=1 and integrated
strength KNL = 0.6 m⁻¹, contains infinite cubic coefficients. Concatenation contaminates 336 map
coefficients with NaN, including R₂₁, which should be −0.6000 m⁻¹; finite second-order and
short-length controls isolate the singular thin-lens construction. For a thin-lens FODO cell using
cubic lens maps, horizontal tune νₓ has d²νₓ/dδ² and d³νₓ/dδ³ equal to NaN instead of 0.2030 and
−0.06786, where δ is the relative momentum offset.

The script builds a thin-lens quadrupole `MULT,L=0,ORDER=1,KNL=0.6` in serial Elegant (and with
`--mpi` in 3-rank Pelegant) with a third-order map, reads the element's raw matrix from the matrix
printout and the exported concatenated map, and compares them with the ideal thin-lens kick. The
same lens in a thin-lens FODO cell gives the higher-order chromaticities, which are compared with
the derivatives of the tune of the cubic lens maps (truncated at δ²). Both control maps (a
second-order map of the same lens and a third-order map of a 1 μm long lens) must be finite, and
their R₂₁ coefficients must match the zero-length and finite-length hard-edge expectations,
respectively.
"""

import re
import sys

import numpy as np
from _cubic_optics import fodo_tune_derivatives
from _elegant import modes, run, verdict

KNL = 0.6  # integrated strength, m^-1
RTOL, ATOL = 1e-6, 1e-12


def single_lens(length, matrix_order):
    lattice = f"E: MULT,L={length!r},ORDER=1,KNL={KNL!r}{matrix_order}\nBL: LINE=(E)\n"
    # The element matrix is printed to standard output (the "individual matrices" printout),
    # because the exported SDDS map holds only the concatenated product.
    commands = (
        "&run_setup lattice=lattice.lte, use_beamline=BL, p_central=1000, default_order=3 &end\n"
        "&run_control n_steps=1, n_passes=1 &end\n"
        "&matrix_output SDDS_output=map.sdds, SDDS_output_order=3, printout=/dev/stdout,\n"
        " printout_order=3, individual_matrices=1 &end\n"
        "&bunched_beam n_particles_per_bunch=12 &end\n"
        "&track &end\n"
    )
    return {"lattice.lte": lattice, "run.ele": commands}


def raw_cubic(log):
    """Cubic coefficients U_ijkl of the element's own matrix, from its printout.

    A line `Qijk:` lists U_ijk1 ... U_ijkk. The element section ends where the effective
    (concatenated) matrix section begins.
    """
    start = log.index("E#1 pInput")
    section = log[start : log.index("Effective element matrix", start)]
    cubic = {}
    for label, values in re.findall(
        r"^Q(\d{3}):((?:[ \t]+\S+)+)[ \t]*$", section, flags=re.MULTILINE
    ):
        for l, value in enumerate(values.split(), start=1):
            cubic[f"U{label}{l}"] = float(value)
    return cubic


def run_lens(mode, length, matrix_order=""):
    out = run(single_lens(length, matrix_order), mode, outputs=("map.sdds",))
    exported = out["map.sdds"]
    row = len(exported[0, "ElementName"]) - 1  # last row: the concatenated map through the lens
    return {c.name: exported[0, c.name][row] for c in exported.columns}, raw_cubic(out["log"])


def fmt(values):
    return "[" + ", ".join(f"{v:.10g}" for v in np.atleast_1d(values)) + "]"


def compare(label, observed, expected, rtol=RTOL, atol=ATOL):
    """Print one case; return True when observed and expected disagree (NaN counts as disagreeing)."""
    observed, expected = np.atleast_1d(observed), np.atleast_1d(expected)
    bad = not np.allclose(observed, expected, rtol=rtol, atol=atol)
    print(f"{label}: observed={fmt(observed)} expected={fmt(expected)}")
    return bad


def ideal_lens_terms():
    """Thin-lens kick to first order in the transverse coordinates.

    x' -> x' - KNL x/(1 + delta), y' -> y' + KNL y/(1 + delta); x is unchanged. An exact
    finite-angle canonical-momentum kick is not an additive slope kick; the coefficients below
    are the transverse-linear chromatic terms. The cubic map keeps the delta dependence through
    delta^2, 1/(1 + delta) = 1 - delta + delta^2, so the xp row has R21 = -KNL, T261 = +KNL
    (x delta) and U2661 = -KNL (x delta^2). The position row has no momentum dependence:
    R11 = 1, U1441 = 0.
    """
    return {"R11": 1.0, "R21": -KNL, "R43": KNL, "T261": KNL, "U2661": -KNL, "U1441": 0.0}


def thick_lens_r21(length):
    """Exact hard-edge quadrupole R21 = -k sin(k L) with k^2 = KNL/L, for the finite-length control."""
    k = np.sqrt(KNL / length)
    return -k * np.sin(k * length)


FODO_LATTICE = (
    "QF: MULT,ORDER=1,KNL=0.6\nQD: MULT,ORDER=1,KNL=-0.54\nD: DRIF,L=1\nBL: LINE=(QF,D,QD,D)\n"
)
FODO_COMMANDS = (
    "&run_setup lattice=lattice.lte, use_beamline=BL, p_central=1000, default_order=3 &end\n"
    "&run_control n_steps=1, n_passes=1 &end\n"
    "&twiss_output filename=twiss.sdds, concat_order=3, higher_order_chromaticity=1,\n"
    " higher_order_chromaticity_points=11, higher_order_chromaticity_range=0.01 &end\n"
    "&bunched_beam n_particles_per_bunch=12 &end\n"
    "&track &end\n"
)

defect = False
for mode in modes():
    # Controls first: a second-order map of the zero-length lens, and a third-order map of a
    # 1 micron lens of the same integrated strength, must be finite, and their R21 must match the
    # zero-length and finite-length hard-edge expectations, respectively.
    exported, raw = run_lens(mode, 0, ",MATRIX_ORDER=2")
    finite = all(np.isfinite(v) for v in exported.values() if isinstance(v, (float, np.floating)))
    if not finite or compare(f"{mode} control order 2, R21", exported["R21"], -KNL):
        sys.exit("The second-order control is not finite and exact: the comparison is not valid.")
    short = 1e-6
    exported, raw = run_lens(mode, short)
    finite = all(np.isfinite(v) for v in exported.values() if isinstance(v, (float, np.floating)))
    if not finite or compare(
        f"{mode} control L=1e-6 order 3, R21", exported["R21"], thick_lens_r21(short)
    ):
        sys.exit("The short-lens control is not finite and exact: the comparison is not valid.")

    # The defect: the zero-length lens at third order.
    exported, raw = run_lens(mode, 0)
    ideal = ideal_lens_terms()
    names = list(ideal)
    observed = [exported[n] for n in names]
    print(f"{mode} third-order map {names}")
    defect |= compare(f"{mode} thin lens, exported map", observed, [ideal[n] for n in names])
    bad_columns = sorted(
        n for n, v in exported.items() if isinstance(v, (float, np.floating)) and not np.isfinite(v)
    )
    print(
        f"{mode} thin lens, nonfinite columns of the exported map: observed={len(bad_columns)} expected=0"
    )
    defect |= bool(bad_columns)
    bad_raw = {n: v for n, v in raw.items() if not np.isfinite(v)}
    print(
        f"{mode} thin lens, nonfinite raw cubic coefficients of the element: "
        f"observed={len(bad_raw)} {bad_raw} expected=0"
    )
    defect |= bool(bad_raw)
    print(f"{mode} thin lens, raw U1441: observed={raw['U1441']} expected=0.0")

    # Consequence: the higher-order chromaticities of a thin-lens FODO cell.
    twiss = run(
        {"lattice.lte": FODO_LATTICE, "run.ele": FODO_COMMANDS}, mode, outputs=("twiss.sdds",)
    )["twiss.sdds"]
    for plane in "xy":
        observed = [twiss[f"dnu{plane}/dp{order}"][0] for order in (2, 3)]
        defect |= compare(
            f"{mode} FODO dnu{plane}/dp2, dnu{plane}/dp3",
            observed,
            fodo_tune_derivatives(plane, thin=True),
            1e-9,
            1e-12,
        )
print(f"comparison: rtol={RTOL:g}, atol={ATOL:g}; chromaticities rtol=1e-9, atol=1e-12")
verdict(defect)
