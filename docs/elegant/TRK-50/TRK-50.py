"""TRK-50: KQUAD backtracking applies inverse linear fringe factors in forward order

Backtracking a KQUAD with noncommuting inner and outer linear fringe factors fails to invert its forward fringe map. The tracker inverts the two factors but retains their forward order, while the inverse of their product requires the opposite order. For a 0.3000 m quadrupole with K₁ = 8.000 m⁻² and a smooth 0.1200 m fringe transition, the reverse vertical focusing coefficient differs from the inverse forward map by 1.043×10⁻⁶ m⁻¹. A finite particle acquires a 2.089 nrad horizontal slope error after a round trip. Single-factor controls and the analytic backward matrix pass.

The script integrates a smooth raised-cosine field profile to obtain physical fringe moments. Four-point particle stencils measure the separate fringe factors and their full product, whose inverse supplies the expected reverse map; finite particles and disabled fringes provide controls.
"""

import numpy as np
from _elegant import verdict
from _slicing import close, derivative, discrepancy, linear_rays, track
from scipy.integrate import quad

WIDTH = 0.06
K1 = 8.0
LENGTH = 0.3


def profile(s):
    # Raised-cosine transition, constant field to the left and zero to the right.
    return (1 - np.sin(np.pi * s / (2 * WIDTH))) / 2


def integrals():
    result = {}
    for suffix, lo, hi, hardedge in (("M", -WIDTH, 0, 1), ("P", 0, WIDTH, 0)):

        def difference(s, hardedge=hardedge):
            return profile(s) - hardedge

        for j in range(4):
            result[f"I{j}{suffix}"] = quad(
                lambda s, j=j: difference(s) * s**j,
                lo,
                hi,
                epsabs=1e-18,
                epsrel=3e-13,
            )[0]
        # The nested quadrature follows the manual's ordered double integral.
        result[f"LAMBDA2{suffix}"] = quad(
            lambda s, hi=hi: (
                difference(s)
                * quad(
                    lambda u: difference(u) * (u - s),
                    s,
                    hi,
                    epsabs=1e-19,
                    epsrel=3e-13,
                )[0]
            ),
            lo,
            hi,
            epsabs=1e-19,
            epsrel=3e-13,
        )[0]
    return result


def lattice(values, edge, slices, linear=True):
    pars = ",".join(f"{name}={value:.17g}" for name, value in values.items())
    return (
        f"Q: KQUAD,L={LENGTH},K1={K1},INTEGRATION_ORDER=6,N_SLICES={slices},"
        f"EDGE1_EFFECTS={int(edge == 1)},EDGE2_EFFECTS={int(edge == 2)},"
        f"EDGE1_LINEAR={int(linear)},EDGE2_LINEAR={int(linear)},{pars}\nBL: LINE=(Q)\n"
    )


def main():
    moments = integrals()
    failures = []
    rows = []
    for edge in (1, 2):
        for slices in (48, 96):
            for h in (2e-5, 1e-5):
                maps = {}
                # Measure the two separate factors through forward tracking.
                # The full inverse must reverse their order. No native reverse
                # fringe formula is used to compute the expected result.
                for part in ("body", "inner", "outer", "both"):
                    values = {
                        name: value
                        if part == "both"
                        or (part == "inner" and name.endswith("M"))
                        or (part == "outer" and name.endswith("P"))
                        else 0
                        for name, value in moments.items()
                    }
                    lat = lattice(values, edge, slices)
                    prefix = f"C2_{edge}_{slices}_{h}_{part}"
                    fwd = track(prefix + "_fwd", lat, linear_rays(h), matrix=True)
                    rev = track(prefix + "_rev", lat, linear_rays(h), reverse=True, matrix=True)
                    jf = derivative(fwd["q"], h)
                    jr = derivative(rev["q"], h)
                    maps[part] = jf
                    inv = np.linalg.inv(jf)
                    err = float(np.max(np.abs(jr - inv)))
                    if part != "both":
                        close(jr, inv, 2e-11, "single fringe factor inverse")
                    close(fwd["R"][:4, :4], jf, 2e-11, "analytic forward")
                    close(rev["R"][:4, :4], inv, 2e-11, "analytic inverse")
                    if part == "both":
                        row = dict(
                            edge=edge,
                            slices=slices,
                            h=h,
                            error=err,
                            inverse_r43=inv[3, 2],
                            reverse_r43=jr[3, 2],
                            analytic_error=float(np.max(np.abs(rev["R"][:4, :4] - inv))),
                            matrix_roundtrip=float(np.max(np.abs(jr @ jf - np.eye(4)))),
                        )
                        rows.append(row)

                body = maps["body"]
                if edge == 2:
                    m1 = maps["inner"] @ np.linalg.inv(body)
                    m2 = maps["outer"] @ np.linalg.inv(body)
                    composed = m2 @ m1 @ body
                    wrong_reverse = np.linalg.inv(body) @ np.linalg.inv(m2) @ np.linalg.inv(m1)
                else:
                    # Entry is the reflected exit; reflection reverses the
                    # factor order when the two partial maps are combined.
                    e1 = np.linalg.inv(body) @ maps["inner"]
                    e2 = np.linalg.inv(body) @ maps["outer"]
                    composed = body @ e1 @ e2
                    wrong_reverse = np.linalg.inv(e1) @ np.linalg.inv(e2) @ np.linalg.inv(body)
                close(composed, maps["both"], 2e-11, "separately measured fringe composition")
                prediction_error = float(np.max(np.abs(jr - wrong_reverse)))
                failures.append(discrepancy(jr, np.linalg.inv(maps["both"]), wrong_reverse, 2e-11))
                rows[-1]["wrong_order_prediction_error"] = prediction_error
                row = rows[-1]
                print(
                    f"edge={edge} slices={slices} h={h:g}: reverse R43={row['reverse_r43']:.16g} "
                    f"expected={row['inverse_r43']:.16g} (inverse forward particle map); "
                    f"inverse error={row['error']:.4g}; analytic error={row['analytic_error']:.4g}; "
                    f"wrong-order prediction error={prediction_error:.4g}"
                )
    # Finite orbit and suppressed-linear controls separate this from cubic
    # fringe truncation and from the numerical body integration.
    q = np.array([[0.002, 0.0009, -0.001, 0.0006, 0, 0]])
    for linear in (False, True):
        lat = lattice(moments, 2, 96, linear=linear)
        fwd = track(f"C2_orbit_{linear}_fwd", lat, q)
        rev = track(f"C2_orbit_{linear}_rev", lat, fwd["q"], reverse=True)
        delta = rev["q"] - q
        if not linear:
            close(delta, 0, 2e-12, "disabled linear fringe")
        print(
            f"finite orbit linear={int(linear)}: horizontal slope error={delta[0, 1]:.16g} rad "
            "expected=0 (roundtrip identity)"
        )
    print("controls: body, separate factors, analytic inverse and disabled fringes pass")
    verdict(any(failures))


if __name__ == "__main__":
    main()
