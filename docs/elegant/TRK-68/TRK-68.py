"""TRK-68: Positive LEFFECTIVE backtracking keeps the forward quadrupole body length

Backtracking QUAD or KQUAD with positive LEFFECTIVE retains a forward effective body length while making the end drifts too negative. The reverse map therefore fails to undo the forward focusing, even with zero misalignment, steering and fringe effects. For insertion length 0.6000 m, effective length 0.5000 m and normalized gradient K₁=1.400 m⁻², a KQUAD ray starting at x=1.000 mm and x′=0 returns at x=1.374 mm and x′=−1.313 mrad instead of its input. Independent Hamiltonian integration and closed-form matrices establish the discrepancy. Native tracking, analytic quadrupole maps and tracking-derived KQUAD matrices are affected; explicit drift–body–drift representation and nonpositive LEFFECTIVE controls reverse correctly.

The script tracks fixed rays forward and then natively backward through one aligned quadrupole with positive LEFFECTIVE and composes the forward and backward first-order matrices. Closed-form drift and quadrupole matrices give the forward map, its inverse and the map of the implemented backward interval sequence; the tiny-ray round trips and matrix products are compared with both. An explicit EDRIFT–KQUAD–EDRIFT line, LEFFECTIVE=0 and a zero-gradient element are controls.
"""

import numpy as np
from _elegant import verdict
from _slicing import close, track

P0 = 1500.0
L = 0.6
# Tiny rays isolate the linear map; the 1 mm ray is the Summary's case.
H = 1e-7
RAYS = np.array(
    [
        [H, 0, 0, 0, 0, 0],
        [0, H, 0, 0, 0, 0],
        [0, 0, H, 0, 0, 0],
        [0, 0, 0, H, 0, 0],
        [1e-3, 0, 0, 0, 0, 0],
    ],
    dtype=float,
)


def drift(length):
    return np.array([[1.0, length], [0.0, 1.0]])


def body(k1, length):
    # Hard-edge quadrupole x'' = -k1 x over a signed length.
    s = np.sqrt(abs(k1))
    if k1 > 0:
        c, sn = np.cos(s * length), np.sin(s * length)
        return np.array([[c, sn / s], [-s * sn, c]])
    c, sn = np.cosh(s * length), np.sinh(s * length)
    return np.array([[c, sn / s], [s * sn, c]])


def maps(k1, leff):
    """Forward F, correct inverse and implemented backward map, x and y planes."""
    d = (L - leff) / 2
    # Backtracking negates L but keeps positive LEFFECTIVE, so the body stays +leff
    # with K1*leff integrated strength while each end drift becomes (-L - leff)/2.
    b = (-L - leff) / 2
    out = {}
    for plane, k in (("x", k1), ("y", -k1)):
        forward = drift(d) @ body(k, leff) @ drift(d)
        out[plane] = (forward, np.linalg.inv(forward), drift(b) @ body(k, leff) @ drift(b))
    return out


def blocks(r):
    return {"x": r[:2, :2], "y": r[2:4, 2:4]}


def kquad(k1, leff, extra=""):
    return (
        f"Q: KQUAD,L={L},LEFFECTIVE={leff:.17g},K1={k1:.17g},N_SLICES=64,INTEGRATION_ORDER=6"
        f"{extra}\nBL: LINE=(Q)\n"
    )


def quad(k1, leff):
    return (
        f'Q: QUAD,L={L},LEFFECTIVE={leff:.17g},K1={k1:.17g},FRINGE_TYPE="integrals"\nBL: LINE=(Q)\n'
    )


def roundtrip(lattice):
    fwd = track("forward", lattice, RAYS, p0=P0, matrix=True)
    back = track("backward", lattice, fwd["q"], p0=P0, reverse=True, matrix=True)
    return fwd, back


def linear_columns(q):
    # Column j of the 4x4 linear map is the response to the tiny ray along axis j.
    return q[:4, :4].T / H


def main():
    failures = []
    cases = [
        ("KQUAD", 1.4, 0.5, kquad(1.4, 0.5)),
        ("KQUAD", 1.4, 0.6, kquad(1.4, 0.6)),
        ("KQUAD", 1.4, 0.75, kquad(1.4, 0.75)),
        ("KQUAD", -1.4, 0.5, kquad(-1.4, 0.5)),
        ("KQUAD TRACKING_MATRIX=1", 1.4, 0.5, kquad(1.4, 0.5, ",TRACKING_MATRIX=1")),
        ("QUAD", 1.4, 0.5, quad(1.4, 0.5)),
    ]
    for label, k1, leff, lattice in cases:
        expected = maps(k1, leff)
        fwd, back = roundtrip(lattice)
        # The forward map is correct: tracking and matrix agree with D(d) Q(leff) D(d).
        for plane, ref in blocks(linear_columns(fwd["q"])).items():
            close(ref, expected[plane][0], 1e-12, f"{label} forward {plane} tracking")
        for plane, ref in blocks(fwd["R"]).items():
            close(ref, expected[plane][0], 1e-12, f"{label} forward {plane} matrix")
        tracked = blocks(linear_columns(back["q"]))
        product = blocks(back["R"] @ fwd["R"])
        for plane in ("x", "y"):
            faulty = expected[plane][2] @ expected[plane][0]
            close(tracked[plane], faulty, 1e-12, f"{label} {plane} implemented-sequence law")
            close(product[plane], faulty, 1e-12, f"{label} {plane} matrix law")
            failures.append(not np.allclose(tracked[plane], np.eye(2), rtol=0, atol=1e-12))
            failures.append(not np.allclose(product[plane], np.eye(2), rtol=0, atol=1e-12))
        big = back["q"][4, :2]
        # Exact-Hamiltonian terms at 1 mm stay near 1e-6 of the linear law.
        linear = (expected["x"][2] @ expected["x"][0])[:, 0] * 1e-3
        close(big, linear, 1e-5 * abs(linear[0]), f"{label} 1 mm ray against linear law")
        print(
            f"{label} K1={k1} LEFFECTIVE={leff}: 1 mm ray roundtrip (x,x')={big.tolist()} "
            f"expected=[0.001, 0.0] (inverse transport); linear roundtrip x-block="
            f"{tracked['x'].tolist()} matrix Rback*Rforward x-block={product['x'].tolist()} "
            f"expected identity; implemented-sequence law D(b)Q(+l)D(b)*F x-block="
            f"{(expected['x'][2] @ expected['x'][0]).tolist()}"
        )

    controls = [
        (
            "EDRIFT-KQUAD-EDRIFT",
            (
                "E: EDRIFT,L=0.05\nQ: KQUAD,L=0.5,K1=1.4,N_SLICES=64,INTEGRATION_ORDER=6\n"
                "BL: LINE=(E,Q,E)\n"
            ),
        ),
        ("KQUAD LEFFECTIVE=0", kquad(1.4, 0.0)),
        ("KQUAD K1=0 LEFFECTIVE=0.5", kquad(0.0, 0.5)),
        ("QUAD LEFFECTIVE=-1", quad(1.4, -1.0)),
    ]
    for label, lattice in controls:
        fwd, back = roundtrip(lattice)
        error = close(back["q"][:, :4], RAYS[:, :4], 3e-14, f"{label} inverse")
        matrix = close(back["R"] @ fwd["R"], np.eye(6), 3e-14, f"{label} matrix inverse")
        if label == "EDRIFT-KQUAD-EDRIFT":
            # The same physical magnet as the principal case, forward map unchanged.
            for plane, ref in blocks(fwd["R"]).items():
                close(ref, maps(1.4, 0.5)[plane][0], 1e-12, f"{label} forward {plane}")
        print(
            f"control {label}: roundtrip max error={error:.4g} expected=0; "
            f"Rback*Rforward - I max={matrix:.4g} expected=0"
        )
    verdict(any(failures))


if __name__ == "__main__":
    main()
