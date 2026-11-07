"""TRK-67: Backtracking keeps the forward entrance placement of misaligned magnets

Native backtracking keeps entrance-referenced magnet placement instead of expressing it at the physical exit, so a forward pass followed by backtracking does not restore the particle. A 0.6000 m quadrupole with K₁ = 1.400 m⁻² and 1.000 mrad yaw returns an initially on-axis particle at x = −145.0 µm and slope x′ = 462.7 µrad instead of zero. Linearized and exact entrance methods fail for quadrupole, sextupole and octupole trackers, and analytic quadrupole maps share the first-order error; displaced bends also keep the wrong displacement frame, including with the original alignment method. Independent field integration and geometric exit re-expression establish the correct inverse. Exact centre-referenced controls pass.

The script tracks three fixed rays forward and then natively backward through each placed magnet and compares the round trip with the input. A separate backward input that places the same physical magnet from its forward exit (the rigid-body exit anchor for straight magnets, the exit-frame displacement for legacy bends) restores the input, and centre-referenced, original-method and nominal controls pass. At 0.1 µrad the straight-magnet round trips and the analytic affine maps are compared with a closed-form first-order law built from quadrupole matrices and rigid frame geometry.
"""

import numpy as np
from _elegant import verdict
from _slicing import close, track

P0 = 1500.0
L = 0.6
K1 = 1.4
ANGLE = 0.04
# On-axis, and off-axis rays at 3% above and 2% below reference momentum.
RAYS = np.array(
    [
        [0, 0, 0, 0, 0, 0],
        [5e-4, 2e-4, -3e-4, 1e-4, 0, 0.03],
        [-4e-4, -1e-4, 2e-4, -2e-4, 0, -0.02],
    ],
    dtype=float,
)
BODY = {
    "KQUAD": f"K1={K1},N_SLICES=64,INTEGRATION_ORDER=6",
    "KSEXT": "K2=35,N_SLICES=64,INTEGRATION_ORDER=6",
    # N_KICKS=0 keeps the deprecated alias from overriding N_SLICES.
    "KOCT": "K3=6000,N_SLICES=64,N_KICKS=0,INTEGRATION_ORDER=6",
    "DQCOR": f"K1={K1},N_SLICES=64,INTEGRATION_ORDER=6",
    "QUAD": f"K1={K1}",
}


def straight(kind, method, d=(0.0, 0.0, 0.0), pitch=0.0, yaw=0.0):
    return (
        f"Q: {kind},L={L},{BODY[kind]},MALIGN_METHOD={method},DX={d[0]:.17g},DY={d[1]:.17g},"
        f"DZ={d[2]:.17g},PITCH={pitch:.17g},YAW={yaw:.17g}\nBL: LINE=(Q)\n"
    )


def exit_anchor(d, pitch, yaw):
    # The magnet axis is n = Rx(pitch) Ry(yaw) e_z = (sin yaw, -cos yaw sin pitch,
    # cos pitch cos yaw) (malign_mat.c rotation matrix).  The physical exit point is
    # d + L n, so the backward entrance displacement is d + L (n - e_z).
    n = np.array([np.sin(yaw), -np.cos(yaw) * np.sin(pitch), np.cos(pitch) * np.cos(yaw)])
    return np.asarray(d) + L * (n - np.array([0, 0, 1]))


def first_order_law(pitch, yaw):
    # Backward transport takes the particle relative to the forward entrance anchor, so
    # in the magnet frame it starts displaced by a = L (n - e_z) from the true exit.
    # The inverse body map Q(L)^-1 = [[C, -S], [-C', C]] carries (a, 0) to
    # (a C, -a C'), and the backward exit removes a again:
    #   x = a_x (cos mu - 1),   x' = a_x sqrt(K1) sin mu        (focusing plane)
    #   y = a_y (cosh mu - 1),  y' = -a_y sqrt(K1) sinh mu      (defocusing plane)
    # with mu = sqrt(K1) L.  The on-axis error is odd in the angle, so the next
    # term is cubic.
    a = L * np.array([np.sin(yaw), -np.cos(yaw) * np.sin(pitch)])
    k, mu = np.sqrt(K1), np.sqrt(K1) * L
    return np.array(
        [
            a[0] * (np.cos(mu) - 1),
            a[0] * k * np.sin(mu),
            a[1] * (np.cosh(mu) - 1),
            -a[1] * k * np.sinh(mu),
        ]
    )


def roundtrip(forward_lattice, backward_lattice, matrix=False):
    fwd = track("forward", forward_lattice, RAYS, p0=P0, matrix=matrix)
    back = track("backward", backward_lattice, fwd["q"], p0=P0, reverse=True, matrix=matrix)
    return fwd, back


def bend(kind, d, method=None, yaw=0.0):
    common = f"L={L},ANGLE={ANGLE},K1=0.2,DX={d[0]:.17g},DY={d[1]:.17g},DZ={d[2]:.17g}"
    if kind == "CSBEND":
        extra = ",N_SLICES=16,INTEGRATION_ORDER=4,EDGE1_EFFECTS=0,EDGE2_EFFECTS=0"
        if method is not None:
            extra += f",MALIGN_METHOD={method},EYAW={yaw:.17g}"
    elif kind == "CSRCSBEND":
        extra = (
            ",N_SLICES=16,INTEGRATION_ORDER=4,EDGE1_EFFECTS=0,EDGE2_EFFECTS=0,"
            "CSR=0,BINS=64,STEADY_STATE=1,SYNCH_RAD=0,ISR=0"
        )
    else:
        extra = ""
    return f"Q: {kind},{common}{extra}\nBL: LINE=(Q)\n"


def main():
    failures = []
    zero = np.zeros(4)

    # Principal case: 1 mrad yaw or pitch, exact entrance method 3, KQUAD.
    for label, pitch, yaw in (("yaw", 0.0, 1e-3), ("pitch", 1e-3, 0.0)):
        lat = straight("KQUAD", 3, pitch=pitch, yaw=yaw)
        _, back = roundtrip(lat, lat)
        observed = back["q"][0, :4]
        failures.append(not np.allclose(observed, zero, rtol=0, atol=3e-14))
        law = first_order_law(pitch, yaw)
        # At 1 mrad only the cubic remainder, a few θ² relative, separates the two.
        close(observed, law, 1e-5 * np.max(np.abs(law)), "Faulty-law magnitude")
        fixed = straight("KQUAD", 3, exit_anchor((0, 0, 0), pitch, yaw), pitch, yaw)
        _, corrected = roundtrip(lat, fixed)
        error = close(corrected["q"][:, :4], RAYS[:, :4], 3e-14, "Exit-anchored inverse")
        print(
            f"KQUAD method=3 {label}=1e-3: roundtrip (x,x',y,y')={observed.tolist()} "
            f"expected={zero.tolist()} (inverse transport); first-order law={law.tolist()}; "
            f"exit-anchored backward input max error={error:.4g}"
        )

    # Linear limit: tracking and analytic affine maps follow the closed-form law.
    for kind in ("KQUAD", "QUAD"):
        for method in (1, 3):
            for label, pitch, yaw in (("yaw", 0.0, 1e-7), ("pitch", 1e-7, 0.0)):
                lat = straight(kind, method, pitch=pitch, yaw=yaw)
                fwd, back = roundtrip(lat, lat, matrix=True)
                observed = back["q"][0, :4]
                law = first_order_law(pitch, yaw)
                # The cubic remainder is about θ² = 1e-14 relative; 1e-12 relative allows
                # the double-precision rounding of the composed maps and still resolves
                # any other first-order law.
                tol = 1e-12 * np.max(np.abs(law))
                failures.append(not np.allclose(observed, zero, rtol=0, atol=tol))
                close(observed, law, tol, f"{kind} method {method} first-order law")
                # Analytic matrices: the on-axis round trip is C_back + R_back C_forward.
                affine = (back["C"] + back["R"] @ fwd["C"])[:4]
                close(affine, law, tol, f"{kind} method {method} analytic affine law")
                print(
                    f"{kind} method={method} {label}=1e-7: roundtrip (x,x',y,y')={observed.tolist()} "
                    f"expected={zero.tolist()} (inverse transport); first-order law={law.tolist()}; "
                    f"analytic affine roundtrip={affine.tolist()}"
                )

    # Other exact-method trackers with a displaced, yawed placement.
    d = (1e-3, -2e-4, 0.0)
    for kind in ("KSEXT", "KOCT", "DQCOR"):
        lat = straight(kind, 3, d, yaw=1e-3)
        _, back = roundtrip(lat, lat)
        error = float(np.max(np.abs(back["q"][:, :4] - RAYS[:, :4])))
        failures.append(error > 3e-14)
        _, corrected = roundtrip(lat, straight(kind, 3, exit_anchor(d, 0, 1e-3), yaw=1e-3))
        fixed = close(corrected["q"][:, :4], RAYS[:, :4], 3e-14, f"{kind} exit-anchored inverse")
        print(
            f"{kind} method=3 DX=1e-3 DY=-2e-4 yaw=1e-3: roundtrip max error={error:.4g} "
            f"expected=0 (inverse transport); exit-anchored backward input max error={fixed:.4g}"
        )

    # Straight-magnet controls: centre-referenced exact, original method and nominal.
    for method, pitch in ((4, 7e-4), (0, 7e-4), (3, 0.0)):
        yaw = 1e-3 if method != 3 else 0.0
        lat = straight("KQUAD", method, d, pitch=pitch, yaw=yaw)
        _, back = roundtrip(lat, lat)
        error = close(back["q"][:, :4], RAYS[:, :4], 3e-14, f"KQUAD method {method} control")
        print(
            f"control KQUAD method={method} pitch={pitch} yaw={yaw}: roundtrip max error={error:.4g} expected=0"
        )

    # CSBEND exact entrance method; the centre method is the control.
    for method in (3, 4):
        lat = bend("CSBEND", (0, 0, 0), method=method, yaw=1e-3)
        _, back = roundtrip(lat, lat)
        error = float(np.max(np.abs(back["q"][:, :4] - RAYS[:, :4])))
        if method == 3:
            failures.append(error > 5e-13)
        else:
            close(back["q"][:, :4], RAYS[:, :4], 5e-13, "CSBEND method 4 control")
        print(
            f"CSBEND method={method} yaw=1e-3: central roundtrip (x,x')={back['q'][0, :2].tolist()} "
            f"max error={error:.4g} expected=0 (inverse transport)"
        )

    # Legacy translation of displaced bends (original method, or no method selector).
    for kind, d in (
        ("CSBEND", (1e-3, 0, 0)),
        ("CSBEND", (0, 0, 1e-3)),
        ("SBEN", (1e-3, 0, 0)),
        ("RBEN", (1e-3, 0, 0)),
        ("CSRCSBEND", (1e-3, 0, 0)),
    ):
        lat = bend(kind, d)
        _, back = roundtrip(lat, lat)
        error = float(np.max(np.abs(back["q"][:, :4] - RAYS[:, :4])))
        failures.append(error > 5e-14)
        # The same displacement vector in the exit frame, rotated by the bend angle.
        c, s = np.cos(ANGLE), np.sin(ANGLE)
        exit_d = (d[0] * c + d[2] * s, d[1], d[2] * c - d[0] * s)
        _, corrected = roundtrip(lat, bend(kind, exit_d))
        fixed = close(corrected["q"][:, :4], RAYS[:, :4], 5e-14, f"{kind} exit-frame inverse")
        print(
            f"{kind} DX={d[0]} DZ={d[2]}: central roundtrip (x,x')={back['q'][0, :2].tolist()} "
            f"max error={error:.4g} expected=0 (inverse transport); "
            f"exit-frame backward input max error={fixed:.4g}"
        )
    nominal = bend("CSBEND", (0, 0, 0))
    _, back = roundtrip(nominal, nominal)
    error = close(back["q"][:, :4], RAYS[:, :4], 5e-14, "Nominal CSBEND control")
    print(f"control nominal CSBEND: roundtrip max error={error:.4g} expected=0")
    verdict(any(failures))


if __name__ == "__main__":
    main()
