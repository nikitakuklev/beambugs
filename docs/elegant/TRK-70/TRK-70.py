"""TRK-70: CCBEND backtracking reverses the longitudinal displacement DZ

Backtracking a longitudinally displaced CCBEND reverses its displacement DZ when the beamline's ends are swapped. The backward pass therefore traverses a different physical magnet and does not undo forward tracking. A uniform 0.6000 m, 0.04000 rad dipole displaced longitudinally by DZ = 1.000 mm returns an initially on-axis particle at x = 79.99 µm instead of zero, exactly matching the independent wrong-sign law 4DZ sin(θ/2), where θ is the bend angle. All five alignment methods fail; fixed-face circle geometry and compensating the internal displacement flip restore the inverse.

The script tracks three fixed rays forward and then natively backward through one CCBEND with DZ = 1 mm, optimization and edge kicks off, and compares the round trip with the input. For a uniform dipole it compares the on-axis return with the closed-form wrong-sign law, and for the uniform dipole and a K₁ = 0.2 m⁻² bend of either sign an independent rigid-face Cartesian integration gives the forward pass and, inverted through the magnet displaced by −DZ, the native backward pass. Every MALIGN_METHOD is run; a backward input with DZ = −1 mm, which cancels the internal flip, restores the input, and transverse-offset and nominal controls pass.
"""

import numpy as np
from _ccbend import ANGLE, RAYS, Magnet, close, error, roundtrip
from _elegant import verdict

DZ = 1e-3
INVERSE_TOL = 3e-14  # a correct inverse returns every ray to rounding of 64 sixth-order slices
ORACLE_TOL = 1e-15  # native transport against the independent integration


def main():
    failures = []

    # Uniform dipole.  A magnet moved along its axis by s carries every ray's trajectory
    # along the incoming direction (the field is uniform and the faces are planes normal to
    # the axis); for a ray parallel to the reference this displaces the outgoing ray by
    # 2 s sin(theta/2) in the exit frame, while a rectangular uniform dipole maps a parallel
    # offset onto itself (R11 = 1, R21 = 0).  Forward through +DZ and inversely through -DZ
    # therefore returns the on-axis ray at x = 4 DZ sin(theta/2), x' = 0.
    params = dict(K1=0.0, DZ=DZ, MALIGN_METHOD=4)
    fwd, back, _ = roundtrip(params)
    law = float(4 * DZ * np.sin(ANGLE / 2))
    observed = back[0, :4]
    failures.append(error(back, RAYS) > INVERSE_TOL)
    close([observed], [[law, 0, 0, 0]], 1e-16, "Wrong-sign circle law")
    forward_error = close(
        fwd, Magnet(k1=0.0, d=(0, 0, DZ)).forward(RAYS), ORACLE_TOL, "Dipole forward"
    )
    backward_error = close(
        back, Magnet(k1=0.0, d=(0, 0, -DZ)).inverse(fwd), ORACLE_TOL, "Dipole backward"
    )
    _, corrected, _ = roundtrip(params, dict(params, DZ=-DZ))
    fixed = close(corrected, RAYS, INVERSE_TOL, "Dipole compensated inverse")
    print(
        f"uniform dipole MALIGN_METHOD=4 DZ=1e-3: roundtrip (x,x',y,y')={observed.tolist()} "
        f"expected=[0.0, 0.0, 0.0, 0.0] (inverse transport); wrong-sign law 4*DZ*sin(theta/2)={law!r}; "
        f"forward vs integration={forward_error:.4g}; backward vs integrated inverse with DZ=-1e-3="
        f"{backward_error:.4g}; backward input DZ=-1e-3 max error={fixed:.4g}"
    )

    # Combined-function bend, every alignment method.  Methods 1 and 2 are linearized, so
    # only the exact methods 0, 3 and 4 are compared with the integration.
    for method in range(5):
        params = dict(DZ=DZ, MALIGN_METHOD=method)
        fwd, back, _ = roundtrip(params)
        roundtrip_error = error(back, RAYS)
        failures.append(roundtrip_error > INVERSE_TOL)
        _, corrected, _ = roundtrip(params, dict(params, DZ=-DZ))
        fixed = close(corrected, RAYS, INVERSE_TOL, f"Method {method} compensated inverse")
        line = (
            f"K1=0.2 MALIGN_METHOD={method} DZ=1e-3: roundtrip (x,x')={back[0, :2].tolist()} "
            f"max error={roundtrip_error:.4g} expected=0 (inverse transport); "
            f"backward input DZ=-1e-3 max error={fixed:.4g}"
        )
        if method in (0, 3, 4):
            forward_error = close(fwd, Magnet(d=(0, 0, DZ)).forward(RAYS), ORACLE_TOL, "Forward")
            backward_error = close(back, Magnet(d=(0, 0, -DZ)).inverse(fwd), ORACLE_TOL, "Backward")
            line += (
                f"; forward vs integration={forward_error:.4g}; backward vs integrated inverse "
                f"with DZ=-1e-3={backward_error:.4g}"
            )
        print(line)

    # Negative bend angle.
    params = dict(ANGLE=-ANGLE, DZ=DZ, MALIGN_METHOD=4)
    fwd, back, _ = roundtrip(params)
    roundtrip_error = error(back, RAYS)
    failures.append(roundtrip_error > INVERSE_TOL)
    forward_error = close(
        fwd, Magnet(angle=-ANGLE, d=(0, 0, DZ)).forward(RAYS), ORACLE_TOL, "Forward"
    )
    backward_error = close(
        back, Magnet(angle=-ANGLE, d=(0, 0, -DZ)).inverse(fwd), ORACLE_TOL, "Backward"
    )
    print(
        f"K1=0.2 ANGLE=-0.04 MALIGN_METHOD=4 DZ=1e-3: roundtrip (x,x')={back[0, :2].tolist()} "
        f"max error={roundtrip_error:.4g} expected=0 (inverse transport); forward vs integration="
        f"{forward_error:.4g}; backward vs integrated inverse with DZ=-1e-3={backward_error:.4g}"
    )

    # Controls: transverse offsets and the nominal magnet.
    for label, params in (
        ("MALIGN_METHOD=4 DX=1e-3", dict(DX=1e-3, MALIGN_METHOD=4)),
        ("MALIGN_METHOD=4 DY=1e-3", dict(DY=1e-3, MALIGN_METHOD=4)),
        ("nominal", {}),
    ):
        _, back, _ = roundtrip(params)
        control = close(back, RAYS, INVERSE_TOL, f"Control {label}")
        print(f"control K1=0.2 {label}: roundtrip max error={control:.4g} expected=0")
    verdict(any(failures))


if __name__ == "__main__":
    main()
