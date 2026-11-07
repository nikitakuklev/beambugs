"""TRK-63: CCBEND nominal-radius fringes survive dipole-field cancellation

Cancelling CCBEND's dipole field with a fractional strength error leaves a force from its nominal-radius fringe. A horizontal ray 1.500 mm above the midplane of a 0.6500 m, 0.04500 rad magnet acquires y′ = −4.674 µrad in the default fringe model, although field-free Cartesian transport requires zero vertical slope. The default-order integral-based fringe also fails with zero fringe integrals. Both bend signs and total-field or dipole-only cancellation reproduce; disabling fringes restores the straight-ray result.

Intersect fixed Cartesian rays with independently surveyed exit planes at zero physical dipole field. Compare both fringe models, cancellation paths, bend signs and integration refinements, with disabled-fringe and inactive-order controls.
"""

from _slicing import close, track

# isort: split
import _elegant as native
import numpy as np


def straight(rays, length, angle):
    # Reference curvature fixes plane geometry, independently of the physical field.
    rho = length / angle
    origin = np.array([-rho * (1 - np.cos(angle)), 0, rho * np.sin(angle)])
    tangent = np.array([-np.sin(angle), 0, np.cos(angle)])
    horizontal = np.array([np.cos(angle), 0, np.sin(angle)])
    answer = []
    for q in rays:
        start, direction = np.array([q[0], q[2], 0]), np.array([q[1], q[3], 1])
        distance = np.dot(origin - start, tangent) / np.dot(direction, tangent)
        end = start + distance * direction - origin
        answer.append(
            [
                np.dot(end, horizontal),
                np.dot(direction, horizontal) / np.dot(direction, tangent),
                end[1],
                direction[1] / np.dot(direction, tangent),
            ]
        )
    return np.array(answer)


def main():
    rays = np.zeros((4, 6))
    rays[:, :4] = [
        [0, 0, 0.0015, 0],
        [0.0007, 0.0001, -0.0008, 0.0002],
        [0, 0, 0, 0],
        [0, 0, -0.0015, 0],
    ]
    faults = []
    for angle in (0.045, -0.045):
        expected = straight(rays, 0.65, angle)
        for cancellation in ("FSE=-1", "FSE_DIPOLE=-1", "FSE=-.25,FSE_DIPOLE=-.75"):
            coarse = {}
            for slices in (8, 96):
                for fringe, order in ((-1, 3), (0, 3), (1, 3), (0, 0), (1, 1)):
                    lattice = f"B:CCBEND,L=.65,ANGLE={angle},{cancellation},OPTIMIZE_FSE=0,OPTIMIZE_DX=0,REFERENCE_CORRECTION=0,N_SLICES={slices},FRINGEMODEL={fringe},EDGE_ORDER={order},HGAP=0\nBL:LINE=(B)\n"
                    obs = track("field-off", lattice, rays, p0=3200)["raw"][:, :4]
                    if fringe == -1 or order == 0 or (fringe == 1 and order == 1):
                        close(obs, expected, 3e-14, "Cartesian control")
                    else:
                        close(obs[0, 3], -obs[3, 3], 2e-17, "opposite vertical rays")
                        close(obs[2], expected[2], 3e-14, "midplane control")
                        error = float(np.max(abs(obs - expected)))
                        if error <= 3e-14:
                            faults.append(False)
                        elif abs(obs[0, 3]) > 1e-6:
                            faults.append(True)
                        else:
                            raise RuntimeError(f"Unexpected intermediate fringe error: {error}")
                        if slices == 8:
                            coarse[fringe] = obs[0, 3]
                        else:
                            close(obs[0, 3], coarse[fringe], 2e-17, "refinement invariance")
                    print(
                        f"angle={angle:g} {cancellation} N={slices} fringe={fringe} order={order}: y={obs[0, 2]:.17g} m, yprime={obs[0, 3]:.17g}; expected y=.0015 m, yprime=0 (Cartesian ray-plane intersection)"
                    )
    native.verdict(any(faults))


if __name__ == "__main__":
    main()
