"""TRK-60: Native spin tracking rejects interior solenoid boundary conversions

Native spin tracking incorrectly treats interior solenoid slices as having no boundary conversions. The native canonical-momentum convention requires these conversions even at artificial cuts, where adjacent exit and entrance transformations cancel. A 0.6 m solenoid divided into six pieces supplies conversions at only two of the six entrances. Independent Lorentz–Thomas–BMT transport and isolated boundary-helper checks establish the resulting loss of spin action; the native orbital map is preserved. Marker-created splits and a solenoid–quadrupole element with zero quadrupole strength exhibit the same failure.

Inspect the native boundary flags for six solenoid slices and compare their spin with an independent linear Lorentz–Thomas–BMT matrix. Local diagnostic helper copies separate boundary initialization from the distance fault; parsed marker splits and a zero-quadrupole solenoid–quadrupole provide controls.
"""

import numpy as np
from _bmad import compare, verdict
from _slicing import analytic, branch, close, spin


def main():
    whole, constants, _ = spin(0, 0)
    one, _, _ = spin(0, 1)
    six, _, edges = spin(0, 6)
    expected = analytic(constants)
    close(one, whole, 0, "One full-length native slice")
    close(six[:4], whole[:4], 1e-20, "Native orbital invariance")
    flags = [int(edge[1]) for edge in edges]
    defect = compare(
        "six entrance boundary flags; longitudinal-field conversion contract",
        flags,
        [1] * 6,
        atol=0,
    )
    print(f"six-slice Sx={six[6]:.17g}; analytic_Lorentz_BMT={expected[4]:.17g}; orbit preserved")
    if defect:
        distance_only, _, _ = spin(0, 6, variant="distance")
        both, _, corrected_edges = spin(0, 6, variant="both")
        close(both[6:8], expected[4:], 1.5e-12 / 6**4 + 5e-15, "Both-helper analytic control")
        if [int(edge[1]) for edge in corrected_edges] != [1] * 6:
            raise RuntimeError("Initialization diagnostic did not restore all boundaries")
        compare(
            "distance-only correction Sx,Sy; missing boundary action survives",
            distance_only[6:8],
            expected[4:],
            atol=2e-13,
        )
        print(
            f"both-helper control: Sx={both[6]:.17g}, Sy={both[7]:.17g}; analytic_max_error={np.max(abs(both[6:8] - expected[4:])):.3g}"
        )
    for kind in ("solenoid", "sol_quad"):
        parsed_whole, _ = branch("bmad_standard", False, kind)
        parsed_split, structures = branch("bmad_standard", True, kind)
        # Parsed solenoid pieces have positive length; marker records have other keys.
        pieces = [row for row in structures if "S#" in row[1]]
        if len(pieces) != 4:
            raise RuntimeError(f"Expected four parsed {kind} pieces: {structures}")
        observed = [int(row[7]) for row in pieces]
        print(
            f"parsed {kind} entrance flags: observed={observed} expected={[1] * 4}; longitudinal-field contract"
        )
        compare(
            f"parsed {kind} split Sx,Sy; same physical whole element",
            parsed_split[6:8],
            parsed_whole[6:8],
            atol=2e-12,
        )
    verdict(defect)


if __name__ == "__main__":
    main()
