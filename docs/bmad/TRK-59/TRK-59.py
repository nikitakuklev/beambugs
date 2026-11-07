"""TRK-59: Slicing suppresses Runge–Kutta hard-edge solenoid spin rotations

With spin fringe tracking enabled, slicing a hard-edge solenoid suppresses its Runge–Kutta fringe spin rotations, even for one full-length slice. The same failure occurs with marker-created splits and with adaptive or fixed-step integration in space or time. A 6 MeV/c electron entering a 0.6 m, 22.02 mT solenoid 1 mm off axis leaves with horizontal spin −5.085×10⁻⁶ instead of −2.208×10⁻⁴, a 97.70% underestimate. Independent Lorentz–Thomas–BMT transfer matrices establish the missing edge action; orbital transport and axial precession are preserved.

Compare four RK methods with an independent linear Lorentz–Thomas–BMT matrix, a full-length slice, and marker-created splits. Direct-field, disabled-fringe, and axial-precession controls isolate the edge-spin selector; a millimetre-amplitude case measures the practical error.
"""

from _bmad import compare, verdict
from _slicing import analytic, branch, close, spin


def main():
    discrepancies = []
    for method, label in enumerate(("s-adaptive", "s-fixed", "time-adaptive", "time-fixed"), 1):
        whole, constants, _ = spin(method, 0)
        expected = analytic(constants)
        body_only = analytic(constants, fringe=False)
        close(whole[6:8], expected[4:], 2e-13, "Whole-element analytic spin")
        for n in (1, 2, 6):
            sliced, _, _ = spin(method, n)
            close(sliced[[0, 2, 1, 3]], expected[:4], 3e-17, "Analytic orbit")
            mismatch = compare(
                f"{label} {n} slices Sx,Sy; Lorentz–BMT matrix",
                sliced[6:8],
                expected[4:],
                atol=2e-13,
            )
            discrepancies.append(mismatch)
            if mismatch:
                close(sliced[6:8], body_only[4:], 2e-13, "Body-only spin signature")
        direct, _, _ = spin(method, 1, direct=1)
        off, _, _ = spin(method, 0, fringe=0)
        axis_whole, _, _ = spin(method, 0, axis=1)
        axis_slice, _, _ = spin(method, 1, axis=1)
        close(direct, whole, 0, "Direct-field full-length slice")
        close(off[6:8], body_only[4:], 2e-13, "Fringe-disabled analytic spin")
        close(axis_slice, axis_whole, 0, "Axial precession")
        print(
            f"{label} controls: whole_Sx={whole[6]:.17g}, direct/whole and axial/slice max_error=0"
        )
    parsed_whole, _ = branch("runge_kutta", False)
    parsed_split, _ = branch("runge_kutta", True)
    close(parsed_split[:4], parsed_whole[:4], 3e-17, "Marker-split orbit")
    discrepancies.append(
        compare(
            "marker-created splits Sx,Sy; same physical whole solenoid",
            parsed_split[6:8],
            parsed_whole[6:8],
            atol=2e-13,
        )
    )
    whole_mm, _, _ = spin(1, 0, amp=500)
    sliced_mm, _, _ = spin(1, 1, amp=500)
    discrepancies.append(
        compare(
            "1 mm entrance x Sx; whole-element representation",
            sliced_mm[6],
            whole_mm[6],
            atol=2e-12,
        )
    )
    print(f"1 mm horizontal-spin underestimate={100 * (1 - sliced_mm[6] / whole_mm[6]):.6f}%")
    verdict(any(discrepancies))


if __name__ == "__main__":
    main()
