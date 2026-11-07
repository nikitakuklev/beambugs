---
title: "TRK-59: Slicing suppresses Runge–Kutta hard-edge solenoid spin rotations"
hide:
  - toc
---

# TRK-59 — Slicing suppresses Runge–Kutta hard-edge solenoid spin rotations

| | |
|---|---|
| Code | [Bmad](../index.md) · Tracking, RF, fields and spin |
| Priority | P2 |
| Severity | Wrong physics |
| Evidence | Reproduced, independently rechecked |
| Last reproduced | 2026-11-06 on Bmad release [20261031-0](https://github.com/bmad-sim/bmad-ecosystem/tree/20261031-0) (`c4c629f`) |
| Tags | <span class="bb-tag">partly-source-only</span> [?](../../methodology.md#tags) |
| Source | [slice API, bmad/code/create_element_slice.f90:1–29](https://github.com/bmad-sim/bmad-ecosystem/blob/c4c629fc9071d3a56cb1ba5bc714baba65e9c5d3/bmad/code/create_element_slice.f90#L1-L29)<br>[delegated field lookup, bmad/code/create_element_slice.f90:212–217](https://github.com/bmad-sim/bmad-ecosystem/blob/c4c629fc9071d3a56cb1ba5bc714baba65e9c5d3/bmad/code/create_element_slice.f90#L212-L217)<br>[edge-spin selector, bmad/modules/runge_kutta_mod.f90:93–103](https://github.com/bmad-sim/bmad-ecosystem/blob/c4c629fc9071d3a56cb1ba5bc714baba65e9c5d3/bmad/modules/runge_kutta_mod.f90#L93-L103)<br>[passes that flag to the edge helper, bmad/modules/runge_kutta_mod.f90:142–149](https://github.com/bmad-sim/bmad-ecosystem/blob/c4c629fc9071d3a56cb1ba5bc714baba65e9c5d3/bmad/modules/runge_kutta_mod.f90#L142-L149)<br>[selector and edge call, bmad/modules/time_tracker_mod.f90:198–208](https://github.com/bmad-sim/bmad-ecosystem/blob/c4c629fc9071d3a56cb1ba5bc714baba65e9c5d3/bmad/modules/time_tracker_mod.f90#L198-L208)<br>[Body spin integration, bmad/modules/runge_kutta_mod.f90:793–801](https://github.com/bmad-sim/bmad-ecosystem/blob/c4c629fc9071d3a56cb1ba5bc714baba65e9c5d3/bmad/modules/runge_kutta_mod.f90#L793-L801)<br>[resolves delegated fields, bmad/low_level/apply_element_edge_kick.f90:85–94](https://github.com/bmad-sim/bmad-ecosystem/blob/c4c629fc9071d3a56cb1ba5bc714baba65e9c5d3/bmad/low_level/apply_element_edge_kick.f90#L85-L94)<br>[spin enable gate, bmad/low_level/apply_element_edge_kick.f90:106–113](https://github.com/bmad-sim/bmad-ecosystem/blob/c4c629fc9071d3a56cb1ba5bc714baba65e9c5d3/bmad/low_level/apply_element_edge_kick.f90#L106-L113)<br>[solenoid edge action, bmad/low_level/apply_element_edge_kick.f90:334–360](https://github.com/bmad-sim/bmad-ecosystem/blob/c4c629fc9071d3a56cb1ba5bc714baba65e9c5d3/bmad/low_level/apply_element_edge_kick.f90#L334-L360)<br>[local-field spin contract, bmad/doc/methods.tex:472–482](https://github.com/bmad-sim/bmad-ecosystem/blob/c4c629fc9071d3a56cb1ba5bc714baba65e9c5d3/bmad/doc/methods.tex#L472-L482)<br>[default-enabled spin fringes, bmad/doc/attributes.tex:3374–3382](https://github.com/bmad-sim/bmad-ecosystem/blob/c4c629fc9071d3a56cb1ba5bc714baba65e9c5d3/bmad/doc/attributes.tex#L3374-L3382) |

With spin fringe tracking enabled, slicing a hard-edge solenoid suppresses its Runge–Kutta fringe spin rotations, even for one full-length slice. The same failure occurs with marker-created splits and with adaptive or fixed-step integration in space or time. A 6 MeV/c electron entering a 0.6 m, 22.02 mT solenoid 1 mm off axis leaves with horizontal spin −5.085×10⁻⁶ instead of −2.208×10⁻⁴, a 97.70% underestimate. Independent Lorentz–Thomas–BMT transfer matrices establish the missing edge action; orbital transport and axial precession are preserved.

Run `python TRK-59.py` with [_bmad.py](../helpers/_bmad.py) and [_slicing.py](../helpers/_slicing.py) in the same directory.

??? example "Reproducer: TRK-59.py"

    [Download TRK-59.py](TRK-59.py)

    ```python
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

    ```

## Provenance

| Role | Model | Effort | Date | Code examined |
|---|---|---|---|---|
| Found | `gpt-6-astra` | xhigh | 2026-11-06 | 20261031-0 (`c4c629fc9071d3a56cb1ba5bc714baba65e9c5d3`), source, production, GNU Fortran 15.3.0, CPU, double, OpenMP one thread, MPI disabled |
| Confirmed | `gpt-6.1-sol` | xhigh | 2026-11-06 | 20261031-0 (`c4c629fc9071d3a56cb1ba5bc714baba65e9c5d3`), source, production, GNU Fortran 15.3.0, CPU, double, OpenMP one thread, MPI disabled |

