---
title: "TRK-60: Native spin tracking rejects interior solenoid boundary conversions"
hide:
  - toc
---

# TRK-60 — Native spin tracking rejects interior solenoid boundary conversions

| | |
|---|---|
| Code | [Bmad](../index.md) · Tracking, RF, fields and spin |
| Priority | P2 |
| Failure | Wrong physics |
| Evidence | Reproduced, independently rechecked |
| Last reproduced | 2026-11-06 on Bmad release [20261031-0](https://github.com/bmad-sim/bmad-ecosystem/tree/20261031-0) (`c4c629f`) |
| Tags | <span class="bb-tag">partly-source-only</span> [?](../../methodology.md#tags) |
| Source | [boundary initialization contract, bmad/low_level/init_fringe_info.f90:1–12](https://github.com/bmad-sim/bmad-ecosystem/blob/c4c629fc9071d3a56cb1ba5bc714baba65e9c5d3/bmad/low_level/init_fringe_info.f90#L1-L12)<br>[boundary-location contract, bmad/low_level/calc_next_fringe_edge.f90:1–15](https://github.com/bmad-sim/bmad-ecosystem/blob/c4c629fc9071d3a56cb1ba5bc714baba65e9c5d3/bmad/low_level/calc_next_fringe_edge.f90#L1-L15)<br>[interior-slice rejection, bmad/low_level/init_fringe_info.f90:109–129](https://github.com/bmad-sim/bmad-ecosystem/blob/c4c629fc9071d3a56cb1ba5bc714baba65e9c5d3/bmad/low_level/init_fringe_info.f90#L109-L129)<br>[the all-nowhere result, bmad/low_level/init_fringe_info.f90:68–75](https://github.com/bmad-sim/bmad-ecosystem/blob/c4c629fc9071d3a56cb1ba5bc714baba65e9c5d3/bmad/low_level/init_fringe_info.f90#L68-L75)<br>[temporary-orbit reconstruction, bmad/spin/track1_spin_integration.f90:48–95](https://github.com/bmad-sim/bmad-ecosystem/blob/c4c629fc9071d3a56cb1ba5bc714baba65e9c5d3/bmad/spin/track1_spin_integration.f90#L48-L95)<br>[native canonical-momentum convention, bmad/doc/charged-tracking.tex:1627–1657](https://github.com/bmad-sim/bmad-ecosystem/blob/c4c629fc9071d3a56cb1ba5bc714baba65e9c5d3/bmad/doc/charged-tracking.tex#L1627-L1657)<br>[conversion to kinetic momenta, bmad/low_level/solenoid_track_and_mat.f90:44–55](https://github.com/bmad-sim/bmad-ecosystem/blob/c4c629fc9071d3a56cb1ba5bc714baba65e9c5d3/bmad/low_level/solenoid_track_and_mat.f90#L44-L55)<br>[spin dispatcher, bmad/spin/track1_spin.f90:53–103](https://github.com/bmad-sim/bmad-ecosystem/blob/c4c629fc9071d3a56cb1ba5bc714baba65e9c5d3/bmad/spin/track1_spin.f90#L53-L103)<br>[finite-tolerance spin quadrature, bmad/spin/track1_spin_integration.f90:130–179](https://github.com/bmad-sim/bmad-ecosystem/blob/c4c629fc9071d3a56cb1ba5bc714baba65e9c5d3/bmad/spin/track1_spin_integration.f90#L130-L179) |

Native spin tracking incorrectly treats interior solenoid slices as having no boundary conversions. The native canonical-momentum convention requires these conversions even at artificial cuts, where adjacent exit and entrance transformations cancel. A 0.6 m solenoid divided into six pieces supplies conversions at only two of the six entrances. Independent Lorentz–Thomas–BMT transport and isolated boundary-helper checks establish the resulting loss of spin action; the native orbital map is preserved. Marker-created splits and a solenoid–quadrupole element with zero quadrupole strength exhibit the same failure.

Run `python TRK-60.py` with [_bmad.py](../helpers/_bmad.py) and [_slicing.py](../helpers/_slicing.py) in the same directory.

??? example "Reproducer: TRK-60.py"

    [Download TRK-60.py](TRK-60.py)

    ```python
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

    ```

## Provenance

| Role | Model | Effort | Date | Code examined |
|---|---|---|---|---|
| Found | `gpt-6-astra` | xhigh | 2026-11-06 | 20261031-0 (`c4c629fc9071d3a56cb1ba5bc714baba65e9c5d3`), source, production, GNU Fortran 15.3.0, CPU, double, OpenMP one thread, MPI disabled |
| Confirmed | `gpt-6.1-sol` | xhigh | 2026-11-06 | 20261031-0 (`c4c629fc9071d3a56cb1ba5bc714baba65e9c5d3`), source, production, GNU Fortran 15.3.0, CPU, double, OpenMP one thread, MPI disabled |

