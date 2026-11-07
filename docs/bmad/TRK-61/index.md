---
title: "TRK-61: Slice boundary distances count the parent displacement twice"
hide:
  - toc
---

# TRK-61 — Slice boundary distances count the parent displacement twice

| | |
|---|---|
| Code | [Bmad](../index.md) · Tracking, RF, fields and spin |
| Priority | P2 |
| Severity | Wrong physics |
| Evidence | Reproduced, independently rechecked |
| Last reproduced | 2026-11-06 on Bmad 20261031-0 (`c4c629f`, committed 2026-10-31) |
| Tags | <span class="bb-tag">partly-source-only</span> [?](../../methodology.md#tags) |
| Source | [longitudinal-field boundary contract, bmad/low_level/calc_next_fringe_edge.f90:1–15](https://github.com/bmad-sim/bmad-ecosystem/blob/c4c629fc9071d3a56cb1ba5bc714baba65e9c5d3/bmad/low_level/calc_next_fringe_edge.f90#L1-L15)<br>[parent offset, particle position and clipped boundaries, bmad/low_level/calc_next_fringe_edge.f90:124–147](https://github.com/bmad-sim/bmad-ecosystem/blob/c4c629fc9071d3a56cb1ba5bc714baba65e9c5d3/bmad/low_level/calc_next_fringe_edge.f90#L124-L147)<br>[returned edge and distance, bmad/low_level/calc_next_fringe_edge.f90:214–222](https://github.com/bmad-sim/bmad-ecosystem/blob/c4c629fc9071d3a56cb1ba5bc714baba65e9c5d3/bmad/low_level/calc_next_fringe_edge.f90#L214-L222)<br>[native spin reconstruction, bmad/spin/track1_spin_integration.f90:48–95](https://github.com/bmad-sim/bmad-ecosystem/blob/c4c629fc9071d3a56cb1ba5bc714baba65e9c5d3/bmad/spin/track1_spin_integration.f90#L48-L95) |

Native spin tracking adds a parent-to-slice displacement twice when calculating the distance to a slice boundary. Its reconstructed orbit therefore receives a spurious drift even when the particle is already at that boundary. For the last 0.1 m slice of a 0.6 m solenoid, the reported entrance distance is −0.5 m instead of zero. In a two-piece, 22.02 mT solenoid at 6 MeV/c and 1 mm entrance displacement, the vertical spin magnitude is 4.770×10⁻⁵ instead of 2.994×10⁻⁵, a 59.31% excess. Independent boundary geometry and Thomas–BMT transport establish the error.

Run `python TRK-61.py` with [_bmad.py](../helpers/_bmad.py) and [_slicing.py](../helpers/_slicing.py) in the same directory.

??? example "Reproducer: TRK-61.py"

    [Download TRK-61.py](TRK-61.py)

    ```python
    """TRK-61: Slice boundary distances count the parent displacement twice

    Native spin tracking adds a parent-to-slice displacement twice when calculating the distance to a slice boundary. Its reconstructed orbit therefore receives a spurious drift even when the particle is already at that boundary. For the last 0.1 m slice of a 0.6 m solenoid, the reported entrance distance is −0.5 m instead of zero. In a two-piece, 22.02 mT solenoid at 6 MeV/c and 1 mm entrance displacement, the vertical spin magnitude is 4.770×10⁻⁵ instead of 2.994×10⁻⁵, a 59.31% excess. Independent boundary geometry and Thomas–BMT transport establish the error.

    Measure the boundary distance for a particle already at the last slice entrance and compare two-slice spin with an independent Lorentz–Thomas–BMT matrix. A local distance-only helper copy isolates the spurious drift, with full-length and marker-split controls.
    """

    import numpy as np
    from _bmad import compare, verdict
    from _slicing import analytic, branch, close, spin


    def main():
        whole, constants, _ = spin(0, 0)
        one, _, _ = spin(0, 1)
        close(one, whole, 0, "Zero parent-offset control")
        _, _, edges = spin(0, 6)
        last = edges[-1]
        if int(last[1]) != 1:
            raise RuntimeError("Final entrance boundary absent")
        # EDGE: index, has_fringe, particle_at, orbit.s, slice.s_start, s_edge_body, ds_edge.
        orbit_s, slice_start, edge_s, distance = map(float, last[3:])
        expected_distance = edge_s - (orbit_s - slice_start)
        close(expected_distance, 0, 1e-16, "Particle already at entrance")
        defect = compare(
            "last 0.1 m slice entrance distance (m); slice-frame geometry",
            distance,
            expected_distance,
            atol=1e-15,
        )
        two, _, two_edges = spin(0, 2)
        if [int(edge[1]) for edge in two_edges] != [1, 1]:
            raise RuntimeError("Two-slice test requires both boundary conversions")
        expected = analytic(constants)
        compare(
            "two native slices Sx,Sy; analytic Lorentz–BMT",
            two[6:8],
            expected[4:],
            atol=1.5e-12 / 2**4 + 5e-15,
        )
        if defect:
            corrected, _, _ = spin(0, 2, variant="distance")
            close(
                corrected[6:8], expected[4:], 1.5e-12 / 2**4 + 5e-15, "Distance-only analytic control"
            )
            print(
                f"distance-only two-slice control: Sx={corrected[6]:.17g}, Sy={corrected[7]:.17g}; analytic_max_error={np.max(abs(corrected[6:8] - expected[4:])):.3g}"
            )
            original_mm, _, _ = spin(0, 2, amp=500)
            corrected_mm, _, _ = spin(0, 2, amp=500, variant="distance")
            print(
                f"1 mm entrance x vertical spin: observed={original_mm[7]:.17g} expected_distance_corrected={corrected_mm[7]:.17g}; magnitude_excess={100 * (original_mm[7] / corrected_mm[7] - 1):.6f}%"
            )
        _, structures = branch("bmad_standard", True)
        pieces = [row for row in structures if "S#" in row[1]]
        if len(pieces) != 4 or len(pieces[-1]) != 10:
            raise RuntimeError(f"Expected final parsed boundary diagnostic: {structures}")
        compare(
            "last marker-created slice entrance distance (m); particle at boundary",
            float(pieces[-1][-1]),
            0,
            atol=1e-15,
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

