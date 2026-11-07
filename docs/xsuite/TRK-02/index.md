---
title: "TRK-02: Tracking optimization drops a full bend's multipole fringe"
hide:
  - toc
---

# TRK-02 — Tracking optimization drops a full bend's multipole fringe

| | |
|---|---|
| Code | [Xsuite](../index.md) · Tracking |
| Priority | P2 |
| Severity | Wrong physics |
| Evidence | Probe-pinned |
| Last reproduced | 2026-10-26 on xtrack release [v0.118.4](https://github.com/xsuite/xtrack/tree/v0.118.4) (`3cd8bfc`) |
| Pull request | [[gh:https://github.com/xsuite/xtrack/pull/911]], fix |
| Source | [`ThinSliceBendEntry.get_equivalent_element`, slice_elements_edge.py:52–73](https://github.com/xsuite/xtrack/blob/ad5e808cdc5b9a881eac2f41e32a0b7d11cc7028/xtrack/beam_elements/slice_elements_edge.py#L52-L73) |

`Line.optimize_for_tracking()` replaces the full edge of a sliced `Bend` with a dipole-only edge, silently dropping the fringe field of the bend's quadrupole component. Optimization should leave the sliced map unchanged, and the unoptimized line agrees with the manual's multipole-fringe generating function. For a straight 0.4 m `Bend` with no dipole field, quadrupole strength `k1` = 0.7 m⁻², a full entrance edge and `Teapot(1)` slicing, a particle entering at (x, y) = (20, 10) mm with normalized momenta (`px`, `py`) = (0.001, −0.002) exits at y = 9.738 mm instead of 9.737 mm, 0.9015 µm too high. Entrance and exit edges with a normal quadrupole component are both affected; without one, optimization preserves the map.

Run `python TRK-02.py` with [_common.py](../helpers/_common.py) in the same directory.

??? example "Reproducer: TRK-02.py"

    [Download TRK-02.py](TRK-02.py)

    ```python
    """TRK-02: Tracking optimization drops a full bend's multipole fringe

    `Line.optimize_for_tracking()` replaces the full edge of a sliced `Bend` with a dipole-only edge,
    silently dropping the fringe field of the bend's quadrupole component. Optimization should leave the
    sliced map unchanged, and the unoptimized line agrees with the manual's multipole-fringe generating
    function. For a straight 0.4 m `Bend` with no dipole field, quadrupole strength `k1` = 0.7 m⁻², a
    full entrance edge and `Teapot(1)` slicing, a particle entering at (x, y) = (20, 10) mm with
    normalized momenta (`px`, `py`) = (0.001, −0.002) exits at y = 9.738 mm instead of 9.737 mm, 0.9015
    µm too high. Entrance and exit edges with a normal quadrupole component are both affected; without
    one, optimization preserves the map.

    The script builds this bend, and a control with `k1` = 0, slices it with `Teapot(1)` and tracks the
    particle before and after optimization. The expected exit coordinates come from the
    quadrupole-fringe generating function followed by the same `Teapot(1)` body of half drift, thin
    quadrupole kick and half drift; the entrance edge alone and the unoptimized line must match them.
    After optimization the `k1` = 0 control still matches, and the `k1` = 0.7 m⁻² bend shows the bug.
    """

    import numpy as np
    import xtrack as xt

    RTOL, ATOL = 2e-13, 2e-16


    def coordinates(particles):
        return np.array(
            [getattr(particles, field)[0] for field in ("x", "px", "y", "py", "zeta", "delta")]
        )


    def analytic_map(k1):
        x, px, y, py = 0.02, 0.001, 0.01, -0.002
        # Entrance generating function S=q.P - (P_x*f_x+P_y*f_y)/(1+delta), with
        # f_x = -k1 (x^3 + 3 x y^2)/12, f_y = k1 (3 x^2 y + y^3)/12: position q' = q - f,
        # momenta p' = J^-T p with J = d(q - f)/dq, longitudinal kick px' f_x + py' f_y.
        fx = -k1 * (x**3 + 3 * x * y * y) / 12
        fy = k1 * (3 * x * x * y + y**3) / 12
        a = k1 * (x * x + y * y) / 4
        b = k1 * x * y / 2
        jacobian = np.array([[1 + a, b], [-b, 1 - a]])
        px, py = np.linalg.solve(jacobian.T, [px, py])
        x, y = x - fx, y - fy
        zeta = px * fx + py * fy
        edge = np.array([x, px, y, py, zeta, 0.0])
        # Teapot(1): expanded half-drift, integrated quadrupole kick, half-drift.
        half_length = 0.2
        x, y = x + half_length * px, y + half_length * py
        zeta -= half_length * (px * px + py * py) / 2
        px, py = px - k1 * 0.4 * x, py + k1 * 0.4 * y
        x, y = x + half_length * px, y + half_length * py
        zeta -= half_length * (px * px + py * py) / 2
        return edge, np.array([x, px, y, py, zeta, 0.0])


    def evaluate():
        cases = {}
        for k1 in (0.0, 0.7):
            bend = xt.Bend(
                length=0.4, angle=0, k0=0, k1=k1, edge_entry_model="full", edge_exit_active=False
            )
            line = xt.Line(elements={"b": bend})
            line.slice_thick_elements(slicing_strategies=[xt.Strategy(xt.Teapot(1))])
            line.build_tracker()
            initial = xt.Particles(p0c=1e9, x=0.02, px=0.001, y=0.01, py=-0.002)
            edge_expected, expected = analytic_map(k1)
            edge_particle = initial.copy()
            line["b..entry_map"].track(edge_particle)
            np.testing.assert_allclose(coordinates(edge_particle), edge_expected, rtol=RTOL, atol=ATOL)
            before = initial.copy()
            line.track(before)
            np.testing.assert_allclose(coordinates(before), expected, rtol=RTOL, atol=ATOL)
            line.optimize_for_tracking(verbose=False)
            after = initial.copy()
            line.track(after)
            assert after.state[0] == 1
            if k1 == 0:
                np.testing.assert_allclose(coordinates(after), expected, rtol=RTOL, atol=ATOL)
            cases[f"optimized k1={k1}, x,px,y,py,zeta,delta"] = coordinates(after), expected
        return cases


    if __name__ == "__main__":
        from _common import prepare, report

        prepare()
        print("Expected: quadrupole fringe generating function followed by the same Teapot(1) body.")
        report(evaluate(), rtol=RTOL, atol=ATOL)

    ```

## Provenance

| Role | Model | Effort | Date | Code examined |
|---|---|---|---|---|
| Found | `gpt-6-astra` | xhigh | 2026-10-26 | xtrack 0.118.2 (`ad5e808cdc5b`), installed packages, prebuilt kernels, CPU, serial |
| Confirmed | `gpt-6.1-sol` | xhigh | 2026-10-30 | xtrack 0.118.2 (`ad5e808cdc5b`), installed packages, prebuilt kernels (compilation forbidden), CPU, double, serial |

