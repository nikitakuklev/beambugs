---
title: "OPT-23: Periodic spin Twiss omits the longitudinal fixed-point constraint"
hide:
  - toc
---

# OPT-23 — Periodic spin Twiss omits the longitudinal fixed-point constraint

| | |
|---|---|
| Code | [Xsuite](../index.md) · Optics and matching |
| Priority | P2 |
| Failure | Wrong physics |
| Evidence | Reproduced, independently rechecked |
| Last reproduced | 2026-11-06 on xtrack release [v0.118.2](https://github.com/xsuite/xtrack/tree/v0.118.2) (`ad5e808`) |
| Pull request | [[gh:https://github.com/xsuite/xtrack/pull/923]], fix |
| Discussed in | [[gh:https://github.com/xsuite/xtrack/pull/922]] |
| Source | [_find_spin_fixed_point, spin.py:23–38](https://github.com/xsuite/xtrack/blob/ad5e808cdc5b9a881eac2f41e32a0b7d11cc7028/xtrack/twiss/spin.py#L23-L38)<br>[residual function, lines 41–56](https://github.com/xsuite/xtrack/blob/ad5e808cdc5b9a881eac2f41e32a0b7d11cc7028/xtrack/twiss/spin.py#L41-L56)<br>[ActionCall.get_targets, optimize.py:2838–2843](https://github.com/xsuite/xdeps/blob/73147cf7d9f1fc7b0f49ddb55c0ce037b70b7353/xdeps/optimize/optimize.py#L2838-L2843)<br>[twiss.py:188–192](https://github.com/xsuite/xtrack/blob/ad5e808cdc5b9a881eac2f41e32a0b7d11cc7028/xtrack/twiss/twiss.py#L188-L192)<br>[spin_polarization.rst:4–6](https://github.com/xsuite/xsuite/blob/e0913e0a18028b2fce06991aebed6150876124b3/docs/spin_polarization.rst#L4-L6)<br>[xtrack.tex:6752–6768](https://github.com/xsuite/xsuite/blob/e0913e0a18028b2fce06991aebed6150876124b3/docs/physics_manual/xtrack.tex#L6752-L6768) |

Periodic spin Twiss can return a spin direction that changes after one turn when the invariant axis is slightly tilted from vertical. Its optimizer constrains only horizontal and vertical changes, which become dependent when the axis has zero longitudinal component. For a flat electron ring viewed through a 1.000 mrad coordinate rotation, the returned longitudinal spin component is 1.819e-4 instead of zero and changes by −3.638e-4 in one turn. Analytic coordinate conjugation, an independent matrix nullspace and direct tracking establish the discrepancy. The exactly vertical control passes.

Run `python OPT-23.py` with [_spin_ring.py](../helpers/_spin_ring.py), [_common.py](../helpers/_common.py) and [_prebuilt.py](../helpers/_prebuilt.py) in the same directory.

??? example "Reproducer: OPT-23.py"

    [Download OPT-23.py](OPT-23.py)

    ```python
    """OPT-23: Periodic spin Twiss omits the longitudinal fixed-point constraint

    Periodic spin Twiss can return a spin direction that changes after one turn when the invariant axis is slightly tilted from vertical. Its optimizer constrains only horizontal and vertical changes, which become dependent when the axis has zero longitudinal component. For a flat electron ring viewed through a 1.000 mrad coordinate rotation, the returned longitudinal spin component is 1.819e-4 instead of zero and changes by −3.638e-4 in one turn. Analytic coordinate conjugation, an independent matrix nullspace and direct tracking establish the discrepancy. The exactly vertical control passes.

    The script constructs a flat 32 m electron ring and conjugates it by a 1 mrad coordinate
    rotation. It compares the returned axis with the analytic conjugated vertical direction,
    an independent nullspace of a directly tracked spin matrix, and one-turn tracking.
    The exactly vertical ring is a control; a fixed implementation prints NOT REPRODUCED.
    """

    from _spin_ring import G, kernels, matrix, np, report, ring, setup, track_spin


    def main():
        setup()
        cases = {}
        for tilt in (0.0, 1e-3):
            line = ring(tilt)
            expected = np.array([-np.sin(tilt), np.cos(tilt), 0.0])
            A = matrix(line)
            _, singular, vh = np.linalg.svd(A - np.eye(3))
            axis = vh[-1] * np.sign(vh[-1] @ expected)
            np.testing.assert_allclose(axis, expected, rtol=0, atol=2e-14)
            np.testing.assert_allclose(track_spin(line, expected), expected, rtol=0, atol=2e-14)
            # The flat-ring BMT map has spin tune G*gamma; it is away from an
            # integer resonance, so the invariant axis is unique as an axis line.
            tune = np.max(np.angle(np.linalg.eigvals(A))) / (2 * np.pi)
            np.testing.assert_allclose(tune, G * 200, rtol=0, atol=3e-14)
            tw = line.twiss(method="4d", spin=True)
            observed = np.array([tw.spin_x[0], tw.spin_y[0], tw.spin_z[0]])
            residual = track_spin(line, observed) - observed
            axis_error = np.linalg.norm(np.cross(observed, expected))
            np.testing.assert_allclose(np.linalg.norm(observed), 1, rtol=0, atol=2e-14)
            if tilt == 0:
                np.testing.assert_allclose(residual, 0, rtol=0, atol=2e-14)
                assert axis_error < 2e-14
            print(
                f"tilt={tilt:g}: observed_axis={observed} expected_axis={expected} "
                f"one_turn_change={residual} axis_line_error={axis_error:.17g} "
                f"nonzero_singular_values={singular[:2]} "
                "derivation=coordinate conjugation and independently tracked nullspace"
            )
            cases[f"tilt={tilt:g} one-turn invariant; derivation=A*n-n=0"] = residual, np.zeros(3)
            cases[f"tilt={tilt:g} axis line; derivation=n=(-sin(t),cos(t),0)"] = axis_error, 0.0
        kernels()
        report(cases, rtol=0, atol=2e-14)


    if __name__ == "__main__":
        main()

    ```

## Provenance

| Role | Model | Effort | Date | Code examined |
|---|---|---|---|---|
| Found | `gpt-6-astra` | xhigh | 2026-11-06 | xtrack 0.118.2 (`ad5e808cdc5b`), installed packages, prebuilt kernels (compilation forbidden), CPU, double, serial |
| Confirmed | `gpt-6.1-sol` | xhigh | 2026-11-06 | xtrack 0.118.2 (`ad5e808cdc5b`), installed packages, prebuilt kernels (compilation forbidden), CPU, double, serial |

