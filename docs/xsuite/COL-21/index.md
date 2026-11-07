---
title: "COL-21: Wake kicks omit source charge and witness charge-to-mass scaling"
hide:
  - toc
---

# COL-21 — Wake kicks omit source charge and witness charge-to-mass scaling

| | |
|---|---|
| Code | [Xsuite](../index.md) · Collective effects |
| Priority | P3 |
| Severity | Wrong physics |
| Evidence | Reproduced, independently rechecked |
| Last reproduced | 2026-11-07 on xfields release [v0.27.7](https://github.com/xsuite/xfields/tree/v0.27.7) (`3f06c87`) |
| Tags | <span class="bb-tag">partly-source-only</span> [?](../../methodology.md#tags) |
| Source | [`_ConvData.track`, convolution.py:185–198](https://github.com/xsuite/xfields/blob/223b4fa134a1435be9b45ef21ae8903f96c5f685/xfields/beam_elements/waketracker/convolution.py#L185-L198)<br>[Source products, lines 200–216](https://github.com/xsuite/xfields/blob/223b4fa134a1435be9b45ef21ae8903f96c5f685/xfields/beam_elements/waketracker/convolution.py#L200-L216)<br>[number populations and means, element_with_slicer.py:192–211](https://github.com/xsuite/xfields/blob/223b4fa134a1435be9b45ef21ae8903f96c5f685/xfields/beam_elements/element_with_slicer.py#L192-L211)<br>[number weight, uniform_bin_slicer.h:79–82](https://github.com/xsuite/xfields/blob/223b4fa134a1435be9b45ef21ae8903f96c5f685/xfields/slicers/slicers_src/uniform_bin_slicer.h#L79-L82)<br>[accumulates number-weighted moments, lines 130–150](https://github.com/xsuite/xfields/blob/223b4fa134a1435be9b45ef21ae8903f96c5f685/xfields/slicers/slicers_src/uniform_bin_slicer.h#L130-L150)<br>[Particles species and number-weight definitions, particles.py:209–235](https://github.com/xsuite/xtrack/blob/ad5e808cdc5b9a881eac2f41e32a0b7d11cc7028/xtrack/particles/particles.py#L209-L235)<br>[mixed-species ensemble and δ, xtrack.tex:327–350](https://github.com/xsuite/xsuite/blob/e0913e0a18028b2fce06991aebed6150876124b3/docs/physics_manual/xtrack.tex#L327-L350)<br>[mass-scaled transverse momenta, lines 393–403](https://github.com/xsuite/xsuite/blob/e0913e0a18028b2fce06991aebed6150876124b3/docs/physics_manual/xtrack.tex#L393-L403)<br>[transverse, xfields.tex:2128–2144](https://github.com/xsuite/xsuite/blob/e0913e0a18028b2fce06991aebed6150876124b3/docs/physics_manual/xfields.tex#L2128-L2144)<br>[longitudinal wake equations, lines 2233–2237](https://github.com/xsuite/xsuite/blob/e0913e0a18028b2fce06991aebed6150876124b3/docs/physics_manual/xfields.tex#L2233-L2237) |

When source charges or witness charge-to-mass ratios differ from their reference values, wake kicks use the wrong species normalization. The code uses the reference charge squared instead of each source–witness charge product and omits the witness mass ratio required by the stored momentum coordinates. With a causal longitudinal wake, an opposite-charge witness at 2.100 GeV/c receives Δδ = −9.066 × 10⁻⁷ instead of +9.066 × 10⁻⁷, where δ is relative momentum deviation. A direct sum over source–witness pairs also finds wrong transverse kicks and a nonzero wake from exactly cancelling opposite-charge sources. Matching-reference homogeneous beams have the correct species normalization.

Run `python COL-21.py` with [_prebuilt.py](../helpers/_prebuilt.py) and [_common.py](../helpers/_common.py) in the same directory.

??? example "Reproducer: COL-21.py"

    [Download COL-21.py](COL-21.py)

    ```python
    """COL-21: Wake kicks omit source charge and witness charge-to-mass scaling

    When source charges or witness charge-to-mass ratios differ from their reference values, wake kicks use the wrong species normalization. The code uses the reference charge squared instead of each source–witness charge product and omits the witness mass ratio required by the stored momentum coordinates. With a causal longitudinal wake, an opposite-charge witness at 2.100 GeV/c receives Δδ = −9.066 × 10⁻⁷ instead of +9.066 × 10⁻⁷, where δ is relative momentum deviation. A direct sum over source–witness pairs also finds wrong transverse kicks and a nonzero wake from exactly cancelling opposite-charge sources. Matching-reference homogeneous beams have the correct species normalization.

    An exponentially decaying causal wake and sources on slice centers give exact sampled pair forces without a bin or self-wake endpoint ambiguity. A direct sum over physical source charges and witness species momenta supplies the expectation independently of the FFT implementation. It uses serial CPU execution with one numerical thread and installed prebuilt kernels, with compilation forbidden.
    """

    from _prebuilt import CTX, identity, kernels, np, xt

    # Import Xfields after the shared helper configures thread pools and compilation.
    # isort: split
    import xfields as xf
    from scipy.constants import e

    PC = 2.1e9
    MASS = 938272089.43
    np.set_printoptions(precision=17, linewidth=10000)
    RTOL = 3e-14
    matches = []


    class Component:
        def __init__(self, plane, source_power=0, test_power=0):
            self.plane = plane
            self.kick = "delta" if plane == "z" else "p" + plane
            self.source_exponents = (source_power, 0)
            self.test_exponents = (0, test_power)
            self.source_moments = ["num_particles"] + (["x"] if source_power else [])
            self._source_moment_exponents = [1] * len(self.source_moments)
            self.amplitude = 7e11 if plane == "z" else 9e13

        def function_vs_zeta(self, zeta, beta0, dzeta):
            # All nonzero pair separations are >= 0.5 m: neither the endpoint
            # convention nor the cell-integrated self-wake can enter.
            return self.amplitude * np.where(zeta < -0.1, np.exp(zeta / 2.0), 0.0)


    def run(
        label,
        charges,
        q0=1.0,
        mr=1.0,
        mass0=MASS,
        p0c=PC,
        cancellation=False,
        control=False,
        split=1,
        zero=False,
    ):
        initial_charges = np.asarray(charges).copy()
        for plane, sp, tp in (("z", 0, 0), ("x", 1, 0), ("y", 0, 1)):
            component = Component(plane, sp, tp)
            if zero:
                component.amplitude = 0.0
            zeta = np.array([1.75, 0.75, -1.25, -0.75])
            if cancellation:
                zeta[1] = zeta[0]
            weights = np.array([2e10, 2e10 if cancellation else 3e10, 0.0, 0.0])
            xs = np.array([0.0013, 0.0013 if cancellation else -0.0007, 0.0004, -0.0002])
            ys = np.array([0.0001, -0.0003, -0.0008, 0.0006])
            charges = np.array(initial_charges, dtype=float)
            ratios = np.broadcast_to(np.array(mr, dtype=float), (4,)).copy()
            # Splitting preserves number and charge without changing any force.
            if split > 1:
                zeta = np.repeat(zeta, split)
                weights = np.repeat(weights / split, split)
                xs, ys = np.repeat(xs, split), np.repeat(ys, split)
                charges, ratios = np.repeat(charges, split), np.repeat(ratios, split)
            particles = xt.Particles(
                _context=CTX,
                mass0=mass0,
                p0c=p0c,
                q0=q0,
                mass_ratio=ratios,
                charge_ratio=charges / q0,
                delta=np.zeros(len(zeta)),
                zeta=zeta,
                x=xs,
                y=ys,
                weight=weights,
            )
            beta = p0c / np.hypot(p0c, mass0)
            source_coordinate = xs**sp
            witness_coordinate = ys**tp
            # Direct O(n^2) pair sum, in physical charges Qs and Qw (units of e).
            # delta and transverse momenta are normalized to mr_w * p0c.
            pair_wake = component.function_vs_zeta(zeta[:, None] - zeta[None, :], beta, 0.5)
            signed_source = pair_wake @ (weights * charges * source_coordinate)
            sign = -1 if plane == "z" else 1
            expected = sign * e * charges * witness_coordinate * signed_source / (ratios * p0c * beta)
            # Independent number-source formula predicts exactly the faulty path.
            faulty = (
                sign
                * e
                * q0**2
                * witness_coordinate
                * (pair_wake @ (weights * source_coordinate))
                / (p0c * beta)
            )
            tracker = xf.WakeTracker(
                _context=CTX, components=[component], zeta_range=(-2.0, 2.0), num_slices=8, num_turns=1
            )
            before = np.array(getattr(particles, component.kick)).copy()
            tracker.track(particles)
            observed = np.array(getattr(particles, component.kick)) - before
            assert np.all(particles.state == 1)
            witnesses = slice(2 * split, None)
            obs, exp, bad = observed[witnesses], expected[witnesses], faulty[witnesses]
            scale = float(np.max(np.abs(bad)))
            atol = scale * 16 * np.finfo(float).eps
            predicted = bool(np.allclose(obs, bad, rtol=RTOL, atol=atol))
            passed = bool(np.allclose(obs, exp, rtol=RTOL, atol=atol))
            if control:
                assert passed, (label, plane)
            print(
                f"{label} plane={plane} split={split}: observed={obs} expected={exp} "
                f"derivation=sigma*e*Qw*fw*sum(Ns*Qs*fs*W)/(mr_w*p0c*beta0) rtol={RTOL} atol={atol:.17g} control={control} physical_match={passed} missing_factor_match={predicted}"
            )
            matches.append(passed)


    if __name__ == "__main__":
        identity()
        print(
            "DERIVATION: sigma*e*Qw*fw*sum(Ns*Qs*fs*W)/(mr_w*p0c*beta0); sigma=-1 for delta, +1 transverse"
        )
        run("matching", [1, 1, 1, 1], control=True)
        run("matching-negative", [-1, -1, -1, -1], q0=-1, control=True)
        run("matching-double", [2, 2, 2, 2], q0=2, control=True)
        run("source-mass", [1, 1, 1, 1], mr=[2, 2, 1, 1], control=True)
        run("zero-wake", [1, -1, -1, 1], control=True, zero=True)
        run("witness-sign", [1, 1, -1, -1])
        run("source-sign", [-1, -1, 1, 1])
        run("reference-charge", [1, 1, 1, 1], q0=2)
        run("reference-mass", [1, 1, 1, 1], mr=2, mass0=MASS / 2, p0c=PC / 2)
        run("witness-mass", [1, 1, 1, 1], mr=[1, 1, 2, 2])
        run("neutral-source", [1, -1, 1, -1], cancellation=True)
        run("neutral-source", [1, -1, 1, -1], cancellation=True, split=2)
        kernels()
        print(
            f"COMPARISONS total={len(matches)} physical_pass={sum(matches)} physical_fail={len(matches) - sum(matches)}"
        )
        print("NOT REPRODUCED" if all(matches) else "DEFECT REPRODUCED")

    ```

## Provenance

| Role | Model | Effort | Date | Code examined |
|---|---|---|---|---|
| Found | `gpt-6-astra` | xhigh | 2026-11-06 | xfields 0.27.5 (`223b4fa134a1`), installed packages, prebuilt kernels (compilation forbidden), CPU, double, serial |
| Confirmed | `gpt-6.1-sol` | xhigh | 2026-11-06 | xfields 0.27.5 (`223b4fa134a1`), installed packages, prebuilt kernels (compilation forbidden), CPU, double, serial |

