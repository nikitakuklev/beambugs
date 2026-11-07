---
title: "TRK-26: Backtracking silently skips spin transport and its rejection"
hide:
  - toc
---

# TRK-26 — Backtracking silently skips spin transport and its rejection

| | |
|---|---|
| Code | [Xsuite](../index.md) · Tracking |
| Priority | P2 |
| Severity | Wrong physics |
| Evidence | Reproduced, independently rechecked |
| Last reproduced | 2026-11-07 on xtrack main (`f09e0fb`, head build) |
| Tags | <span class="bb-tag">partly-source-only</span> [?](../../methodology.md#tags) |
| Issue | [[gh:https://github.com/xsuite/xsuite/issues/647]] |
| Source | [track_magnet.h:104–163](https://github.com/xsuite/xtrack/blob/ad5e808cdc5b9a881eac2f41e32a0b7d11cc7028/xtrack/beam_elements/elements_src/track_magnet.h#L104-L163)<br>[lines 341–343](https://github.com/xsuite/xtrack/blob/ad5e808cdc5b9a881eac2f41e32a0b7d11cc7028/xtrack/beam_elements/elements_src/track_magnet.h#L341-L343)<br>[spin refusal, track_magnet_radiation.h:113–115](https://github.com/xsuite/xtrack/blob/ad5e808cdc5b9a881eac2f41e32a0b7d11cc7028/xtrack/beam_elements/elements_src/track_magnet_radiation.h#L113-L115)<br>[negates length](https://github.com/xsuite/xtrack/blob/ad5e808cdc5b9a881eac2f41e32a0b7d11cc7028/xtrack/beam_elements/elements_src/legacy_solenoid.h#L27-L29)<br>[gates spin on positive length](https://github.com/xsuite/xtrack/blob/ad5e808cdc5b9a881eac2f41e32a0b7d11cc7028/xtrack/beam_elements/elements_src/legacy_solenoid.h#L119-L126)<br>[SplineBoris calls the shared spin helper directly](https://github.com/xsuite/xtrack/blob/ad5e808cdc5b9a881eac2f41e32a0b7d11cc7028/xtrack/beam_elements/elements_src/track_splineboris.h#L293-L302)<br>[spin transported with the orbital coordinates](https://github.com/xsuite/xsuite/blob/e0913e0a18028b2fce06991aebed6150876124b3/docs/spin_polarization.rst#L4-L6)<br>[Thomas–BMT precession and rotation](https://github.com/xsuite/xsuite/blob/e0913e0a18028b2fce06991aebed6150876124b3/docs/physics_manual/xtrack.tex#L6649-L6679)<br>[backtracking restricted to supported elements](https://github.com/xsuite/xsuite/blob/e0913e0a18028b2fce06991aebed6150876124b3/docs/track.rst#L31-L36)<br>[the documented −33 XT_SPIN_BACKTRACK state](https://github.com/xsuite/xsuite/blob/e0913e0a18028b2fce06991aebed6150876124b3/docs/dev_guide_xtbe_lost_part_codes.rst#L21-L21) |

Backtracking a spin-enabled solenoid or bend silently leaves the spin unchanged while returning a live particle. Unsupported spin reversal should return the documented rejection state, or apply the inverse Thomas–BMT rotation. For a 3 GeV/c proton in a 0.9000 m solenoid of strength 0.6500 m⁻¹, forward and backward tracking leaves spin (−0.03779, −0.5988, 0.8000) instead of its initial (0.6000, 0, 0.8000). An analytic axial-field rotation and an opposite-field control establish the discrepancy; a SplineBoris field-map solenoid correctly rejects reversal.

Run `python TRK-26.py` with [_prebuilt.py](../helpers/_prebuilt.py) and [_common.py](../helpers/_common.py) in the same directory.

??? example "Reproducer: TRK-26.py"

    [Download TRK-26.py](TRK-26.py)

    ```python
    """TRK-26: Backtracking silently skips spin transport and its rejection

    Backtracking a spin-enabled solenoid or bend silently leaves the spin unchanged while returning a live particle. Unsupported spin reversal should return the documented rejection state, or apply the inverse Thomas–BMT rotation. For a 3 GeV/c proton in a 0.9000 m solenoid of strength 0.6500 m⁻¹, forward and backward tracking leaves spin (−0.03779, −0.5988, 0.8000) instead of its initial (0.6000, 0, 0.8000). An analytic axial-field rotation and an opposite-field control establish the discrepancy; a SplineBoris field-map solenoid correctly rejects reversal.

    Track axial protons through unsliced and thick-sliced solenoids and compare with a signed analytic Thomas–BMT rotation. Check the inverse map or documented rejection, an opposite-field inverse control, a bend, and a field-map solenoid.
    """

    import math

    from _prebuilt import CTX, identity, kernels, np, xt

    L, KS, G = 0.9, 0.65, 1.79284734463
    S0 = np.array([0.6, 0.0, 0.8])


    def spin(p):
        return np.array([getattr(p, f)[0] for f in ("spin_x", "spin_y", "spin_z")])


    def particle(delta):
        return xt.Particles(
            _context=CTX, p0c=3e9, delta=delta, anomalous_magnetic_moment=G, spin_x=0.6, spin_z=0.8
        )


    def rotation(angle):
        # On axis, BMT integrates to phi=-(1+G)*ks*L/(1+delta).
        return np.array([0.6 * math.cos(angle), 0.6 * math.sin(angle), 0.8])


    def make(kind, sign=1, sliced=False):
        if kind == "Solenoid":
            element = xt.Solenoid(_context=CTX, length=L, ks=sign * KS)
        elif kind == "SplineBoris":
            field = sign * KS * 3e9 / 299792458
            profile = xt.Spline4(val_start=field, der_start=0, val_end=field, der_end=0, mean=field)
            element = xt.SplineBoris(_context=CTX, length=L, bs=profile, n_steps=2)
        else:
            element = xt.UniformSolenoid(
                _context=CTX,
                length=L,
                ks=sign * KS,
                edge_entry_active=False,
                edge_exit_active=False,
                num_multipole_kicks=3,
                integrator="yoshida-4",
            )
        line = xt.Line(elements=[element])
        if sliced:
            line.slice_thick_elements([xt.Strategy(xt.Uniform(3, mode="thick"))])
        line.skip_end_turn_actions = True
        line.build_tracker(_context=CTX)
        line.configure_spin(spin_model="auto")
        return line


    def main():
        identity()
        failures = 0
        for kind, sliced in (
            ("UniformSolenoid", False),
            ("UniformSolenoid", True),
            ("Solenoid", False),
            ("SplineBoris", False),
        ):
            line = make(kind, sliced=sliced)
            for delta in (0.0, 0.07):
                angle = -(1 + G) * KS * L / (1 + delta)
                p = particle(delta)
                line.track(p)
                forward = spin(p)
                np.testing.assert_allclose(forward, rotation(angle), atol=3e-15, rtol=0)
                line.track(p, backtrack=True)
                ok = p.state[0] == -33 or (
                    p.state[0] > 0 and np.allclose(spin(p), S0, atol=3e-15, rtol=0)
                )
                failures += not ok
                print(
                    f"ROUNDTRIP {kind} sliced={sliced} delta={delta}: observed={spin(p)} "
                    f"expected={S0} or state=-33 state={p.state[0]} "
                    f"forward_error={np.max(abs(forward - rotation(angle))):.17g} "
                    f"max_error={np.max(abs(spin(p) - S0)):.17g} "
                    f"s={p.s[0]:.17g} zeta={p.zeta[0]:.17g} derivation=signed axial BMT inverse pass={ok}"
                )
                fresh = particle(delta)
                line.track(fresh, backtrack=True)
                ok = fresh.state[0] == -33 or (
                    fresh.state[0] > 0
                    and np.allclose(spin(fresh), rotation(-angle), atol=3e-15, rtol=0)
                )
                failures += not ok
                print(
                    f"BACKWARD {kind} delta={delta}: observed={spin(fresh)} "
                    f"expected={rotation(-angle)} or state=-33 pass={ok}"
                )
        p = particle(0.07)
        make("UniformSolenoid").track(p)
        make("UniformSolenoid", sign=-1).track(p)
        np.testing.assert_allclose(spin(p), S0, atol=3e-15, rtol=0)
        print(f"OPPOSITE_FIELD: observed={spin(p)} expected={S0} derivation=opposite BMT generator")
        bend = xt.Bend(
            _context=CTX,
            length=0.8,
            angle=0.08,
            k0=0.1,
            edge_entry_active=False,
            edge_exit_active=False,
        )
        line = xt.Line(elements=[bend])
        line.skip_end_turn_actions = True
        line.build_tracker(_context=CTX)
        line.configure_spin(spin_model="auto")
        p = xt.Particles(_context=CTX, p0c=3e9, spin_x=1, anomalous_magnetic_moment=G)
        line.track(p)
        line.track(p, backtrack=True)
        ok = p.state[0] == -33 or (
            p.state[0] > 0 and np.allclose(spin(p), [1, 0, 0], atol=3e-15, rtol=0)
        )
        failures += not ok
        print(f"BEND: observed={spin(p)} expected=[1,0,0] or state=-33 state={p.state[0]} pass={ok}")
        kernels()
        print(f"FAILED_INVERSE_CASES {failures}")
        print("DEFECT REPRODUCED" if failures else "NOT REPRODUCED")


    if __name__ == "__main__":
        main()

    ```

## Provenance

| Role | Model | Effort | Date | Code examined |
|---|---|---|---|---|
| Found | `gpt-6-astra` | xhigh | 2026-11-06 | xtrack 0.118.2 (`ad5e808cdc5b`), installed packages, prebuilt kernels (compilation forbidden), CPU, double, serial |
| Confirmed | `gpt-6.1-sol` | xhigh | 2026-11-06 | xtrack 0.118.2 (`ad5e808cdc5b`), installed packages, prebuilt kernels (compilation forbidden), CPU, double, serial |
| Confirmed | `claude-opus-5-5` | high | 2026-11-07 | xtrack 0.118.2 (`ad5e808cdc5b`), installed packages, prebuilt kernels (compilation forbidden), CPU, double, serial |

