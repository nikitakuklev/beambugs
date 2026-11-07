---
title: "TRK-23: Optimized thin bends scale geometric arrival shifts by charge-to-mass ratio"
hide:
  - toc
---

# TRK-23 — Optimized thin bends scale geometric arrival shifts by charge-to-mass ratio

| | |
|---|---|
| Code | [Xsuite](../index.md) · Tracking |
| Priority | P2 |
| Severity | Wrong physics |
| Evidence | Reproduced, independently rechecked |
| Last reproduced | 2026-11-07 on xtrack main (`f09e0fb`, head build) |
| Source | [`SimpleThinBend_track_local_particle`, simplethinbend.h:13–51](https://github.com/xsuite/xtrack/blob/ad5e808cdc5b9a881eac2f41e32a0b7d11cc7028/xtrack/beam_elements/elements_src/simplethinbend.h#L13-L51)<br>[general magnet kick, track_magnet_kick.h:151–156](https://github.com/xsuite/xtrack/blob/ad5e808cdc5b9a881eac2f41e32a0b7d11cc7028/xtrack/beam_elements/elements_src/track_magnet_kick.h#L151-L156)<br>[SimpleThinBend contract, simple_thin_bend.py:11–23](https://github.com/xsuite/xtrack/blob/ad5e808cdc5b9a881eac2f41e32a0b7d11cc7028/xtrack/beam_elements/simple_thin_bend.py#L11-L23)<br>[`use_simple_bends`, line.py:5707–5725](https://github.com/xsuite/xtrack/blob/ad5e808cdc5b9a881eac2f41e32a0b7d11cc7028/xtrack/line.py#L5707-L5725)<br>[`optimize_for_tracking`, line.py:5295–5296](https://github.com/xsuite/xtrack/blob/ad5e808cdc5b9a881eac2f41e32a0b7d11cc7028/xtrack/line.py#L5295-L5296)<br>[ThinSliceBend conversion, slice_elements_thin.py:184–208](https://github.com/xsuite/xtrack/blob/ad5e808cdc5b9a881eac2f41e32a0b7d11cc7028/xtrack/beam_elements/slice_elements_thin.py#L184-L208)<br>[eligibility guard, line.py:7956–7968](https://github.com/xsuite/xtrack/blob/ad5e808cdc5b9a881eac2f41e32a0b7d11cc7028/xtrack/line.py#L7956-L7968)<br>[mixed-species coordinates, xtrack.tex:327–351](https://github.com/xsuite/xsuite/blob/e0913e0a18028b2fce06991aebed6150876124b3/docs/physics_manual/xtrack.tex#L327-L351)<br>[thin-bend Hamiltonian and map, lines 1332–1349](https://github.com/xsuite/xsuite/blob/e0913e0a18028b2fce06991aebed6150876124b3/docs/physics_manual/xtrack.tex#L1332-L1349)<br>[optimization guide, optimize_for_tracking.rst:4–8](https://github.com/xsuite/xsuite/blob/e0913e0a18028b2fce06991aebed6150876124b3/docs/optimize_for_tracking.rst#L4-L8) |

Optimizing eligible thin dipoles, including dipoles obtained by slicing bends, changes arrival times for particles with a different charge-to-mass ratio from the reference species. The optimized map multiplies the geometric arrival-coordinate shift Δζ by χ, the relative charge-to-mass ratio, although this path-length term of the curved reference frame is independent of χ. With a 1 GeV/c proton reference, a 10 mrad bend, 2 mm horizontal offset and 1% momentum offset, χ = 0.5 gives Δζ = −9.954 μm instead of −19.91 μm. Independent path-length and canonical-derivative checks establish the correct result. Unoptimized tracking and the χ = 1 control preserve the geometric term.

Run `python TRK-23.py` with [_prebuilt.py](../helpers/_prebuilt.py) and [_common.py](../helpers/_common.py) in the same directory.

??? example "Reproducer: TRK-23.py"

    [Download TRK-23.py](TRK-23.py)

    ```python
    """TRK-23: Optimized thin bends scale geometric arrival shifts by charge-to-mass ratio

    Optimizing eligible thin dipoles, including dipoles obtained by slicing bends, changes arrival times for particles with a different charge-to-mass ratio from the reference species. The optimized map multiplies the geometric arrival-coordinate shift Δζ by χ, the relative charge-to-mass ratio, although this path-length term of the curved reference frame is independent of χ. With a 1 GeV/c proton reference, a 10 mrad bend, 2 mm horizontal offset and 1% momentum offset, χ = 0.5 gives Δζ = −9.954 μm instead of −19.91 μm. Independent path-length and canonical-derivative checks establish the correct result. Unoptimized tracking and the χ = 1 control preserve the geometric term.

    Compute the geometric flight time from physical momentum and rest energy, then compare native and optimized thin bends at several charge states and originating lengths. Perturb physical energy to check the reciprocal canonical derivatives independently of Xtrack's energy conversion. Check ordinary Bend slicing, reference-species particles and zero reference angle using installed serial prebuilt kernels with compilation forbidden.
    """

    import math

    from _prebuilt import CTX, identity, kernels, np, xt

    P0 = 1e9
    M0 = xt.PROTON_MASS_EV
    BETA0 = P0 / math.hypot(P0, M0)
    matches = []


    def speed(delta):
        # P=rm*p0*(1+delta), m=rm*m0: the mass ratio cancels from beta=P*c/E.
        momentum = P0 * (1 + delta)
        return momentum / np.hypot(momentum, M0)


    def compare(label, observed, expected, tolerance, derivation, control=False):
        passed = bool(np.allclose(observed, expected, atol=tolerance, rtol=0))
        if control:
            assert passed, (label, observed, expected)
        matches.append(passed)
        print(f"{label}: observed={observed} expected={expected} "
              f"derivation={derivation} match={passed}")


    def element(path, length, angle, field):
        if path == "direct":
            return xt.SimpleThinBend(_context=CTX, knl=[field], hxl=angle, length=length)
        line = xt.Line(elements={"b": xt.Multipole(
            _context=CTX, knl=[field], hxl=angle, length=length, isthick=False)})
        if path == "simple":
            line.use_simple_bends()
        elif path == "optimize":
            line.build_tracker(_context=CTX)
            line.optimize_for_tracking(compile=False, verbose=False)
        return line.element_dict["b"]


    def thin_maps():
        angle, field, x, delta = -0.008, -0.0075, 0.003, 0.02
        chis = np.array([0.25, 0.5, 1.0, 2.0])
        for path in ("native", "simple", "optimize", "direct"):
            for length in (0.0, 1.2):
                el = element(path, length, angle, field)
                p = xt.Particles(_context=CTX, p0c=P0, mass_ratio=2,
                                 charge_ratio=2 * chis, x=x, delta=delta)
                el.track(p)
                # Extra path theta*x takes time theta*x/(beta*c), so
                # Delta zeta=-beta0*c*Delta t, with no charge factor.
                expected = -angle * x * BETA0 / speed(delta)
                expected_px = angle * (1 + delta) - chis * field
                if length:
                    expected_px -= chis * field * angle * x / length
                np.testing.assert_allclose(p.px, expected_px, atol=2e-17, rtol=0)
                np.testing.assert_allclose(p.rvv, speed(delta) / BETA0, atol=5e-16, rtol=0)
                for chi, zeta in zip(chis, np.asarray(p.zeta)):
                    compare(f"geometry {path} L={length} chi={chi}", float(zeta),
                            expected, 2e-19, "-theta*x*beta0/beta from extra path",
                            control=path == "native" or chi == 1)

        # In canonical order (x,px,y,py,zeta,pzeta), R26=-R51=theta*beta0/beta.
        # Invert pzeta=(E/rm-E0)/(beta0*p0*c) independently using physical energy.
        for path in ("native", "direct"):
            el = element(path, 0, angle, field)
            for step in (1e-5, 5e-6):
                energy = math.hypot(P0 * (1 + delta), M0)
                epm = energy + np.array([-step, step]) * BETA0 * P0
                delta_pm = np.sqrt(epm * epm - M0 * M0) / P0 - 1
                p = xt.Particles(_context=CTX, p0c=P0, mass_ratio=2, charge_ratio=1,
                                 x=[x, x, x - step, x + step],
                                 delta=[*delta_pm, delta, delta])
                el.track(p)
                r26 = float((p.px[1] - p.px[0]) / (2 * step))
                r51 = float((np.asarray(p.zeta)[3] - np.asarray(p.zeta)[2]) / (2 * step))
                expected = angle * BETA0 / speed(delta)
                compare(f"canonical {path} h={step} R26", r26, expected, 2e-12,
                        "theta*beta0/beta from physical-energy derivative", control=True)
                compare(f"canonical {path} h={step} R51", r51, -expected, 1e-14,
                        "-theta*beta0/beta from path derivative", control=path == "native")
                compare(f"canonical {path} h={step} R26+R51", r26 + r51, 0.0, 2e-12,
                        "reciprocal derivatives of the thin-bend Hamiltonian",
                        control=path == "native")

        for path in ("native", "direct"):
            el = element(path, 0.8, 0.01, 0.01)
            p = xt.Particles(_context=CTX, p0c=P0, mass_ratio=2, charge_ratio=1,
                             x=0.002, delta=0.01)
            el.track(p)
            compare(f"10mrad {path} chi=0.5", float(np.asarray(p.zeta)[0]),
                    -0.01 * 0.002 * BETA0 / speed(0.01), 2e-19,
                    "-theta*x*beta0/beta", control=path == "native")
            p = xt.Particles(_context=CTX, p0c=P0, mass_ratio=2, charge_ratio=1, x=x, delta=0)
            element(path, 1.2, 0, field).track(p)
            compare(f"zero-angle {path}", float(np.asarray(p.zeta)[0]), 0.0, 0.0,
                    "theta=0 gives zero geometric path", control=True)


    def sliced_bends():
        for chi in (0.5, 1.0, 2.0):
            line = xt.Line(elements={"b": xt.Bend(
                length=1.2, angle=0.008, k0=0.008 / 1.2,
                edge_entry_active=False, edge_exit_active=False)})
            line.slice_thick_elements([xt.Strategy(slicing=xt.Teapot(2))], with_progress=False)
            line.build_tracker(_context=CTX)
            p = xt.Particles(_context=CTX, p0c=P0, mass_ratio=2, charge_ratio=2 * chi,
                             x=0.003, delta=0.02)
            reference = p.copy()
            line.track(reference)
            line.optimize_for_tracking(compile=False, verbose=False)
            # Optimization disables automatic prebuilt selection. Keep compilation
            # forbidden and explicitly select the installed tracking library.
            line.use_prebuilt_kernels = True
            assert sum(isinstance(line.element_dict[n], xt.SimpleThinBend)
                       for n in line.element_names) == 2
            line.track(p)
            np.testing.assert_allclose([p.x[0], p.px[0], p.y[0], p.py[0]],
                                       [reference.x[0], reference.px[0],
                                        reference.y[0], reference.py[0]], atol=3e-17, rtol=0)
            compare(f"Teapot(2) chi={chi}", float(np.asarray(p.zeta)[0]),
                    float(np.asarray(reference.zeta)[0]), 3e-17,
                    "unchanged drifts and native weighted reference-bend kicks",
                    control=chi == 1)


    if __name__ == "__main__":
        identity()
        thin_maps()
        sliced_bends()
        kernels()
        print(f"COMPARISONS total={len(matches)} pass={sum(matches)} fail={len(matches) - sum(matches)}")
        print("NOT REPRODUCED" if all(matches) else "DEFECT REPRODUCED")

    ```

## Provenance

| Role | Model | Effort | Date | Code examined |
|---|---|---|---|---|
| Found | `gpt-6-astra` | xhigh | 2026-11-06 | xtrack 0.118.2 (`ad5e808cdc5b`), installed packages, prebuilt kernels (compilation forbidden), CPU, double, serial |
| Confirmed | `gpt-6.1-sol` | xhigh | 2026-11-06 | xtrack 0.118.2 (`ad5e808cdc5b`), installed packages, prebuilt kernels (compilation forbidden), CPU, double, serial |
| Confirmed | `claude-opus-5-5` | high | 2026-11-07 | xtrack 0.118.2 (`ad5e808cdc5b`), installed packages, prebuilt kernels (compilation forbidden), CPU, double, serial |

