---
title: "TRK-24: Thin-element simplification discards quadrupole and dipole spin transport"
hide:
  - toc
---

# TRK-24 — Thin-element simplification discards quadrupole and dipole spin transport

| | |
|---|---|
| Code | [Xsuite](../index.md) · Tracking |
| Priority | P2 |
| Failure | Wrong physics |
| Evidence | Reproduced, independently rechecked |
| Last reproduced | 2026-11-07 on xtrack release [v0.118.6](https://github.com/xsuite/xtrack/tree/v0.118.6) (`f09e0fb`) |
| Source | [optimize_for_tracking, line.py:5219–5240](https://github.com/xsuite/xtrack/blob/ad5e808cdc5b9a881eac2f41e32a0b7d11cc7028/xtrack/line.py#L5219-L5240)<br>[invokes both simplifiers, lines 5295–5299](https://github.com/xsuite/xtrack/blob/ad5e808cdc5b9a881eac2f41e32a0b7d11cc7028/xtrack/line.py#L5295-L5299)<br>[use_simple_quadrupoles and use_simple_bends, lines 5688–5725](https://github.com/xsuite/xtrack/blob/ad5e808cdc5b9a881eac2f41e32a0b7d11cc7028/xtrack/line.py#L5688-L5725)<br>[their predicates, lines 7943–7968](https://github.com/xsuite/xtrack/blob/ad5e808cdc5b9a881eac2f41e32a0b7d11cc7028/xtrack/line.py#L7943-L7968)<br>[SimpleThinQuadrupole_track_local_particle, simplethinquadrupole.h:13–33](https://github.com/xsuite/xtrack/blob/ad5e808cdc5b9a881eac2f41e32a0b7d11cc7028/xtrack/beam_elements/elements_src/simplethinquadrupole.h#L13-L33)<br>[SimpleThinBend_track_local_particle, simplethinbend.h:13–53](https://github.com/xsuite/xtrack/blob/ad5e808cdc5b9a881eac2f41e32a0b7d11cc7028/xtrack/beam_elements/elements_src/simplethinbend.h#L13-L53)<br>[track_magnet.h:104–162](https://github.com/xsuite/xtrack/blob/ad5e808cdc5b9a881eac2f41e32a0b7d11cc7028/xtrack/beam_elements/elements_src/track_magnet.h#L104-L162)<br>[quadrupole slice conversion, slice_elements_thin.py:26–58](https://github.com/xsuite/xtrack/blob/ad5e808cdc5b9a881eac2f41e32a0b7d11cc7028/xtrack/beam_elements/slice_elements_thin.py#L26-L58)<br>[bend conversion, lines 176–208](https://github.com/xsuite/xtrack/blob/ad5e808cdc5b9a881eac2f41e32a0b7d11cc7028/xtrack/beam_elements/slice_elements_thin.py#L176-L208)<br>[The compile flag, line.py:5100–5105](https://github.com/xsuite/xtrack/blob/ad5e808cdc5b9a881eac2f41e32a0b7d11cc7028/xtrack/line.py#L5100-L5105)<br>[spin tracked with the six orbital coordinates, spin_polarization.rst:4–6](https://github.com/xsuite/xsuite/blob/e0913e0a18028b2fce06991aebed6150876124b3/docs/spin_polarization.rst#L4-L6)<br>[Thomas–BMT rotation, xtrack.tex:6649–6679](https://github.com/xsuite/xsuite/blob/e0913e0a18028b2fce06991aebed6150876124b3/docs/physics_manual/xtrack.tex#L6649-L6679)<br>[tracking optimization, optimize_for_tracking.rst:4–8](https://github.com/xsuite/xsuite/blob/e0913e0a18028b2fce06991aebed6150876124b3/docs/optimize_for_tracking.rst#L4-L8) |

Optimizing normal thin-sliced quadrupoles or bends with radiation off replaces their field-bearing kicks by simple elements that do not transport spin. The orbit is preserved while spin precession disappears, including when spin tracking is enabled after optimization. For a 0.7500 m quadrupole of strength 0.8000 m⁻² with four uniform thin slices, a horizontal unit initial spin on an off-axis 2.100 GeV/c proton exits with longitudinal component zero instead of 0.007231. An independent planar Thomas–BMT rotation sum gives the preserved value. Calling use_simple_quadrupoles or use_simple_bends directly also loses spin; rolls or shifts that prevent replacement preserve it.

Run `python TRK-24.py` with [_prebuilt.py](../helpers/_prebuilt.py) and [_common.py](../helpers/_common.py) in the same directory.

??? example "Reproducer: TRK-24.py"

    [Download TRK-24.py](TRK-24.py)

    ```python
    """TRK-24: Thin-element simplification discards quadrupole and dipole spin transport

    Optimizing normal thin-sliced quadrupoles or bends with radiation off replaces their field-bearing kicks by simple elements that do not transport spin. The orbit is preserved while spin precession disappears, including when spin tracking is enabled after optimization. For a 0.7500 m quadrupole of strength 0.8000 m⁻² with four uniform thin slices, a horizontal unit initial spin on an off-axis 2.100 GeV/c proton exits with longitudinal component zero instead of 0.007231. An independent planar Thomas–BMT rotation sum gives the preserved value. Calling use_simple_quadrupoles or use_simple_bends directly also loses spin; rolls or shifts that prevent replacement preserve it.

    Compare spin before and after optimization with an independent planar Thomas–BMT kick sum and a separately calculated drift/kick orbit. Exercise late spin configuration and both public simplification methods, with rolled, shifted, radiating, thick and zero-field-length controls.
    """

    from _prebuilt import CTX, identity, kernels, np, xt

    G = 1.79284734463
    L, K1, X, PX, N = 0.75, 0.8, 0.0023, 0.00017, 4
    records = []


    def launch(**kw):
        options = dict(x=X, px=PX, spin_x=1.0, spin_y=0.0, spin_z=0.0, anomalous_magnetic_moment=G)
        options.update(kw)
        return particle(**options)


    def values(line, **kw):
        p = launch(**kw)
        line.track(p)
        assert np.all(p.state == 1)
        return np.array([p.spin_x[0], p.spin_y[0], p.spin_z[0]]), np.array(
            [getattr(p, key)[0] for key in ("x", "px", "y", "py", "zeta", "delta")]
        )


    def quad_oracle(n=N):
        # Uniform slicing has n+1 equal drifts and n equal field kicks.
        x, px = X, PX
        total = 0.0
        for _ in range(n):
            x += L / (n + 1) * px
            total += K1 * L / n * x
            px -= K1 * L / n * x
        x += L / (n + 1) * px
        gamma = np.sqrt(1 + (2.1e9 / xt.PROTON_MASS_EV) ** 2)
        angle = (1 + G * gamma) * total
        return np.array([np.cos(angle), 0.0, np.sin(angle)]), np.array([x, px]), angle


    def report(label, before, after, expected=None):
        bs, bo = before
        ss, oo = after
        err = float(np.max(np.abs(ss - bs)))
        orbit_error = float(np.max(np.abs(oo - bo)))
        assert orbit_error < 8e-17, (label, orbit_error)
        if expected is not None:
            assert np.max(np.abs(bs - expected)) < 3e-15, (label, bs, expected)
        defect = not np.allclose(ss, bs, rtol=0, atol=3e-15)
        print(
            "SPIN",
            label,
            "before",
            bs,
            "observed",
            ss,
            "expected",
            bs if expected is None else expected,
            "spin_error",
            err,
            "orbit_error",
            orbit_error,
            "derivation=planar BMT or unchanged configured map",
            "defect",
            defect,
        )
        records.append(
            dict(
                case=label,
                before=bs.tolist(),
                after=ss.tolist(),
                expected=None if expected is None else expected.tolist(),
                spin_error=err,
                orbit_error=orbit_error,
                defect=defect,
            )
        )
        return defect


    def make_quad(mode="thin", roll=0.0, shift=0.0, radiation=False):
        q = xt.Quadrupole(
            length=L,
            k1=K1,
            rot_s_rad=roll,
            shift_x=shift,
            edge_entry_active=False,
            edge_exit_active=False,
        )
        line = xt.Line(elements={"q": q})
        if mode != "unsliced":
            line.slice_thick_elements(
                slicing_strategies=[xt.Strategy(slicing=xt.Uniform(N, mode=mode))]
            )
        line.build_tracker(_context=CTX)
        line.configure_spin("auto")
        if radiation:
            line.configure_radiation(model="mean")
        return line


    def particle(**kwargs):
        return xt.Particles(_context=CTX, p0c=2.1e9, mass0=xt.PROTON_MASS_EV, q0=1, **kwargs)


    def finish(records):
        kernels()
        failures = sum(record["defect"] for record in records)
        print(f"COMPARISONS total={len(records)} pass={len(records) - failures} fail={failures}")
        print("DEFECT REPRODUCED" if failures else "NOT REPRODUCED")


    identity()
    expected, orbit_expected, angle = quad_oracle()
    print("ANALYTIC", "spin", expected, "orbit_x_px", orbit_expected, "angle", angle)
    for label, mode, roll, shift, radiation in [
        ("normal-thin", "thin", 0.0, 0.0, False),
        ("rolled-thin", "thin", 0.2, 0.0, False),
        ("shifted-thin", "thin", 0.0, 0.0003, False),
        ("mean-radiation-thin", "thin", 0.0, 0.0, True),
        ("normal-thick-slices", "thick", 0.0, 0.0, False),
        ("normal-unsliced", "unsliced", 0.0, 0.0, False),
    ]:
        line = make_quad(mode, roll, shift, radiation)
        before = values(line)
        if label == "normal-thin":
            assert np.max(np.abs(before[1][:2] - orbit_expected)) < 3e-18
        line.optimize_for_tracking(verbose=False)
        print(
            "REPRESENTATION",
            label,
            [type(line.get(n)).__name__ for n in line.element_names],
            "spin_model",
            line._spin_model,
            "NO_SYNRAD",
            line.config.get("XTRACK_MULTIPOLE_NO_SYNRAD", False),
        )
        defect = report(label, before, values(line), expected if label == "normal-thin" else None)
        if label != "normal-thin":
            assert not defect

    # The same wrong lowering also happens when spin is enabled after optimization.
    line = make_quad()
    before = values(line)
    line.configure_spin(None)
    line.optimize_for_tracking(verbose=False)
    line.configure_spin("auto")
    report("spin-enabled-after-optimization", before, values(line), expected)

    # Finite field length matters: zero-length multipoles already have no native
    # BMT field step. Direct public simplification reaches the same two kernels.
    for kind, strength in [("quadrupole", 0.28), ("dipole", 0.002)]:
        for length in (0.4, 0.0):
            normal = [0.0, strength] if kind == "quadrupole" else [strength]
            el = xt.Multipole(knl=normal, length=length)
            line = xt.Line(elements={"m": el})
            line.build_tracker(_context=CTX)
            line.configure_spin("auto")
            before = values(line, px=0.0)
            gamma = np.sqrt(1 + (2.1e9 / xt.PROTON_MASS_EV) ** 2)
            angle = (1 + G * gamma) * strength * (X if kind == "quadrupole" else 1.0) if length else 0.0
            expected_direct = np.array([np.cos(angle), 0.0, np.sin(angle)])
            line.discard_tracker()
            if kind == "quadrupole":
                line.use_simple_quadrupoles()
            else:
                line.use_simple_bends()
            line.build_tracker(_context=CTX)
            print("DIRECT_REPRESENTATION", kind, length, type(line.get("m")).__name__)
            defect = report(
                f"direct-{kind}-length-{length}", before, values(line, px=0.0), expected_direct
            )
            if length == 0.0:
                assert not defect

    # Ordinary thin Bend slices also lower to the spin-free bend kernel. At the
    # reference orbit the field spin angle relative to the curved frame is G*gamma*h*L.
    h, bend_length = 0.01, 0.5
    line = xt.Line(
        elements={
            "b": xt.Bend(
                length=bend_length,
                k0=h,
                angle=h * bend_length,
                edge_entry_active=False,
                edge_exit_active=False,
            )
        }
    )
    line.slice_thick_elements(slicing_strategies=[xt.Strategy(slicing=xt.Uniform(4))])
    line.build_tracker(_context=CTX)
    line.configure_spin("auto")
    before = values(line, x=0.0, px=0.0)
    gamma = np.sqrt(1 + (2.1e9 / xt.PROTON_MASS_EV) ** 2)
    angle = G * gamma * h * bend_length
    expected_bend = np.array([np.cos(angle), 0.0, np.sin(angle)])
    line.optimize_for_tracking(verbose=False)
    print("BEND_REPRESENTATION", [type(line.get(n)).__name__ for n in line.element_names])
    report("thin-bend", before, values(line, x=0.0, px=0.0), expected_bend)
    finish(records)

    ```

## Provenance

| Role | Model | Effort | Date | Code examined |
|---|---|---|---|---|
| Found | <span class="bb-nowrap">`gpt-6-astra`</span> | xhigh | <span class="bb-nowrap">2026-11-06</span> | xtrack 0.118.2 (`ad5e808cdc5b`), installed packages, prebuilt kernels (compilation forbidden), CPU, double, serial |
| Confirmed | <span class="bb-nowrap">`gpt-6.1-sol`</span> | xhigh | <span class="bb-nowrap">2026-11-06</span> | xtrack 0.118.2 (`ad5e808cdc5b`), installed packages, prebuilt kernels (compilation forbidden), CPU, double, serial |
| Confirmed | <span class="bb-nowrap">`claude-opus-5-5`</span> | high | <span class="bb-nowrap">2026-11-07</span> | xtrack 0.118.2 (`ad5e808cdc5b`), installed packages, prebuilt kernels (compilation forbidden), CPU, double, serial |

