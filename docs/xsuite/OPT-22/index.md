---
title: "OPT-22: Native TPSA particle extraction restores the launch reference"
hide:
  - toc
---

# OPT-22 — Native TPSA particle extraction restores the launch reference

| | |
|---|---|
| Code | [Xsuite](../index.md) · Optics and matching |
| Priority | P4 |
| Failure | Silent wrong result |
| Evidence | Reproduced, independently rechecked |
| Last reproduced | 2026-11-07 on xtrack release [v0.118.6](https://github.com/xsuite/xtrack/tree/v0.118.6) (`f09e0fb`) |
| Tags | <span class="bb-tag">experimental</span> [?](../../methodology.md#tags) |
| Pull request | [[gh:https://github.com/xsuite/xtrack/pull/879#issuecomment-6659447765]] |
| Discussed in | [[gh:https://github.com/xsuite/xtrack/pull/879#issuecomment-6659483842]] |
| Source | [ParticlesTpsa.to_particles and current reference getters, particles.py:189–205](https://github.com/xsuite/xtrack/blob/ad5e808cdc5b9a881eac2f41e32a0b7d11cc7028/xtrack/tpsa/particles.py#L189-L205)<br>[element_types.py:127–148](https://github.com/xsuite/xtrack/blob/ad5e808cdc5b9a881eac2f41e32a0b7d11cc7028/xtrack/prebuilt_kernel_definitions/element_types.py#L127-L148)<br>[public API:1090–1101](https://github.com/xsuite/xsuite/blob/e0913e0a18028b2fce06991aebed6150876124b3/docs/apireference.rst#L1090-L1101) |

Converting a tracked native Taylor map to an ordinary particle after a reference-momentum change combines exit coordinates with the launch reference. The conversion must retain the current reference to preserve physical energy and momentum. A proton with physical momentum 1.080 GeV/c, rebased from 900.0 MeV/c to 1.440 GeV/c, is extracted with 675.0 MeV/c instead. An independent relativistic energy calculation and ordinary particle tracking agree with the native map before extraction. A real cavity energy gain does not remove the error; unchanged-reference extraction passes.

Run `python OPT-22.py` with [_prebuilt.py](../helpers/_prebuilt.py) and [_common.py](../helpers/_common.py) in the same directory.

??? example "Reproducer: OPT-22.py"

    [Download OPT-22.py](OPT-22.py)

    ```python
    """OPT-22: Native TPSA particle extraction restores the launch reference

    Converting a tracked native Taylor map to an ordinary particle after a reference-momentum change combines exit coordinates with the launch reference. The conversion must retain the current reference to preserve physical energy and momentum. A proton with physical momentum 1.080 GeV/c, rebased from 900.0 MeV/c to 1.440 GeV/c, is extracted with 675.0 MeV/c instead. An independent relativistic energy calculation and ordinary particle tracking agree with the native map before extraction. A real cavity energy gain does not remove the error; unchanged-reference extraction passes.

    The script tracks second-order native Taylor maps and ordinary particles through both reference-change elements and a zero-frequency cavity with zero or 230 keV gain. It compares physical momenta with an independent relativistic energy calculation, and checks both full-line and direct element tracking before extraction. An exact downstream drift exposes the flight-time error; unchanged-reference extraction is the control.
    """

    from typing import Any

    from _prebuilt import CTX, identity, kernels, np, xt


    def close(a, b, *, atol=3e-10, rtol=3e-10):
        np.testing.assert_allclose(a, b, atol=atol, rtol=rtol)


    def make_line(elements, p):
        line = xt.Line(elements=elements)
        line.particle_ref = p.copy()
        line.build_tracker(_context=CTX)
        return line


    identity()
    failed = 0
    P0 = 900e6
    MASS = xt.PROTON_MASS_EV
    P = 1.08e9
    GAIN = 230e3

    for kind in ("absolute", "increment"):
        for ratio in (1.0, 1.6, 0.75):
            for gain in (0.0, GAIN):
                if kind == "absolute":
                    element = xt.ReferenceEnergyChange(p0c=P0 * ratio)
                else:
                    element = xt.ReferenceEnergyIncrease(Delta_p0c=P0 * (ratio - 1))
                args: dict[str, Any] = dict(
                    p0c=P0, mass0=MASS, delta=P / P0 - 1, x=0.002, px=0.0013, py=-0.0009, zeta=0.01
                )
                scalar = xt.Particles(_context=CTX, **args)
                taylor = xt.ParticlesTpsa(order=2, **args)
                line = make_line(
                    {
                        "rebase": element,
                        "cavity": xt.Cavity(voltage=gain, frequency=0, phase=np.pi / 2),
                    },
                    scalar,
                )
                line.track(scalar)
                line.track(taylor)
                direct = xt.ParticlesTpsa(order=2, **args)
                for entry in line.elements:
                    entry.track(direct)
                close(direct.const_part, taylor.const_part, atol=3e-14, rtol=3e-13)
                close(direct.p0c, taylor.p0c, atol=1e-7, rtol=3e-15)
                extracted = taylor.to_particles()
                expected_energy = np.hypot(P, MASS) + gain
                expected_p = np.sqrt((expected_energy - MASS) * (expected_energy + MASS))
                native = np.array(
                    [
                        taylor.p0c * (1 + taylor.const_part[5]),
                        taylor.p0c * taylor.const_part[1],
                        taylor.p0c * taylor.const_part[3],
                    ]
                )
                expected = np.array([expected_p, P0 * args["px"], P0 * args["py"]])
                close(native, expected, atol=1e-6, rtol=3e-15)
                close(scalar.energy[0], expected_energy, atol=1e-6, rtol=3e-15)
                close(taylor.p0c, P0 * ratio, atol=1e-7, rtol=3e-15)
                close(
                    taylor.const_part,
                    [getattr(scalar, k)[0] for k in ("x", "px", "y", "py", "zeta", "delta")],
                    atol=3e-14,
                    rtol=3e-13,
                )
                restored = np.array(
                    [
                        extracted.p0c[0] * (1 + extracted.delta[0]),
                        extracted.p0c[0] * extracted.px[0],
                        extracted.p0c[0] * extracted.py[0],
                    ]
                )
                ok = np.allclose(restored, expected, atol=1e-6, rtol=3e-15)
                if not ok:
                    close(restored * ratio, expected, atol=1e-6, rtol=3e-15)
                direct_out = direct.to_particles()
                close(direct_out.p0c, extracted.p0c, atol=1e-7, rtol=3e-15)
                close(direct_out.delta, extracted.delta, atol=3e-14, rtol=3e-13)
                # Continued tracking probes the meaning of extraction beyond metadata.
                drift = xt.DriftExact(length=0.4)
                xs = scalar.x[0]
                xe = extracted.x[0]
                drift.track(scalar)
                drift.track(extracted)
                # Slopes are invariant to a uniform momentum scaling; flight time is not.
                close(scalar.x[0] - xs, extracted.x[0] - xe, atol=1e-16, rtol=3e-13)
                extracted_zeta = np.asarray(extracted.zeta)[0]
                scalar_zeta = np.asarray(scalar.zeta)[0]
                print(
                    f"{kind} a={ratio:g} gain={gain:g}: observed_p_Px_Py={restored} expected_p_Px_Py={expected} observed_zeta={extracted_zeta:.15g} expected_zeta={scalar_zeta:.15g} derivation=Eout=hypot(Pin,mass0)+gain,Pout=sqrt(Eout**2-mass0**2),PxPy=constant physical_match={ok}"
                )
                if ratio == 1:
                    close(restored, expected, atol=1e-6, rtol=3e-15)
                    close(extracted_zeta, scalar_zeta, atol=2e-15, rtol=3e-13)
                else:
                    failed += not ok
    kernels()
    print("DEFECT REPRODUCED" if failed else "NOT REPRODUCED")

    ```

## Provenance

| Role | Model | Effort | Date | Code examined |
|---|---|---|---|---|
| Found | `gpt-6-astra` | xhigh | 2026-11-06 | xtrack 0.118.2 (`ad5e808cdc5b`), installed packages, prebuilt kernels (compilation forbidden), CPU, double, serial |
| Confirmed | `gpt-6.1-sol` | xhigh | 2026-11-06 | xtrack 0.118.2 (`ad5e808cdc5b`), installed packages, prebuilt kernels (compilation forbidden), CPU, double, serial |

