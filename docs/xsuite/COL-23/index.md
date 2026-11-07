---
title: "COL-23: Gaussian space-charge construction and reload discard placement"
hide:
  - toc
---

# COL-23 — Gaussian space-charge construction and reload discard placement

| | |
|---|---|
| Code | [Xsuite](../index.md) · Collective effects |
| Priority | P2 |
| Severity | Wrong physics |
| Evidence | Reproduced, independently rechecked |
| Last reproduced | 2026-11-07 on xfields release [v0.27.7](https://github.com/xsuite/xfields/tree/v0.27.7) (`3f06c87`) |
| Source | [SpaceChargeBiGaussian.to_dict, spacecharge.py:197–206](https://github.com/xsuite/xfields/blob/223b4fa134a1435be9b45ef21ae8903f96c5f685/xfields/beam_elements/spacecharge.py#L197-L206)<br>[The constructor, lines 208–257](https://github.com/xsuite/xfields/blob/223b4fa134a1435be9b45ef21ae8903f96c5f685/xfields/beam_elements/spacecharge.py#L208-L257)<br>[base_element.py:412–420](https://github.com/xsuite/xtrack/blob/ad5e808cdc5b9a881eac2f41e32a0b7d11cc7028/xtrack/base_element.py#L412-L420)<br>[BeamElement.from_dict, lines 872–879](https://github.com/xsuite/xtrack/blob/ad5e808cdc5b9a881eac2f41e32a0b7d11cc7028/xtrack/base_element.py#L872-L879)<br>[line element reconstruction, line.py:7935–7941](https://github.com/xsuite/xtrack/blob/ad5e808cdc5b9a881eac2f41e32a0b7d11cc7028/xtrack/line.py#L7935-L7941)<br>[HybridClass._static_from_dict, hybrid_class.py:325–332](https://github.com/xsuite/xobjects/blob/dfe34ae824bd8b202015f4350e655238b5bb9511/xobjects/hybrid_class.py#L325-L332)<br>[environment and line JSON round trips, environment.rst:1058–1083](https://github.com/xsuite/xsuite/blob/e0913e0a18028b2fce06991aebed6150876124b3/docs/environment.rst#L1058-L1083)<br>[placement definitions, apireference.rst:683–710](https://github.com/xsuite/xsuite/blob/e0913e0a18028b2fce06991aebed6150876124b3/docs/apireference.rst#L683-L710)<br>[frozen/quasi-frozen Gaussian modes, spacechargeauto.rst:4–9](https://github.com/xsuite/xsuite/blob/e0913e0a18028b2fce06991aebed6150876124b3/docs/spacechargeauto.rst#L4-L9) |

Constructing a Gaussian space-charge lens (`SpaceChargeBiGaussian`) with placement keywords, or reloading a placed lens from a dictionary or JSON, silently resets its placement. Stored shifts, rotations and rotation anchor are discarded by the constructor. For a 0.3000 m frozen round lens with 1.400 mm transverse rms size, 120.0 mm longitudinal rms size, population 1.200×10¹⁰ and shifts (4.000, −3.000) mm, a 2.100 GeV/c proton at the lens center receives normalized transverse momentum kicks (4.778×10⁻⁷, −3.584×10⁻⁷) instead of (0, 0). Gauss's law and explicitly assigned alignment agree; unaligned reloads preserve the kick.

Run `python COL-23.py` with [_prebuilt.py](../helpers/_prebuilt.py) and [_common.py](../helpers/_common.py) in the same directory.

??? example "Reproducer: COL-23.py"

    [Download COL-23.py](COL-23.py)

    ```python
    """COL-23: Gaussian space-charge construction and reload discard placement

    Constructing a Gaussian space-charge lens (`SpaceChargeBiGaussian`) with placement keywords, or reloading a placed lens from a dictionary or JSON, silently resets its placement. Stored shifts, rotations and rotation anchor are discarded by the constructor. For a 0.3000 m frozen round lens with 1.400 mm transverse rms size, 120.0 mm longitudinal rms size, population 1.200×10¹⁰ and shifts (4.000, −3.000) mm, a 2.100 GeV/c proton at the lens center receives normalized transverse momentum kicks (4.778×10⁻⁷, −3.584×10⁻⁷) instead of (0, 0). Gauss's law and explicitly assigned alignment agree; unaligned reloads preserve the kick.

    Track round and elliptical frozen lenses through direct construction, dictionary and JSON reconstruction, with explicitly assigned placement and unaligned controls. Compare their kicks with Gauss's law or an independent Poisson integral, and check all eight inherited placement fields. JSON round trips use in-memory streams.
    """

    import copy
    import io

    from _prebuilt import CTX, identity, kernels, np, xt

    # isort: split
    # _prebuilt sets numerical thread limits before Xfields imports NumPy.
    import xfields as xf
    from scipy.integrate import quad

    # Use the pinned model's constants to isolate placement from constant rounding.
    QE = 1.60217662e-19
    EPS = 8.854187817620e-12
    LENGTH, POP, SZ = 0.3, 1.2e10, 0.12
    records = []


    def make(sigmas, alignment=None):
        return xf.SpaceChargeBiGaussian(
            _context=CTX,
            length=LENGTH,
            sigma_x=sigmas[0],
            sigma_y=sigmas[1],
            longitudinal_profile=xf.LongitudinalProfileQGaussian(number_of_particles=POP, sigma_z=SZ),
            **(alignment or {}),
        )


    def oracle(points, sigmas, shift, roll):
        """Gauss law for equal sigmas; dimensionless Poisson integral otherwise."""
        probe = particle()
        b, g = float(probe.beta0[0]), float(probe.gamma0[0])
        # With mass0 in eV and unit reference species, the kick coefficient is
        # L*e*lambda/(mass0*beta0^2*gamma0^3), lambda=N/(sqrt(2*pi)*sigma_z).
        factor = LENGTH * QE * POP / (np.sqrt(2 * np.pi) * SZ * probe.mass0 * b * b * g**3)
        c, s = np.cos(roll), np.sin(roll)
        rotation = np.array([[c, -s], [s, c]])
        local = (points - shift) @ rotation
        field = np.empty_like(local)
        sx, sy = sigmas
        for k, r in enumerate(local):
            rr = r @ r
            if rr == 0:
                field[k] = 0
            elif sx == sy:
                field[k] = -np.expm1(-rr / (2 * sx * sx)) * r / (2 * np.pi * EPS * rr)
            else:
                # Scale t by sx^2 before integrating the 2D Gaussian Green function.
                a = np.array([1.0, (sy / sx) ** 2])
                z = r * r / (sx * sx)
                for i in range(2):

                    def integrand(v, variances, coordinates, axis):
                        den = variances + v
                        return np.exp(-0.5 * np.sum(coordinates / den)) / (
                            (variances[axis] + v) * np.sqrt(den.prod())
                        )

                    integral, err = quad(
                        integrand, 0, np.inf, args=(a, z, i), epsabs=2e-13, epsrel=2e-13
                    )
                    assert err < 2e-12
                    field[k, i] = r[i] * integral / (4 * np.pi * EPS * sx * sx)
        return factor * field @ rotation.T


    def kick(el, points):
        pp = particle(x=points[:, 0], y=points[:, 1])
        el.track(pp)
        assert np.all(pp.state == 1)
        return np.column_stack([pp.px, pp.py])


    def particle(**kwargs):
        return xt.Particles(_context=CTX, p0c=2.1e9, mass0=xt.PROTON_MASS_EV, q0=1, **kwargs)


    def finish(records):
        kernels()
        failures = sum(record["defect"] for record in records)
        print(f"COMPARISONS total={len(records)} pass={len(records) - failures} fail={failures}")
        print("DEFECT REPRODUCED" if failures else "NOT REPRODUCED")


    identity()
    for label, sigmas, shift, roll in [
        ("round-shift", (0.0014, 0.0014), np.array([0.004, -0.003]), 0.0),
        ("elliptic-roll", (0.0017, 0.0009), np.array([0.0, 0.0]), 0.27),
        ("elliptic-combined", (0.0017, 0.0009), np.array([0.002, -0.001]), 0.27),
        ("unaligned", (0.0017, 0.0009), np.array([0.0, 0.0]), 0.0),
    ]:
        points = np.array([shift, shift + [0.0008, -0.0005], shift + [-0.0012, 0.0003]])
        alignment = dict(shift_x=shift[0], shift_y=shift[1], rot_s_rad=roll)
        original = make(sigmas)
        for key, val in alignment.items():
            setattr(original, key, val)
        expected = oracle(points, sigmas, shift, roll)
        control = kick(original, points)
        assert np.allclose(control, expected, atol=3e-21, rtol=4e-14), (label, control, expected)
        print(
            "ANALYTIC_CONTROL",
            label,
            "expected",
            expected.tolist(),
            "observed",
            control.tolist(),
            "max_abs_error",
            np.max(np.abs(control - expected)),
        )
        serialized = original.to_dict()
        print("SERIALIZED_ALIGNMENT", label, {k: serialized.get(k, 0) for k in alignment})
        line = xt.Line(elements={"lens": original})
        line_json = io.StringIO()
        line.to_json(line_json)
        env_json = io.StringIO()
        line.env.to_json(env_json)
        paths = {
            "constructor": make(sigmas, alignment),
            "element-dict": xf.SpaceChargeBiGaussian.from_dict(copy.deepcopy(serialized)),
            "line-dict": xt.Line.from_dict(line.to_dict()).get("lens"),
            "line-json": xt.Line.from_json(io.StringIO(line_json.getvalue())).get("lens"),
            "environment-json": xt.Environment.from_json(io.StringIO(env_json.getvalue())).get("lens"),
            "copy": original.copy(),
        }
        for path, lens in paths.items():
            observed = kick(lens, points)
            align = [float(getattr(lens, name)) for name in alignment]
            error = float(np.max(np.abs(observed - expected)))
            defect = not np.allclose(observed, expected, atol=3e-21, rtol=4e-14)
            if path == "copy" or label == "unaligned":
                assert not defect, (label, path, error)
            print(
                "ROUNDTRIP",
                label,
                path,
                "alignment",
                align,
                "observed",
                observed.tolist(),
                "expected",
                expected.tolist(),
                "max_abs_error",
                error,
                "derivation=Gauss law or independent Poisson integral",
                "defect",
                defect,
            )
            records.append(
                dict(
                    case=label,
                    path=path,
                    alignment=align,
                    observed=observed.tolist(),
                    expected=expected.tolist(),
                    error=error,
                    defect=defect,
                )
            )

    # Scalar reconstruction scope: all eight inherited placement fields are real
    # serialized xofields, not arbitrary Python metadata.
    fields = dict(
        shift_x=0.002,
        shift_y=-0.001,
        shift_s=0.003,
        rot_s_rad=0.27,
        rot_x_rad=0.001,
        rot_y_rad=-0.002,
        rot_s_rad_no_frame=0.05,
        rot_shift_anchor=0.4,
    )
    el = make((0.0017, 0.0009))
    for name, value in fields.items():
        setattr(el, name, value)
    dct = el.to_dict()
    restored = xf.SpaceChargeBiGaussian.from_dict(copy.deepcopy(dct))
    for name, expected in fields.items():
        assert dct[name] == expected
    constructed = make((0.0017, 0.0009), fields)
    quasi = xf.SpaceChargeBiGaussian(
        _context=CTX,
        length=LENGTH,
        sigma_x=0.0017,
        sigma_y=0.0009,
        longitudinal_profile=xf.LongitudinalProfileQGaussian(number_of_particles=POP, sigma_z=SZ),
        update_on_track=True,
        **fields,
    )
    assert quasi._update_flag
    for label, lens in [
        ("element-dict-fields", restored),
        ("constructor-fields", constructed),
        ("quasi-frozen-fields", quasi),
    ]:
        expected_fields = np.array(list(fields.values()))
        observed_fields = np.array([getattr(lens, name) for name in fields])
        defect = not np.array_equal(observed_fields, expected_fields)
        print(
            "PLACEMENT_FIELDS",
            label,
            "observed",
            observed_fields.tolist(),
            "expected",
            expected_fields.tolist(),
            "derivation=serialized xofields",
            "defect",
            defect,
        )
        records.append(dict(case=label, defect=defect))
    # A field-map centroid is stored in the nested field map and survives reload;
    # it must be distinguished from the inherited placement shift.
    center = np.array([0.004, -0.003])
    points = np.array([center, center + [0.0008, -0.0005]])
    base = make((0.0014, 0.0014))
    base.mean_x, base.mean_y = center
    restored = xf.SpaceChargeBiGaussian.from_dict(copy.deepcopy(base.to_dict()))
    expected = oracle(points, (0.0014, 0.0014), center, 0.0)
    observed = kick(restored, points)
    assert np.allclose(observed, expected, atol=3e-21, rtol=4e-14)
    assert restored.mean_x == center[0] and restored.mean_y == center[1]
    print(
        "FIELDMAP_CENTROID_CONTROL",
        "observed",
        observed.tolist(),
        "expected",
        expected.tolist(),
        "max_abs_error",
        np.max(abs(observed - expected)),
    )
    records.append(
        dict(
            case="fieldmap-centroid",
            path="element-dict",
            observed=observed.tolist(),
            expected=expected.tolist(),
            error=float(np.max(abs(observed - expected))),
            defect=False,
        )
    )
    finish(records)

    ```

## Provenance

| Role | Model | Effort | Date | Code examined |
|---|---|---|---|---|
| Found | `gpt-6-astra` | xhigh | 2026-11-06 | xfields 0.27.5 (`223b4fa134a1`), installed packages, prebuilt kernels (compilation forbidden), CPU, double, serial |
| Confirmed | `gpt-6.1-sol` | xhigh | 2026-11-06 | xfields 0.27.5 (`223b4fa134a1`), installed packages, prebuilt kernels (compilation forbidden), CPU, double, serial |
| Confirmed | `claude-opus-5-5` | high | 2026-11-07 | xfields 0.27.5 (`223b4fa134a1`), installed packages, prebuilt kernels (compilation forbidden), CPU, double, serial |

