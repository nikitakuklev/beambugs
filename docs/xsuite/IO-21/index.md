---
title: "IO-21: Exported magnet body slices repeat the parent's integrated multipole strengths"
hide:
  - toc
---

# IO-21 — Exported magnet body slices repeat the parent's integrated multipole strengths

| | |
|---|---|
| Code | [Xsuite](../index.md) · Lattices, particles and I/O |
| Priority | P2 |
| Failure | Wrong physics |
| Evidence | Reproduced, independently rechecked |
| Last reproduced | 2026-11-07 on xtrack release [v0.118.6](https://github.com/xsuite/xtrack/tree/v0.118.6) (`f09e0fb`) |
| Source | [`quadrupole_to_mad_str`, mad_writer.py:559–572](https://github.com/xsuite/xtrack/blob/ad5e808cdc5b9a881eac2f41e32a0b7d11cc7028/xtrack/mad_writer.py#L559-L572)<br>[array serializer, lines 96–112](https://github.com/xsuite/xtrack/blob/ad5e808cdc5b9a881eac2f41e32a0b7d11cc7028/xtrack/mad_writer.py#L96-L112)<br>[`sextupole_to_mad_str`, lines 501–514](https://github.com/xsuite/xtrack/blob/ad5e808cdc5b9a881eac2f41e32a0b7d11cc7028/xtrack/mad_writer.py#L501-L514)<br>[`octupole_to_mad_str`, lines 530–543](https://github.com/xsuite/xtrack/blob/ad5e808cdc5b9a881eac2f41e32a0b7d11cc7028/xtrack/mad_writer.py#L530-L543)<br>[`bend_to_mad_str`, lines 448–481](https://github.com/xsuite/xtrack/blob/ad5e808cdc5b9a881eac2f41e32a0b7d11cc7028/xtrack/mad_writer.py#L448-L481)<br>[thick `multipole_to_mad_str` branches, lines 312–358](https://github.com/xsuite/xtrack/blob/ad5e808cdc5b9a881eac2f41e32a0b7d11cc7028/xtrack/mad_writer.py#L312-L358)<br>[slice dispatch, lines 670–680](https://github.com/xsuite/xtrack/blob/ad5e808cdc5b9a881eac2f41e32a0b7d11cc7028/xtrack/mad_writer.py#L670-L680)<br>[`Line.to_madx_sequence`, line.py:877–894](https://github.com/xsuite/xtrack/blob/ad5e808cdc5b9a881eac2f41e32a0b7d11cc7028/xtrack/line.py#L877-L894)<br>[`Line.to_madng`, lines 897–918](https://github.com/xsuite/xtrack/blob/ad5e808cdc5b9a881eac2f41e32a0b7d11cc7028/xtrack/line.py#L897-L918)<br>[`line_to_madng`, madng_interface.py:1038–1056](https://github.com/xsuite/xtrack/blob/ad5e808cdc5b9a881eac2f41e32a0b7d11cc7028/xtrack/madng_interface.py#L1038-L1056)<br>[`build_madng_model`, lines 98–124](https://github.com/xsuite/xtrack/blob/ad5e808cdc5b9a881eac2f41e32a0b7d11cc7028/xtrack/madng_interface.py#L98-L124)<br>[lines 251–253](https://github.com/xsuite/xtrack/blob/ad5e808cdc5b9a881eac2f41e32a0b7d11cc7028/xtrack/madng_interface.py#L251-L253)<br>[additional-multipole docstring, _common.py:405–412](https://github.com/xsuite/xtrack/blob/ad5e808cdc5b9a881eac2f41e32a0b7d11cc7028/xtrack/beam_elements/_common.py#L405-L412)<br>[Thick slicing, slicing.py:442–460](https://github.com/xsuite/xtrack/blob/ad5e808cdc5b9a881eac2f41e32a0b7d11cc7028/xtrack/slicing.py#L442-L460)<br>[quadrupole slice kernel, thick_slice_quadrupole.h:23–35](https://github.com/xsuite/xtrack/blob/ad5e808cdc5b9a881eac2f41e32a0b7d11cc7028/xtrack/beam_elements/elements_src/thick_slice_quadrupole.h#L23-L35)<br>[track_magnet.h:338–367](https://github.com/xsuite/xtrack/blob/ad5e808cdc5b9a881eac2f41e32a0b7d11cc7028/xtrack/beam_elements/elements_src/track_magnet.h#L338-L367)<br>[thick-element KNL/KSL, ptc-general.tex:502–516](https://github.com/MethodicalAcceleratorDesign/MAD-X/blob/3865cb633e66fb4cf53d2f37b00c919dbd6ab495/doc/latexuguide/ptc-general.tex#L502-L516)<br>[PTC translation, madx_ptc_module.f90:1951–1983](https://github.com/MethodicalAcceleratorDesign/MAD-X/blob/3865cb633e66fb4cf53d2f37b00c919dbd6ab495/src/madx_ptc_module.f90#L1951-L1983) |

Exporting native magnet body slices to MAD-X or MAD-NG repeats the parent's integrated normal and skew multipole strengths instead of scaling them by each slice's weight. Four equal slices of a 1.200 m quadrupole turn integrated normal/skew sextupole strengths of 0.02000/0.01000 m⁻² into totals of 0.08000/0.04000 m⁻². In PTC and MAD-NG, a 3.000 GeV/c proton's exit vertical normalized momentum pᵧ is then 5.288×10⁻⁴ instead of 5.283×10⁻⁴, matching independent integration with fourfold field errors. Native tracking, unsliced export and correctly weighted text agree with the intended field.

Run `python IO-21.py` with [_madx_io.py](../helpers/_madx_io.py), [_prebuilt.py](../helpers/_prebuilt.py) and [_common.py](../helpers/_common.py) in the same directory.

??? example "Reproducer: IO-21.py"

    [Download IO-21.py](IO-21.py)

    ```python
    """IO-21: Exported magnet body slices repeat the parent's integrated multipole strengths

    Exporting native magnet body slices to MAD-X or MAD-NG repeats the parent's integrated normal and skew multipole strengths instead of scaling them by each slice's weight. Four equal slices of a 1.200 m quadrupole turn integrated normal/skew sextupole strengths of 0.02000/0.01000 m⁻² into totals of 0.08000/0.04000 m⁻². In PTC and MAD-NG, a 3.000 GeV/c proton's exit vertical normalized momentum pᵧ is then 5.288×10⁻⁴ instead of 5.283×10⁻⁴, matching independent integration with fourfold field errors. Native tracking, unsliced export and correctly weighted text agree with the intended field.

    Export thick slices of six magnet families with nonzero normal and skew KNL/KSL of orders 0-3 in both writer dialects and compare every exported coefficient with its slice weight times the parent's integrated value; a single weight-1 slice is the control. Then track one ray through a sliced 1.2 m quadrupole with integrated sextupole errors in PTC (body slices exported to MAD-X) and in MAD-NG (the whole sliced line through Line.to_madng), against direct ODE integration of the intended field and of a fourfold field. Controls are native Xtrack tracking, unsliced export, independently handwritten weighted MAD-X text and sliced export without higher-order terms. The script uses cpymad, pymadng and installed serial prebuilt Xtrack kernels with compilation forbidden; scratch files stay in automatically removed directories beside the script.
    """

    import re
    from pathlib import Path
    from tempfile import TemporaryDirectory

    from _madx_io import consume, ptc_track
    from _prebuilt import CTX, identity, kernels, np, xt

    # Import after the helpers cap numerical threads and forbid compilation.
    # isort: split
    from importlib.metadata import version

    from scipy.integrate import solve_ivp
    from xtrack import mad_writer as writer

    LENGTH = 1.2
    K1 = 0.3
    PC = 3e9
    INITIAL = [0.003, 1e-4, 0.002, -2e-4]
    # Orders 0-3: dipole, quadrupole, sextupole and octupole integrated strengths.
    KNL = [5e-5, 1e-3, 0.02, 0.6]
    KSL = [3e-5, 2e-3, 0.01, 0.3]
    SEXT = (0.02, 0.01)
    FAMILIES = {
        "Quadrupole": lambda: xt.Quadrupole(length=LENGTH, k1=K1, knl=KNL, ksl=KSL),
        "Sextupole": lambda: xt.Sextupole(length=LENGTH, k2=0.1, knl=KNL, ksl=KSL),
        "Octupole": lambda: xt.Octupole(length=LENGTH, k3=1.0, knl=KNL, ksl=KSL),
        "Bend": lambda: xt.Bend(length=LENGTH, angle=0.02, knl=KNL, ksl=KSL),
        "RBend": lambda: xt.RBend(length_straight=LENGTH, angle=0.02, knl=KNL, ksl=KSL),
        "Multipole": lambda: xt.Multipole(length=LENGTH, isthick=True, knl=KNL, ksl=KSL),
    }
    SCHEMES = {
        "Uniform(4)": lambda: xt.Uniform(4, mode="thick"),
        "Teapot(4)": lambda: xt.Teapot(4, mode="thick"),
        "Uniform(1)": lambda: xt.Uniform(1, mode="thick"),
    }
    failures = []


    def sliced(element, scheme):
        line = xt.Line(elements={"m": element})
        line.particle_ref = xt.Particles(p0c=PC)
        if scheme is not None:
            line.slice_thick_elements([xt.Strategy(slicing=scheme)])
        return line


    def body_slices(line):
        return [nn for nn in line.element_names if re.fullmatch(r"\w+\.\.\d+", nn)]


    def arrays(text):
        # Exported tokens read `knl := {a,b,...}` and `ksl := {a,b,...}`; MAD-NG also
        # writes relative `dknl`/`dksl` arrays, which the lookbehind skips.
        out = []
        for key in ("knl", "ksl"):
            match = re.search(r"(?<!d)" + key + r" := \{([^}]*)\}", text)
            assert match is not None, text
            out.append(np.array([float(v) for v in match.group(1).split(",")]))
        return out


    def coefficient_checks():
        for family, make in FAMILIES.items():
            for scheme_name, scheme in SCHEMES.items():
                line = sliced(make(), scheme())
                names = body_slices(line)
                weights = np.array([line[nn].weight for nn in names])
                assert abs(weights.sum() - 1) < 1e-15
                for dialect in (writer.MadType.MADX, writer.MadType.MADNG):
                    texts = [writer.element_to_mad_str(nn, nn, line, mad_type=dialect, substituted_vars=[]) for nn in names]
                    knl, ksl = zip(*(arrays(t) for t in texts))
                    knl, ksl = np.array(knl)[:, :4], np.array(ksl)[:, :4]
                    # Field-integral additivity: slice i carries w_i * KNL_parent, and the
                    # slices sum to the parent for every order.
                    expected_knl, expected_ksl = np.outer(weights, KNL), np.outer(weights, KSL)
                    ok = (np.allclose(knl, expected_knl, rtol=1e-15, atol=0)
                          and np.allclose(ksl, expected_ksl, rtol=1e-15, atol=0))
                    if scheme_name == "Uniform(1)":
                        assert ok, (family, dialect, knl)
                    else:
                        failures.append(not ok)
                    print(f"{family} {scheme_name} {dialect.name}: weights={weights.tolist()} "
                          f"exported sextupole knl={knl[:, 2].tolist()} expected={expected_knl[:, 2].tolist()}; "
                          f"order 0-3 sums knl={knl.sum(0).tolist()} ksl={ksl.sum(0).tolist()} "
                          f"expected knl={KNL} ksl={KSL} derivation=field-integral additivity "
                          f"{'pass' if ok else 'FAIL'}")
        # Control: a dipole-only thick Multipole is written as a KICKER whose kicks
        # the writer scales by the slice weight itself.
        line = sliced(xt.Multipole(length=LENGTH, isthick=True, knl=[5e-5], ksl=[3e-5]),
                      xt.Uniform(4, mode="thick"))
        kicks = [[float(v) for v in re.findall(r"[hv]kick = ([^,]+)", writer.element_to_mad_str(nn, nn, line))]
                 for nn in body_slices(line)]
        assert np.allclose(kicks, [[-1.25e-5, 7.5e-6]] * 4, rtol=1e-15, atol=0), kicks
        print(f"dipole-only Multipole Uniform(4) MADX kicker (hkick, vkick) per slice={kicks} "
              "expected=[-1.25e-05, 7.5e-06] each derivation=w*(-KNL0, KSL0) pass")


    def oracle(multiplier, sext=SEXT, exact=False):
        # Straight-element Hamiltonian of a uniform quadrupole with uniform sextupole
        # errors at the reference energy: with z = x + iy and
        # F = k1 z + (K2L + i K2SL) z^2 / (2 L), px' = -Re F and py' = Im F. KNL/KSL
        # are integrated, so the field divides by L. The expanded (paraxial) drift
        # gives x' = px, y' = py, as Xtrack and PTC with exact=false use; the exact
        # drift that MAD-NG uses gives x' = px/ps, y' = py/ps, ps = sqrt(1-px^2-py^2).
        kn, ks = sext

        def rhs(_, u):
            z = u[0] + 1j * u[2]
            f = K1 * z + multiplier * (kn + 1j * ks) * z**2 / (2 * LENGTH)
            ps = np.sqrt(1 - u[1] ** 2 - u[3] ** 2) if exact else 1.0
            return [u[1] / ps, -f.real, u[3] / ps, f.imag]

        first = solve_ivp(rhs, (0, LENGTH), INITIAL, method="DOP853", rtol=2.3e-14, atol=2e-17)
        second = solve_ivp(rhs, (0, LENGTH), INITIAL, method="RK45", rtol=2.3e-14, atol=2e-17)
        # A second adaptive Runge-Kutta order checks convergence.
        np.testing.assert_allclose(first.y[:, -1], second.y[:, -1], atol=2e-16, rtol=0)
        return first.y[:, -1]


    def quadrupole(sext=SEXT):
        return xt.Quadrupole(length=LENGTH, k1=K1, knl=[0, 0, sext[0]], ksl=[0, 0, sext[1]],
                             model="mat-kick-mat", integrator="yoshida-6", num_multipole_kicks=64)


    def handwritten(n, sext=SEXT):
        # Independently written sequence: n quadrupoles of length L/n with KNL/n.
        text = f"testseq: sequence, l={LENGTH};\n"
        for i in range(n):
            text += (f"h{i}: quadrupole, l={LENGTH / n:.17g}, k1={K1}, "
                     f"knl={{0,0,{sext[0] / n:.17g}}}, ksl={{0,0,{sext[1] / n:.17g}}}, "
                     f"at={(i + 0.5) * LENGTH / n:.17g};\n")
        return text + "endsequence;\n"


    def madng_track(line):
        # Public MAD-NG path: Line.to_madng writes the MAD-NG sequence and loads it.
        with TemporaryDirectory(prefix="io21-ng-", dir=Path(__file__).resolve().parent) as scratch:
            mng = line.to_madng(sequence_name="seq", temp_fname=str(Path(scratch) / "line"),
                                stdout=str(Path(scratch) / "stdout.txt"))
            x, px, y, py = INITIAL
            mng.send(f"""
                local _, mflw = track {{ sequence=seq, method=6, nslice=20, save=false,
                    X0={{ x={x:.17g}, px={px:.17g}, y={y:.17g}, py={py:.17g}, t=0, pt=0 }} }}
                local k2, k2s = 0, 0
                for i = 1, #seq do
                    local e = seq[i]
                    if e.kind == 'quadrupole' and not e.kill_body then
                        k2, k2s = k2 + (e.knl[3] or 0), k2s + (e.ksl[3] or 0)
                    end
                end
                py:send(MAD.vector {{ mflw[1].x, mflw[1].px, mflw[1].y, mflw[1].py, k2, k2s }})
                py:send(MAD.env.version)
            """)
            values = np.asarray(mng.recv()).ravel()
            ng_version = mng.recv()
            mng.close()
        print(f"CONSUMER pymadng={version('pymadng')} MAD-NG={ng_version}")
        return values[:4], values[4:]


    def compare(label, observed, reference, derivation, tolerance, defect_expected, sums=None):
        ok = bool(np.allclose(observed, reference, atol=tolerance, rtol=0))
        if not defect_expected:
            assert ok, (label, observed, reference)
        else:
            failures.append(not ok)
        extra = f" read-back integrated K2L,K2SL={[float(v) for v in sums]}" if sums is not None else ""
        print(f"{label} (x,px,y,py): observed={list(map(float, observed))} "
              f"expected={reference.tolist()}{extra} derivation={derivation} tol={tolerance:g} "
              f"{'pass' if ok else 'FAIL'}")
        return ok


    def dynamics_checks():
        intended, fourfold = oracle(1), oracle(4)
        pure = oracle(1, (0.0, 0.0))

        whole = sliced(quadrupole(), xt.Uniform(4, mode="thick"))
        body = whole.select(start="m..0", end="m..3")
        p = xt.Particles(_context=CTX, p0c=PC, x=INITIAL[0], px=INITIAL[1], y=INITIAL[2], py=INITIAL[3])
        whole.build_tracker(_context=CTX)
        whole.track(p)
        compare("native four slices", [p.x[0], p.px[0], p.y[0], p.py[0]], intended,
                "DOP853 intended field", 3e-16, False)

        # MAD-X refuses the whole sliced line before tracking (`kill_body` on the
        # generated entry/exit maps is not a MAD-X keyword); the body slices, selected
        # from the same native slices, isolate the body serialization.
        refusal = consume(whole.to_madx_sequence("testseq"))
        print(f"MAD-X whole sliced line: status={refusal['status']} error={refusal.get('error')!r}")

        for label, source, defect in (
            ("PTC unsliced export", sliced(quadrupole(), None).to_madx_sequence("testseq"), False),
            ("PTC handwritten weighted text", handwritten(4), False),
            ("PTC exported four body slices", body.to_madx_sequence("testseq"), True),
        ):
            magnets, end = ptc_track(source, INITIAL, PC / 1e9)
            sums = [sum(m[2][2] for m in magnets), sum(m[3][2] for m in magnets)]
            compare(label, end, intended, "DOP853 intended field", 3e-16, defect, sums)
            if defect:
                compare(label + " vs fourfold field", end, fourfold, "DOP853 fourfold field", 3e-16, False)

        pure_body = sliced(quadrupole((0.0, 0.0)), xt.Uniform(4, mode="thick")).select(start="m..0", end="m..3")
        _, end = ptc_track(pure_body.to_madx_sequence("testseq"), INITIAL, PC / 1e9)
        compare("PTC four body slices, no higher orders", end, pure, "DOP853 pure quadrupole", 3e-16, False)

        intended, fourfold = oracle(1, exact=True), oracle(4, exact=True)
        pure = oracle(1, (0.0, 0.0), exact=True)
        for label, line, reference, defect in (
            ("MAD-NG unsliced", sliced(quadrupole(), None), intended, False),
            ("MAD-NG four slices, no higher orders",
             sliced(quadrupole((0.0, 0.0)), xt.Uniform(4, mode="thick")), pure, False),
            ("MAD-NG four slices", whole, intended, True),
        ):
            end, sums = madng_track(line)
            compare(label, end, reference, "DOP853 exact-drift field", 3e-16, defect, sums)
            if defect:
                compare(label + " vs fourfold field", end, fourfold,
                        "DOP853 exact-drift fourfold field", 3e-16, False)


    def main():
        identity()
        print(f"PACKAGE cpymad {version('cpymad')}; pymadng {version('pymadng')}")
        coefficient_checks()
        dynamics_checks()
        kernels()
        print(f"COMPARISONS expected-correct cases fail={sum(failures)} of {len(failures)}")
        print("DEFECT REPRODUCED" if any(failures) else "NOT REPRODUCED")


    if __name__ == "__main__":
        main()

    ```

## Provenance

| Role | Model | Effort | Date | Code examined |
|---|---|---|---|---|
| Found | <span class="bb-nowrap">`claude-opus-5-5`</span> | high | <span class="bb-nowrap">2026-11-06</span> | xtrack 0.118.2 (`ad5e808cdc5b`) |
| Confirmed | <span class="bb-nowrap">`gpt-6.1-sol`</span> | xhigh | <span class="bb-nowrap">2026-11-06</span> | xtrack 0.118.2 (`ad5e808cdc5b`), installed packages, prebuilt kernels (compilation forbidden), CPU, double, serial; cpymad 1.19.0, MAD-X 5.09.03 PTC |
| Confirmed<br><small class="bb-scope">MAD-NG tracking of the whole sliced line</small> | <span class="bb-nowrap">`claude-opus-5-5`</span> | high | <span class="bb-nowrap">2026-11-06</span> | xtrack 0.118.2 (`ad5e808cdc5b`), installed packages, prebuilt kernels (compilation forbidden), CPU, double, serial; cpymad 1.19.0, MAD-X 5.09.03 PTC; pymadng 0.9.6, MAD-NG 1.1.14 |

