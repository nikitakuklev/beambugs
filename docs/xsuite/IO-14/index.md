---
title: "IO-14: MAD-X export loses embedded quadrupole dipole steering"
hide:
  - toc
---

# IO-14 — MAD-X export loses embedded quadrupole dipole steering

| | |
|---|---|
| Code | [Xsuite](../index.md) · Lattices, particles and I/O |
| Priority | P2 |
| Failure | Wrong physics |
| Evidence | Reproduced, independently rechecked |
| Last reproduced | 2026-11-07 on xtrack release [v0.118.6](https://github.com/xsuite/xtrack/tree/v0.118.6) (`f09e0fb`) |
| Source | [`quadrupole_to_mad_str`, mad_writer.py:566–574](https://github.com/xsuite/xtrack/blob/ad5e808cdc5b9a881eac2f41e32a0b7d11cc7028/xtrack/mad_writer.py#L566-L574)<br>[array serializer, lines 96–112](https://github.com/xsuite/xtrack/blob/ad5e808cdc5b9a881eac2f41e32a0b7d11cc7028/xtrack/mad_writer.py#L96-L112)<br>[tokens, lines 125–126](https://github.com/xsuite/xtrack/blob/ad5e808cdc5b9a881eac2f41e32a0b7d11cc7028/xtrack/mad_writer.py#L125-L126)<br>[additional-multipole contract, _common.py:405–412](https://github.com/xsuite/xtrack/blob/ad5e808cdc5b9a881eac2f41e32a0b7d11cc7028/xtrack/beam_elements/_common.py#L405-L412)<br>[Quadrupole API, quadrupole.py:55–62](https://github.com/xsuite/xtrack/blob/ad5e808cdc5b9a881eac2f41e32a0b7d11cc7028/xtrack/beam_elements/quadrupole.py#L55-L62)<br>[`Line.to_madx_sequence`, line.py:877–894](https://github.com/xsuite/xtrack/blob/ad5e808cdc5b9a881eac2f41e32a0b7d11cc7028/xtrack/line.py#L877-L894)<br>[QUADRUPOLE attributes, elements.tex:284–354](https://github.com/MethodicalAcceleratorDesign/MAD-X/blob/3865cb633e66fb4cf53d2f37b00c919dbd6ab495/doc/latexuguide/elements.tex#L284-L354)<br>[PTC additional options, ptc-general.tex:472–533](https://github.com/MethodicalAcceleratorDesign/MAD-X/blob/3865cb633e66fb4cf53d2f37b00c919dbd6ab495/doc/latexuguide/ptc-general.tex#L472-L533)<br>[Twiss quadrupole dispatch, twiss.f90:3711–3712](https://github.com/MethodicalAcceleratorDesign/MAD-X/blob/3865cb633e66fb4cf53d2f37b00c919dbd6ab495/src/twiss.f90#L3711-L3712)<br>[`tmquad`, lines 6167–6175](https://github.com/MethodicalAcceleratorDesign/MAD-X/blob/3865cb633e66fb4cf53d2f37b00c919dbd6ab495/src/twiss.f90#L6167-L6175)<br>[TRACK dispatch, trrun.f90:748–749](https://github.com/MethodicalAcceleratorDesign/MAD-X/blob/3865cb633e66fb4cf53d2f37b00c919dbd6ab495/src/trrun.f90#L748-L749)<br>[`tttquad`, lines 4462–4487](https://github.com/MethodicalAcceleratorDesign/MAD-X/blob/3865cb633e66fb4cf53d2f37b00c919dbd6ab495/src/trrun.f90#L4462-L4487)<br>[PTC quadrupole translation, madx_ptc_module.f90:983–1050](https://github.com/MethodicalAcceleratorDesign/MAD-X/blob/3865cb633e66fb4cf53d2f37b00c919dbd6ab495/src/madx_ptc_module.f90#L983-L1050) |

Exporting a line to a MAD-X sequence writes the embedded normal or skew dipole kick of a thick quadrupole as KNL or KSL on the thick QUADRUPOLE, which MAD-X's own Twiss and TRACK ignore, also after MAKETHIN; only PTC applies it. A centered 3 GeV/c proton crossing a 1.2 m quadrupole with normalized gradient k₁ = 0.3000 m⁻² and integrated normal dipole strength 50.00 μrad then exits at zero horizontal position and momentum instead of x = −29.46 μm and pₓ = −4.732×10⁻⁵. Independent integration and an equivalent sequence with an explicit midpoint corrector establish the missing steering. The quadrupole's length and focusing remain intact.

Run `python IO-14.py` with [_prebuilt.py](../helpers/_prebuilt.py) and [_common.py](../helpers/_common.py) in the same directory.

??? example "Reproducer: IO-14.py"

    [Download IO-14.py](IO-14.py)

    ```python
    """IO-14: MAD-X export loses embedded quadrupole dipole steering

    Exporting a line to a MAD-X sequence writes the embedded normal or skew dipole kick of a thick quadrupole as KNL or KSL on the thick QUADRUPOLE, which MAD-X's own Twiss and TRACK ignore, also after MAKETHIN; only PTC applies it. A centered 3 GeV/c proton crossing a 1.2 m quadrupole with normalized gradient k₁ = 0.3000 m⁻² and integrated normal dipole strength 50.00 μrad then exits at zero horizontal position and momentum instead of x = −29.46 μm and pₓ = −4.732×10⁻⁵. Independent integration and an equivalent sequence with an explicit midpoint corrector establish the missing steering. The quadrupole's length and focusing remain intact.

    Integrate the linear transverse equations independently around a midpoint dipole impulse and compare native tracking with ordinary MAD-X Twiss and TRACK. Exercise normal, skew and combined steering, a zero-gradient case, an explicit-corrector representation, MAKETHIN and an off-axis zero-steering control; a PTC control tracks the combined export against a uniform body dipole field. The script uses cpymad and installed serial prebuilt Xtrack kernels with compilation forbidden; MAD-X trajectory tables are read from memory. MAD-X scratch files are confined to an automatically removed directory beside the script.
    """

    from pathlib import Path
    from tempfile import TemporaryDirectory

    from _prebuilt import CTX, identity, kernels, np, xt

    # Import after the helper caps numerical threads and forbids compilation.
    # isort: split
    import cpymad
    from cpymad.madx import Madx
    from scipy.integrate import solve_ivp

    LENGTH = 1.2
    GRADIENT = 0.3
    NORMAL = 5e-5
    SKEW = 3e-5
    PC = 3e9
    matches = []


    def oracle(initial, normal, skew, gradient, distributed=False):
        # Paraxial Hamiltonian: x'=px, px'=-k1*x, y'=py, py'=k1*y.
        # One normal/skew midpoint impulse changes (px,py) by (-normal,+skew).
        # PTC spreads thick-element KNL/KSL over the body: a uniform dipole field
        # adds (-normal/L, +skew/L) to (px', py') instead.
        if distributed:
            def ode(s, u):
                x, px, y, py = u
                return [px, -gradient * x - normal / LENGTH, py, gradient * y + skew / LENGTH]

            return solve_ivp(ode, (0, LENGTH), initial, method="DOP853",
                             rtol=3e-14, atol=1e-19).y[:, -1]

        def ode(s, u):
            x, px, y, py = u
            return [px, -gradient * x, py, gradient * y]

        first = solve_ivp(ode, (0, LENGTH / 2), initial, method="DOP853",
                          rtol=3e-14, atol=1e-19).y[:, -1]
        first += [0, -normal, 0, skew]
        result = solve_ivp(ode, (LENGTH / 2, LENGTH), first, method="DOP853",
                           rtol=3e-14, atol=1e-19).y[:, -1]
        if gradient > 0 and not any(initial):
            root = np.sqrt(gradient)
            phase = root * LENGTH / 2
            analytic = [-normal * np.sin(phase) / root, -normal * np.cos(phase),
                        skew * np.sinh(phase) / root, skew * np.cosh(phase)]
            np.testing.assert_allclose(result, analytic, atol=2e-19, rtol=0)
        return result


    def make_line(normal, skew, gradient, explicit):
        if explicit:
            elements = {
                "qa": xt.Quadrupole(length=LENGTH / 2, k1=gradient),
                "c": xt.Multipole(knl=[normal], ksl=[skew]),
                "qb": xt.Quadrupole(length=LENGTH / 2, k1=gradient),
            }
        else:
            elements = {"q": xt.Quadrupole(
                length=LENGTH, k1=gradient, knl=[normal], ksl=[skew],
                model="mat-kick-mat", integrator="uniform", num_multipole_kicks=1)}
        return xt.Line(elements=elements)


    def madx_run(source, initial, makethin, ptc):
        with TemporaryDirectory(prefix="io14-", dir=Path(__file__).resolve().parent) as scratch, \
                Madx(stdout=False, cwd=scratch) as mad:
            print("MADX_VERSION", str(mad.version))
            mad.input(source)
            mad.beam(particle="proton", pc=PC / 1e9)
            mad.use(sequence="seq")
            if makethin:
                mad.input("select, flag=makethin, class=quadrupole, slice=8; "
                          "makethin, sequence=seq, style=teapot; use, sequence=seq;")
            x, px, y, py = initial
            tw = mad.twiss(betx=9, bety=12, x=x, px=px, y=y, py=py)
            tw_end = np.array([tw.x[-1], tw.px[-1], tw.y[-1], tw.py[-1]])
            # Read trackone from memory. MAD-X's additional output/checkpoint files
            # remain inside the subprocess's automatically cleaned scratch directory.
            mad.input("track, onepass=true, onetable=true, file=false; "
                      f"start, x={x:.17g}, px={px:.17g}, y={y:.17g}, py={py:.17g}; "
                      "run, turns=1; endtrack;")
            table = mad.table.trackone
            assert len(table.x) >= 2 and table.turn[-1] == 1
            np.testing.assert_allclose([table.s[-1], tw.s[-1]], LENGTH, atol=2e-15, rtol=0)
            track_end = np.array([table.x[-1], table.px[-1], table.y[-1], table.py[-1]])
            if not ptc:
                return tw_end, track_end, None
            mad.input("ptc_create_universe; "
                      "ptc_create_layout, model=2, method=6, nst=20, exact=false; "
                      f"ptc_start, x={x:.17g}, px={px:.17g}, y={y:.17g}, py={py:.17g}; "
                      "ptc_track, icase=4, onetable=true, dump=false; ptc_track_end; ptc_end;")
            table = mad.table.trackone
            np.testing.assert_allclose(table.s[-1], LENGTH, atol=2e-15, rtol=0)
            ptc_end = np.array([table.x[-1], table.px[-1], table.y[-1], table.py[-1]])
            return tw_end, track_end, ptc_end


    def main():
        identity()
        print("PACKAGE cpymad", cpymad.__version__, cpymad.__file__)
        for label, normal, skew, gradient, initial, explicit, thin in (
            ("normal", NORMAL, 0.0, GRADIENT, [0.0] * 4, False, False),
            ("skew", 0.0, SKEW, GRADIENT, [0.0] * 4, False, False),
            ("combined", NORMAL, SKEW, GRADIENT, [0.0] * 4, False, False),
            ("zero-gradient", NORMAL, SKEW, 0.0, [0.0] * 4, False, False),
            ("body-control", 0.0, 0.0, GRADIENT, [2e-4, 3e-5, -1e-4, 2e-5], False, False),
            ("explicit-corrector", NORMAL, SKEW, GRADIENT, [0.0] * 4, True, False),
            ("MAKETHIN", NORMAL, SKEW, GRADIENT, [0.0] * 4, False, True),
        ):
            line = make_line(normal, skew, gradient, explicit)
            p = xt.Particles(_context=CTX, p0c=PC, x=initial[0], px=initial[1],
                             y=initial[2], py=initial[3])
            for el in line._elements:
                el.track(p)
            native = np.array([p.x[0], p.px[0], p.y[0], p.py[0]])
            expected = oracle(initial, normal, skew, gradient)
            source = line.to_madx_sequence("seq")
            ptc = label == "combined"
            twiss, track, ptc_track = madx_run(source, initial, thin, ptc)
            cases = [("native", native, 2e-17, expected, "half bodies and midpoint impulse"),
                     ("Twiss", twiss, 2e-17, expected, "half bodies and midpoint impulse"),
                     ("TRACK", track, 2e-12, expected, "half bodies and midpoint impulse")]
            if ptc:
                # Control: PTC applies the exported thick-quadrupole KNL/KSL.
                cases.append(("PTC", ptc_track, 2e-17,
                              oracle(initial, normal, skew, gradient, distributed=True),
                              "uniform body dipole field"))
            for engine, observed, tolerance, reference, derivation in cases:
                passed = bool(np.allclose(observed, reference, atol=tolerance, rtol=0))
                if engine in ("native", "PTC") or explicit or not (normal or skew):
                    assert passed, (label, engine, observed, reference)
                matches.append(passed)
                print(f"{label} {engine} (x,px,y,py): observed={observed.tolist()} "
                      f"expected={reference.tolist()} "
                      f"derivation=DOP853 {derivation} match={passed}")
        kernels()
        print(f"COMPARISONS total={len(matches)} pass={sum(matches)} fail={len(matches) - sum(matches)}")
        print("NOT REPRODUCED" if all(matches) else "DEFECT REPRODUCED")


    if __name__ == "__main__":
        main()

    ```

## Provenance

| Role | Model | Effort | Date | Code examined |
|---|---|---|---|---|
| Found | `gpt-6-astra` | xhigh | 2026-11-06 | xtrack 0.118.2 (`ad5e808cdc5b`), installed packages, prebuilt kernels (compilation forbidden), CPU, double, serial; cpymad 1.19.0, MAD-X 5.09.03 ordinary Twiss and TRACK |
| Confirmed | `gpt-6.1-sol` | xhigh | 2026-11-06 | xtrack 0.118.2 (`ad5e808cdc5b`), installed packages, prebuilt kernels (compilation forbidden), CPU, double, serial; cpymad 1.19.0, MAD-X 5.09.03 ordinary Twiss and TRACK |
| Confirmed | `claude-opus-5-5` | high | 2026-11-07 | xtrack 0.118.2 (`ad5e808cdc5b`), installed packages, prebuilt kernels (compilation forbidden), CPU, double, serial; cpymad 1.19.0, MAD-X 5.09.03 ordinary Twiss, TRACK and PTC |

