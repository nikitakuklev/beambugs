---
title: "TRK-25: Shared steering multipole tables carry one element's factor into another"
hide:
  - toc
---

# TRK-25 — Shared steering multipole tables carry one element's factor into another

| | |
|---|---|
| Code | [Elegant](../index.md) · Tracking |
| Priority | P2 |
| Severity | Wrong physics |
| Evidence | Reproduced, independently rechecked |
| Last reproduced | 2026-11-04 on Elegant master (`0355dd6`) |
| Source | [`readErrorMultipoleData`, multipole.c:267–283](https://github.com/rtsoliday/elegant/blob/0355dd6aca12c861971fbf8f3ae78a0ded991734/src/multipole.c#L267-L283)<br>[shallow copies, multipole.c:91–105](https://github.com/rtsoliday/elegant/blob/0355dd6aca12c861971fbf8f3ae78a0ded991734/src/multipole.c#L91-L105)<br>[steering overwrite, multipole.c:1799–1812](https://github.com/rtsoliday/elegant/blob/0355dd6aca12c861971fbf8f3ae78a0ded991734/src/multipole.c#L1799-L1812)<br>[KQUAD guard, multipole.c:830](https://github.com/rtsoliday/elegant/blob/0355dd6aca12c861971fbf8f3ae78a0ded991734/src/multipole.c#L830)<br>[KSEXT guard, multipole.c:906](https://github.com/rtsoliday/elegant/blob/0355dd6aca12c861971fbf8f3ae78a0ded991734/src/multipole.c#L906)<br>[steering consumer, multipole.c:1445–1456](https://github.com/rtsoliday/elegant/blob/0355dd6aca12c861971fbf8f3ae78a0ded991734/src/multipole.c#L1445-L1456)<br>[KQUAD steering manual, kquad.tex:38](https://github.com/rtsoliday/elegant/blob/0355dd6aca12c861971fbf8f3ae78a0ded991734/doc/kquad.tex#L38)<br>[KSEXT steering manual, ksext.tex:38](https://github.com/rtsoliday/elegant/blob/0355dd6aca12c861971fbf8f3ae78a0ded991734/doc/ksext.tex#L38)<br>[factor metadata, track_data.c:1278](https://github.com/rtsoliday/elegant/blob/0355dd6aca12c861971fbf8f3ae78a0ded991734/src/track_data.c#L1278) |

KQUAD and KSEXT magnets sharing a STEERING_MULTIPOLES filename share mutable coefficients scaled by each magnet's STEERING_MULTIPOLE_FACTOR. A magnet can therefore use another magnet's factor on later passes; separate files with identical contents preserve the intended fields. In a three-pass KQUAD test, a steering magnet at factor 1.25 followed by a zero-field magnet at factor 0 gives a 49 mₑc electron horizontal slopes x′ = −4.280×10⁻³ and −4.566×10⁻³ on passes 2 and 3, instead of −4.272×10⁻³ and −4.561×10⁻³. Across the test beam, the largest vertical-slope error reaches 2.094×10⁻⁴ on pass 3. Independent Lorentz integration and equal-factor controls show that changing the zero-field magnet's factor cannot physically change the trajectory.

Download these shared helpers into the same directory as the reproducer scripts:

- [Download _elegant.py](../helpers/_elegant.py) · [View source](../helpers/index.md#_elegant)
- [Download _multipole.py](../helpers/_multipole.py) · [View source](../helpers/index.md#_multipole)

??? example "Reproducer: TRK-25.py"

    [Download TRK-25.py](TRK-25.py)

    ```python
    """TRK-25: Shared steering multipole tables carry one element's factor into another

    KQUAD and KSEXT magnets sharing a STEERING_MULTIPOLES filename share mutable coefficients scaled by
    each magnet's STEERING_MULTIPOLE_FACTOR. A magnet can therefore use another magnet's factor on later
    passes; separate files with identical contents preserve the intended fields. In a three-pass KQUAD
    test, a steering magnet at factor 1.25 followed by a zero-field magnet at factor 0 gives a 49 mₑc
    electron horizontal slopes x′ = −4.280×10⁻³ and −4.566×10⁻³ on passes 2 and 3, instead of
    −4.272×10⁻³ and −4.561×10⁻³. Across the test beam, the largest vertical-slope error reaches
    2.094×10⁻⁴ on pass 3. Independent Lorentz integration and equal-factor controls show that changing
    the zero-field magnet's factor cannot physically change the trajectory.

    The script tracks 12 electrons for three passes through a KQUAD or KSEXT magnet A, with steering
    kicks and a steering table at factor 1.25, followed by a magnet B of zero strength and zero kicks
    whose table has a factor of 1.25, 0 or -0.5. B names either a second file with identical bytes or
    the same file as A, and each pass is compared with an independent integration of A at its own
    factor followed by B's drift. The controls are B with a separate file and B with equal factors; a
    further check integrates A with B's factor from the second pass on, which identifies the shared
    array as the cause.
    """

    import sys

    import numpy as np
    from _elegant import modes, particles, run, verdict
    from _multipole import P0, beam, canonical, observable, straight_body, table_text

    LENGTH = 0.13  # m, both magnets
    KICK = (-0.002, 0.0015)  # HKICK, VKICK of A
    FACTOR_A = 1.25
    KINDS = {"KQUAD": (1, -0.7), "KSEXT": (2, -17.0)}  # element: (main order m, strength K_m)
    PASSES = 3
    ATOL = 1e-13  # observed agreement is ~2e-15
    P_ATOL = 2e-13  # no radiation, so the momentum is conserved
    SPLIT = 1e-6  # a changed factor moves the third pass by 1e-4 and more


    def oracle(initial, main_order, strength, factor_after_first):
        """Exit x, xp, y, yp after each pass through A followed by B's drift.

        A uses `factor_after_first` from the second pass on. B is a pure drift (zero strength, zero kicks): its steering factor multiplies zero kicks and
        cannot change the physical trajectory.
        """
        state = canonical(initial)
        outputs = []
        for turn in range(PASSES):
            factor = FACTOR_A if turn == 0 else factor_after_first
            state = straight_body(state, initial, main_order, strength, LENGTH, factor, KICK)
            state = straight_body(state, initial, main_order, 0.0, LENGTH)
            outputs.append(observable(state, initial))
        return outputs


    def track(mode, kind, factor_b, shared, initial):
        m, strength = KINDS[kind]
        second_file = "steer.sdds" if shared else "identical.sdds"
        lattice = (
            f"A: {kind},L={LENGTH},K{m}={strength},HKICK={KICK[0]},VKICK={KICK[1]},"
            f'STEERING_MULTIPOLES="steer.sdds",STEERING_MULTIPOLE_FACTOR={FACTOR_A},'
            "N_SLICES=96,SYNCH_RAD=0,ISR=0\n"
            f"B: {kind},L={LENGTH},K{m}=0,HKICK=0,VKICK=0,"
            f'STEERING_MULTIPOLES="{second_file}",STEERING_MULTIPOLE_FACTOR={factor_b},'
            "N_SLICES=96,SYNCH_RAD=0,ISR=0\n"
            'W: WATCH,FILENAME="end.sdds",MODE="coordinates"\n'
            "BL: LINE=(A,B,W)\n"
        )
        commands = (
            f"&run_setup lattice=lattice.lte, use_beamline=BL, p_central={P0}, output=out.sdds, "
            "random_number_seed=753219 &end\n"
            f"&run_control n_steps=1, n_passes={PASSES} &end\n"
            "&sdds_beam input=beam.sdds &end\n"
            "&track soft_failure=0 &end\n"
        )
        table = table_text([2, 4], [0.07, -0.025])
        files = {
            "lattice.lte": lattice,
            "run.ele": commands,
            "steer.sdds": table,
            "identical.sdds": table,
        }
        watch = run(files, mode, {"beam.sdds": initial}, outputs=("end.sdds",))["end.sdds"]
        if watch.n_pages != PASSES:
            sys.exit(f"{mode} {kind}: expected {PASSES} watch pages")
        result = []
        for page in range(PASSES):
            out = particles(watch, page)
            if len(out["p"]) != len(initial["p"]):
                sys.exit(f"{mode} {kind}: particles were lost")
            if not np.allclose(out["p"], initial["p"], rtol=0, atol=P_ATOL, equal_nan=False):
                sys.exit(f"{mode} {kind}: the momentum is not conserved on page {page}")
            result.append(np.column_stack((out["x"], out["xp"], out["y"], out["yp"])))
        return result


    def last_xp(passes):
        return "[" + ", ".join(repr(float(a[-1, 1])) for a in passes) + "]"


    def compare(label, observed, expected):
        """Print the last particle's xp after each pass; return the largest error per pass.

        Nonfinite values make the comparison invalid: the run terminates instead of returning a NaN
        that every later `> ATOL` test would accept.
        """
        for values in (*observed, *expected):
            if not np.all(np.isfinite(values)):
                sys.exit(f"{label}: a nonfinite value makes the comparison invalid")
        errors = [float(np.max(np.abs(o - e))) for o, e in zip(observed, expected, strict=True)]
        print(
            f"{label}: observed={last_xp(observed)} expected={last_xp(expected)} "
            f"max|diff| per pass=[{', '.join(f'{e:.4g}' for e in errors)}]"
        )
        return errors


    defect = False
    initial = beam()
    for mode in modes():
        for kind, (m, strength) in KINDS.items():
            expected = oracle(initial, m, strength, FACTOR_A)
            for factor_b in (FACTOR_A, 0.0, -0.5):
                label = f"{mode} {kind} factor_B={factor_b:g}"
                separate = track(mode, kind, factor_b, False, initial)
                shared = track(mode, kind, factor_b, True, initial)
                if max(compare(f"{label} separate files", separate, expected)) > ATOL:
                    sys.exit(f"{label}: the separate-file control disagrees with the oracle")
                errors = compare(f"{label} shared file", shared, expected)
                # Cause: A is evaluated with B's factor from the second pass on.
                contaminated = oracle(initial, m, strength, factor_b)
                if not all(np.all(np.isfinite(c)) for c in contaminated):
                    sys.exit(f"{label}: the contaminated-factor oracle is not finite")
                if not all(
                    np.allclose(s, c, rtol=0, atol=ATOL, equal_nan=False)
                    for s, c in zip(shared, contaminated, strict=True)
                ):
                    sys.exit(f"{label}: the shared-file result is not A evaluated with B's factor")
                if factor_b == FACTOR_A:
                    if max(errors) > ATOL:
                        sys.exit(f"{label}: equal factors disagree with the oracle")
                else:
                    if errors[0] > ATOL:
                        sys.exit(f"{label}: the first pass already disagrees")
                    defect |= max(errors[1:]) > SPLIT
    print(f"comparison: atol={ATOL:g} for agreement, {SPLIT:g} for disagreement")
    verdict(defect)

    ```

## Provenance

| Role | Model | Effort | Date | Code examined |
|---|---|---|---|---|
| Found | `gpt-6-astra` | xhigh | 2026-10-29 | master (`75a2e1a1a58b`), source with SDDS a60b9bbed7cc, CPU, serial and 3-rank Pelegant |
| Confirmed | `gpt-6.1-sol` | xhigh | 2026-10-29 | master (`75a2e1a1a58b`), source with SDDS a60b9bbed7cc, CPU, serial and 3-rank Pelegant |

