---
title: "COL-14: BEAMBEAM kicks change arrival time because s is not rescaled"
hide:
  - toc
---

# COL-14 — BEAMBEAM kicks change arrival time because s is not rescaled

| | |
|---|---|
| Code | [Elegant](../index.md) · Collective effects |
| Priority | P3 |
| Severity | Wrong physics |
| Evidence | Reproduced, independently rechecked |
| Last reproduced | 2026-11-04 on Elegant master (`0355dd6`, head build) |
| Source | [Gaussian branch update, do_tracking.c:6322](https://github.com/rtsoliday/elegant/blob/0355dd6aca12c861971fbf8f3ae78a0ded991734/src/do_tracking.c#L6322)<br>[helper update, ionEffects.cc:3617](https://github.com/rtsoliday/elegant/blob/0355dd6aca12c861971fbf8f3ae78a0ded991734/src/ionEffects.cc#L3617)<br>[coordinate definition, elegant.tex:372–379](https://github.com/rtsoliday/elegant/blob/0355dd6aca12c861971fbf8f3ae78a0ded991734/doc/elegant.tex#L372-L379) |

A BEAMBEAM kick changes momentum but leaves the stored longitudinal coordinate s = βct unchanged, where t is arrival time, c is the speed of light and β is speed divided by c. The output time therefore changes by the ratio of incoming to outgoing speed, although the zero-length element must preserve time and rescale s. For an electron of momentum 1 mₑc at horizontal position x = 0.2300 mm, a 1.300 nC uniform opposing bunch of radius 1.300 mm changes an input time of 1 µs by −1.936×10⁻¹¹ s, a relative error of −1.936×10⁻⁵. Zero input time masks the error.

Run `python COL-14.py` with [_elegant.py](../helpers/_elegant.py) in the same directory.

??? example "Reproducer: COL-14.py"

    [Download COL-14.py](COL-14.py)

    ```python
    """COL-14: BEAMBEAM kicks change arrival time because s is not rescaled

    A BEAMBEAM kick changes momentum but leaves the stored longitudinal coordinate s = βct unchanged,
    where t is arrival time, c is the speed of light and β is speed divided by c. The output time
    therefore changes by the ratio of incoming to outgoing speed, although the zero-length element must
    preserve time and rescale s. For an electron of momentum 1 mₑc at horizontal position x = 0.2300 mm,
    a 1.300 nC uniform opposing bunch of radius 1.300 mm changes an input time of 1 µs by −1.936×10⁻¹¹
    s, a relative error of −1.936×10⁻⁵. Zero input time masks the error.

    The script sends electrons at x = 0.23 mm with six arrival times (zero, positive and negative,
    from 1 ns to 0.25 s) through one BEAMBEAM element of each distribution at p = 1, 100 and 1000 mₑc.
    It compares the output time with the input time and with the prediction for an unchanged s,
    t_in β_old/β_new, where β is taken from the momentum Elegant returns. The control is zero
    opposing charge, which must leave both momentum and time unchanged.
    """

    import sys

    import numpy as np
    from _elegant import modes, run, verdict

    TIMES = np.array([0, 1e-9, -2e-9, 1e-6, 0.125, -0.25] * 2)  # s; the 1 us entry is index 3
    N = len(TIMES)
    X0, SIZE = 0.23e-3, 1.3e-3  # m
    RTOL, ATOL = 2e-15, 2e-24  # relative, and absolute in s (zero-time entries)


    def beta(p):
        return p / np.sqrt(1 + p * p)


    def track(mode, distribution, p0, charge):
        lattice = (
            f"K: BEAMBEAM,DISTRIBUTION={distribution},CHARGE={charge!r},XSIZE={SIZE!r},YSIZE={SIZE!r}\n"
            "BL: LINE=(K)\n"
        )
        commands = (
            "&global_settings echo_namelists=0 &end\n"
            f"&run_setup lattice=lattice.lte,use_beamline=BL,p_central={p0!r},output=out.sdds,"
            "losses=lost.sdds &end\n"
            "&run_control n_steps=1,n_passes=1 &end\n"
            "&sdds_beam input=beam.sdds &end\n"
            "&track soft_failure=0 &end\n"
        )
        beam = {"x": np.full(N, X0), "t": TIMES, "p": np.full(N, p0)}
        result = run(
            {"lattice.lte": lattice, "run.ele": commands},
            mode,
            {"beam.sdds": beam},
            outputs=("out.sdds", "lost.sdds"),
        )
        out, lost = result["out.sdds"], result["lost.sdds"]
        if len(out[0, "p"]) != N or len(lost[0, "particleID"]) != 0:
            sys.exit(f"{mode} {distribution} p={p0}: particles were lost")
        # The global_settings echo prints "warning_limit"; any other mention is a warning.
        if "warning" in result["log"].lower().replace("warning_limit", ""):
            sys.exit(f"{mode} {distribution} p={p0}: Elegant printed a warning")
        p, t = np.asarray(out[0, "p"]), np.asarray(out[0, "t"])
        # Invalid coordinates are a failed run, never evidence for the defect.
        if not (np.all(np.isfinite(p)) and np.all(p > 0) and np.all(np.isfinite(t))):
            sys.exit(f"{mode} {distribution} p={p0}: non-finite times or non-positive momentum")
        return p, t


    defect = False
    for mode in modes():
        for p0 in (1.0, 100.0, 1000.0):
            for distribution in ("uniform", "parabolic", "gaussian"):
                # The charge scales with p to keep the angular kick small but resolvable.
                for charge in (0.0, 1.3e-9 * p0):
                    p, t = track(mode, distribution, p0, charge)
                    name = f"{mode} {distribution} p_in={p0:g} charge={charge:.3g} C"
                    # The correct map conserves t (s = beta c t is rescaled with the velocity).
                    changed = not np.allclose(t, TIMES, rtol=RTOL, atol=ATOL)
                    # An unchanged s = beta c t predicts t_out = t_in beta_old/beta_new. It is the
                    # diagnosis of a changed time only; a map that preserves t needs no diagnosis.
                    s_unchanged = TIMES * beta(p0) / beta(p)
                    if changed and not np.allclose(t, s_unchanged, rtol=RTOL, atol=ATOL):
                        sys.exit(f"{name}: output time does not follow the unchanged-s prediction.")
                    print(
                        f"{name}: p_out={p[0]:.16g} t_in={TIMES[3]:.16g} "
                        f"observed t_out={t[3]:.16g} expected t_out={TIMES[3]:.16g}, "
                        f"observed t_out/t_in={t[3] / TIMES[3]:.16g}"
                    )
                    if charge == 0:
                        if changed or not np.array_equal(p, np.full(N, p0)):
                            sys.exit("The zero-charge control changed the momentum or the time.")
                    else:
                        defect |= changed
    print(f"comparison: rtol={RTOL:g}, atol={ATOL:g} s")
    verdict(defect)

    ```

## Provenance

| Role | Model | Effort | Date | Code examined |
|---|---|---|---|---|
| Found | `gpt-6-astra` | xhigh | 2026-10-28 | master (`75a2e1a1a58b`), source with SDDS a60b9bbed7cc, CPU, serial |
| Confirmed | `gpt-6.1-sol` | xhigh | 2026-10-28 | master (`75a2e1a1a58b`), source with SDDS a60b9bbed7cc, CPU, serial |

