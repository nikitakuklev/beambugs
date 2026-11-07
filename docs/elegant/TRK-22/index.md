---
title: "TRK-22: Downstream RFCA omits the accumulated clock phase for non-harmonic frequencies"
hide:
  - toc
---

# TRK-22 — Downstream RFCA omits the accumulated clock phase for non-harmonic frequencies

| | |
|---|---|
| Code | [Elegant](../index.md) · Tracking |
| Priority | P2 |
| Severity | Wrong physics |
| Evidence | Reproduced, independently rechecked |
| Last reproduced | 2026-11-04 on Elegant master (`0355dd6`, head build) |
| Tags | <span class="bb-tag">partly-source-only</span> [?](../../methodology.md#tags) |
| Source | [`trackRfCavityWithWakes`, simple_rfca.c:731](https://github.com/rtsoliday/elegant/blob/0355dd6aca12c861971fbf8f3ae78a0ded991734/src/simple_rfca.c#L731)<br>[matrix expressions, simple_rfca.c:1008](https://github.com/rtsoliday/elegant/blob/0355dd6aca12c861971fbf8f3ae78a0ded991734/src/simple_rfca.c#L1008)<br>[`trackingClockPhase`, simple_rfca.c:122](https://github.com/rtsoliday/elegant/blob/0355dd6aca12c861971fbf8f3ae78a0ded991734/src/simple_rfca.c#L122)<br>[manual, rfca.tex:34–41](https://github.com/rtsoliday/elegant/blob/0355dd6aca12c861971fbf8f3ae78a0ded991734/doc/rfca.tex#L34-L41)<br>[`checkRFCAChangeTConflicts`, do_tracking.c:6334–6364](https://github.com/rtsoliday/elegant/blob/0355dd6aca12c861971fbf8f3ae78a0ded991734/src/do_tracking.c#L6334-L6364) |

An RFCA with CHANGE_T = 1 removes whole periods of its own RF, a time T, from every particle's arrival time and banks it in a clock; the manual promises that cavities at non-harmonic frequencies f receive a compensating phase 2πfT. A downstream RFCA does not, so its energy kick is evaluated at the reduced time, and no warning is printed. With a 0.7 GHz cavity removing T = 15.71 ns and a 0.77 GHz, 180 kV cavity after it, an electron of p = 3.5 mₑc leaves with 3.605 mₑc instead of 3.382, a 36.00° phase error; harmonic frequencies are correct. With automatic fiducials a later pass is also wrong (15.63 mₑc instead of 15.00 on pass 3).

Run `python TRK-22.py` with [_elegant.py](../helpers/_elegant.py) in the same directory.

??? example "Reproducer: TRK-22.py"

    [Download TRK-22.py](TRK-22.py)

    ```python
    """TRK-22: Downstream RFCA omits the accumulated clock phase for non-harmonic frequencies

    An RFCA with CHANGE_T = 1 removes whole periods of its own RF, a time T, from every particle's
    arrival time and banks it in a clock; the manual promises that cavities at non-harmonic frequencies
    f receive a compensating phase 2πfT. A downstream RFCA does not, so its energy kick is evaluated at
    the reduced time, and no warning is printed. With a 0.7 GHz cavity removing T = 15.71 ns and a 0.77
    GHz, 180 kV cavity after it, an electron of p = 3.5 mₑc leaves with 3.605 mₑc instead of 3.382, a
    36.00° phase error; harmonic frequencies are correct. With automatic fiducials a later pass is also
    wrong (15.63 mₑc instead of 15.00 on pass 3).

    The script sends 24 electrons or positrons through a zero-length RFCA with and without CHANGE_T and
    then through a second RFCA at 1 to 2.7 times its frequency, and compares the exit momentum with the
    energy-kick formula in laboratory time. Controls are the unwrapped run, the harmonic ratios 1 and 2,
    and the wrapped run with PHASE raised by 2πfT. It also tracks several passes with an automatic
    fiducial against an absolute-time recurrence and reports whether a warning is printed.
    """

    import math
    import sys

    import numpy as np
    from _elegant import modes, particles, run, verdict

    # SDDS constants.h, as Elegant uses them.
    C, ME_MEV = 299792458.0, 0.51099906
    COUNT = 24
    P_ATOL = 6e-14  # one-impulse cases
    CHAIN_ATOL = 3e-13  # recurrence over several passes
    RATIOS = (1.0, 1.1, 1.45, 2.0, 2.7)  # downstream frequency / upstream frequency


    def impulse(p, time, voltage, frequency, phase, tref=0.0, sign=1):
        """Momentum after a thin RF kick, p in m_e c.

        gamma' = gamma + sign*V/(m c^2/e) sin(2 pi f (t - t_ref) + phi), with PHASE = 90 degrees at
        crest; sign = -1 for positrons. All times are laboratory times.
        """
        gamma = np.hypot(p, 1) + sign * voltage / (1e6 * ME_MEV) * np.sin(
            2 * np.pi * frequency * (time - tref) + np.deg2rad(phase)
        )
        return np.sqrt((gamma - 1) * (gamma + 1))


    def track(mode, species, p0, t_in, lattice, passes=1, watch=False):
        commands = (
            f"&change_particle name={species} &end\n"
            f"&run_setup lattice=lattice.lte, use_beamline=BL, p_central={p0!r}, output=out.sdds, "
            "default_order=2, random_number_seed=94821 &end\n"
            f"&run_control n_steps=1, n_passes={passes} &end\n"
            "&sdds_beam input=beam.sdds &end\n"
            "&track soft_failure=0 &end\n"
        )
        files = {"lattice.lte": lattice, "run.ele": commands}
        outputs = ("out.sdds", "first.sdds") if watch else ("out.sdds",)
        beam = {"beam.sdds": {"t": np.broadcast_to(t_in, (COUNT,)).copy(), "p": np.full(COUNT, p0)}}
        result = run(files, mode, beam, outputs=outputs)
        final = []
        for name in outputs:
            out = particles(result[name])
            if len(out["p"]) != COUNT or not all(np.all(np.isfinite(v)) for v in out.values()):
                sys.exit(f"{mode} {species}: particles were lost or not finite")
            # laboratory time is the stored time plus the clock register tOffset
            t_offset = float(result[name]["tOffset"][0])
            t_lab = out["t"] + t_offset
            if not np.isfinite(t_offset) or not np.all(np.isfinite(t_lab)):
                sys.exit(f"{mode} {species}: the clock or laboratory times are not finite")
            final.append((out["p"], t_lab))
        return final, result["log"]


    def nonharmonic(mode):
        """Upstream cavity with CHANGE_T, then a zero-length cavity at f2 = ratio*f1; True on defect."""
        p0, f1, v1, phase1 = 3.5, 0.7e9, 35000.0, 13.0
        v2, phase2, tref2 = 180000.0, -37.0, 2e-9
        removed = 11 / f1  # the wrapped upstream cavity removes eleven RF periods of f1
        times = removed + 0.17e-9 + np.linspace(-12e-12, 12e-12, COUNT)
        defect, warned = False, None
        for species, sign in (("electron", 1), ("positron", -1)):
            first_p = impulse(p0, times, v1, f1, phase1, sign=sign)
            for ratio in RATIOS:
                f2 = ratio * f1
                expected = impulse(first_p, times, v2, f2, phase2, tref2, sign)
                # Reducing every time by T shifts the phase of the cavity at f2 by 2 pi f2 T
                # (mod 2 pi); adding it to PHASE undoes the reduction.
                clock_phase = 360 * math.remainder(f2 * removed, 1)
                results = {}
                for name, wrap, extra in (
                    ("plain", 0, 0.0),
                    ("wrapped", 1, 0.0),
                    ("compensated", 1, clock_phase),
                ):
                    lattice = (
                        f"A: RFCA,L=0,VOLT={v1!r},FREQ={f1!r},PHASE={phase1!r},T_REFERENCE=0,"
                        f"CHANGE_T={wrap},N_KICKS=1\n"
                        'W: WATCH,FILENAME="first.sdds",MODE=coordinates\n'
                        f"B: RFCA,L=0,VOLT={v2!r},FREQ={f2!r},PHASE={phase2 + extra!r},"
                        f"T_REFERENCE={tref2!r},N_KICKS=1\n"
                        "BL: LINE=(A,W,B)\n"
                    )
                    (final, first), log = track(mode, species, p0, times, lattice, watch=True)
                    if np.any(np.abs(first[0] - first_p) > 3e-14) or np.any(
                        np.abs(final[1] - times) > 3e-23
                    ):
                        sys.exit(f"{mode} {species} ratio={ratio}: the upstream cavity is wrong")
                    results[name] = final[0]
                    if name == "wrapped" and ratio == 1.1 and species == "electron":
                        warned = "CHANGE_T=1 conflicts" in log
                label = f"{mode} {species} f2/f1={ratio:g} f2={f2:g}"
                for name in ("plain", "compensated"):
                    if np.any(np.abs(results[name] - expected) > P_ATOL):
                        sys.exit(f"{label}: the {name} run disagrees with the energy oracle")
                wrapped = results["wrapped"]
                print(
                    f"{label}: observed=[{float(wrapped[0])!r}] expected=[{float(expected[0])!r}] "
                    f"clock phase={clock_phase:.4g} deg"
                )
                bad = bool(np.any(np.abs(wrapped - expected) > P_ATOL))
                if ratio == 1.0 or ratio == 2.0:
                    if bad:
                        sys.exit(f"{label}: the harmonic control is not invariant")
                    continue
                # The wrapped result equals the impulse at the reduced time t - T.
                reduced = impulse(first_p, times - removed, v2, f2, phase2, tref2, sign)
                if np.any(np.abs(wrapped - reduced) > P_ATOL):
                    sys.exit(f"{label}: the error is not explained by the reduced time")
                defect |= bad
        # The code's own check for elements that do not handle CHANGE_T lists no RFCA, so the
        # wrapped non-harmonic run gives no warning. This line is information, not a verdict.
        print(f"{mode} wrapped f2/f1=1.1 run prints the CHANGE_T conflict warning: observed=[{warned}]")
        return defect


    def multipass(mode):
        """Cavity at f1 with CHANGE_T, drift, then a downstream cavity with an automatic fiducial."""
        p0, f1, f2 = 15.0, 0.8e9, 1.08e9
        t_in = 7 / f1 + 0.21e-9
        length = 2.5e-9 * C * p0 / math.hypot(p0, 1)  # a drift of 2.5 ns at the entrance momentum
        defect = False
        for passes in (1, 3, 5):
            # Absolute-time recurrence: the downstream phase is fixed at the first arrival time and
            # later passes advance by the drift at the current velocity.
            p, t, tref = p0, t_in, None
            for _ in range(passes):
                p = impulse(p, t, 12000.0, f1, 0.0)
                t += length / C * np.hypot(p, 1) / p
                if tref is None:
                    tref = t
                p = impulse(p, t, 110000.0, f2, 61.0, tref)
            observed = []
            for wrap in (0, 1):
                lattice = (
                    f"A: RFCA,L=0,VOLT=12000,FREQ={f1!r},PHASE=0,T_REFERENCE=0,CHANGE_T={wrap},N_KICKS=1\n"
                    f"D: EDRIFT,L={length!r}\n"
                    f"B: RFCA,L=0,VOLT=110000,FREQ={f2!r},PHASE=61,N_KICKS=1,FIDUCIAL=tmean\n"
                    "BL: LINE=(A,D,B)\n"
                )
                ((final, _),), _ = track(mode, "electron", p0, t_in, lattice, passes)
                observed.append(float(final[0]))
            label = f"{mode} automatic fiducial, passes={passes}"
            print(f"{label}, unwrapped: observed=[{observed[0]!r}] expected=[{float(p)!r}]")
            print(f"{label}, wrapped: observed=[{observed[1]!r}] expected=[{float(p)!r}]")
            if abs(observed[0] - p) > CHAIN_ATOL:
                sys.exit(f"{label}: the unwrapped run disagrees with the recurrence")
            bad = abs(observed[1] - p) > CHAIN_ATOL
            if passes == 1:
                if bad:
                    sys.exit(f"{label}: the first pass is not invariant")
                continue
            defect |= bad
        return defect


    def main():
        defect = False
        for mode in modes():
            defect |= nonharmonic(mode)
            defect |= multipass(mode)
        print(f"comparison: atol one impulse={P_ATOL:g}, recurrence={CHAIN_ATOL:g}")
        verdict(defect)


    main()

    ```

## Provenance

| Role | Model | Effort | Date | Code examined |
|---|---|---|---|---|
| Found | `gpt-6-astra` | xhigh | 2026-10-28 | master (`75a2e1a1a58b`), source with SDDS a60b9bbed7cc, CPU, serial |
| Confirmed | `gpt-6.1-sol` | xhigh | 2026-10-28 | master (`75a2e1a1a58b`), source with SDDS a60b9bbed7cc, CPU, serial |

