---
title: "TRK-06: RFDF drifts after a kick at the speed the particle had before it"
hide:
  - toc
---

# TRK-06 — RFDF drifts after a kick at the speed the particle had before it

| | |
|---|---|
| Code | [Elegant](../index.md) · Tracking |
| Priority | P3 |
| Severity | Wrong physics |
| Evidence | Reproduced, independently rechecked |
| Last reproduced | 2026-11-04 on Elegant master (`0355dd6`, head build) |
| Tags | <span class="bb-tag">partly-source-only</span> [?](../../methodology.md#tags) |
| Source | [initial beta, track_rf.c:166](https://github.com/rtsoliday/elegant/blob/0355dd6aca12c861971fbf8f3ae78a0ded991734/src/track_rf.c#L166)<br>[drift timing, track_rf.c:173–189](https://github.com/rtsoliday/elegant/blob/0355dd6aca12c861971fbf8f3ae78a0ded991734/src/track_rf.c#L173-L189)<br>[kick and momentum update, track_rf.c:198–215](https://github.com/rtsoliday/elegant/blob/0355dd6aca12c861971fbf8f3ae78a0ded991734/src/track_rf.c#L198-L215)<br>[final beta, track_rf.c:217–222](https://github.com/rtsoliday/elegant/blob/0355dd6aca12c861971fbf8f3ae78a0ded991734/src/track_rf.c#L217-L222)<br>[RFDF kick model, rfdf.tex:15–33](https://github.com/rtsoliday/elegant/blob/0355dd6aca12c861971fbf8f3ae78a0ded991734/doc/rfdf.tex#L15-L33)<br>[CUDA RFDF kernel, gpu_cuda_runtime.cu:3639–3684](https://github.com/rtsoliday/elegant/blob/0355dd6aca12c861971fbf8f3ae78a0ded991734/src/gpu/gpu_cuda_runtime.cu#L3639-L3684) |

RFDF computes particle speed at entrance and reuses it for drift timing after kicks change the momentum. For one midpoint kick, vacuum kinematics require the final half drift to use the outgoing speed, as a phase-matched drift–thin-deflector–drift control confirms. In magnetic-deflection mode, a 0.2 m, 300 kV, 100 MHz element with an electron entering at x = 0.04 m and momentum p = 2 mₑc gives 747.7 ps instead of 746.0 ps, 1.671 ps late. At p = 100 mₑc the error is 15.42 as.

Run `python TRK-06.py` with [_elegant.py](../helpers/_elegant.py) in the same directory.

??? example "Reproducer: TRK-06.py"

    [Download TRK-06.py](TRK-06.py)

    ```python
    """TRK-06: RFDF drifts after a kick at the speed the particle had before it

    RFDF computes particle speed at entrance and reuses it for drift timing after kicks change the
    momentum. For one midpoint kick, vacuum kinematics require the final half drift to use the outgoing
    speed, as a phase-matched drift–thin-deflector–drift control confirms. In magnetic-deflection mode, a 0.2 m, 300 kV, 100 MHz element with an electron entering at x = 0.04 m and
    momentum p = 2 mₑc gives 747.7 ps instead of 746.0 ps, 1.671 ps late. At p = 100 mₑc the error is
    15.42 as.

    The script tracks identical particles through a one-kick RFDF and compares the exit time with
    the exact time of flight over the two half drifts, built from the entrance speed and the exit speed
    and slopes that the same run reports. The kick itself is checked against a drift, a thin RFDF and
    a drift, with the finite element's phase shifted so both kicks occur at the same RF phase. Controls:
    the thin-element split reproduces the exact time, and an RFDF with zero voltage, which does not
    change the speed, agrees.
    """

    import sys

    import numpy as np
    from _elegant import modes, particles, run, verdict

    C = 299792458.0  # m/s, c_mks of SDDS
    LENGTH, FREQUENCY, VOLTAGE, X0 = 0.2, 1e8, 3e5, 0.04
    KICK_PHASE = 70.0  # degrees, RF phase at the kick plane
    COPIES = 12  # identical particles, so that each MPI worker tracks some
    RTOL = 1e-12


    def beta(p):
        return p / np.sqrt(1 + p * p)


    def track(mode, p0, lattice):
        """Exit (x, xp, y, yp, p, t) of the identical particles; they start at x = X0, t = 0."""
        commands = (
            f"&run_setup lattice=lattice.lte, use_beamline=BL, p_central={p0!r}, "
            "output=out.sdds &end\n"
            "&run_control n_steps=1, n_passes=1 &end\n"
            "&sdds_beam input=beam.sdds &end\n"
            "&track &end\n"
        )
        beam = {"x": np.full(COPIES, X0), "p": np.full(COPIES, p0)}
        out = particles(
            run({"lattice.lte": lattice, "run.ele": commands}, mode, {"beam.sdds": beam})["out.sdds"]
        )
        if len(out["x"]) != COPIES:
            sys.exit(f"{mode}: {COPIES - len(out['x'])} particles lost")
        columns = np.column_stack([out["x"], out["xp"], out["y"], out["yp"], out["p"], out["t"]])
        if not (columns == columns[:1]).all():
            sys.exit(f"{mode}: identical particles left the element differently")
        return columns[0]


    def rfdf(length, phase, voltage):
        return (
            f"R: RFDF,L={length!r},VOLTAGE={voltage!r},FREQUENCY={FREQUENCY!r},PHASE={float(phase)!r},"
            "N_KICKS=1,MAGNETIC_DEFLECTION=1\n"
        )


    def exact_time(p_in, exit_state):
        """Exit time of a ray that drifts L/2 at entrance speed, is kicked, and drifts L/2 again.

        A particle entering with zero transverse slopes covers L/2 in L/(2 beta_in c). After the kick
        it has the exit momentum and slopes, and covers the slanted path (L/2) sqrt(1 + xp^2 + yp^2)
        at beta_out c. The kick itself is taken from the run being checked.
        """
        _, xp, _, yp, p_out, _ = exit_state
        return LENGTH / (2 * C) * (1 / beta(p_in) + np.sqrt(1 + xp**2 + yp**2) / beta(p_out))


    def compare(label, observed, expected):
        """Print one case; return True when observed and expected disagree."""
        bad = not np.isclose(observed, expected, rtol=RTOL, atol=0)
        print(
            f"{label}: observed={observed:.16g} expected={expected:.16g} observed-expected={observed - expected:.4g}"
        )
        return bad


    defect = False
    print("exit time (s) of the RFDF; expected from the two half drifts at the entrance and exit speed")
    for mode in modes():
        for p0 in (2.0, 100.0):
            # The finite element kicks at its middle plane, where the RF phase seen by the particle is
            # PHASE + omega*(t_part - t_light) with t_part = L/(2 beta c) and t_light = L/(2 c). Lowering
            # PHASE by that term (in degrees) makes the kick phase KICK_PHASE, that of the thin element.
            shift = 360 * FREQUENCY * LENGTH / (2 * C) * (1 / beta(p0) - 1)
            finite = track(mode, p0, rfdf(LENGTH, KICK_PHASE - shift, VOLTAGE) + "BL: LINE=(R)\n")
            thin = rfdf(0, KICK_PHASE, VOLTAGE)
            half = f"D: EDRIFT,L={LENGTH / 2!r}\n"
            split = track(mode, p0, half + thin + "BL: LINE=(D,R,D)\n")
            # The kick is the same in both: transverse state and momentum agree.
            if not np.allclose(finite[:5], split[:5], rtol=0, atol=3e-14):
                sys.exit(
                    f"{mode} p={p0:g}: the finite and split kicks differ, the comparison is not valid."
                )
            if compare(f"{mode} p={p0:g} control, split thin RFDF", split[5], exact_time(p0, split)):
                sys.exit(
                    "The split element disagrees with the exact time: the comparison is not valid."
                )
            zero = track(mode, p0, rfdf(LENGTH, KICK_PHASE - shift, 0) + "BL: LINE=(R)\n")
            if compare(f"{mode} p={p0:g} control, VOLTAGE=0", zero[5], exact_time(p0, zero)):
                sys.exit(
                    "The zero-voltage RFDF disagrees with the exact time: the comparison is not valid."
                )
            defect |= compare(f"{mode} p={p0:g} finite RFDF", finite[5], exact_time(p0, finite))
    print(f"comparison: rtol={RTOL:g}")
    verdict(defect)

    ```

## Provenance

| Role | Model | Effort | Date | Code examined |
|---|---|---|---|---|
| Found | `gpt-6-astra` | xhigh | 2026-10-15 | 2026.4.0 (`d177ff5347c7`), source reading, not executed for this claim |
| Confirmed | `gpt-6-astra` | xhigh | 2026-10-15 | 2026.4.0 (`d177ff5347c7`), source with SDDS 3fa650ee9a1c, CPU, serial |
| Confirmed | `gpt-6.1-sol` | xhigh | 2026-10-15 | 2026.4.0 (`d177ff5347c7`), source with SDDS 3fa650ee9a1c, CPU, serial |
| Confirmed | `gpt-6-astra` | high | 2026-10-25 | master (`75a2e1a1a58b`), source with SDDS a60b9bbed7cc, CPU, serial and 3-rank Pelegant |
| Confirmed | `gpt-6.1-sol` | high | 2026-10-25 | master (`75a2e1a1a58b`), source with SDDS a60b9bbed7cc, CPU, serial and 3-rank Pelegant |

