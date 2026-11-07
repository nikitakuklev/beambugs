---
title: "SCT-24: Pelegant's distributed IBSCATTER integration uses the wrong reference momenta in accelerating lines"
hide:
  - toc
---

# SCT-24 — Pelegant's distributed IBSCATTER integration uses the wrong reference momenta in accelerating lines

| | |
|---|---|
| Code | [Elegant](../index.md) · Scattering and pressure |
| Priority | P2 |
| Failure | Wrong physics |
| Evidence | Reproduced, independently rechecked |
| Last reproduced | 2026-11-05 on Elegant commit [`0355dd6`](https://github.com/rtsoliday/elegant/commit/0355dd6aca12), committed 2026-11-03 |
| Tags | <span class="bb-tag">partly-source-only</span> [?](../../methodology.md#tags) |
| Source | [`init_IBS` samples, ibs_tracking.c:844–847](https://github.com/rtsoliday/elegant/blob/0355dd6aca12c861971fbf8f3ae78a0ded991734/src/ibs_tracking.c#L844-L847)<br>[`track_IBS` distributed call, ibs_tracking.c:347–370](https://github.com/rtsoliday/elegant/blob/0355dd6aca12c861971fbf8f3ae78a0ded991734/src/ibs_tracking.c#L347-L370)<br>[exit normalization, zibs.c:47–53](https://github.com/rtsoliday/elegant/blob/0355dd6aca12c861971fbf8f3ae78a0ded991734/src/zibs.c#L47-L53)<br>[per-sample energy, zibs.c:101–108](https://github.com/rtsoliday/elegant/blob/0355dd6aca12c861971fbf8f3ae78a0ded991734/src/zibs.c#L101-L108)<br>[`track_IBS` full-line call, ibs_tracking.c:386–392](https://github.com/rtsoliday/elegant/blob/0355dd6aca12c861971fbf8f3ae78a0ded991734/src/ibs_tracking.c#L386-L392)<br>[ibs_tracking.c:465](https://github.com/rtsoliday/elegant/blob/0355dd6aca12c861971fbf8f3ae78a0ded991734/src/ibs_tracking.c#L465)<br>[`inflateEmittanceZ`, ibs_tracking.c:733–740](https://github.com/rtsoliday/elegant/blob/0355dd6aca12c861971fbf8f3ae78a0ded991734/src/ibs_tracking.c#L733-L740)<br>[IBSCATTER parameters, track_data.c:2522](https://github.com/rtsoliday/elegant/blob/0355dd6aca12c861971fbf8f3ae78a0ded991734/src/track_data.c#L2522)<br>[IBSCATTER element flags, track_data.c:4237](https://github.com/rtsoliday/elegant/blob/0355dd6aca12c861971fbf8f3ae78a0ded991734/src/track_data.c#L4237)<br>[IBSCATTER, ibscatter.tex:6–10](https://github.com/rtsoliday/elegant/blob/0355dd6aca12c861971fbf8f3ae78a0ded991734/doc/ibscatter.tex#L6-L10)<br>[ibscatter.tex:17–20](https://github.com/rtsoliday/elegant/blob/0355dd6aca12c861971fbf8f3ae78a0ded991734/doc/ibscatter.tex#L17-L20) |

When Pelegant tracks an accelerating line through IBSCATTER on more than three ranks with PARALLEL_INTEGRATION=1 (the default), each worker integrates the intrabeam-scattering (IBS) growth rates over its part of the line with that part's optics but the reference momenta of the line's first, equally long stretch, and scales the emittances measured at the element as if measured at that stretch's end. The correct rates integrate the whole line, as serial Elegant and PARALLEL_INTEGRATION=0 do; at constant energy all agree. For a 500 pC electron bunch accelerated from 61.32 to 85.32 MeV over 24.00 m, five ranks raise the rms momentum spread by the fraction 9.208×10⁻⁴ instead of 4.219×10⁻⁴, 2.183 times too much; the transverse increments are 2.462 and 2.384 times too large.

Run `python SCT-24.py` with [_elegant.py](../helpers/_elegant.py) in the same directory.

??? example "Reproducer: SCT-24.py"

    [Download SCT-24.py](SCT-24.py)

    ```python
    """SCT-24: Pelegant's distributed IBSCATTER integration uses the wrong reference momenta in accelerating lines

    When Pelegant tracks an accelerating line through IBSCATTER on more than three ranks with
    PARALLEL_INTEGRATION=1 (the default), each worker integrates the intrabeam-scattering (IBS) growth
    rates over its part of the line with that part's optics but the reference momenta of the line's
    first, equally long stretch, and scales the emittances measured at the element as if measured at
    that stretch's end. The correct rates integrate the whole line, as serial Elegant and
    PARALLEL_INTEGRATION=0 do; at constant energy all agree. For a 500 pC electron bunch accelerated
    from 61.32 to 85.32 MeV over 24.00 m, five ranks raise the rms momentum spread by the fraction
    9.208×10⁻⁴ instead of 4.219×10⁻⁴, 2.183 times too much; the transverse increments are 2.462 and
    2.384 times too large.

    Setup: always runs 5-rank Pelegant (Open MPI `mpirun` on `PATH`, oversubscribed) beside serial
    Elegant, because the distributed IBSCATTER integration runs only on more than three ranks.

    The script tracks a 1024-particle, 500 pC electron beam through 48 cells of a 0.3 m drift and a 0.2
    m, 0.5 MV DC cavity to a single-slice, smooth, linac-mode IBSCATTER, in serial elegant and in 5-rank
    Pelegant. Each increment of the rms xp, yp and p is measured against a FACTOR=0 run; the expected
    longitudinal increment is the flight time times the trapezoid average of serial elegant's local
    rates, which splitting the line cannot change. The controls are 5-rank Pelegant with
    PARALLEL_INTEGRATION=0 and the same line at zero cavity voltage, which must both equal serial
    elegant to rounding.
    """

    import sys

    import numpy as np
    from _elegant import modes, particles, run, verdict

    # The distributed branch of track_IBS runs only with more than three ranks and more than five
    # lattice samples per worker; the shared runner's three ranks take the full-line call instead.
    RANKS = 5
    WORKERS = RANKS - 1

    C = 299792458.0  # m/s
    B0, CHARGE, COUNT = 120.0, 5e-10, 1024  # entrance beta*gamma, C, particles
    EX, EY, SD, SZ = 5e-8, 2e-8, 3e-4, 1e-3  # m, m, rms dp/p, m
    BX, AX, BY, AY = 2.0, 0.4, 3.0, -0.2  # Twiss parameters at the start, m and 1
    CELLS, VOLT = 48, 5e5  # cells of a 0.3 m drift and a 0.2 m cavity; V per cavity
    RTOL, ATOL = 1e-9, 1e-12  # rms increments of equal-integral runs agree to rounding
    MIN_ERROR = 1e-6  # relative change of the longitudinal increment counted as a defect


    def beam():
        """Exact Twiss-matched moments: orthonormal columns scaled by sqrt(COUNT) have zero mean and
        unit covariance, so the sample emittances, spread and length equal the requested ones."""
        rng = np.random.default_rng(2405)
        s = rng.normal(size=(COUNT, 6))
        s -= s.mean(axis=0)
        s = np.linalg.qr(s)[0] * np.sqrt(COUNT)
        velocity = C * B0 / np.hypot(1, B0)
        return {
            "x": np.sqrt(EX * BX) * s[:, 0],
            "xp": np.sqrt(EX / BX) * (s[:, 1] - AX * s[:, 0]),
            "y": np.sqrt(EY * BY) * s[:, 2],
            "yp": np.sqrt(EY / BY) * (s[:, 3] - AY * s[:, 2]),
            "t": SZ / velocity * s[:, 4],
            "p": B0 * (1 + SD * s[:, 5]),
            "particleID": np.arange(1, COUNT + 1),
        }


    def track(columns, mode, voltage, factor=1.0, parallel=1):
        """Final particles and, for serial runs, the IBSCATTER diagnostic file.

        The line is CELLS cells of a drift and a crest-phased DC cavity (FREQ=0, PHASE=90) that
        raises the reference momentum (CHANGE_P0=1), then one linac-mode IBSCATTER with one slice
        and the smooth update. PARALLEL_INTEGRATION=1 is the element's default.
        """
        lattice = (
            f"Q: CHARGE,TOTAL={CHARGE!r}\n"
            "D: DRIF,L=0.3\n"
            f"A: RFCA,L=0.2,VOLT={voltage!r},PHASE=90,FREQ=0,CHANGE_P0=1,END1_FOCUS=0,END2_FOCUS=0\n"
            "I: IBSCATTER,ISRING=0,BUNCHED_BEAM_MODE=0,SMOOTH=1,NSLICE=1,"
            f'FACTOR={factor!r},PARALLEL_INTEGRATION={parallel},FILENAME="ibs.sdds"\n'
            "CELL: LINE=(D,A)\n"
            f"BL: LINE=(Q,{CELLS}*CELL,I)\n"
        )
        commands = (
            f"&run_setup lattice=lattice.lte, use_beamline=BL, p_central={B0!r}, "
            "output=out.sdds &end\n"
            "&run_control n_steps=1, n_passes=1 &end\n"
            f"&twiss_output matched=0, beta_x={BX!r}, alpha_x={AX!r}, beta_y={BY!r}, "
            f"alpha_y={AY!r} &end\n"
            "&sdds_beam input=beam.sdds &end\n"
            "&track &end\n"
        )
        # FACTOR=0 writes no diagnostic file; MPI rate columns are not used (they are zero there).
        outputs = ("out.sdds", "ibs.sdds") if mode == "serial" and factor else ("out.sdds",)
        result = run(
            {"lattice.lte": lattice, "run.ele": commands},
            mode,
            {"beam.sdds": columns},
            outputs=outputs,
            ranks=RANKS,
        )
        out = particles(result["out.sdds"])
        if len(out["p"]) != COUNT or not np.array_equal(out["particleID"], columns["particleID"]):
            sys.exit(f"{mode} V={voltage:g}: the output does not hold the {COUNT} input particles")
        return out, result.get("ibs.sdds")


    def rms(out):
        return np.array([np.std(out[name]) for name in ("xp", "yp", "p")])


    def fmt(values):
        return "[" + ", ".join(f"{v:.10g}" for v in np.ravel(values)) + "]"


    columns = beam()
    (mpi,) = modes(mpi_only=True)
    defect = False
    print("fractional rms increments of (xp, yp, p) caused by IBSCATTER, FACTOR=1")
    for voltage in (VOLT, 0.0):
        # FACTOR=0 leaves the beam unscattered; its rms values are the reference of each increment,
        # so the cavities' own change of the momentum spread cancels.
        baseline = rms(track(columns, "serial", voltage, factor=0.0)[0])
        out, ibs = track(columns, "serial", voltage)
        serial = rms(out) / baseline - 1
        s = np.asarray(ibs[0, "s"], dtype=float)
        if len(s) // WORKERS <= 5:
            sys.exit(f"{len(s)} lattice samples: too few for the distributed branch")
        # Independent expectation for the longitudinal plane. The growth rate applied to a single
        # slice is the trapezoid average of the local rates over the samples s,
        # Gz = int dIBSRatez ds / (s_end - s_start); splitting the s range into partitions does not
        # change it. The smooth update multiplies every centred momentum deviation by 1 + dt Gz
        # (dt the flight time to the element), so the rms momentum grows by exactly dt Gz.
        rate = np.trapezoid(np.asarray(ibs[0, "dIBSRatez"], dtype=float), s) / (s[-1] - s[0])
        dt = float(np.ravel(ibs["dt"].data)[0])
        expected_z = dt * rate
        print(
            f"V={voltage:g} V per cavity, {len(s)} samples: serial observed={fmt(serial)}; "
            f"expected z=dt*Gz={expected_z:.10g} (dt={dt:.10g} s, Gz={rate:.10g} 1/s)"
        )
        if not np.isclose(serial[2], expected_z, rtol=RTOL, atol=ATOL):
            sys.exit("Serial elegant disagrees with its own integrated rate: comparison invalid")
        if voltage:
            # Control: the same ranks with PARALLEL_INTEGRATION=0 integrate the whole line on every
            # worker; this must reproduce serial elegant.
            control = rms(track(columns, mpi, voltage, parallel=0)[0]) / baseline - 1
            print(f"V={voltage:g}: {WORKERS + 1}-rank, PARALLEL_INTEGRATION=0: observed={fmt(control)}")
            if not np.allclose(control, serial, rtol=RTOL, atol=ATOL):
                sys.exit("Replicated MPI integration differs from serial: comparison invalid")
        distributed = rms(track(columns, mpi, voltage)[0]) / baseline - 1
        ratio = distributed / serial
        print(
            f"V={voltage:g}: {WORKERS + 1}-rank, PARALLEL_INTEGRATION=1: observed={fmt(distributed)} "
            f"expected={fmt(serial)} ratio={fmt(ratio)}"
        )
        if not voltage:
            # Constant-energy control through the same distributed branch.
            if not np.allclose(distributed, serial, rtol=RTOL, atol=ATOL):
                sys.exit("Constant-energy distributed integration differs from serial: invalid")
        else:
            defect = abs(ratio[2] - 1) > MIN_ERROR
    print(
        f"comparison: equal-integral runs rtol={RTOL:g}, atol={ATOL:g}; "
        f"defect if the z ratio differs from 1 by more than {MIN_ERROR:g}"
    )
    verdict(defect)

    ```

## Provenance

| Role | Model | Effort | Date | Code examined |
|---|---|---|---|---|
| Found | `gpt-6-astra` | xhigh | 2026-11-05 | master (`0355dd6aca12`), source with SDDS a3f730a14c49, CPU, serial and 5-rank Pelegant |
| Confirmed | `gpt-6.1-sol` | xhigh | 2026-11-05 | master (`0355dd6aca12`), source with SDDS a3f730a14c49, CPU, serial and 3-, 5- and 7-rank Pelegant |

