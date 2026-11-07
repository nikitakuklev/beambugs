---
title: "COL-47: LRWAKE keeps more turns of wake history than TURNS_TO_KEEP after an end bunch is lost"
hide:
  - toc
---

# COL-47 — LRWAKE keeps more turns of wake history than TURNS_TO_KEEP after an end bunch is lost

| | |
|---|---|
| Code | [Elegant](../index.md) · Collective effects |
| Priority | P2 |
| Severity | Wrong physics |
| Evidence | Reproduced, independently rechecked |
| Last reproduced | 2026-11-05 on Elegant master (`0355dd6`, committed 2026-11-03) |
| Tags | <span class="bb-tag">partly-source-only</span> [?](../../methodology.md#tags) |
| Source | [occupied range, lrwake.c:59–84](https://github.com/rtsoliday/elegant/blob/0355dd6aca12c861971fbf8f3ae78a0ded991734/src/lrwake.c#L59-L84)<br>[growth check, lrwake.c:90–103](https://github.com/rtsoliday/elegant/blob/0355dd6aca12c861971fbf8f3ae78a0ded991734/src/lrwake.c#L90-L103)<br>[caller, lrwake.c:292–300](https://github.com/rtsoliday/elegant/blob/0355dd6aca12c861971fbf8f3ae78a0ded991734/src/lrwake.c#L292-L300)<br>[early return, lrwake.c:532–533](https://github.com/rtsoliday/elegant/blob/0355dd6aca12c861971fbf8f3ae78a0ded991734/src/lrwake.c#L532-L533)<br>[capacity, lrwake.c:681–688](https://github.com/rtsoliday/elegant/blob/0355dd6aca12c861971fbf8f3ae78a0ded991734/src/lrwake.c#L681-L688)<br>[shift and append, lrwake.c:383–396](https://github.com/rtsoliday/elegant/blob/0355dd6aca12c861971fbf8f3ae78a0ded991734/src/lrwake.c#L383-L396)<br>[history sum, lrwake.c:418–424](https://github.com/rtsoliday/elegant/blob/0355dd6aca12c861971fbf8f3ae78a0ded991734/src/lrwake.c#L418-L424)<br>[lrwake.c:442–444](https://github.com/rtsoliday/elegant/blob/0355dd6aca12c861971fbf8f3ae78a0ded991734/src/lrwake.c#L442-L444)<br>[TURNS_TO_KEEP, track_data.c:2584](https://github.com/rtsoliday/elegant/blob/0355dd6aca12c861971fbf8f3ae78a0ded991734/src/track_data.c#L2584)<br>[history buffer, lrwake.tex:22–33](https://github.com/rtsoliday/elegant/blob/0355dd6aca12c861971fbf8f3ae78a0ded991734/doc/lrwake.tex#L22-L33)<br>[bunch IDs, elegant.tex:498–503](https://github.com/rtsoliday/elegant/blob/0355dd6aca12c861971fbf8f3ae78a0ded991734/doc/elegant.tex#L498-L503) |

When an end bunch of a multibunch beam is lost, LRWAKE keeps more turns of wake sources than TURNS_TO_KEEP requests. Its history holds TURNS_TO_KEEP times the initial bunch count but advances on each pass by only the surviving count, so stale turns are never dropped; the correct loss sums the sources of the requested turns, as tracking without a loss confirms. For two 1.000 nC bunches at 511.0 MeV/c in a constant 1.000×10¹² V/C longitudinal wake with TURNS_TO_KEEP = 2, the surviving bunch loses 3.000 keV per pass instead of 1.000 keV from the third pass on, whichever end bunch was lost. With TURNS_TO_KEEP = 1 it keeps losing 1.000 keV per pass instead of nothing.

Run `python COL-47.py` with [_elegant.py](../helpers/_elegant.py) in the same directory.

??? example "Reproducer: COL-47.py"

    [Download COL-47.py](COL-47.py)

    ```python
    """COL-47: LRWAKE keeps more turns of wake history than TURNS_TO_KEEP after an end bunch is lost

    When an end bunch of a multibunch beam is lost, LRWAKE keeps more turns of wake sources than
    TURNS_TO_KEEP requests. Its history holds TURNS_TO_KEEP times the initial bunch count but advances
    on each pass by only the surviving count, so stale turns are never dropped; the correct loss sums
    the sources of the requested turns, as tracking without a loss confirms. For two 1.000 nC bunches
    at 511.0 MeV/c in a constant 1.000×10¹² V/C longitudinal wake with TURNS_TO_KEEP = 2, the
    surviving bunch loses 3.000 keV per pass instead of 1.000 keV from the third pass on, whichever
    end bunch was lost. With TURNS_TO_KEEP = 1 it keeps losing 1.000 keV per pass instead of nothing.

    The script tracks two bunches of eight electrons each through a ring of one LRWAKE with a
    constant longitudinal wake, a WATCH and an aperture that removes the first or the last bunch on
    its first pass. The expected energy loss of the surviving bunch on every pass sums the sources of
    the requested turns; runs without a loss and with a single bunch are the controls, and a queue of
    fixed record capacity that advances by the surviving bunch count predicts the faulty values. Run
    with --mpi to add 3-rank Pelegant.
    """

    import sys

    import numpy as np
    from _elegant import modes, particles, run, verdict

    REST_EV = 510999.06  # SDDS constants.h: electron rest energy, eV
    P0 = 1000.0  # electron momentum, units of m_e c (511.0 MeV/c)
    Q = 1e-9  # C per bunch
    W = 1e12  # V/C: constant wake for every delay of at least 0.1 ns, W(0) = 0
    NP = 8  # macroparticles per bunch, all identical within a bunch
    SPACING = 1e-9  # s, between bunch arrival times
    LOST_X = 1e-3  # m, offset of the bunch the aperture removes
    PASSES = 7
    # One ulp of p = 1000 is 5.809e-8 eV of energy; observed losses are within two ulps of Q W multiples.
    ATOL = 1e-6  # eV

    # (bunches, index of the bunch lost at the aperture on pass 0 or None, TURNS_TO_KEEP)
    CASES = ((1, None, 2), (2, None, 2), (2, 0, 2), (2, 1, 2), (2, 0, 1), (2, 1, 1))


    def table():
        """ASCII SDDS wake: 0 at t = 0, W from 0.1 ns to 1 us (every delay here is 1 ns to 31 ns)."""
        return (
            "SDDS1\n"
            "&column name=t, units=s, type=double &end\n"
            "&column name=wz, units=V/C, type=double &end\n"
            "&data mode=ascii, no_row_counts=1 &end\n"
            f"0 0\n1e-10 {W!r}\n1e-6 {W!r}\n"
        )


    def survivor(bunches, lost):
        """Index of the witness bunch: the survivor, or the last bunch when none is lost."""
        if lost is None:
            return bunches - 1
        return 1 - lost


    def observed_loss(mode, bunches, lost, turns):
        """Energy loss in eV of the witness bunch on each pass, from the WATCH after the wake."""
        ids = np.arange(1, bunches * NP + 1)
        x = np.zeros(bunches * NP)
        if lost is not None:
            x[lost * NP : (lost + 1) * NP] = LOST_X
        beam = {
            "x": x,
            "t": np.repeat(np.arange(bunches) * SPACING, NP),
            "p": np.full(bunches * NP, P0),
            "particleID": ids,
            "IDSlotsPerBunch": NP,
        }
        # RCOL removes the offset bunch after the WATCH; the 3 m drift makes each pass ~10.01 ns.
        lattice = (
            f"Q: CHARGE,TOTAL={bunches * Q!r}\n"
            f'W: LRWAKE,INPUTFILE="wake.sdds",TCOLUMN="t",WZCOLUMN="wz",TURNS_TO_KEEP={turns}\n'
            'OBS: WATCH,FILENAME="watch.sdds",MODE="coordinates"\n'
            f"AP: RCOL,X_MAX={LOST_X / 2!r},Y_MAX=1\n"
            "D: DRIF,L=3\n"
            "BL: LINE=(Q,W,OBS,AP,D)\n"
        )
        commands = (
            "&global_settings echo_namelists=0 &end\n"
            f"&run_setup lattice=lattice.lte,use_beamline=BL,p_central={P0!r},output=out.sdds &end\n"
            f"&run_control n_steps=1,n_passes={PASSES} &end\n"
            "&sdds_beam input=beam.sdds,use_bunched_mode=1,track_pages_separately=0 &end\n"
            "&track &end\n"
        )
        result = run(
            {"lattice.lte": lattice, "run.ele": commands, "wake.sdds": table()},
            mode,
            {"beam.sdds": beam},
            outputs=("watch.sdds",),
        )
        watch = result["watch.sdds"]
        if watch.n_pages != PASSES:
            sys.exit(f"{mode}: {watch.n_pages} WATCH pages instead of {PASSES}")
        witness = survivor(bunches, lost)
        p_in, losses = P0, []
        for n in range(PASSES):
            out = particles(watch, n)
            kept = ids if lost is None or n == 0 else np.delete(ids, np.s_[lost * NP : (lost + 1) * NP])
            if not np.array_equal(out["particleID"], kept):
                sys.exit(f"{mode}: unexpected particle IDs on pass {n}")
            p = out["p"][(out["particleID"] - 1) // NP == witness]
            if np.ptp(p) != 0:
                sys.exit(f"{mode}: identical witnesses received different kicks")
            p_out = p[0]
            # Delta E = mc^2 (gamma_in - gamma_out), written without subtracting two large energies.
            losses.append(REST_EV * (p_in - p_out) * (p_in + p_out) / (np.hypot(p_in, 1) + np.hypot(p_out, 1)))
            p_in = p_out
        return np.array(losses)


    def present(bunches, lost, n):
        """Bunches that reach the wake on pass n (the lost bunch passes it once, on pass 0)."""
        return [b for b in range(bunches) if lost is None or n == 0 or b != lost]


    def expected_loss(bunches, lost, turns):
        """Turn-sum oracle: Q W per source bunch within the TURNS_TO_KEEP retained turns.

        The latest turn fills one of the TURNS_TO_KEEP slots (the manual's slot 1), so the witness on
        pass n sees every bunch of passes n-K+1 .. n-1 and the bunches ahead of it on pass n; its own
        bunch gives W(0) = 0. The no-loss controls fix this convention.
        """
        witness = survivor(bunches, lost)
        counts = []
        for n in range(PASSES):
            prior = sum(len(present(bunches, lost, m)) for m in range(max(0, n - turns + 1), n))
            ahead = sum(1 for b in present(bunches, lost, n) if b < witness)
            counts.append(prior + ahead)
        return Q * W * np.array(counts)


    def queue_prediction(bunches, lost, turns):
        """Fixed-capacity queue: K B0 records shifted by the current bunch count on every pass.

        B0 is the initial bunch count. On each pass the oldest len(current) records are dropped, the
        current bunches are appended, and bunch i of the current pass sums the records before its own.
        """
        capacity = turns * bunches
        history = [0] * capacity  # 1 marks a record holding one bunch of charge Q
        witness = survivor(bunches, lost)
        counts = []
        for n in range(PASSES):
            current = present(bunches, lost, n)
            history = history[len(current) :] + [1] * len(current)
            i = current.index(witness)
            counts.append(sum(history[: capacity - len(current) + i]))
        return Q * W * np.array(counts)


    def fmt(values):
        return "[" + ", ".join(f"{v:.10g}" for v in values + 0.0) + "]"


    def main():
        defect = False
        for mode in modes():
            for bunches, lost, turns in CASES:
                label = (
                    f"{mode} {bunches} bunch{'es' if bunches > 1 else ''}, "
                    f"{'no loss' if lost is None else ('first' if lost == 0 else 'last') + ' bunch lost'}, "
                    f"TURNS_TO_KEEP={turns}"
                )
                observed = observed_loss(mode, bunches, lost, turns)
                expected = expected_loss(bunches, lost, turns)
                print(f"{label}: survivor loss per pass (eV) observed={fmt(observed)}")
                print(f"{label}: expected={fmt(expected)} (Q W per source bunch in the kept turns)")
                agrees = np.allclose(observed, expected, rtol=0, atol=ATOL)
                if lost is None:
                    if not agrees:
                        sys.exit(f"{label}: control disagrees with the turn sum; invalid comparison")
                    continue
                queue = queue_prediction(bunches, lost, turns)
                print(f"{label}: fixed-capacity queue prediction={fmt(queue)}")
                if not agrees and not np.allclose(observed, queue, rtol=0, atol=ATOL):
                    sys.exit(f"{label}: losses follow neither the turn sum nor the fixed-capacity queue")
                defect |= not agrees
        print(f"comparison: rtol=0, atol={ATOL:g} eV")
        verdict(defect)


    if __name__ == "__main__":
        main()

    ```

## Provenance

| Role | Model | Effort | Date | Code examined |
|---|---|---|---|---|
| Found | `gpt-6-astra` | xhigh | 2026-11-05 | master (`0355dd6aca12`), source with SDDS a3f730a14c49, CPU, serial and 3-rank Pelegant |
| Confirmed | `gpt-6.1-sol` | xhigh | 2026-11-05 | master (`0355dd6aca12`), source with SDDS a3f730a14c49, CPU, serial and 3-rank Pelegant |

