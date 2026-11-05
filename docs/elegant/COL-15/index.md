---
title: "COL-15: FTRFMODE ignores RIGID_UNTIL_PASS"
hide:
  - toc
---

# COL-15 — FTRFMODE ignores RIGID_UNTIL_PASS

| | |
|---|---|
| Code | [Elegant](../index.md) · Collective effects |
| Priority | P2 |
| Severity | Wrong physics |
| Evidence | Reproduced, independently rechecked |
| Last reproduced | 2026-11-04 on Elegant master (`0355dd6`) |
| Source | [RIGID_UNTIL_PASS definition, track_data.c:2278](https://github.com/rtsoliday/elegant/blob/0355dd6aca12c861971fbf8f3ae78a0ded991734/src/track_data.c#L2278)<br>[kick application, ftrfmode.c:536–564](https://github.com/rtsoliday/elegant/blob/0355dd6aca12c861971fbf8f3ae78a0ded991734/src/ftrfmode.c#L536-L564)<br>[FTRFMODE dispatch, do_tracking.c:2418](https://github.com/rtsoliday/elegant/blob/0355dd6aca12c861971fbf8f3ae78a0ded991734/src/do_tracking.c#L2418)<br>[binned gate, trfmode.c:512](https://github.com/rtsoliday/elegant/blob/0355dd6aca12c861971fbf8f3ae78a0ded991734/src/trfmode.c#L512)<br>[binless gate, trfmode.c:898](https://github.com/rtsoliday/elegant/blob/0355dd6aca12c861971fbf8f3ae78a0ded991734/src/trfmode.c#L898) |

FTRFMODE, a dipole resonator with modes read from a file, ignores RIGID_UNTIL_PASS, which should prevent beam kicks until the specified pass. On the first pass, settings 1 and 9 produce full kicks identical to setting 0, while an equivalent single-mode TRFMODE leaves the beam unchanged. For a 16-electron bunch with momentum 750 mₑc, charge 0.7000 nC and horizontal offset 8.000 mm, an 800.0 MHz mode with quality factor Q = 12 and shunt impedance Rₛ = 6×10⁷ Ω/m produces momentum changes up to 0.02984 mₑc and horizontal slope changes up to 1.557×10⁻⁴ where none are allowed.

Download these shared helpers into the same directory as the reproducer scripts:

- [Download _elegant.py](../helpers/_elegant.py) · [View source](../helpers/index.md#_elegant)

??? example "Reproducer: COL-15.py"

    [Download COL-15.py](COL-15.py)

    ```python
    """COL-15: FTRFMODE ignores RIGID_UNTIL_PASS

    FTRFMODE, a dipole resonator with modes read from a file, ignores RIGID_UNTIL_PASS, which should
    prevent beam kicks until the specified pass. On the first pass, settings 1 and 9 produce full kicks
    identical to setting 0, while an equivalent single-mode TRFMODE leaves the beam unchanged. For a
    16-electron bunch with momentum 750 mₑc, charge 0.7000 nC and horizontal offset 8.000 mm, an
    800.0 MHz mode with quality factor Q = 12 and shunt impedance Rₛ = 6×10⁷ Ω/m produces momentum
    changes up to 0.02984 mₑc and horizontal slope changes up to 1.557×10⁻⁴ where none are allowed.

    The script tracks the same bunch through one FTRFMODE and through the equivalent single-mode TRFMODE
    for one pass with RIGID_UNTIL_PASS = 0, 1 and 9, and records the largest momentum and slope
    change. A threshold of 9 excludes a kick on the first pass whether passes are counted from zero or
    one. The controls are TRFMODE, which must kick only for 0, and agreement of the two elements'
    kicks at 0, which shows that the mode is excited identically.
    """

    import sys

    import numpy as np
    from _elegant import modes, particles, run, verdict

    FREQ, QUALITY, RS = 8e8, 12.0, 6e7  # Hz, dimensionless, Ohm/m
    CHARGE, P0, N = 7e-10, 750.0, 16  # C, units of m_e c
    SPACING, DT, BINS = 1.25e-11, 5e-13, 1024  # s between particles, bin size, bin count
    X0 = 8e-3  # m
    # Mode table of FTRFMODE: one row, columns Frequency, Q, ShuntImpedance.
    MODES = (
        "SDDS1\n"
        '&column name=Frequency, type=double, units="Hz" &end\n'
        '&column name=Q, type=double, units="" &end\n'
        '&column name=ShuntImpedance, type=double, units="Ohm/m" &end\n'
        "&data mode=ascii &end\n"
        f"1\n{FREQ!r} {QUALITY!r} {RS!r}\n"
    )


    def track(mode, kind, rigid):
        """Output particle columns after one pass through the resonator."""
        params = f"BIN_SIZE={DT!r},N_BINS={BINS},RIGID_UNTIL_PASS={rigid}"
        if kind == "TRFMODE":
            params += f',FREQ={FREQ!r},Q={QUALITY!r},RS={RS!r},PLANE="x",BINLESS=0'
        else:
            params += ',FILENAME="modes.sdds",YFACTOR=0'  # horizontal mode only, like TRFMODE
        lattice = f"C: CHARGE,TOTAL={CHARGE!r}\nK: {kind},{params}\nBL: LINE=(C,K)\n"
        commands = (
            "&global_settings echo_namelists=0 &end\n"
            f"&run_setup lattice=lattice.lte,use_beamline=BL,p_central={P0!r},output=out.sdds &end\n"
            "&run_control n_steps=1,n_passes=1 &end\n"
            "&sdds_beam input=beam.sdds &end\n"
            "&track soft_failure=0 &end\n"
        )
        beam = {"x": np.full(N, X0), "t": np.arange(N) * SPACING, "p": np.full(N, P0)}
        result = run(
            {"lattice.lte": lattice, "run.ele": commands, "modes.sdds": MODES},
            mode,
            {"beam.sdds": beam},
        )
        out = particles(result["out.sdds"])
        if len(out["p"]) != N or not all(np.all(np.isfinite(v)) for v in out.values()):
            sys.exit(f"{mode} {kind} rigid={rigid}: particles lost or non-finite output")
        return out


    def fmt(values):
        return "[" + ", ".join(f"{v:.16g}" for v in values) + "]"


    defect = False
    for mode in modes():
        outputs = {}
        for kind in ("TRFMODE", "FTRFMODE"):
            for rigid in (0, 1, 9):
                out = outputs[kind, rigid] = track(mode, kind, rigid)
                dp = np.max(np.abs(out["p"] - P0))
                dxp = np.max(np.abs(out["xp"]))
                # Rigid for the first traversal: no coordinate may change. The incoming slopes are
                # zero, so the outgoing slope and momentum change are the kick. Rigid 0 has no
                # expectation; the kick there is the mode's excitation.
                print(
                    f"{mode} {kind} RIGID_UNTIL_PASS={rigid}: observed max|dp|,max|xp|={fmt([dp, dxp])} "
                    f"expected={fmt([0, 0]) if rigid else 'nonzero'}"
                )
                if rigid and kind == "TRFMODE" and (dp > 1e-15 or dxp > 1e-15):
                    sys.exit("The TRFMODE control kicks the beam while rigid.")
                if rigid and kind == "FTRFMODE":
                    defect |= dp > 1e-15 or dxp > 1e-15
                if not rigid and (dp == 0 or dxp == 0):
                    sys.exit(f"{kind} does not excite the beam with RIGID_UNTIL_PASS=0.")
        # Matched-mode control: the same single mode gives the same kick in both elements.
        for name in ("p", "xp"):
            gap = np.max(np.abs(outputs["FTRFMODE", 0][name] - outputs["TRFMODE", 0][name]))
            print(f"{mode} RIGID_UNTIL_PASS=0: max|FTRFMODE - TRFMODE| of {name} = {gap:.3g}")
            if gap > 1e-15:
                sys.exit(f"TRFMODE and FTRFMODE kicks of {name} disagree at RIGID_UNTIL_PASS=0.")
        for rigid in (1, 9):
            same = all(
                np.array_equal(outputs["FTRFMODE", 0][k], outputs["FTRFMODE", rigid][k])
                for k in outputs["FTRFMODE", 0]
            )
            print(f"{mode} FTRFMODE RIGID_UNTIL_PASS={rigid}: all columns bit-identical to 0: {same}")
    verdict(defect)

    ```

## Provenance

| Role | Model | Effort | Date | Code examined |
|---|---|---|---|---|
| Found | `gpt-6-astra` | xhigh | 2026-10-28 | master (`75a2e1a1a58b`), source with SDDS a60b9bbed7cc, CPU, serial |
| Confirmed | `gpt-6.1-sol` | xhigh | 2026-10-28 | master (`75a2e1a1a58b`), source with SDDS a60b9bbed7cc, CPU, serial |

