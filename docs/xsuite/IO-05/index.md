---
title: "IO-05: MAD-X bend roll errors are imported as design tilts"
hide:
  - toc
---

# IO-05 — MAD-X bend roll errors are imported as design tilts

| | |
|---|---|
| Code | [Xsuite](../index.md) · Lattices, particles and I/O |
| Priority | P1 |
| Severity | Wrong physics |
| Evidence | Probe-pinned |
| Last reproduced | 2026-10-26 on xtrack main (`3cd8bfc`, head build) |
| Source | [`Alignment`, mad_loader.py:403–420](https://github.com/xsuite/xtrack/blob/ad5e808cdc5b9a881eac2f41e32a0b7d11cc7028/xtrack/mad_loader.py#L403-L420)<br>[`make_composite_element`, lines 723–729](https://github.com/xsuite/xtrack/blob/ad5e808cdc5b9a881eac2f41e32a0b7d11cc7028/xtrack/mad_loader.py#L723-L729) |

Xtrack's MAD-X importer (via cpymad, the Python interface to MAD-X) adds a bend's `EALIGN` roll `DPSI` to its design tilt in `rot_s_rad`, rotating the reference bending plane instead of applying an alignment roll in the unchanged design frame. A pure sector-bend roll must produce vertical steering, as MAD-X and an Xtrack control using `rot_s_rad_no_frame` establish. For a 1 m `SBEND` of angle 0.001 rad with `DPSI` = 0.01 rad, zero design tilt and beam-direction flag `bv = 1`, the exit vertical momentum `py` is below 8×10⁻²¹ instead of −1.000×10⁻⁵, and the exit `y` is 3.253×10⁻²¹ m instead of −5.000×10⁻⁶ m. MAD-X sector-bend roll-error studies lose essentially all this vertical steering.

Run `python IO-05.py` with [_common.py](../helpers/_common.py) in the same directory.

??? example "Reproducer: IO-05.py"

    [Download IO-05.py](IO-05.py)

    ```python
    """IO-05: MAD-X bend roll errors are imported as design tilts

    Xtrack's MAD-X importer (via cpymad, the Python interface to MAD-X) adds a bend's `EALIGN` roll
    `DPSI` to its design tilt in `rot_s_rad`, rotating the reference bending plane instead of applying
    an alignment roll in the unchanged design frame. A pure sector-bend roll must produce vertical
    steering, as MAD-X and an Xtrack control using `rot_s_rad_no_frame` establish. For a 1 m `SBEND` of
    angle 0.001 rad with `DPSI` = 0.01 rad, zero design tilt and beam-direction flag `bv = 1`, the exit
    vertical momentum `py` is below 8×10⁻²¹ instead of −1.000×10⁻⁵, and the exit `y` is 3.253×10⁻²¹ m
    instead of −5.000×10⁻⁶ m. MAD-X sector-bend roll-error studies lose essentially all this vertical
    steering.

    The script imports a 1 m MAD-X `SBEND` of angle 0.001 or 0.0005 rad with `EALIGN` `DPSI` of 0 or
    ±0.01 rad and tracks a 1 GeV/c proton from zero coordinates. It compares the exit `x`, `px`, `y` and
    `py` with an open MAD-X Twiss started from zero coordinates. The zero-roll import and the same
    imported bend with the roll moved to `rot_s_rad_no_frame` must agree with MAD-X; the leading-order
    vertical kick −angle·sin(`DPSI`) is a looser control.
    """

    import importlib.metadata

    import numpy as np
    import xtrack as xt
    from cpymad.madx import Madx

    RTOL, ATOL = 2e-12, 2e-15


    def evaluate():
        cases = {}
        for angle in (0.001, 0.0005):
            for roll in (0.0, 0.01, -0.01):
                with Madx(stdout=False) as mad:
                    mad.input(f"""
                        b:sbend,l=1,angle={angle};
                        seq:sequence,l=1; b,at=0.5; endsequence;
                        beam,particle=proton,pc=1; use,sequence=seq;
                        select,flag=error,pattern=b; ealign,dpsi={roll};
                    """)
                    twiss = mad.twiss(betx=1, bety=1, x=0, px=0, y=0, py=0)
                    expected = np.array([twiss.x[-1], twiss.px[-1], twiss.y[-1], twiss.py[-1]])
                    line = xt.Line.from_madx_sequence(
                        mad.sequence.seq,
                        allow_thick=True,
                        apply_madx_errors=True,
                    )
                    seed = xt.Particles(p0c=1e9, mass0=mad.sequence.seq.beam.mass * 1e9)
                    observed = seed.copy()
                    line.track(observed)
                    assert observed.state[0] == 1
                    output = np.array([observed.x[0], observed.px[0], observed.y[0], observed.py[0]])
                    # A correctly specified physical roll, rather than a design-frame
                    # tilt, agrees with MAD-X to this tighter absolute tolerance.
                    line["b"].rot_s_rad = 0
                    line["b"].rot_s_rad_no_frame = roll
                    physical_roll = seed.copy()
                    line.track(physical_roll)
                    physical_output = np.array(
                        [
                            physical_roll.x[0],
                            physical_roll.px[0],
                            physical_roll.y[0],
                            physical_roll.py[0],
                        ]
                    )
                    np.testing.assert_allclose(physical_output, expected, rtol=RTOL, atol=ATOL)
                    if roll == 0:
                        np.testing.assert_allclose(output, expected, rtol=RTOL, atol=ATOL)
                    else:
                        # Independent leading-order bending check, used only as a
                        # control; the pinned reference is the full MAD-X result.
                        np.testing.assert_allclose(
                            expected[3], -angle * np.sin(roll), rtol=2e-7, atol=0
                        )
                        cases[f"angle={angle}, dpsi={roll}, exit x/px/y/py"] = output, expected
        return cases


    if __name__ == "__main__":
        from _common import prepare, report

        prepare()
        print(f"cpymad {importlib.metadata.version('cpymad')}")
        with Madx(stdout=False) as mad:
            print(f"MAD-X {mad.version}")
        print(
            "Expected: the imported EALIGN roll preserves the MAD-X steering orbit in the unchanged design frame."
        )
        print(
            "Zero-roll, opposite-roll, two bend angles and explicitly specified physical-roll controls are checked."
        )
        report(evaluate(), rtol=RTOL, atol=ATOL)

    ```

## Provenance

| Role | Model | Effort | Date | Code examined |
|---|---|---|---|---|
| Found | `gpt-6-astra` | xhigh | 2026-10-26 | xtrack 0.118.2 (`ad5e808cdc5b`), installed packages, CPU, serial |
| Confirmed | `gpt-6.1-sol` | low | 2026-10-26 | xtrack 0.118.2 (`ad5e808cdc5b`), installed packages, CPU, serial |

