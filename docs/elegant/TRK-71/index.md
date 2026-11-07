---
title: "TRK-71: Negative-angle CCBEND and LGBEND reverse EPITCH and EYAW in their π-rotated internal frame"
hide:
  - toc
---

# TRK-71 — Negative-angle CCBEND and LGBEND reverse EPITCH and EYAW in their π-rotated internal frame

| | |
|---|---|
| Code | [Elegant](../index.md) · Tracking |
| Priority | P2 |
| Severity | Wrong physics |
| Evidence | Reproduced, independently rechecked |
| Last reproduced | 2026-11-06 on Elegant commit [`0355dd6`](https://github.com/rtsoliday/elegant/commit/0355dd6aca12), committed 2026-11-03 |
| Tags | <span class="bb-tag">partly-source-only</span> [?](../../methodology.md#tags) |
| Source | [src/ccbend.c:365–377](https://github.com/rtsoliday/elegant/blob/0355dd6aca12c861971fbf8f3ae78a0ded991734/src/ccbend.c#L365-L377)<br>[src/ccbend.c:492–506](https://github.com/rtsoliday/elegant/blob/0355dd6aca12c861971fbf8f3ae78a0ded991734/src/ccbend.c#L492-L506)<br>[src/ccbend.c:526–529](https://github.com/rtsoliday/elegant/blob/0355dd6aca12c861971fbf8f3ae78a0ded991734/src/ccbend.c#L526-L529)<br>[src/ccbend.c:639–642](https://github.com/rtsoliday/elegant/blob/0355dd6aca12c861971fbf8f3ae78a0ded991734/src/ccbend.c#L639-L642)<br>[src/malign_mat.c:516–570](https://github.com/rtsoliday/elegant/blob/0355dd6aca12c861971fbf8f3ae78a0ded991734/src/malign_mat.c#L516-L570)<br>[src/malign_mat.c:1421–1437](https://github.com/rtsoliday/elegant/blob/0355dd6aca12c861971fbf8f3ae78a0ded991734/src/malign_mat.c#L1421-L1437)<br>[src/lgbend.c:323–336](https://github.com/rtsoliday/elegant/blob/0355dd6aca12c861971fbf8f3ae78a0ded991734/src/lgbend.c#L323-L336)<br>[src/lgbend.c:343–357](https://github.com/rtsoliday/elegant/blob/0355dd6aca12c861971fbf8f3ae78a0ded991734/src/lgbend.c#L343-L357)<br>[src/lgbend.c:392–395](https://github.com/rtsoliday/elegant/blob/0355dd6aca12c861971fbf8f3ae78a0ded991734/src/lgbend.c#L392-L395)<br>[src/lgbend.c:538–541](https://github.com/rtsoliday/elegant/blob/0355dd6aca12c861971fbf8f3ae78a0ded991734/src/lgbend.c#L538-L541)<br>[src/csbend.c:1088–1107](https://github.com/rtsoliday/elegant/blob/0355dd6aca12c861971fbf8f3ae78a0ded991734/src/csbend.c#L1088-L1107)<br>[src/csbend.c:1269–1271](https://github.com/rtsoliday/elegant/blob/0355dd6aca12c861971fbf8f3ae78a0ded991734/src/csbend.c#L1269-L1271)<br>[doc/ccbend.tex:89–100](https://github.com/rtsoliday/elegant/blob/0355dd6aca12c861971fbf8f3ae78a0ded991734/doc/ccbend.tex#L89-L100)<br>[src/track_data.c:3822–3824](https://github.com/rtsoliday/elegant/blob/0355dd6aca12c861971fbf8f3ae78a0ded991734/src/track_data.c#L3822-L3824)<br>[doc/malign_method.tex:1–22](https://github.com/rtsoliday/elegant/blob/0355dd6aca12c861971fbf8f3ae78a0ded991734/doc/malign_method.tex#L1-L22)<br>[doc/elegant.tex:12336–12339](https://github.com/rtsoliday/elegant/blob/0355dd6aca12c861971fbf8f3ae78a0ded991734/doc/elegant.tex#L12336-L12339)<br>[src/track_data.c:3857](https://github.com/rtsoliday/elegant/blob/0355dd6aca12c861971fbf8f3ae78a0ded991734/src/track_data.c#L3857-L3857)<br>[doc/lgbend.tex:40–44](https://github.com/rtsoliday/elegant/blob/0355dd6aca12c861971fbf8f3ae78a0ded991734/doc/lgbend.tex#L40-L44)<br>[src/track_data.c:4044–4045](https://github.com/rtsoliday/elegant/blob/0355dd6aca12c861971fbf8f3ae78a0ded991734/src/track_data.c#L4044-L4045) |

With a negative-angle CCBEND and MALIGN_METHOD 1–4, EPITCH and EYAW are applied with the opposite physical sign because they are left unchanged in the internally π-rotated frame. For a 0.6000 m, −0.04000 rad bend with quadrupole gradient K₁ = 0.2000 m⁻² and a 1.000 mrad error, exact body-centered tracking differs from laboratory Lorentz integration by 7.442 µm for pitch and 7.160 µm for yaw, and agrees with the opposite-signed rotation within 10⁻¹⁶. Rotation symmetry about the beam axis independently establishes the sign error. LGBEND, whose placement is always exact and body-centered, reverses both angles in the same way. Positive-angle placement, DX, DY and the documented pitch/yaw-free method 0 controls pass.

Run `python TRK-71.py` with [_elegant.py](../helpers/_elegant.py) and [_ccbend.py](../helpers/_ccbend.py) in the same directory.

??? example "Reproducer: TRK-71.py"

    [Download TRK-71.py](TRK-71.py)

    ```python
    """TRK-71: Negative-angle CCBEND and LGBEND reverse EPITCH and EYAW in their π-rotated internal frame

    With a negative-angle CCBEND and MALIGN_METHOD 1–4, EPITCH and EYAW are applied with the opposite physical sign because they are left unchanged in the internally π-rotated frame. For a 0.6000 m, −0.04000 rad bend with quadrupole gradient K₁ = 0.2000 m⁻² and a 1.000 mrad error, exact body-centered tracking differs from laboratory Lorentz integration by 7.442 µm for pitch and 7.160 µm for yaw, and agrees with the opposite-signed rotation within 10⁻¹⁶. Rotation symmetry about the beam axis independently establishes the sign error. LGBEND, whose placement is always exact and body-centered, reverses both angles in the same way. Positive-angle placement, DX, DY and the documented pitch/yaw-free method 0 controls pass.

    The script tracks three fixed reference-momentum rays through one CCBEND with a 1 mrad EPITCH or EYAW, optimization and edge kicks off, and compares bends of both signs with an independent rigid-face Cartesian integration of the magnet rotated about laboratory axes. For MALIGN_METHOD 1 to 4 it also compares the negative bend with the positive bend rotated by π about the beam axis, which must carry the opposite pitch and yaw. It then shows that native backtracking of a positive bend, whose reversed copy has a negative angle, fails to invert unless the backward pass's angles are negated, and repeats the integration and symmetry checks on a uniform two-segment LGBEND written inline. Method 0, DX, DY and LGBEND ETILT are controls.
    """

    import _elegant as native
    import numpy as np
    from _ccbend import Magnet, close, error, track
    from _elegant import verdict

    P0 = 1500.0  # reference momentum, units of m c (the value _ccbend.track uses)
    THETA = 0.04  # magnitude of the bend angle; arc length 0.6 m and K1 = 0.2 m^-2 from _ccbend
    ERROR = 1e-3  # pitch or yaw, rad
    # On-axis and two off-axis rays at reference momentum: (x, x', y, y', delta).
    RAYS = np.array(
        [[0, 0, 0, 0, 0], [7e-4, 3e-4, -4e-4, 2e-4, 0], [-5e-4, -2e-4, 3e-4, -1e-4, 0]],
        dtype=float,
    )
    # A rotation by pi about the beam axis reverses (x, x', y, y') and keeps delta.  It carries a
    # bend of angle theta onto one of angle -theta (the field B_y(x, y) and B_x(x, y) both change
    # sign at the reflected point) and conjugates the error rotations:
    # Rz(pi) Rx(a) Rz(-pi) = Rx(-a), Rz(pi) Ry(a) Rz(-pi) = Ry(-a), Rz(pi) Rz(c) Rz(-pi) = Rz(c).
    # With laboratory-referenced errors the negative bend therefore obeys
    # F_-(q; pitch a) = FLIP F_+(FLIP q; pitch -a), and likewise for yaw, while roll keeps its sign.
    FLIP = np.diag([-1.0, -1.0, -1.0, -1.0, 1.0])
    ORACLE_TOL = 1e-15  # native transport against the independent integration
    SYMMETRY_TOL = (
        1e-18  # the internal rotation by pi is exact apart from sin(pi) ~ 1.2e-16 cross terms
    )
    INVERSE_TOL = 3e-14  # a correct inverse returns every ray to rounding of 64 sixth-order slices
    DEFECT_MIN = 1e-12  # a discrepancy this large is not rounding
    ANGLES = (("EPITCH", "pitch"), ("EYAW", "yaw"))


    def ccbend(rays, angle, **params):
        return track(rays, ANGLE=angle, **params)[0]


    def rotated(run, angle, rays, **params):
        """FLIP F(FLIP q): transport of the bend rotated by pi about the beam axis."""
        return run(rays @ FLIP, angle, **params) @ FLIP


    def lgbend_configuration(theta, length=0.6):
        """Uniform-field LGBEND of one rectangular magnet, written as two equal segments.

        Circular geometry with zero gradients and no fringe terms: each segment spans half the
        chord; the reference enters at x = 0 with slope theta/2 to the chord, reaches the sagitta
        rho (1 - cos(theta/2)) at the shared face, which is perpendicular to the chord, and leaves
        at x = 0.  The vertex is where the entry and exit tangents meet.
        """
        rho = length / theta
        chord = 2 * rho * np.sin(theta / 2)
        half = theta / 2
        first = {
            "LONGIT_L": chord / 2, "ANGLE": half, "ENTRY_X": 0, "ENTRY_ANGLE": half,
            "EXIT_X": rho * (1 - np.cos(half)), "EXIT_ANGLE": 0, "K1": 0, "K2": 0, "PREDRIFT": 0,
            "ZEntry": 0, "XEntry": 0, "ZExit": chord, "XExit": 0, "ZVertex": chord / 2,
            "XVertex": chord / 2 * np.tan(half),
        }  # fmt: skip
        second = {
            "LONGIT_L": chord / 2, "ANGLE": half, "EXIT_X": 0, "EXIT_ANGLE": -half, "K1": 0, "K2": 0,
            "POSTDRIFT": 0,
        }  # fmt: skip
        pages = "".join(
            f"{len(page)}\n" + "".join(f"{k} {v:.17g}\n" for k, v in page.items())
            for page in (first, second)
        )
        return (
            "SDDS1\n&column name=ParameterName,type=string &end\n"
            "&column name=ParameterValue,type=double &end\n&data mode=ascii &end\n" + pages
        )


    def lgbend(rays, angle, **params):
        """Track rays through one uniform LGBEND, fresh process, optimization and edges off."""
        q = np.array(rays, dtype=float, ndmin=2)
        beam = dict(zip(("x", "xp", "y", "yp"), q[:, :4].T, strict=True))
        beam.update(t=np.zeros(len(q)), p=P0 * (1 + q[:, 4]), particleID=np.arange(1, len(q) + 1))
        extra = "".join(f",{k}={v:.17g}" for k, v in params.items())
        element = (
            'B: LGBEND,CONFIGURATION="bend.sdds",N_SLICES=64,INTEGRATION_ORDER=6,'
            f"OPTIMIZE_FSE=0,EDGE_ORDER=0{extra}\nBL: LINE=(B)\n"
        )
        commands = (
            "&run_setup lattice=lattice.lte, use_beamline=BL, "
            f"p_central={P0:.17g}, default_order=1, output=out.sdds &end\n"
            "&run_control n_steps=1 &end\n&sdds_beam input=beam.sdds &end\n&track soft_failure=0 &end\n"
        )
        files = {"lattice.lte": element, "run.ele": commands, "bend.sdds": lgbend_configuration(angle)}
        data = native.particles(native.run(files, "serial", {"beam.sdds": beam})["out.sdds"])
        if not np.array_equal(data["particleID"], beam["particleID"]):
            raise RuntimeError("Particle IDs changed or particles were lost")
        out = np.column_stack([data[n] for n in ("x", "xp", "y", "yp", "p")])
        out[:, 4] = out[:, 4] / P0 - 1
        return out


    def placement(run, label, k1, failures, **method):
        """Both bend signs against the rigid-face integration with laboratory-axis rotations."""
        for name, kw in ANGLES:
            positive = run(RAYS, THETA, **{name: ERROR}, **method)
            control = close(
                positive, Magnet(angle=THETA, k1=k1, **{kw: ERROR}).forward(RAYS), ORACLE_TOL,
                f"{label} positive {name}",
            )  # fmt: skip
            negative = run(RAYS, -THETA, **{name: ERROR}, **method)
            lab = error(negative, Magnet(angle=-THETA, k1=k1, **{kw: ERROR}).forward(RAYS))
            opposite = close(
                negative, Magnet(angle=-THETA, k1=k1, **{kw: -ERROR}).forward(RAYS), ORACLE_TOL,
                f"{label} negative {name} against the opposite rotation",
            )  # fmt: skip
            failures.append(lab > DEFECT_MIN)
            print(
                f"{label} {name}=1e-3: ANGLE=-0.04 max error vs laboratory {kw} integration={lab!r} "
                f"expected=0; vs opposite {kw} -1e-3={opposite:.4g}; "
                f"control ANGLE=+0.04 vs integration={control:.4g}"
            )


    def main():
        failures = []

        # CCBEND, exact body-centred placement.
        placement(ccbend, "CCBEND K1=0.2 MALIGN_METHOD=4", 0.2, failures, MALIGN_METHOD=4)

        # Rotation symmetry for every method that applies pitch and yaw; the linearized methods 1
        # and 2 are checked only this way, since their approximation is not the exact map.
        for method in (1, 2, 3, 4):
            for name, _ in ANGLES:
                negative = ccbend(RAYS, -THETA, **{name: ERROR}, MALIGN_METHOD=method)
                law = error(
                    negative, rotated(ccbend, THETA, RAYS, **{name: -ERROR}, MALIGN_METHOD=method)
                )
                same = close(
                    negative, rotated(ccbend, THETA, RAYS, **{name: ERROR}, MALIGN_METHOD=method),
                    SYMMETRY_TOL, f"Method {method} {name} rotated same-sign bend",
                )  # fmt: skip
                failures.append(law > DEFECT_MIN)
                print(
                    f"CCBEND MALIGN_METHOD={method} {name}=1e-3 ANGLE=-0.04: max error vs rotated "
                    f"positive bend with {name}=-1e-3={law!r} expected=0 (rotation by pi about the "
                    f"beam axis); vs rotated bend with {name}=+1e-3={same:.4g}"
                )

        # Native backtracking of a positive bend tracks a reversed copy with negative angle, so the
        # backward pass takes the rotated frame; negating its angles restores the inverse.
        for name, _ in ANGLES:
            params = dict(MALIGN_METHOD=4, **{name: ERROR})
            forward = ccbend(RAYS, THETA, **params)
            back = track(forward, reverse=True, ANGLE=THETA, **params)[0]
            roundtrip = error(back, RAYS)
            compensated = track(forward, reverse=True, ANGLE=THETA, MALIGN_METHOD=4, **{name: -ERROR})[
                0
            ]
            fixed = close(compensated, RAYS, INVERSE_TOL, f"Backtracking {name} compensated")
            failures.append(roundtrip > INVERSE_TOL)
            print(
                f"CCBEND backtracking ANGLE=+0.04 MALIGN_METHOD=4 {name}=1e-3: roundtrip max error="
                f"{roundtrip!r} expected=0 (inverse transport); backward pass with {name}=-1e-3 "
                f"max error={fixed:.4g}"
            )

        # Controls: method 0 ignores pitch and yaw; DX and DY are carried into the rotated frame.
        for name, _ in ANGLES:
            ignored = close(
                ccbend(RAYS, -THETA, **{name: ERROR}, MALIGN_METHOD=0), ccbend(RAYS, -THETA, MALIGN_METHOD=0),
                0.0, f"Method 0 {name}",
            )  # fmt: skip
            print(
                f"control CCBEND MALIGN_METHOD=0 {name}=1e-3 ANGLE=-0.04: vs no error={ignored:.4g} expected=0"
            )
        for name, d in (("DX", (1e-3, 0, 0)), ("DY", (0, 1e-3, 0))):
            offset = close(
                ccbend(RAYS, -THETA, **{name: 1e-3}, MALIGN_METHOD=4),
                Magnet(angle=-THETA, d=d).forward(RAYS), ORACLE_TOL, f"Control {name}",
            )  # fmt: skip
            print(
                f"control CCBEND MALIGN_METHOD=4 {name}=1e-3 ANGLE=-0.04: vs laboratory-offset "
                f"integration={offset:.4g}"
            )

        # LGBEND always uses exact body-centred placement.
        placement(lgbend, "LGBEND uniform", 0.0, failures)
        for name, _ in ANGLES:
            negative = lgbend(RAYS, -THETA, **{name: ERROR})
            law = error(negative, rotated(lgbend, THETA, RAYS, **{name: -ERROR}))
            same = close(
                negative, rotated(lgbend, THETA, RAYS, **{name: ERROR}), SYMMETRY_TOL,
                f"LGBEND {name} rotated same-sign bend",
            )  # fmt: skip
            failures.append(law > DEFECT_MIN)
            print(
                f"LGBEND uniform {name}=1e-3 ANGLE=-0.04: max error vs rotated positive bend with "
                f"{name}=-1e-3={law!r} expected=0; vs rotated bend with {name}=+1e-3={same:.4g}"
            )
        # A roll commutes with the rotation by pi, so ETILT keeps its sign under the same law.
        roll = close(
            lgbend(RAYS, -THETA, ETILT=ERROR), rotated(lgbend, THETA, RAYS, ETILT=ERROR), SYMMETRY_TOL,
            "LGBEND ETILT",
        )  # fmt: skip
        print(
            f"control LGBEND uniform ETILT=1e-3 ANGLE=-0.04: vs rotated bend with ETILT=+1e-3={roll:.4g}"
        )
        verdict(all(failures))


    if __name__ == "__main__":
        main()

    ```

## Provenance

| Role | Model | Effort | Date | Code examined |
|---|---|---|---|---|
| Found | `claude-opus-5-5` | high | 2026-11-06 | master (`0355dd6aca12`), source with SDDS a3f730a14c49, CPU, double, serial, one thread |
| Confirmed | `gpt-6.1-sol` | xhigh | 2026-11-06 | master (`0355dd6aca12`), source with SDDS a3f730a14c49, CPU, double, serial, one thread |
| Confirmed<br><small>rotation-symmetry law, LGBEND and backtracking of positive bends</small> | `claude-opus-5-5` | high | 2026-11-06 | master (`0355dd6aca12`), source with SDDS a3f730a14c49, CPU, double, serial, one thread |

