---
title: "TRK-69: CCBEND backtracking reverses its automatic centering offset"
hide:
  - toc
---

# TRK-69 — CCBEND backtracking reverses its automatic centering offset

| | |
|---|---|
| Code | [Elegant](../index.md) · Tracking |
| Priority | P2 |
| Severity | Wrong physics |
| Evidence | Reproduced, independently rechecked |
| Last reproduced | 2026-11-06 on Elegant commit [`0355dd6`](https://github.com/rtsoliday/elegant/commit/0355dd6aca12), committed 2026-11-03 |
| Tags | <span class="bb-tag">known-limitation</span> <span class="bb-tag">partly-source-only</span> [?](../../methodology.md#tags) |
| Source | [src/track_data.c:3850–3851](https://github.com/rtsoliday/elegant/blob/0355dd6aca12c861971fbf8f3ae78a0ded991734/src/track_data.c#L3850-L3851)<br>[src/ccbend.c:256–266](https://github.com/rtsoliday/elegant/blob/0355dd6aca12c861971fbf8f3ae78a0ded991734/src/ccbend.c#L256-L266)<br>[src/ccbend.c:283–285](https://github.com/rtsoliday/elegant/blob/0355dd6aca12c861971fbf8f3ae78a0ded991734/src/ccbend.c#L283-L285)<br>[src/get_beamline.c:734–742](https://github.com/rtsoliday/elegant/blob/0355dd6aca12c861971fbf8f3ae78a0ded991734/src/get_beamline.c#L734-L742)<br>[src/get_beamline.c:2147–2150](https://github.com/rtsoliday/elegant/blob/0355dd6aca12c861971fbf8f3ae78a0ded991734/src/get_beamline.c#L2147-L2150)<br>[src/ccbend.c:365–377](https://github.com/rtsoliday/elegant/blob/0355dd6aca12c861971fbf8f3ae78a0ded991734/src/ccbend.c#L365-L377)<br>[src/ccbend.c:492–506](https://github.com/rtsoliday/elegant/blob/0355dd6aca12c861971fbf8f3ae78a0ded991734/src/ccbend.c#L492-L506)<br>[src/ccbend.c:530–531](https://github.com/rtsoliday/elegant/blob/0355dd6aca12c861971fbf8f3ae78a0ded991734/src/ccbend.c#L530-L531)<br>[src/ccbend.c:637–638](https://github.com/rtsoliday/elegant/blob/0355dd6aca12c861971fbf8f3ae78a0ded991734/src/ccbend.c#L637-L638)<br>[doc/ccbend.tex:17–34](https://github.com/rtsoliday/elegant/blob/0355dd6aca12c861971fbf8f3ae78a0ded991734/doc/ccbend.tex#L17-L34)<br>[src/track_data.c:4298](https://github.com/rtsoliday/elegant/blob/0355dd6aca12c861971fbf8f3ae78a0ded991734/src/track_data.c#L4298-L4298)<br>[doc/elegant.tex:7283–7288](https://github.com/rtsoliday/elegant/blob/0355dd6aca12c861971fbf8f3ae78a0ded991734/doc/elegant.tex#L7283-L7288)<br>[doc/elegant.tex:12569–12570](https://github.com/rtsoliday/elegant/blob/0355dd6aca12c861971fbf8f3ae78a0ded991734/doc/elegant.tex#L12569-L12570) |

With automatic trajectory centering enabled, as it is by default, CCBEND backtracking applies the fitted transverse offsets without transforming them into its reflected internal frame. In a 0.6000 m, 0.04000 rad bend with normalized gradient K₁ = 0.2000 m⁻², an initially on-axis particle tracked forward and then backward returns at x = −107.3 µm and slope x′ = 355.7 µrad instead of zero. Independent Cartesian field integration identifies reversal through the opposite physical centering offset. Freezing the same forward fit as an explicit magnet displacement restores the inverse; field-strength optimization alone and uniform-dipole controls pass.

Run `python TRK-69.py` with [_ccbend.py](../helpers/_ccbend.py) and [_elegant.py](../helpers/_elegant.py) in the same directory.

??? example "Reproducer: TRK-69.py"

    [Download TRK-69.py](TRK-69.py)

    ```python
    """TRK-69: CCBEND backtracking reverses its automatic centering offset

    With automatic trajectory centering enabled, as it is by default, CCBEND backtracking applies the fitted transverse offsets without transforming them into its reflected internal frame. In a 0.6000 m, 0.04000 rad bend with normalized gradient K₁ = 0.2000 m⁻², an initially on-axis particle tracked forward and then backward returns at x = −107.3 µm and slope x′ = 355.7 µrad instead of zero. Independent Cartesian field integration identifies reversal through the opposite physical centering offset. Freezing the same forward fit as an explicit magnet displacement restores the inverse; field-strength optimization alone and uniform-dipole controls pass.

    The script tracks three fixed rays forward and then natively backward through one CCBEND with the default strength and centering fits and compares the round trip with the input. It freezes the reported fit (FSE, dx) as an explicit magnet with optimization off: displaced by +dx the frozen magnet reproduces the forward pass and reverses correctly, displaced by −dx it reproduces the faulty backward pass. With edge kicks off, an independent rigid-face Cartesian integration of the fitted magnet gives the forward pass for both bend signs and, inverted through the opposite offset, the native backward pass; strength-only fitting and a uniform dipole are controls.
    """

    from _ccbend import DEFAULT, RAYS, Magnet, close, error, roundtrip, track
    from _elegant import verdict

    # Element defaults for the fits, fringe model, edge order and reference correction.
    DEFAULTS = dict(
        OPTIMIZE_FSE=DEFAULT,
        OPTIMIZE_DX=DEFAULT,
        FRINGEMODEL=DEFAULT,
        EDGE_ORDER=DEFAULT,
        REFERENCE_CORRECTION=DEFAULT,
    )
    INVERSE_TOL = 3e-14  # a correct inverse returns every ray to rounding of 64 sixth-order slices
    ORACLE_TOL = 1e-15  # native transport against the independent integration


    def main():
        failures = []

        # Principal case: default fits and fringes.  Both passes report the same fit.
        fwd, back, (fits_f, fits_b) = roundtrip(DEFAULTS)
        fse, dx = fits_f[0]
        if fits_b != fits_f:
            raise RuntimeError(f"Forward and backward fits differ: {fits_f} {fits_b}")
        observed = back[0, :4]
        failures.append(error(back, RAYS) > INVERSE_TOL)
        # The fitted magnet frozen as FSE=fse and an explicit horizontal displacement DX=dx with the
        # fits off; explicit DX is carried into the reflected frame (ccbend.c tilt rotation).
        frozen = dict(DEFAULTS, OPTIMIZE_FSE=0, OPTIMIZE_DX=0, FSE=fse)
        same = close(track(RAYS, DX=dx, **frozen)[0], fwd, 1e-16, "Frozen forward")
        _, frozen_back, _ = roundtrip(dict(frozen, DX=dx))
        frozen_inverse = close(frozen_back, RAYS, INVERSE_TOL, "Frozen magnet inverse")
        opposite = close(
            track(fwd, reverse=True, DX=-dx, **frozen)[0], back, 1e-16, "Opposite-offset law"
        )
        print(
            f"default fits: FSE={fse:.16g} dx={dx:.16g} m in both directions; roundtrip (x,x',y,y')="
            f"{observed.tolist()} expected=[0.0, 0.0, 0.0, 0.0] (inverse transport); frozen DX=+dx magnet: "
            f"forward vs fitted={same:.4g}, roundtrip max error={frozen_inverse:.4g}; native backward vs "
            f"frozen DX=-dx backward={opposite:.4g}"
        )

        # Independent integration with edge kicks off (model 0 keeps its centering objective).
        # A positive bend's fit in the unreflected frame displaces the magnet by +dx; a negative
        # bend fits in its reflected frame, so the same internal dx is the mirror offset -dx in
        # the laboratory.  The backward pass of either uses the opposite laboratory offset.
        fits_on = dict(OPTIMIZE_FSE=1, OPTIMIZE_DX=1)
        for angle in (0.04, -0.04):
            fwd, back, (fits_f, _) = roundtrip(dict(fits_on, ANGLE=angle))
            fse, dx = fits_f[0]
            lab = dx if angle > 0 else -dx
            forward_error = close(
                fwd,
                Magnet(angle=angle, fse=fse, d=(lab, 0, 0)).forward(RAYS),
                ORACLE_TOL,
                f"Oracle forward ANGLE={angle}",
            )
            wrong = Magnet(angle=angle, fse=fse, d=(-lab, 0, 0)).inverse(fwd)
            backward_error = close(
                back, wrong, ORACLE_TOL, f"Oracle opposite-offset backward ANGLE={angle}"
            )
            roundtrip_error = error(back, RAYS)
            failures.append(roundtrip_error > INVERSE_TOL)
            print(
                f"edge kicks off ANGLE={angle}: fitted laboratory offset={lab:.16g} m; roundtrip (x,x')="
                f"{back[0, :2].tolist()} max error={roundtrip_error:.4g} expected=0 (inverse transport); "
                f"forward vs integration={forward_error:.4g}; backward vs integrated inverse with offset "
                f"{-lab:.4g} m={backward_error:.4g}"
            )

        # Centering-only fit.
        _, back, (fits_f, _) = roundtrip(dict(DEFAULTS, OPTIMIZE_FSE=0))
        roundtrip_error = error(back, RAYS)
        failures.append(roundtrip_error > INVERSE_TOL)
        print(
            f"OPTIMIZE_FSE=0 OPTIMIZE_DX=1: dx={fits_f[0][1]:.16g} m; roundtrip (x,x')={back[0, :2].tolist()} "
            f"max error={roundtrip_error:.4g} expected=0 (inverse transport)"
        )

        # Controls: strength-only fit, and a uniform dipole whose fitted offset leaves the field unchanged.
        for label, params in (
            ("OPTIMIZE_FSE=1 OPTIMIZE_DX=0", dict(DEFAULTS, OPTIMIZE_DX=0)),
            ("uniform dipole K1=0, default fits", dict(DEFAULTS, K1=0.0)),
        ):
            _, back, (fits_f, _) = roundtrip(params)
            control = close(back, RAYS, INVERSE_TOL, f"Control {label}")
            print(
                f"control {label}: fit (FSE, dx)={fits_f[0]}; roundtrip max error={control:.4g} expected=0"
            )
        verdict(any(failures))


    if __name__ == "__main__":
        main()

    ```

## Provenance

| Role | Model | Effort | Date | Code examined |
|---|---|---|---|---|
| Found | `claude-opus-5-5` | high | 2026-11-06 | master (`0355dd6aca12`), source with SDDS a3f730a14c49, CPU, double, serial, one thread |
| Confirmed | `gpt-6.1-sol` | xhigh | 2026-11-06 | master (`0355dd6aca12`), source with SDDS a3f730a14c49, CPU, double, serial and three-rank MPI, one thread |
| Confirmed<br><small>rigid-face integration for both bend signs and frozen-fit controls</small> | `claude-opus-5-5` | high | 2026-11-06 | master (`0355dd6aca12`), source with SDDS a3f730a14c49, CPU, double, serial, one thread |

