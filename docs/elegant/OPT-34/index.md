---
title: "OPT-34: RF matrix concatenation loses the accumulated clock transformation"
hide:
  - toc
---

# OPT-34 — RF matrix concatenation loses the accumulated clock transformation

| | |
|---|---|
| Code | [Elegant](../index.md) · Optics |
| Priority | P3 |
| Severity | Wrong physics |
| Evidence | Reproduced, independently rechecked |
| Last reproduced | 2026-11-06 on Elegant master (`0355dd6`, committed 2026-11-03) |
| Tags | <span class="bb-tag">partly-source-only</span> [?](../../methodology.md#tags) |
| Source | [src/concat_mat.c:56–89](https://github.com/rtsoliday/elegant/blob/0355dd6aca12c861971fbf8f3ae78a0ded991734/src/concat_mat.c#L56-L89)<br>[src/matrix_output.c:425–429](https://github.com/rtsoliday/elegant/blob/0355dd6aca12c861971fbf8f3ae78a0ded991734/src/matrix_output.c#L425-L429)<br>[src/compute_matrices.c:2492–2503](https://github.com/rtsoliday/elegant/blob/0355dd6aca12c861971fbf8f3ae78a0ded991734/src/compute_matrices.c#L2492-L2503)<br>[src/analyze.c:695–698](https://github.com/rtsoliday/elegant/blob/0355dd6aca12c861971fbf8f3ae78a0ded991734/src/analyze.c#L695-L698)<br>[src/analyze.c:931–953](https://github.com/rtsoliday/elegant/blob/0355dd6aca12c861971fbf8f3ae78a0ded991734/src/analyze.c#L931-L953)<br>[src/matrix_output.c:451–461](https://github.com/rtsoliday/elegant/blob/0355dd6aca12c861971fbf8f3ae78a0ded991734/src/matrix_output.c#L451-L461)<br>[doc/elegant.tex:247–256](https://github.com/rtsoliday/elegant/blob/0355dd6aca12c861971fbf8f3ae78a0ded991734/doc/elegant.tex#L247-L256)<br>[doc/elegant.tex:364–380](https://github.com/rtsoliday/elegant/blob/0355dd6aca12c861971fbf8f3ae78a0ded991734/doc/elegant.tex#L364-L380)<br>[doc/elegant.tex:5351–5363](https://github.com/rtsoliday/elegant/blob/0355dd6aca12c861971fbf8f3ae78a0ded991734/doc/elegant.tex#L5351-L5363)<br>[doc/rfca.tex:11–17](https://github.com/rtsoliday/elegant/blob/0355dd6aca12c861971fbf8f3ae78a0ded991734/doc/rfca.tex#L11-L17)<br>[src/matrix.c:340–363](https://github.com/rtsoliday/elegant/blob/0355dd6aca12c861971fbf8f3ae78a0ded991734/src/matrix.c#L340-L363) |

With flight distance before an accelerating RFCA, the concatenated matrix leaves the incoming longitudinal coordinate unscaled. For s = βct, where β is speed divided by c and t is arrival time, that coordinate must scale by βout/βin. With 0.4000 m before a thin gap accelerating electrons from 1.500 to 3.000 mc and 0.6000 m afterward, the on-axis output C₅ is 1.000 m instead of 1.056 m, and R₅₆ = ∂sout/∂δin is zero instead of −0.1203 m, where δ is fractional momentum deviation. Relativistic flight times and tracked derivatives agree. TW1's local R₅₅ and transverse Twiss transport pass; raising output order does not restore the longitudinal map.

Run `python OPT-34.py` with [_elegant.py](../helpers/_elegant.py), [_momentum_matrix.py](../helpers/_momentum_matrix.py) and [_diagnostics.py](../helpers/_diagnostics.py) in the same directory.

??? example "Reproducer: OPT-34.py"

    [Download OPT-34.py](OPT-34.py)

    ```python
    """OPT-34: RF matrix concatenation loses the accumulated clock transformation

    With flight distance before an accelerating RFCA, the concatenated matrix leaves the incoming longitudinal coordinate unscaled. For s = βct, where β is speed divided by c and t is arrival time, that coordinate must scale by βout/βin. With 0.4000 m before a thin gap accelerating electrons from 1.500 to 3.000 mc and 0.6000 m afterward, the on-axis output C₅ is 1.000 m instead of 1.056 m, and R₅₆ = ∂sout/∂δin is zero instead of −0.1203 m, where δ is fractional momentum deviation. Relativistic flight times and tracked derivatives agree. TW1's local R₅₅ and transverse Twiss transport pass; raising output order does not restore the longitudinal map.

    The script compares cumulative and individual TW1 matrices with an 80-digit relativistic flight-time derivative and native four-point particle derivatives. It varies reference policy, geometry, order and acceleration sign, with zero-voltage, zero-incoming-distance and transverse Twiss controls.
    """

    import mpmath as mp
    import numpy as np
    from _elegant import verdict
    from _momentum_matrix import C, close, discrepancy, track, value

    mp.mp.dps = 80
    MASS_EV = mp.mpf("510999.06")  # Electron rest energy in the executable's SDDS constants.


    def oracle(pi, pf, before, after):
        pi, pf, before, after = map(mp.mpf, map(str, (pi, pf, before, after)))
        gi, gf = mp.sqrt(1 + pi**2), mp.sqrt(1 + pf**2)
        bi, bf = pi / gi, pf / gf
        ratio = bf / bi
        # The gap holds gamma_f-gamma_i fixed. d(log beta)/d(gamma)=1/(gamma*p^2),
        # d(gamma_f)/d(delta_in)=p_i^2/gamma_i, and d(log beta_i)/d(delta_in)=1/gamma_i^2.
        derivative = ratio * (pi**2 / (gi * gf * pf**2) - 1 / gi**2)
        return tuple(
            map(float, ((gf - gi) * MASS_EV, before * ratio + after, ratio, before * derivative))
        )


    def case(pi, pf, before, after, change, h, *, order=2, geometry="EDRIFT", twiss=False):
        voltage, expected_c5, expected_r55, expected_r56 = oracle(pi, pf, before, after)
        first = (
            f"A: MALIGN,DZ={before:.17g}\n" if geometry == "MALIGN" else f"A: EDRIFT,L={before:.17g}\n"
        )
        lattice = first + (
            f"G: RFCA,L=0,VOLT={voltage:.17g},FREQ=0,PHASE=90,N_KICKS=10,"
            f"CHANGE_P0={change},BODY_FOCUS_MODEL=TW1,END1_FOCUS=0,END2_FOCUS=0,CHANGE_T=0\n"
            f"B: EDRIFT,L={after:.17g}\nBL: LINE=(A,G,B)\n"
        )
        steps = np.array([-2, -1, 1, 2]) * h
        delta, s = np.r_[0, steps, np.zeros(4)], np.r_[np.zeros(5), steps]
        result = track(lattice, pi, delta, s=s, order=order, twiss=twiss)
        data, sout = result["particles"], result["s"]
        pin = pi * (1 + delta)
        gin = np.hypot(pin, 1)
        gout = gin + voltage / float(MASS_EV)
        pout = np.sqrt(gout**2 - 1)
        bin_, bout = pin / gin, pout / gout
        # s belongs to each particle: continuous time gives s_out=(s_in+L1)*beta_out/beta_in+L2.
        close(data["p"], pout, 4e-14 * max(1, pi, pf), "mass-shell endpoints")
        close(data["t"], ((s + before) / bin_ + after / bout) / C, 3e-23, "flight times")
        close(sout, (s + before) * bout / bin_ + after, 3e-14, "equivalent distances")
        close(sout[0], expected_c5, 3e-14, "nominal equivalent distance")

        def fd(values):
            return float((values[0] - 8 * values[1] + 8 * values[2] - values[3]) / (12 * h))

        fd56, fd55 = fd(sout[1:5]), fd(sout[5:9])
        close(fd56, expected_r56, 2e-11, "momentum derivative")
        close(fd55, expected_r55, 2e-11, "clock derivative")
        c5, r55, r56 = (value(result, key) for key in ("C5", "R55", "R56"))
        close(r55, expected_r55, 8e-10, "TW1 local clock scaling")
        failures = [
            discrepancy(c5, expected_c5, before + after, 3e-14, "unscaled clock law"),
            discrepancy(r56, expected_r56, 0, 8e-10, "missing clock derivative law"),
        ]
        # Individual output writes a centroid increment, not a cumulative coordinate.
        ic5, ir56 = (value(result, key, output="individual.sdds", element="G") for key in ("C5", "R56"))
        increment = before * (expected_r55 - 1)
        failures.extend(
            [
                discrepancy(ic5, increment, 0, 3e-14, "individual clock increment"),
                discrepancy(ir56, expected_r56, 0, 8e-10, "individual momentum derivative"),
            ]
        )
        print(
            f"{geometry} {pi:g}->{pf:g} mc L={before:g}/{after:g} m change_p0={change} "
            f"order={order} h={h:g}: C5={c5:.16g} expected={expected_c5:.16g} m; "
            f"R55={r55:.16g} expected={expected_r55:.16g}; "
            f"R56={r56:.16g} expected={expected_r56:.16g} m (80-digit flight-time derivative); "
            f"tracked R56={fd56:.16g}; individual gap C5={ic5:.16g} expected={increment:.16g} m; "
            f"individual R56={ir56:.16g} expected={expected_r56:.16g} m"
        )
        if twiss:
            # In slopes, the gap scales by k=p_i/p_f. With beta_in=1 and alpha_in=0,
            # the normalized transverse Twiss map gives beta_out=(1+R12^2)/k, alpha_out=-R12.
            k, r12 = pi / pf, before + after * pi / pf
            expected_beta = (1 + r12**2) / k
            for plane in ("x", "y"):
                close(
                    value(result, "beta" + plane, output="twiss.sdds"),
                    expected_beta,
                    2e-10,
                    "Twiss beta",
                )
                close(value(result, "alpha" + plane, output="twiss.sdds"), -r12, 2e-10, "Twiss alpha")
            print(
                f"transverse Twiss control: beta_x=beta_y={expected_beta:.12g} m, "
                f"alpha_x=alpha_y={-r12:.12g}, agreeing with the drift-gap-drift slope map"
            )
        return any(failures)


    def main():
        failures = []
        for change in (0, 1):
            for h in (1e-4, 5e-5):
                failures.append(case(1.5, 3.0, 0.4, 0.6, change, h, twiss=True))
        failures.append(case(2.0, 4.0, 0.3, 0.7, 1, 1e-4))
        failures.append(case(1.5, 3.0, 0.4, 0.6, 1, 1e-4, geometry="MALIGN"))
        for order in (1, 3):
            failures.append(case(1.5, 3.0, 0.4, 0.6, 1, 1e-4, order=order))
        failures.append(case(3.0, 1.5, 0.4, 0.6, 1, 1e-4))
        for args in ((1.5, 1.5, 0.4, 0.6, 1, 1e-4), (1.5, 3.0, 0.0, 0.6, 1, 1e-4)):
            if case(*args):
                raise AssertionError("Zero-voltage or zero-incoming-distance control failed")
        verdict(any(failures))


    if __name__ == "__main__":
        main()

    ```

## Provenance

| Role | Model | Effort | Date | Code examined |
|---|---|---|---|---|
| Found | `gpt-6-astra` | xhigh | 2026-11-06 | master (`0355dd6aca12`), source with SDDS a3f730a14c49, CPU, double, serial, one thread |
| Confirmed | `gpt-6.1-sol` | xhigh | 2026-11-06 | master (`0355dd6aca12`), source with SDDS a3f730a14c49, CPU, double, serial, one thread |

