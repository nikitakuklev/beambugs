---
title: "OPT-31: AC-kicker matrices omit waveform derivatives"
hide:
  - toc
---

# OPT-31 — AC-kicker matrices omit waveform derivatives

| | |
|---|---|
| Code | [Bmad](../index.md) · Optics |
| Priority | P2 |
| Failure | Silent wrong result |
| Evidence | Reproduced, independently rechecked |
| Last reproduced | 2026-11-05 on Bmad release [20261031-0](https://github.com/bmad-sim/bmad-ecosystem/tree/20261031-0) (`c4c629f`) |
| Tags | <span class="bb-tag">partly-source-only</span> [?](../../methodology.md#tags) |
| Pull request | [[gh:https://github.com/bmad-sim/bmad-ecosystem/pull/2108]] |
| Source | [track_a_thick_multipole, bmad/low_level/track_a_thick_multipole.f90:94–118](https://github.com/bmad-sim/bmad-ecosystem/blob/c4c629fc9071d3a56cb1ba5bc714baba65e9c5d3/bmad/low_level/track_a_thick_multipole.f90#L94-L118)<br>[multipole matrix update, bmad/modules/multipole_mod.f90:124–128](https://github.com/bmad-sim/bmad-ecosystem/blob/c4c629fc9071d3a56cb1ba5bc714baba65e9c5d3/bmad/modules/multipole_mod.f90#L124-L128)<br>[Bmad manual, Linear Transfer Map (Mat6) Calculation Methods, bmad/doc/methods.tex:253–260](https://github.com/bmad-sim/bmad-ecosystem/blob/c4c629fc9071d3a56cb1ba5bc714baba65e9c5d3/bmad/doc/methods.tex#L253-L260)<br>[Bmad manual, AC Kicker waveform, bmad/doc/elements.tex:181–216](https://github.com/bmad-sim/bmad-ecosystem/blob/c4c629fc9071d3a56cb1ba5bc714baba65e9c5d3/bmad/doc/elements.tex#L181-L216) |

For a thin magnetic AC kicker with a changing waveform, the native matrix omits the dependence of the kick on arrival time. Particle tracking includes the waveform, so differentiating that trajectory gives nonzero longitudinal-to-transverse matrix terms. For a 2 MeV/c electron, a 4 mrad horizontal kick and a 200 kHz sinusoid, M₂₅ is zero instead of 1.340×10⁻⁵ m⁻¹, where M₂₅ is the derivative of horizontal normalized momentum with respect to longitudinal position. Vertical kicks and finite-speed momentum derivatives have the same omission; a constant waveform gives the correct zero terms.

Run `python OPT-31.py` with [_bmad.py](../helpers/_bmad.py) and [_srdt_wiggler_maps.py](../helpers/_srdt_wiggler_maps.py) in the same directory.

??? example "Reproducer: OPT-31.py"

    [Download OPT-31.py](OPT-31.py)

    ```python
    """OPT-31: AC-kicker matrices omit waveform derivatives

    For a thin magnetic AC kicker with a changing waveform, the native matrix omits the dependence of the kick on arrival time. Particle tracking includes the waveform, so differentiating that trajectory gives nonzero longitudinal-to-transverse matrix terms. For a 2 MeV/c electron, a 4 mrad horizontal kick and a 200 kHz sinusoid, M₂₅ is zero instead of 1.340×10⁻⁵ m⁻¹, where M₂₅ is the derivative of horizontal normalized momentum with respect to longitudinal position. Vertical kicks and finite-speed momentum derivatives have the same omission; a constant waveform gives the correct zero terms.

    The script tracks thin horizontal and vertical magnetic kicks with two sinusoids, a ramp and a constant-waveform control at 2 MeV/c. It compares native and auto matrices with analytic waveform derivatives and fourth-order trajectory differences at two stencil sizes.
    """

    import math

    import numpy as np
    from _bmad import compare, verdict
    from _srdt_wiggler_maps import matrices

    LATTICE = """parameter[p0c]=2000000.0
    parameter[particle]=electron
    parameter[geometry]=open
    beginning[beta_a]=9
    beginning[beta_b]=11
    prefix: drift, l=.63
    e1: ac_kicker, l=0, hkick=0.004, vkick=-0.003, t_offset=2.3e-9, frequencies={(200000.0,0.8,0.21)}
    e2: ac_kicker, l=0, hkick=-0.007, vkick=0.005, t_offset=2.3e-9, frequencies={(-800000.0,1.3,-0.13)}
    e3: ac_kicker, l=0, hkick=0.004, vkick=-0.003, t_offset=2.3e-9, interpolation=linear, amp_vs_time={(-1e-6,4.2),(1e-6,-1.8)}
    e4: ac_kicker, l=0, hkick=0.004, vkick=-0.003, t_offset=2.3e-9, frequencies={(0,0.8,0.21)}
    lat: line=(prefix,e1,e2,e3,e4)
    use,lat
    """
    # Frequency, amplitude, phase in cycles, horizontal/vertical integrated kicks.
    settings = [
        (2e5, 0.8, 0.21, 0.004, -0.003),
        (-8e5, 1.3, -0.13, -0.007, 0.005),
        (None, 1.2, 0, 0.004, -0.003),
        (0, 0.8, 0.21, 0.004, -0.003),
    ]
    vectors = [
        (ie, [0, 0, 0, 0, z, delta]) for ie in range(2, 6) for z, delta in [(0, 0), (0.023, 0.19)]
    ]
    defect = False
    for (ie, v), (row, m, fd) in zip(vectors, matrices(LATTICE, vectors), strict=True):
        frequency, amplitude, phase, hk, vk = settings[ie - 2]
        # beta follows from the input momentum and electron rest energy, independently
        # of the tracked orbit. Check the stored value before using the timing law.
        mu = 510998.95069 / 2e6
        r = 1 + v[5]
        beta = r / math.hypot(r, mu)
        np.testing.assert_allclose(row[12], beta, rtol=0, atol=2e-15)
        time = -v[4] / (beta * 299792458.0) - 2.3e-9
        if frequency is None:
            value = amplitude - 3e6 * time
            slope = -3e6
        else:
            angle = 2 * math.pi * (frequency * time + phase)
            value = amplitude * math.cos(angle)
            slope = -2 * math.pi * frequency * amplitude * math.sin(angle)
        kicks = np.array([hk, vk])
        expected_z = -kicks * slope / (beta * 299792458.0)
        # Holding the reference clock fixed, t_eff=-z/(beta*c)-t_offset.
        expected_delta = kicks * slope * v[4] * mu**2 / (299792458.0 * r**2 * np.sqrt(r * r + mu * mu))
        expected = np.column_stack((expected_z, expected_delta))
        np.testing.assert_allclose(row[[7, 9]], kicks * value, rtol=0, atol=8e-17)
        np.testing.assert_allclose(fd[:, [1, 3], 4:6], np.tile(expected, (2, 1, 1)), rtol=0, atol=2e-13)
        bad = compare(
            f"element {ie}, z={v[4]}, delta={v[5]}: M25,M26; M45,M46",
            m[0][np.ix_([1, 3], [4, 5])],
            expected,
            atol=2e-13,
        )
        if ie == 5 and bad:
            raise AssertionError("Constant-waveform control failed")
        defect |= bad
    verdict(defect)

    ```

## Provenance

| Role | Model | Effort | Date | Code examined |
|---|---|---|---|---|
| Found | <span class="bb-nowrap">`gpt-6-astra`</span> | xhigh | <span class="bb-nowrap">2026-10-29</span> | 20261020-0 (`19b8afb46a40`), source-matched 20261021-0 (d00a8f17e55a) libraries, production, CPU, double, serial |
| Confirmed | <span class="bb-nowrap">`gpt-6.1-sol`</span> | xhigh | <span class="bb-nowrap">2026-10-29</span> | 20261020-0 (`19b8afb46a40`), source-matched 20261021-0 (d00a8f17e55a) libraries, production, CPU, double, serial |

