---
title: "XLT-17: The manual and MAD-X RF exporter use the wrong phase sign"
hide:
  - toc
---

# XLT-17 — The manual and MAD-X RF exporter use the wrong phase sign

| | |
|---|---|
| Code | [Bmad](../index.md) · Translators |
| Priority | P1 |
| Failure | Wrong physics |
| Evidence | Probe-pinned |
| Last reproduced | 2026-11-05 on Bmad release [20261031-0](https://github.com/bmad-sim/bmad-ecosystem/tree/20261031-0) (`c4c629f`) |
| Tags | <span class="bb-tag">partly-source-only</span> [?](../../methodology.md#tags) |
| Source | [`write_lattice_mad_format`, bmad/output/write_lattice_mad_format.f90:906–913](https://github.com/bmad-sim/bmad-ecosystem/blob/c4c629fc9071d3a56cb1ba5bc714baba65e9c5d3/bmad/output/write_lattice_mad_format.f90#L906-L913)<br>[bmad/doc/elements.tex:4874–4897](https://github.com/bmad-sim/bmad-ecosystem/blob/c4c629fc9071d3a56cb1ba5bc714baba65e9c5d3/bmad/doc/elements.tex#L4874-L4897)<br>[MAD-X cavity.html:28](https://github.com/MethodicalAcceleratorDesign/MAD-X/blob/7998e738eca575c82a424138517c2a17115f3cdb/doc/usrguide/Introduction/cavity.html#L28) |

Exporting a thin rfcavity to MAD-X follows the manual's phase rule and writes lag = φ₀ + 0.5 instead of 0.5 − φ₀. For matched electron beams, Bmad's kick V sin(2π(φ₀ + fτ)) and MAD-X's V sin(2π(lag − fτ)) require the latter rule, where V is voltage, f is frequency, φ₀ is phase in turns, and τ is arrival-time advance. A 10 kV cavity at φ₀ = 0.125 gives an on-time 1 MeV/c electron +7.071 keV in Bmad and −7.071 keV in the export. At longitudinal coordinate z = 3 cm and relative momentum offset δ = 0.1, the gains are +7.494 and −6.622 keV. Correcting lag restores Bmad's waveform.

Run `python XLT-17.py` with [_bmad.py](../helpers/_bmad.py) and [_translators.py](../helpers/_translators.py) in the same directory.

??? example "Reproducer: XLT-17.py"

    [Download XLT-17.py](XLT-17.py)

    ```python
    """XLT-17: The manual and MAD-X RF exporter use the wrong phase sign

    Exporting a thin rfcavity to MAD-X follows the manual's phase rule and writes lag = φ₀ + 0.5
    instead of 0.5 − φ₀. For matched electron beams, Bmad's kick V sin(2π(φ₀ + fτ)) and MAD-X's V
    sin(2π(lag − fτ)) require the latter rule, where V is voltage, f is frequency, φ₀ is phase in
    turns, and τ is arrival-time advance. A 10 kV cavity at φ₀ = 0.125 gives an on-time 1 MeV/c
    electron +7.071 keV in Bmad and −7.071 keV in the export. At longitudinal coordinate z = 3 cm
    and relative momentum offset δ = 0.1, the gains are +7.494 and −6.622 keV. Correcting lag
    restores Bmad's waveform.

    The script exports with bmad_to_mad_sad_elegant a line of a thin rfcavity (10 kV, phi0 = 0.125,
    harmonic number 1) and a drift, for electrons at p0c = 1 MeV/c, and tracks a particle with
    z = 3 cm and δ = 0.1 and an on-time reference particle through it in Tao and, from the export,
    in MAD-X TRACK. Bmad's energy kick
    is checked against V sin(2π(phi0 + f τ)) with τ the arrival-time advance; MAD-X's kick is checked
    against its own lag formula V sin(2π(lag − f τ)) at the exported lag, and then compared with
    Bmad's. A copy of the export with lag = 0.5 − phi0 is the receiving-code control.
    """

    import re

    import numpy as np
    from _bmad import compare, verdict
    from _translators import (
        export,
        lattice,
        madx,
        madx_track,
        numbers,
        record,
        statements,
        tao,
        value,
    )

    C = 299792458.0  # m/s
    P0C, VOLTAGE, PHI0, LENGTH = 1e6, 1e4, 0.125, 3.0
    CAVITY = f"e: rfcavity, l = 0, voltage = {VOLTAGE}, harmon = 1, phi0 = {PHI0}"


    def line(z, delta):
        return lattice(
            f"{CAVITY}\nd: drift, l = {LENGTH}", "e, d", p0c=P0C, start=[0, 0, 0, 0, z, delta]
        )


    def madx_pt(text, advance, eta0):
        """MAD-X PT at the end of the line, checked against MAD-X's own kick at the deck's lag."""
        with madx(text) as mad:
            # MAD-X T = -c times arrival-time delay, in metres; advance is in seconds.
            end = madx_track(mad, [0, 0, 0, 0, advance * C, eta0])
            lag_used = float(mad.elements.e.lag)
            beam = mad.sequence.lat.beam
            beta0, pc0 = float(beam.beta), float(beam.pc) * 1e9
        # MAD-X derives the frequency from HARMON, the line length and its reference beta, and
        # normalizes the kick by its own reference momentum (its electron mass is not Bmad's).
        f_madx = 1 * beta0 * C / LENGTH
        own = eta0 + VOLTAGE / pc0 * np.sin(2 * np.pi * (lag_used - f_madx * advance))
        if compare(
            f"  MAD-X PT, expected its own formula at lag {lag_used:g}", end[5], own, atol=1e-12
        ):
            raise SystemExit("MAD-X disagrees with its own RF formula: comparison not valid.")
        return float(end[5])


    def energy(pz, mass):
        return float(np.hypot(P0C * (1 + pz), mass))


    defect = False
    for z, delta in ((0.03, 0.1), (0.0, 0.0)):
        print(f"particle z = {z} m, pz = {delta}:")
        lat = line(z, delta)
        out = tao(lat, ["pipe ele:orbit 0", "pipe ele:orbit 1", "pipe ele:gen_attribs 1"])
        z0, pz0, beta = numbers(out["pipe ele:orbit 0"], ["z", "pz", "beta"])
        pz1, p0c = numbers(out["pipe ele:orbit 1"], ["pz", "p0c"])
        frequency = float(record(out["pipe ele:gen_attribs 1"])["RF_FREQUENCY"])
        if compare(
            "  Bmad start (z m, pz), reference p0c (eV/c)", [z0, pz0, p0c], [z, delta, P0C], atol=0
        ):
            raise SystemExit("Unexpected Bmad start or reference: comparison not valid.")
        # Bmad's electron mass from the particle velocity: m = p sqrt(1/beta^2 - 1), p = p0c(1 + pz).
        mass = P0C * (1 + delta) * np.sqrt(1 / beta**2 - 1)

        # Energy deviation eta = (E - E0)/p0c (MAD-X PT); advance = z/(beta*c) is
        # the arrival-time advance in seconds. Positive z arrives early.
        eta0, advance = (energy(delta, mass) - energy(0, mass)) / P0C, float(z / (beta * C))
        waveform = eta0 + VOLTAGE / P0C * float(np.sin(2 * np.pi * (PHI0 + frequency * advance)))
        print(f"  Bmad RF frequency (Hz) from harmon = 1: {frequency!r}")
        if compare(
            "  Bmad PT after the cavity, expected V sin(2 pi (phi0 + f tau))",
            (energy(pz1, mass) - energy(0, mass)) / P0C,
            waveform,
            atol=3e-16,
            rtol=4e-14,
        ):
            raise SystemExit("Bmad disagrees with its waveform: comparison not valid.")

        deck = export(lat, "madx")
        cavity = statements(deck, "madx")["E"]
        print(f"  exported: E: {cavity}")
        defect |= compare(
            "  exported lag, expected 0.5 - phi0 from matched waveforms",
            value(cavity, "lag") % 1.0,
            0.5 - PHI0,
            atol=1e-15,
        )
        # MAD-X and Bmad references differ by their electron mass constants (1.4e-9 relative),
        # which moves PT by 2.5e-12; 1e-10 separates that from the sign of the waveform.
        observed = madx_pt(deck, advance, eta0)
        print(
            f"  energy gain (eV): Bmad {(waveform - eta0) * P0C!r}, MAD-X {(observed - eta0) * P0C!r}"
        )
        defect |= compare(
            "  MAD-X PT from the export, expected Bmad's waveform", observed, waveform, atol=1e-10
        )
        fixed, count = re.subn(r"\blag =\s*[^,;\s]+", f"lag = {0.5 - PHI0!r}", deck)
        if count != 1 or compare(
            "  MAD-X PT with lag = 0.5 - phi0, expected Bmad's waveform",
            madx_pt(fixed, advance, eta0),
            waveform,
            atol=1e-10,
        ):
            raise SystemExit("The lag = 0.5 - phi0 control disagrees: comparison not valid.")
    verdict(defect)

    ```

## Provenance

| Role | Model | Effort | Date | Code examined |
|---|---|---|---|---|
| Found | <span class="bb-nowrap">`gpt-6-astra`</span> | xhigh | <span class="bb-nowrap">2026-10-20</span> | 20261020-0 (`19b8afb46a40`), source, production, CPU, double, serial |
| Confirmed | <span class="bb-nowrap">`gpt-6-astra`</span> | xhigh | <span class="bb-nowrap">2026-10-21</span> | 20261020-0 (`19b8afb46a40`), source, production, CPU, double, serial |
| Confirmed<br><small class="bb-scope">thin electron rfcavity export with matched clocks</small> | <span class="bb-nowrap">`gpt-6.1-sol`</span> | xhigh | <span class="bb-nowrap">2026-11-05</span> | 20261031-0 (`c4c629fc9071`), source, production, GNU Fortran 15.3.0, CPU, double, serial, OpenMP=1 |

