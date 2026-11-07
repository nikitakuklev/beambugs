---
title: "XLT-05: Embedded steering kicks delete the host magnet on export"
hide:
  - toc
---

# XLT-05 — Embedded steering kicks delete the host magnet on export

| | |
|---|---|
| Code | [Bmad](../index.md) · Translators |
| Priority | P1 |
| Failure | Wrong physics |
| Evidence | Reproduced, independently rechecked |
| Last reproduced | 2026-11-05 on Bmad release [20261031-0](https://github.com/bmad-sim/bmad-ecosystem/tree/20261031-0) (`c4c629f`) |
| Tags | <span class="bb-tag">partly-source-only</span> [?](../../methodology.md#tags) |
| Source | [`write_lattice_mad_format`, bmad/output/write_lattice_mad_format.f90:294–324](https://github.com/bmad-sim/bmad-ecosystem/blob/c4c629fc9071d3a56cb1ba5bc714baba65e9c5d3/bmad/output/write_lattice_mad_format.f90#L294-L324)<br>[`write_lattice_elegant_format`, bmad/output/write_lattice_elegant_format.f90:288–316](https://github.com/bmad-sim/bmad-ecosystem/blob/c4c629fc9071d3a56cb1ba5bc714baba65e9c5d3/bmad/output/write_lattice_elegant_format.f90#L288-L316)<br>[MAD:616](https://github.com/bmad-sim/bmad-ecosystem/blob/c4c629fc9071d3a56cb1ba5bc714baba65e9c5d3/bmad/output/write_lattice_mad_format.f90#L616)<br>[Elegant:475](https://github.com/bmad-sim/bmad-ecosystem/blob/c4c629fc9071d3a56cb1ba5bc714baba65e9c5d3/bmad/output/write_lattice_elegant_format.f90#L475)<br>[`init_attribute_name_array`, bmad/modules/attribute_mod.f90:762](https://github.com/bmad-sim/bmad-ecosystem/blob/c4c629fc9071d3a56cb1ba5bc714baba65e9c5d3/bmad/modules/attribute_mod.f90#L762)<br>[`offset_particle`, bmad/code/offset_particle.f90:316–320](https://github.com/bmad-sim/bmad-ecosystem/blob/c4c629fc9071d3a56cb1ba5bc714baba65e9c5d3/bmad/code/offset_particle.f90#L316-L320)<br>[bmad/doc/elements.tex:1890–1893](https://github.com/bmad-sim/bmad-ecosystem/blob/c4c629fc9071d3a56cb1ba5bc714baba65e9c5d3/bmad/doc/elements.tex#L1890-L1893)<br>[lines 672–690](https://github.com/bmad-sim/bmad-ecosystem/blob/c4c629fc9071d3a56cb1ba5bc714baba65e9c5d3/bmad/output/write_lattice_mad_format.f90#L672-L690) |

Exporting a quadrupole or another host element with embedded steering to MAD-X, MAD8 or Elegant replaces the host by two half-kickers around an identity matrix, deleting the host body and its length. A 1 m quadrupole with strength K₁ = 0.5 m⁻² and horizontal kick 0.1 mrad becomes zero length in MAD-X; its horizontal beta function stays at 10.00 m instead of the thick-lens 5.864 m. Unkicked exports and Bmad match the thick-lens result. Sextupoles, solenoids, cavities, monitors and wigglers are replaced in all three formats; a tilted electrostatic separator also loses its tilt and charge sign. Dedicated kickers, sad_mult elements and MAD-X sbends avoid this replacement.

Run `python XLT-05.py` with [_bmad.py](../helpers/_bmad.py) and [_translators.py](../helpers/_translators.py) in the same directory.

??? example "Reproducer: XLT-05.py"

    [Download XLT-05.py](XLT-05.py)

    ```python
    """XLT-05: Embedded steering kicks delete the host magnet on export

    Exporting a quadrupole or another host element with embedded steering to MAD-X, MAD8 or Elegant
    replaces the host by two half-kickers around an identity matrix, deleting the host body and its
    length. A 1 m quadrupole with strength K₁ = 0.5 m⁻² and horizontal kick 0.1 mrad becomes zero
    length in MAD-X; its horizontal beta function stays at 10.00 m instead of the thick-lens 5.864
    m. Unkicked exports and Bmad match the thick-lens result. Sextupoles, solenoids, cavities,
    monitors and wigglers are replaced in all three formats; a tilted electrostatic separator also
    loses its tilt and charge sign. Dedicated kickers, sad_mult elements and MAD-X sbends avoid this
    replacement.

    The script exports with bmad_to_mad_sad_elegant a 1 m quadrupole with K₁ = 0.5 m⁻² and
    hkick = 0.1 mrad, and the same quadrupole without the kick as the control. It runs MAD-X TWISS
    on both exports and compares the length and the exit β_x with the thick-lens transfer matrix.
    It then exports eleven element classes carrying a vertical kick and reads which element class
    each MAD-X, MAD8 and Elegant file gives them, and tracks a tilted elseparator in MAD-X and in
    Bmad against the rotated, charge-signed kick.
    """

    import numpy as np
    from _bmad import compare, verdict
    from _translators import element_type, export, lattice, madx, madx_track, numbers, tao

    P0C = 3e9  # eV/c, electrons
    K1, LENGTH, BETA0 = 0.5, 1.0, 10.0

    # Thick quadrupole, initial beta_x = 10 m, alpha_x = 0: beta_x = M11^2 beta0 + M12^2 / beta0.
    root = np.sqrt(K1)
    m11, m12 = np.cos(root * LENGTH), np.sin(root * LENGTH) / root
    BETA_X = m11**2 * BETA0 + m12**2 / BETA0

    defect = False
    for kick in (0.0, 1e-4):
        lat = lattice(f"q: quadrupole, l = {LENGTH}, k1 = {K1}, hkick = {kick}", "q", p0c=P0C)
        bmad_beta = numbers(tao(lat, ["pipe ele:twiss 1"])["pipe ele:twiss 1"], ["beta_a"])[0]
        with madx(export(lat, "madx")) as mad:
            twiss = mad.twiss(beta0="initial")
            observed = [twiss.s[-1], twiss.betx[-1]]
        label = f"hkick={kick:g}: MAD-X length (m), beta_x (m), expected thick-lens matrix"
        if kick == 0:
            if compare(label, observed, [LENGTH, BETA_X], atol=1e-12):
                raise SystemExit("The unkicked export disagrees with the matrix: comparison not valid.")
        else:
            defect |= compare(label, observed, [LENGTH, BETA_X], atol=1e-12)
        if compare(f"hkick={kick:g}: Bmad beta_x (m)", bmad_beta, BETA_X, atol=1e-12):
            raise SystemExit("Bmad disagrees with the thick-lens matrix: comparison not valid.")

    # Element classes with HKICK/VKICK; each carries vkick = 10 urad, the elseparator its own kicks.
    hosts = {
        "SX": "sextupole, l = 0.2, k2 = 1",
        "OC": "octupole, l = 0.2, k3 = 1",
        "SO": "solenoid, l = 0.3, ks = 0.1",
        "SQ": "sol_quad, l = 0.3, ks = 0.1, k1 = 0.2",
        "RF": "rfcavity, l = 0.4, voltage = 1e5, rf_frequency = 5e8",
        "LC": "lcavity, l = 0.4, voltage = 1e5, rf_frequency = 5e8",
        "MO": "monitor, l = 0.1",
        "INS": "instrument, l = 0.1",
        "TM": "thick_multipole, l = 0.2, b2 = 0.1",
        "WG": "wiggler, l = 1, b_max = 0.1, n_period = 4",
        "ES": "elseparator, l = 1, hkick = 1e-3, vkick = 4e-4, tilt = 0.5",
    }
    body = "\n".join(f"{n}: {d}" + ("" if n == "ES" else ", vkick = 1e-5") for n, d in hosts.items())
    sweep = lattice(body, ", ".join(hosts), p0c=P0C)
    for fmt in ("madx", "mad8", "elegant"):
        text = export(sweep, fmt)
        classes = {name: element_type(text, fmt, name) for name in hosts}
        print(f"{fmt}: element class written for each kicked host: {classes}")
        # A matrix representation can preserve a host's map. The class inventory records the
        # breadth of the current output; the independent optics/kick checks determine the verdict.
        # The control quadrupole without a kick is written as a quadrupole.
        plain = export(lattice("q: quadrupole, l = 1, k1 = 0.5", "q", p0c=P0C), fmt)
        if element_type(plain, fmt, "Q") not in ("quadrupole", "quad", "kquad"):
            raise SystemExit(f"{fmt}: the unkicked quadrupole is not written as a quadrupole.")

    # Tilted elseparator, electrons. Bmad applies HKICK/VKICK for a positive charge in the element
    # frame rotated by TILT: kick = -R(tilt) (hkick, vkick) for an electron.
    hk, vk, tilt = 1e-3, 4e-4, 0.5
    expected = -np.array([hk * np.cos(tilt) - vk * np.sin(tilt), hk * np.sin(tilt) + vk * np.cos(tilt)])
    sep = lattice(f"es: {hosts['ES']}", "es", p0c=P0C)
    bmad_kick = numbers(tao(sep, ["pipe ele:orbit 1"])["pipe ele:orbit 1"], ["px", "py"])
    if compare(
        "elseparator: Bmad (px, py), expected -R(tilt)(hkick, vkick)",
        bmad_kick,
        expected,
        atol=0,
        rtol=1e-6,
    ):
        raise SystemExit("Bmad disagrees with the rotated kick: comparison not valid.")
    with madx(export(sep, "madx")) as mad:
        end = madx_track(mad, [0, 0, 0, 0, 0, 0])
    defect |= compare(
        "elseparator: MAD-X (px, py) of the export, expected -R(tilt)(hkick, vkick)",
        end[[1, 3]],
        expected,
        atol=0,
        rtol=1e-6,
    )
    verdict(defect)

    ```

## Provenance

| Role | Model | Effort | Date | Code examined |
|---|---|---|---|---|
| Found | `gpt-6.1-sol` | xhigh | 2026-10-21 | 20261020-0 (`19b8afb46a40`), source, production, CPU, double, serial |
| Confirmed | `gpt-6-astra` | xhigh | 2026-10-21 | 20261020-0 (`19b8afb46a40`), source, production, CPU, double, serial |

