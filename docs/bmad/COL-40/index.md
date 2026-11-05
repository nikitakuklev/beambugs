---
title: "COL-40: The beam–beam manual names a nonexistent Twiss-reference attribute"
hide:
  - toc
---

# COL-40 — The beam–beam manual names a nonexistent Twiss-reference attribute

| | |
|---|---|
| Code | [Bmad](../index.md) · Collective effects |
| Priority | P4 |
| Severity | Documentation or example mismatch |
| Evidence | Reproduced, independently rechecked |
| Last reproduced | 2026-11-05 on Bmad 20261031-0 (`c4c629f`) |
| Source | [Bmad manual, BeamBeam attribute list, s:beambeam, bmad/doc/elements.tex:315–321](https://github.com/bmad-sim/bmad-ecosystem/blob/c4c629fc9071d3a56cb1ba5bc714baba65e9c5d3/bmad/doc/elements.tex#L315-L321)<br>[Bmad manual, BeamBeam reference-position prose, bmad/doc/elements.tex:352–360](https://github.com/bmad-sim/bmad-ecosystem/blob/c4c629fc9071d3a56cb1ba5bc714baba65e9c5d3/bmad/doc/elements.tex#L352-L360)<br>[S_TWISS_REF registration, bmad/modules/attribute_mod.f90:894–898](https://github.com/bmad-sim/bmad-ecosystem/blob/c4c629fc9071d3a56cb1ba5bc714baba65e9c5d3/bmad/modules/attribute_mod.f90#L894-L898)<br>[strong_beam_sigma_calc reference position, bmad/low_level/strong_beam_sigma_calc.f90:31–43](https://github.com/bmad-sim/bmad-ecosystem/blob/c4c629fc9071d3a56cb1ba5bc714baba65e9c5d3/bmad/low_level/strong_beam_sigma_calc.f90#L31-L43) |

When users copy the beam–beam manual's s_beta_ref attribute to specify the strong-beam Twiss reference position, the parser rejects it. The accepted attribute is s_twiss_ref, which the field-size calculation and other manual prose already use. A beam–beam element with s_twiss_ref parses successfully, while the otherwise identical element with s_beta_ref fails. Both stale names in the manual should be replaced with s_twiss_ref.

Download these shared helpers into the same directory as the reproducer scripts:

- [Download _bmad.py](../helpers/_bmad.py) · [View source](../helpers/index.md#_bmad)
- [Download _collective_contracts.py](../helpers/_collective_contracts.py) · [View source](../helpers/index.md#_collective_contracts)

??? example "Reproducer: COL-40.py"

    [Download COL-40.py](COL-40.py)

    ```python
    """COL-40: The beam–beam manual names a nonexistent Twiss-reference attribute

    When users copy the beam–beam manual's s_beta_ref attribute to specify the strong-beam Twiss reference position, the parser rejects it. The accepted attribute is s_twiss_ref, which the field-size calculation and other manual prose already use. A beam–beam element with s_twiss_ref parses successfully, while the otherwise identical element with s_beta_ref fails. Both stale names in the manual should be replaced with s_twiss_ref.

    The script parses otherwise identical beam–beam elements with the accepted and documented
    attribute names. It requires a successful control and the explicit bad-attribute error for
    s_beta_ref, then checks that the stale name remains in the manual. BMAD_DIST and BMAD_SOURCE
    select the build and source tree.
    """

    from _bmad import verdict
    from _collective_contracts import parse_lattice, source_text

    base = "beambeam, sig_x=0.001, sig_y=0.002, sig_z=0.01, beta_a_strong=1, beta_b_strong=1"
    if not parse_lattice(base + ", s_twiss_ref=0.1")[0]:
        raise RuntimeError("Registered s_twiss_ref control failed")
    accepted, log = parse_lattice(base + ", s_beta_ref=0.1")
    if not accepted and "BAD ATTRIBUTE NAME:" not in log:
        raise RuntimeError(log)
    manual = source_text("bmad/doc/elements.tex")
    documented = "s_beta_ref" in manual.lower()
    print(
        f"s_beta_ref: documented={documented}, parser_accepts={accepted}; expected documented input accepted"
    )
    print("s_twiss_ref: parser_accepts=True (control)")
    verdict(documented and not accepted)

    ```

## Provenance

| Role | Model | Effort | Date | Code examined |
|---|---|---|---|---|
| Found<br><small>source check</small> | `gemini-4-argon-high` |  | 2026-10-20 | 20261020-0 (`19b8afb46a40`) |
| Confirmed<br><small>source check</small> | `gpt-6-astra` | xhigh | 2026-10-20 | 20261020-0 (`19b8afb46a40`) |
| Confirmed | `gpt-6.1-sol` | xhigh | 2026-11-05 | 20261031-0 (`c4c629fc9071`), source, production, GNU Fortran 15.3.0, CPU, double, serial, OpenMP=1 |

