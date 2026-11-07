---
title: "TRK-14: An lcavity with is_on = False still accelerates and focuses the beam"
hide:
  - toc
---

# TRK-14 — An lcavity with is_on = False still accelerates and focuses the beam

| | |
|---|---|
| Code | [Bmad](../index.md) · Tracking, RF, fields and spin |
| Priority | P1 |
| Severity | Wrong physics |
| Evidence | Reproduced, independently rechecked |
| Last reproduced | 2026-10-21 on Bmad 20261020-0 (`19b8afb`, committed 2026-10-20); not reproduced on Bmad 20261031-0 (`c4c629f`), 2026-11-05 |
| Issue | [[gh:https://github.com/bmad-sim/bmad-ecosystem/issues/2222]], filed from this evidence |
| Pull request | [[gh:https://github.com/bmad-sim/bmad-ecosystem/pull/2223]], filed from this evidence |
| Source | [`track_a_lcavity`, bmad/low_level/track_a_lcavity.f90:223–226](https://github.com/bmad-sim/bmad-ecosystem/blob/19b8afb46a400d12a53a92d616e02359b3b2b71f/bmad/low_level/track_a_lcavity.f90#L223-L226)<br>[`fringe_kick`, lines 260–265](https://github.com/bmad-sim/bmad-ecosystem/blob/19b8afb46a400d12a53a92d616e02359b3b2b71f/bmad/low_level/track_a_lcavity.f90#L260-L265)<br>[`this_energy_kick`, line 358](https://github.com/bmad-sim/bmad-ecosystem/blob/19b8afb46a400d12a53a92d616e02359b3b2b71f/bmad/low_level/track_a_lcavity.f90#L358)<br>[`this_pondermotive_transverse_kick`, lines 438–451](https://github.com/bmad-sim/bmad-ecosystem/blob/19b8afb46a400d12a53a92d616e02359b3b2b71f/bmad/low_level/track_a_lcavity.f90#L438-L451)<br>[`track1`, bmad/code/track1.f90:219](https://github.com/bmad-sim/bmad-ecosystem/blob/19b8afb46a400d12a53a92d616e02359b3b2b71f/bmad/code/track1.f90#L219)<br>[`track1_bmad`, bmad/low_level/track1_bmad.f90:112–116](https://github.com/bmad-sim/bmad-ecosystem/blob/19b8afb46a400d12a53a92d616e02359b3b2b71f/bmad/low_level/track1_bmad.f90#L112-L116)<br>[bmad/doc/attributes.tex:2006–2022](https://github.com/bmad-sim/bmad-ecosystem/blob/c4c629fc9071d3a56cb1ba5bc714baba65e9c5d3/bmad/doc/attributes.tex#L2006-L2022) |

Turning an aligned lcavity off with `is_on = False` in standard tracking with nonzero `n_rf_steps` leaves its stored RF voltage and focusing active. The energy kick, entrance and exit fringe kicks, and ponderomotive focusing ignore the off state. The documented result is a drift that retains the cavity's reference energy profile while exerting no force on the particle. A 5 MeV/c electron through a 1 m, 1 MV standing-wave cavity gains 1.000 MeV in either state; a particle initially 1 mm off axis leaves an off 2 MV cavity at x = 0.8237 mm instead of 1 mm.

Run `python TRK-14.py` with [_bmad.py](../helpers/_bmad.py) in the same directory.

??? example "Reproducer: TRK-14.py"

    [Download TRK-14.py](TRK-14.py)

    ```python
    """TRK-14: An lcavity with is_on = False still accelerates and focuses the beam

    Turning an aligned lcavity off with `is_on = False` in standard tracking with nonzero
    `n_rf_steps` leaves its stored RF voltage and focusing active. The energy kick, entrance and
    exit fringe kicks, and ponderomotive focusing ignore the off state. The documented result is a
    drift that retains the cavity's reference energy profile while exerting no force on the
    particle. A 5 MeV/c electron through a 1 m, 1 MV standing-wave cavity gains 1.000 MeV in either
    state; a particle initially 1 mm off axis leaves an off 2 MV cavity at x = 0.8237 mm instead of
    1 mm.

    The script compiles a Fortran program that tracks an electron at p0c = 5 MeV/c, 1 mm off axis,
    through a 1 m, 1 MV, 1 GHz lcavity with the default standing-wave model, first on and then off,
    re-running the lattice bookkeeping after each change. It compares the physical energy gain and the
    exit x and px with a drift for the off cavity. The on cavity is the control: it must give the full
    1 MeV on crest and a focusing kick.
    """

    import numpy as np
    from _bmad import compare, fortran, verdict

    LATTICE = """no_digested
    parameter[particle] = electron
    parameter[p0c] = 5e6
    parameter[geometry] = open
    lc: lcavity, l = 1, rf_frequency = 1e9, voltage = 1e6, phi0 = 0, n_rf_steps = 4
    ln: line = (lc)
    use, ln
    """

    PROGRAM = """
    program off_probe
    use bmad
    implicit none
    type(lat_struct), target :: lat
    type(ele_struct), pointer :: ele
    type(coord_struct) :: a, b
    integer :: j
    real(rp) :: e0, e1
    call bmad_parser('lat.bmad', lat)
    ele => lat%ele(1)
    do j = 1, 2
      ele%is_on = (j == 1)
      call set_ele_status_stale(ele, attribute_group$)
      call lattice_bookkeeper(lat)
      call init_coord(a, [1e-3_rp, 0.0_rp, 0.0_rp, 0.0_rp, 0.0_rp, 0.0_rp], ele=ele, element_end=upstream_end$)
      call track1(a, ele, lat%param, b)
      if (b%state /= alive$) error stop 'particle lost'
      ! Physical total energies before and after the element, eV.
      e0 = hypot(a%p0c * (1 + a%vec(6)), mass_of(a%species))
      e1 = hypot(b%p0c * (1 + b%vec(6)), mass_of(b%species))
      print '(a,l1,4es25.16e3)', 'RESULT ', ele%is_on, e1 - e0, b%vec(1), b%vec(2), b%p0c
    enddo
    end program
    """

    log = fortran(PROGRAM, files={"lat.bmad": LATTICE}).log
    rows = {}
    for line in log.splitlines():
        words = line.split()
        if len(words) == 6 and words[0] == "RESULT":
            rows[words[1]] = np.array([float(w) for w in words[2:]])
    if set(rows) != {"T", "F"}:
        raise RuntimeError(f"Missing results:\n{log}")

    # Control: on crest a 1 MV cavity gives 1 MeV and focuses an off-axis particle (px < 0).
    on = rows["T"]
    if compare("on: energy gain (eV), expected V = 1e6", on[0], 1e6, atol=1e-3):
        raise RuntimeError("The on cavity does not give its voltage; the comparison is not valid.")
    if not on[2] < 0:
        raise RuntimeError(f"The on cavity does not focus: px = {on[2]}")
    print(f"on: exit x (m), px: {float(on[1])!r}, {float(on[2])!r}")
    # Off: the reference momentum still rises to the on value (the documented reference profile),
    # but the particle drifts: no energy change, x stays 1 mm and px stays 0.
    off = rows["F"]
    reference_changed = compare(
        "off: exit reference p0c (eV/c), expected the on cavity's reference p0c",
        off[3],
        on[3],
        atol=0,
        rtol=1e-15,
    )
    if reference_changed:
        raise RuntimeError("The off cavity's reference profile changed; comparison is not valid.")
    # Rounding of the two 5-6 MeV total energies is 1e-9 eV.
    defect = compare("off: energy gain (eV), expected a drift: 0", off[0], 0.0, atol=1e-6)
    defect |= compare(
        "off: exit x (m), px, expected a drift: 1e-3, 0", off[1:3], [1e-3, 0.0], atol=1e-15
    )
    verdict(defect)

    ```

## Provenance

| Role | Model | Effort | Date | Code examined |
|---|---|---|---|---|
| Found | `gpt-6.1-sol` | xhigh | 2026-10-21 | 20261020-0 (`19b8afb46a40`), source, production, CPU, double, serial |
| Confirmed | `gpt-6-astra` | xhigh | 2026-10-21 | 20261020-0 (`19b8afb46a40`), source, production, CPU, double, serial |

