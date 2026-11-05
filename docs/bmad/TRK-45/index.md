---
title: "TRK-45: offset_particle promises multipole kicks that it does not apply"
hide:
  - toc
---

# TRK-45 — offset_particle promises multipole kicks that it does not apply

| | |
|---|---|
| Code | [Bmad](../index.md) · Tracking, RF, fields and spin |
| Priority | P4 |
| Severity | Documentation or example mismatch |
| Evidence | Reproduced, independently rechecked |
| Last reproduced | 2026-11-05 on Bmad 20261031-0 (`c4c629f`) |
| Source | [offset_particle header](https://github.com/bmad-sim/bmad-ecosystem/blob/c4c629fc9071d3a56cb1ba5bc714baba65e9c5d3/bmad/code/offset_particle.f90#L5-L8)<br>[steering dispatch](https://github.com/bmad-sim/bmad-ecosystem/blob/c4c629fc9071d3a56cb1ba5bc714baba65e9c5d3/bmad/code/offset_particle.f90#L157-L171)<br>[entry kicks and tilt](https://github.com/bmad-sim/bmad-ecosystem/blob/c4c629fc9071d3a56cb1ba5bc714baba65e9c5d3/bmad/code/offset_particle.f90#L298-L331)<br>[exit kicks and tilt](https://github.com/bmad-sim/bmad-ecosystem/blob/c4c629fc9071d3a56cb1ba5bc714baba65e9c5d3/bmad/code/offset_particle.f90#L341-L378) |

Calling offset_particle directly with nonzero multipole arrays applies no multipole kick, despite its header promising half kicks from multipole and steering attributes. Its actual kicks come from the steering attributes, and the element trackers apply multipole arrays separately. Six element types reproduce the intended steering-frame behavior, and an isolated quadrupole with nonzero skew-quadrupole and normal-sextupole arrays applies only the coordinate rotation at each edge. The header should describe that division of responsibility.

Download these shared helpers into the same directory as the reproducer scripts:

- [Download _bmad.py](../helpers/_bmad.py) · [View source](../helpers/index.md#_bmad)
- [Download _tracking_contracts_materials.py](../helpers/_tracking_contracts_materials.py) · [View source](../helpers/index.md#_tracking_contracts_materials)

??? example "Reproducer: TRK-45.py"

    [Download TRK-45.py](TRK-45.py)

    ```python
    """TRK-45: offset_particle promises multipole kicks that it does not apply

    Calling offset_particle directly with nonzero multipole arrays applies no multipole kick,
    despite its header promising half kicks from multipole and steering attributes. Its actual
    kicks come from the steering attributes, and the element trackers apply multipole arrays
    separately. Six element types reproduce the intended steering-frame behavior, and an
    isolated quadrupole with nonzero skew-quadrupole and normal-sextupole arrays applies only
    the coordinate rotation at each edge. The header should describe that division of
    responsibility.

    The script embeds the six-element frame probe and compares entry and exit coordinates with
    independent rotations. With steering strengths zero and the default kick option retained, it
    checks each edge separately with nonzero and zero arrays. It reads the promised multipole
    wording from BMAD_SOURCE."""

    import numpy as np
    from _bmad import compare, fortran, source, verdict
    from _tracking_contracts_materials import rows

    PROGRAM = """
    program offset_kicks
    use bmad
    implicit none
    type(ele_struct) ele
    type(coord_struct) initial, orb
    integer i, keys(6)
    character(16) name
    keys = [quadrupole$, sextupole$, kicker$, hkicker$, vkicker$, elseparator$]
    do i = 1, size(keys)
      call init_ele(ele, keys(i))
      ele%ref_species = positron$
      ele%value(p0c$) = 1.0e6_rp
      ele%value(p0c_start$) = 1.0e6_rp
      ele%value(e_tot$) = sqrt(1.0e12_rp + m_electron**2)
      ele%value(e_tot_start$) = ele%value(e_tot$)
      ele%value(tilt_tot$) = 0.37_rp
      if (keys(i) == hkicker$) then
        ele%value(kick$) = 0.003_rp
      elseif (keys(i) == vkicker$) then
        ele%value(kick$) = -0.005_rp
      else
        ele%value(hkick$) = 0.003_rp
        ele%value(vkick$) = -0.005_rp
      endif
      if (i == 1) then
        call multipole_init(ele, magnetic$)
        ele%a_pole(1) = 0.3_rp
        ele%b_pole(2) = 1.7_rp
        ele%scale_multipoles = .false.
      endif
      call init_coord(initial, [0.012_rp, 0.001_rp, -0.007_rp, -0.002_rp, 0.03_rp, 0.2_rp], ele, upstream_end$, positron$)
      orb = initial
      call offset_particle(ele, set$, orb, s_pos=0.0_rp)
      write(name, '(a,i0)') 'entry', i
      write(*, '(3a,6es25.16e3)') 'RESULT ', trim(name), ' ', orb%vec
      call offset_particle(ele, unset$, orb, s_pos=0.0_rp)
      write(name, '(a,i0)') 'exit', i
      write(*, '(3a,6es25.16e3)') 'RESULT ', trim(name), ' ', orb%vec
      if (i == 1) then
        ele%value(hkick$) = 0
        ele%value(vkick$) = 0
        orb = initial
        call offset_particle(ele, set$, orb, s_pos=0.0_rp)
        write(*, '(a,6es25.16e3)') 'RESULT no_hv_entry ', orb%vec
        call offset_particle(ele, unset$, orb, s_pos=0.0_rp)
        write(*, '(a,6es25.16e3)') 'RESULT no_hv ', orb%vec
        ele%a_pole = 0
        ele%b_pole = 0
        orb = initial
        call offset_particle(ele, set$, orb, s_pos=0.0_rp)
        write(*, '(a,6es25.16e3)') 'RESULT empty_entry ', orb%vec
        call offset_particle(ele, unset$, orb, s_pos=0.0_rp)
        write(*, '(a,6es25.16e3)') 'RESULT empty_exit ', orb%vec
      endif
    enddo
    end program
    """
    records = rows(fortran(PROGRAM).log)
    initial = np.array([0.012, 0.001, -0.007, -0.002, 0.03, 0.2])
    c, s = np.cos(0.37), np.sin(0.37)
    rotation = np.array([[c, s], [-s, c]])
    for i in range(1, 7):
        kick = np.array([0.003, -0.005])
        if i == 4:
            kick[1] = 0
        if i == 5:
            kick[0] = 0
        entry, end = initial.copy(), initial.copy()
        entry[[0, 2]] = rotation @ initial[[0, 2]]
        if i <= 2:
            entry[[1, 3]] = rotation @ (initial[[1, 3]] + 0.5 * kick)
            end[[1, 3]] += kick
        else:
            entry[[1, 3]] = rotation @ initial[[1, 3]] + 0.5 * kick
            end[[1, 3]] += rotation.T @ kick
        for label, expected in [(f"entry{i}", entry), (f"exit{i}", end)]:
            if compare(
                f"{label} steering-frame control", records[label], expected, atol=2e-17, rtol=4e-15
            ):
                raise AssertionError("Steering-frame control failed")
    # Check each edge: a round trip alone could hide cancelling erroneous kicks.
    rotated = initial.copy()
    rotated[[0, 2]] = rotation @ initial[[0, 2]]
    rotated[[1, 3]] = rotation @ initial[[1, 3]]
    for label, expected in [("empty_entry", rotated), ("empty_exit", initial)]:
        if compare(
            f"{label}: zero-array rotation control", records[label], expected, atol=2e-17, rtol=4e-15
        ):
            raise AssertionError("Zero-array frame control failed")
    no_multipole_kicks = []
    for label, expected in [("no_hv_entry", rotated), ("no_hv", initial)]:
        no_multipole_kicks.append(
            not compare(
                f"{label}: multipole-only orbit, pure frame transformation expected",
                records[label],
                expected,
                atol=2e-17,
                rtol=4e-15,
            )
        )
    no_multipole_kick = all(no_multipole_kicks)
    header = (source() / "bmad/code/offset_particle.f90").read_text().split("!-", 1)[0]
    promised = "half kicks due to multipole and kick attributes" in header
    print(
        f"header promises multipole kicks={promised}; isolated arrays apply no kick={no_multipole_kick}; expected consistent contract"
    )
    verdict(promised and no_multipole_kick)

    ```

## Provenance

| Role | Model | Effort | Date | Code examined |
|---|---|---|---|---|
| Found | `gemini-4-argon-high` |  | 2026-10-20 | 20261020-0 (`19b8afb46a40`), source, production, CPU, double, serial |
| Confirmed | `gpt-6-astra` | xhigh | 2026-10-20 | 20261020-0 (`19b8afb46a40`), source, production, CPU, double, serial |

