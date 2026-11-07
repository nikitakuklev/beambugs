---
title: "TRK-62: Placed wall3d apertures ignore element placement at laboratory-coordinate checks"
hide:
  - toc
---

# TRK-62 — Placed wall3d apertures ignore element placement at laboratory-coordinate checks

| | |
|---|---|
| Code | [Bmad](../index.md) · Tracking, RF, fields and spin |
| Priority | P3 |
| Severity | Wrong physics |
| Evidence | Reproduced, independently rechecked |
| Last reproduced | 2026-11-06 on Bmad release [20261031-0](https://github.com/bmad-sim/bmad-ecosystem/tree/20261031-0) (`c4c629f`) |
| Source | [check_aperture_limit placement, bmad/low_level/check_aperture_limit.f90:174–185](https://github.com/bmad-sim/bmad-ecosystem/blob/c4c629fc9071d3a56cb1ba5bc714baba65e9c5d3/bmad/low_level/check_aperture_limit.f90#L174-L185)<br>[ordinary wall3d branch, bmad/low_level/check_aperture_limit.f90:326–328](https://github.com/bmad-sim/bmad-ecosystem/blob/c4c629fc9071d3a56cb1ba5bc714baba65e9c5d3/bmad/low_level/check_aperture_limit.f90#L326-L328)<br>[wall-transition branch, bmad/low_level/check_aperture_limit.f90:249–254](https://github.com/bmad-sim/bmad-ecosystem/blob/c4c629fc9071d3a56cb1ba5bc714baba65e9c5d3/bmad/low_level/check_aperture_limit.f90#L249-L254)<br>[wall3d_to_position, bmad/modules/wall3d_mod.f90:1268–1269](https://github.com/bmad-sim/bmad-ecosystem/blob/c4c629fc9071d3a56cb1ba5bc714baba65e9c5d3/bmad/modules/wall3d_mod.f90#L1268-L1269)<br>[wall selection, bmad/modules/wall3d_mod.f90:1200–1211](https://github.com/bmad-sim/bmad-ecosystem/blob/c4c629fc9071d3a56cb1ba5bc714baba65e9c5d3/bmad/modules/wall3d_mod.f90#L1200-L1211)<br>[wall3d aperture contract, bmad/doc/attributes.tex:902–907](https://github.com/bmad-sim/bmad-ecosystem/blob/c4c629fc9071d3a56cb1ba5bc714baba65e9c5d3/bmad/doc/attributes.tex#L902-L907)<br>[aperture-placement contract, bmad/doc/attributes.tex:944–962](https://github.com/bmad-sim/bmad-ecosystem/blob/c4c629fc9071d3a56cb1ba5bc714baba65e9c5d3/bmad/doc/attributes.tex#L944-L962)<br>[independent branch-wall representation, bmad/doc/attributes.tex:1749–1756](https://github.com/bmad-sim/bmad-ecosystem/blob/c4c629fc9071d3a56cb1ba5bc714baba65e9c5d3/bmad/doc/attributes.tex#L1749-L1756)<br>[local-reference coordinates, bmad/doc/attributes.tex:1787–1788](https://github.com/bmad-sim/bmad-ecosystem/blob/c4c629fc9071d3a56cb1ba5bc714baba65e9c5d3/bmad/doc/attributes.tex#L1787-L1788)<br>[wall-vertex and interior-origin restrictions, bmad/doc/attributes.tex:1578–1624](https://github.com/bmad-sim/bmad-ecosystem/blob/c4c629fc9071d3a56cb1ba5bc714baba65e9c5d3/bmad/doc/attributes.tex#L1578-L1624)<br>[wall-transition contract, bmad/doc/attributes.tex:1024–1029](https://github.com/bmad-sim/bmad-ecosystem/blob/c4c629fc9071d3a56cb1ba5bc714baba65e9c5d3/bmad/doc/attributes.tex#L1024-L1029)<br>[entrance, bmad/code/track1.f90:196–196](https://github.com/bmad-sim/bmad-ecosystem/blob/c4c629fc9071d3a56cb1ba5bc714baba65e9c5d3/bmad/code/track1.f90#L196-L196)<br>[exit, bmad/code/track1.f90:329–329](https://github.com/bmad-sim/bmad-ecosystem/blob/c4c629fc9071d3a56cb1ba5bc714baba65e9c5d3/bmad/code/track1.f90#L329-L329)<br>[s-based RK frame boundary, bmad/low_level/track1_runge_kutta.f90:83–109](https://github.com/bmad-sim/bmad-ecosystem/blob/c4c629fc9071d3a56cb1ba5bc714baba65e9c5d3/bmad/low_level/track1_runge_kutta.f90#L83-L109)<br>[in-step aperture call, bmad/modules/runge_kutta_mod.f90:193–195](https://github.com/bmad-sim/bmad-ecosystem/blob/c4c629fc9071d3a56cb1ba5bc714baba65e9c5d3/bmad/modules/runge_kutta_mod.f90#L193-L195) |

With a wall3d aperture and offset_moves_aperture enabled, laboratory-coordinate aperture checks ignore element translation and collimator tilt. A 4 × 10 mm aperture translated by (10.00, −20.00) mm rejects its center and accepts the laboratory origin. A 90° collimator roll also exchanges a false loss and a false survival. Independent interval containment, equivalent placed wall geometry and ordinary rectangular-aperture controls establish the error. Public entrance/exit checks and wall-transition checks are affected; isolated in-step s-based Runge–Kutta checks already receive body coordinates and pass.

Run `python TRK-62.py` with [_bmad.py](../helpers/_bmad.py) in the same directory.

??? example "Reproducer: TRK-62.py"

    [Download TRK-62.py](TRK-62.py)

    ```python
    """TRK-62: Placed wall3d apertures ignore element placement at laboratory-coordinate checks

    With a wall3d aperture and offset_moves_aperture enabled, laboratory-coordinate aperture checks ignore element translation and collimator tilt. A 4 × 10 mm aperture translated by (10.00, −20.00) mm rejects its center and accepts the laboratory origin. A 90° collimator roll also exchanges a false loss and a false survival. Independent interval containment, equivalent placed wall geometry and ordinary rectangular-aperture controls establish the error. Public entrance/exit checks and wall-transition checks are affected; isolated in-step s-based Runge–Kutta checks already receive body coordinates and pass.

    A rectangular wall with unequal side lengths supplies exact interval containment and quarter-turn expectations. Fresh processes compare direct checks and public tracking on collimator and quadrupole hosts, with explicit wall-geometry, rectangular-aperture and fixed-aperture controls. Two-record crossings and isolated body-coordinate Runge–Kutta checks distinguish the caller frames;
    """

    from itertools import product
    from pathlib import Path

    import numpy as np
    from _bmad import _case, _execute, _recipe, _scratch, compare, verdict

    PROGRAM = r"""
    program confirmation
    use bmad
    implicit none
    type(lat_struct), target :: lat
    type(coord_struct) orb,a,b,old
    real(rp) x,y,v(6),xold,yold
    integer n,j,transition
    logical err
    call bmad_parser('lat.bmad',lat,err_flag=err)
    if(err) error stop 'lattice rejected'
    bmad_com%aperture_limit_on=.true.
    open(21,file='input.dat',status='old');read(21,*) n,transition
    write(*,'(a,l2,3es26.17)') 'PLACEMENT ',lat%ele(1)%offset_moves_aperture, &
     lat%ele(1)%value(x_offset_tot$),lat%ele(1)%value(y_offset_tot$),lat%ele(1)%value(tilt_tot$)
    do j=1,n
      read(21,*) x,y,xold,yold
      v=[x,0.0_rp,y,0.0_rp,0.0_rp,0.0_rp]
      call init_coord(orb,v,lat%ele(1),upstream_end$)
      orb%s=lat%ele(1)%s_start+0.5_rp
      lat%ele(1)%aperture_at=continuous$
      if(transition==1) then
        lat%ele(1)%aperture_at=wall_transition$
        v=[xold,0.0_rp,yold,0.0_rp,0.0_rp,0.0_rp]
        call init_coord(old,v,lat%ele(1),upstream_end$)
        old%s=lat%ele(1)%s_start+0.25_rp
        call check_aperture_limit(orb,lat%ele(1),in_between$,lat%param,old)
      else
        call check_aperture_limit(orb,lat%ele(1),in_between$,lat%param)
      endif
      v=[x,0.0_rp,y,0.0_rp,0.0_rp,0.0_rp]
      call init_coord(a,v,lat%ele(1),upstream_end$)
      lat%ele(1)%aperture_at=both_ends$
      call track1(a,lat%ele(1),lat%param,b)
      write(*,'(a,i5,2i3)') 'RESULT ',j,merge(1,0,orb%state==alive$),merge(1,0,b%state==alive$)
    enddo
    close(21)
    end program
    """

    RK_PROGRAM = r"""
    program confirmation
    use bmad
    implicit none
    type(lat_struct),target :: lat
    type(coord_struct) orb
    real(rp) v(6)
    integer i
    logical err
    call bmad_parser('lat.bmad',lat,err_flag=err)
    if(err) error stop 'lattice rejected'
    bmad_com%aperture_limit_on=.true.
    open(21,file='input.dat',status='old')
    do i=1,4
     read(21,*) v
     call init_coord(orb,v,lat%ele(1),upstream_end$)
     ! Isolate the in-step s-based RK path. Public track1 adds lab-frame end checks.
     call track1_runge_kutta(orb,lat%ele(1),lat%param,err)
     if(err) error stop 'RK failed'
     write(*,'(a,i4,i3,2es26.17)') 'RESULT ',i,merge(1,0,orb%state==alive$),orb%vec(1),orb%vec(3)
    enddo
    end program
    """

    POINTS = np.array([[0, 0], [0.003, 0], [0, 0.003], [0, 0.006]])
    VERTICES = np.array([[-0.002, -0.005], [0.002, -0.005], [0.002, 0.005], [-0.002, 0.005]])
    PLACEMENTS = {
        "center": (np.zeros(2), 0.0),
        "plus": (np.array([0.01, -0.02]), 0.0),
        "minus": (np.array([-0.01, 0.02]), 0.0),
        "turn+": (np.zeros(2), np.pi / 2),
        "turn-": (np.zeros(2), -np.pi / 2),
        "both": (np.array([0.01, -0.02]), np.pi / 2),
    }


    def rotation(angle: float) -> np.ndarray:
        return np.array([[np.cos(angle), -np.sin(angle)], [np.sin(angle), np.cos(angle)]])


    def inside(points: np.ndarray) -> np.ndarray:
        # The counterclockwise straight-sided wall is exactly the product of intervals.
        # Every sample is at least 1 mm from the relevant interval boundary.
        return (np.abs(points[:, 0]) < 0.002) & (np.abs(points[:, 1]) < 0.005)


    def lattice(
        key: str,
        shape: str,
        moves: bool,
        d: np.ndarray,
        angle: float,
        explicit: bool = False,
        rk: bool = False,
    ) -> str:
        aperture_angle = angle if key != "quadrupole" else 0.0
        vertices = VERTICES @ rotation(aperture_angle).T if explicit else VERTICES
        origin = d if explicit else np.zeros(2)
        element_d = np.zeros(2) if explicit else d
        element_angle = 0.0 if explicit else angle
        sections = []
        for ss in (0, 1):
            vv = ",".join(f"v({i})={{{x:.17g},{y:.17g}}}" for i, (x, y) in enumerate(vertices, 1))
            sections.append(f"section={{s={ss},r0=({origin[0]:.17g},{origin[1]:.17g}),{vv}}}")
        wall = ", &\n".join(sections)
        return f"""no_digested
    parameter[geometry]=open
    parameter[particle]=electron
    beginning[p0c]=1e9
    e: {key},l=1,aperture_type={shape},aperture_at=continuous, &
     tracking_method={"runge_kutta" if rk else "bmad_standard"}, &
     offset_moves_aperture={"T" if moves else "F"}, &
     x_offset={element_d[0]:.17g},y_offset={element_d[1]:.17g},tilt={element_angle:.17g}, &
     x_limit=.002,y_limit=.005,wall={{{wall}}}
    lat: line=(e)
    use,lat
    """


    def compile_client(work: Path, program: str) -> None:
        work.mkdir()
        (work / "probe.f90").write_text(program)
        comp, link = _recipe(False)
        _execute(comp, work, "", 290)
        _execute(link, work, "", 290)


    def measure(work: Path, number: int, lat: str, data: str, columns: int) -> np.ndarray:
        result = _case(
            [str(work / "probe")], work / str(number), {"lat.bmad": lat, "input.dat": data}, (), "", 290
        )
        values = np.array(
            [
                list(map(int, line.split()[2 : 2 + columns]))
                for line in result.log.splitlines()
                if line.startswith("RESULT ")
            ]
        )
        if values.ndim != 2 or values.shape[1] != columns or not np.isin(values, [0, 1]).all():
            raise RuntimeError(result.log)
        return values


    def main() -> None:
        defect = False
        cases = 0
        failures = 0
        wrong_points = 0
        configs = [
            (host, "wall3d", True, name, False)
            for host, name in product(("rcollimator", "ecollimator", "quadrupole"), PLACEMENTS)
        ]
        configs += [("rcollimator", "wall3d", True, name, True) for name in PLACEMENTS]
        configs += [
            ("rcollimator", "rectangular", True, name, False)
            for name in ("plus", "turn+", "turn-", "both")
        ]
        configs += [(host, "wall3d", False, "both", False) for host in ("rcollimator", "quadrupole")]
        with _scratch() as tmp:
            boundary = Path(tmp) / "boundary"
            compile_client(boundary, PROGRAM)
            for host, shape, moves, name, explicit in configs:
                d, angle = PLACEMENTS[name]
                # Magnetic-element aperture tilt is intentionally ignored; placement
                # disabled keeps the aperture and test particles in the reference frame.
                effective_d = d if moves else np.zeros(2)
                effective_angle = angle if moves and host != "quadrupole" else 0.0
                rot = rotation(effective_angle)
                lab = POINTS @ rot.T + effective_d
                body = POINTS.copy()
                if np.any(effective_d):
                    lab = np.vstack((lab, [0, 0]))
                    body = np.vstack((body, -effective_d @ rot))
                expected = inside(body).astype(int)
                data = f"{len(lab)} 0\n" + "".join(f"{x:.17g} {y:.17g} 0 0\n" for x, y in lab)
                observed = measure(
                    boundary, cases, lattice(host, shape, moves, d, angle, explicit), data, 2
                )
                cases += 1
                if observed.shape != (len(lab), 2) or np.any(observed[:, 0] != observed[:, 1]):
                    raise RuntimeError("direct/public aperture checks disagree")
                label = (
                    f"direct/public {host} {shape} moves={int(moves)} {name} geometry={int(explicit)}"
                )
                bad = compare(
                    label + "; |x_body|<2mm and |y_body|<5mm", observed[:, 0], expected, atol=0
                )
                control = (
                    explicit
                    or not moves
                    or shape == "rectangular"
                    or name == "center"
                    or (host == "quadrupole" and name.startswith("turn"))
                )
                if control and bad:
                    raise RuntimeError("aperture placement control failed")
                failures += int(bad)
                wrong_points += int(np.count_nonzero(observed[:, 0] != expected))
                defect |= bad

            # A wall transition loses both inside->outside and outside->inside crossings.
            old = np.array([[0, 0], [0.003, 0], [0, 0.003]])
            current = np.array([[0.003, 0], [0, 0], [0, 0]])
            expected = (inside(old) == inside(current)).astype(int)
            for name in ("center", "plus", "turn+", "both"):
                d, angle = PLACEMENTS[name]
                rot = rotation(angle)
                aa, bb = old @ rot.T + d, current @ rot.T + d
                data = "3 1\n" + "".join(
                    f"{x:.17g} {y:.17g} {xo:.17g} {yo:.17g}\n"
                    for (x, y), (xo, yo) in zip(bb, aa, strict=True)
                )
                observed = measure(
                    boundary, cases, lattice("rcollimator", "wall3d", True, d, angle), data, 2
                )[:, 0]
                cases += 1
                if observed.shape != (3,):
                    raise RuntimeError("missing wall-transition results")
                bad = compare(
                    f"wall-transition {name}; inside(old)==inside(current)", observed, expected, atol=0
                )
                if name == "center" and bad:
                    raise RuntimeError("centered crossing control failed")
                failures += int(bad)
                defect |= bad

            rk = Path(tmp) / "rk"
            compile_client(rk, RK_PROGRAM)
            for name in ("center", "plus", "turn+", "both"):
                d, angle = PLACEMENTS[name]
                lab = POINTS @ rotation(angle).T + d
                data = "".join(f"{x:.17g} 0 {y:.17g} 0 0 0\n" for x, y in lab)
                observed = measure(
                    rk, cases, lattice("rcollimator", "wall3d", True, d, angle, rk=True), data, 1
                )[:, 0]
                cases += 1
                if observed.shape != (4,):
                    raise RuntimeError("missing in-step RK results")
                if compare(
                    f"isolated body-coordinate RK {name}; interval containment",
                    observed,
                    inside(POINTS).astype(int),
                    atol=0,
                ):
                    raise RuntimeError("body-coordinate RK control failed")
        print(
            f"{cases} configurations; {failures} fail placement; "
            f"{wrong_points} containment errors per API; {cases - failures} controls pass"
        )
        verdict(defect)


    if __name__ == "__main__":
        main()

    ```

## Provenance

| Role | Model | Effort | Date | Code examined |
|---|---|---|---|---|
| Found | `gpt-6-astra` | xhigh | 2026-11-06 | 20261031-0 (`c4c629fc9071d3a56cb1ba5bc714baba65e9c5d3`), source, production, GNU Fortran 15.3.0, CPU, double, OpenMP one thread, MPI disabled |
| Confirmed | `gpt-6.1-sol` | xhigh | 2026-11-06 | 20261031-0 (`c4c629fc9071d3a56cb1ba5bc714baba65e9c5d3`), source, production, GNU Fortran 15.3.0, CPU, double, OpenMP one thread, MPI disabled |

