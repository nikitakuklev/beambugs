---
title: "TRK-35: Exact curved-multipole bends overwrite the kick-factor derivative"
hide:
  - toc
---

# TRK-35 — Exact curved-multipole bends overwrite the kick-factor derivative

| | |
|---|---|
| Code | [Bmad](../index.md) · Tracking, RF, fields and spin |
| Priority | P2 |
| Severity | Silent wrong result |
| Evidence | Probe-pinned |
| Last reproduced | 2026-11-05 on Bmad release [20261031-0](https://github.com/bmad-sim/bmad-ecosystem/tree/20261031-0) (`c4c629f`) |
| Source | [`track_a_bend::apply_multipole_kicks`, bmad/low_level/track_a_bend.f90:397–409](https://github.com/bmad-sim/bmad-ecosystem/blob/c4c629fc9071d3a56cb1ba5bc714baba65e9c5d3/bmad/low_level/track_a_bend.f90#L397-L409)<br>[native `make_mat6_bmad` dispatch, bmad/low_level/make_mat6_bmad.f90:68–75](https://github.com/bmad-sim/bmad-ecosystem/blob/c4c629fc9071d3a56cb1ba5bc714baba65e9c5d3/bmad/low_level/make_mat6_bmad.f90#L68-L75) |

Analytic transfer matrices for bends with exact curved multipoles disagree with derivatives of their particle map. The exact-basis branch overwrites the derivative of the position-dependent kick factor instead of multiplying that derivative by the integration step. In a 0.8 m bend with curvature 0.2 m⁻¹ and quadrupole strength 0.7 m⁻², the horizontal-pure matrix gives −0.5564 m⁻¹ for the response of exit normalized horizontal momentum to entrance horizontal position, instead of −0.5575 m⁻¹ (0.1984% error). The vertical-pure basis has the same defect; exact_multipoles = off passes, and requesting a matrix leaves the particle trajectory unchanged.

Run `python TRK-35.py` with [_bmad.py](../helpers/_bmad.py) and [_orbital_matrices_radiation.py](../helpers/_orbital_matrices_radiation.py) in the same directory.

??? example "Reproducer: TRK-35.py"

    [Download TRK-35.py](TRK-35.py)

    ```python
    """TRK-35: Exact curved-multipole bends overwrite the kick-factor derivative

    Analytic transfer matrices for bends with exact curved multipoles disagree with derivatives of their particle map. The exact-basis branch overwrites the derivative of the position-dependent kick factor instead of multiplying that derivative by the integration step. In a 0.8 m bend with curvature 0.2 m⁻¹ and quadrupole strength 0.7 m⁻², the horizontal-pure matrix gives −0.5564 m⁻¹ for the response of exit normalized horizontal momentum to entrance horizontal position, instead of −0.5575 m⁻¹ (0.1984% error). The vertical-pure basis has the same defect; exact_multipoles = off passes, and requesting a matrix leaves the particle trajectory unchanged.

    The script checks both exact bases and the off control using fourth-order particle finite differences at two step sizes. It verifies stencil convergence and identical central trajectories with and without matrix construction.
    """

    import numpy as np
    from _bmad import compare, fortran, verdict
    from _orbital_matrices_radiation import records

    PROGRAM = """
    program probe
    use bmad

    implicit none
    type(lat_struct), target :: lat
    type(coord_struct) a,b
    real(rp) v(6)
    logical err
    integer i,j,k,q
    real(rp) h,mat(6,6),dm(6,6),xp(6),acc(6),central(6)
    integer,parameter::offset(4)=[-2,-1,1,2],weight(4)=[1,-8,8,-1]
    character(50) label
    call bmad_parser('lat.bmad',lat,err_flag=err)
    if (err) error stop 'parse failed'
    v=[.012_rp,.025_rp,-.009_rp,-.013_rp,.03_rp,.2_rp]
    do i=1,3
     call init_coord(a,v,lat%ele(i),upstream_end$)
     call make_mat6(lat%ele(i),lat%param,a,b,err)
     if (err.or.b%state/=alive$) error stop 'matrix failed'
     mat=lat%ele(i)%mat6;central=b%vec
     call track1(a,lat%ele(i),lat%param,b,err_flag=err)
     if (err.or.b%state/=alive$) error stop 'central tracking failed'
     write(*,'(a,i0,es26.17)') 'RESULT orbit',i,maxval(abs(central-b%vec))
     do k=1,2
      h=4e-5_rp/2**k
      do j=1,6
       acc=0
       do q=1,4
        xp=v;xp(j)=xp(j)+offset(q)*h
        call init_coord(a,xp,lat%ele(i),upstream_end$)
        call track1(a,lat%ele(i),lat%param,b,err_flag=err)
        if (err.or.b%state/=alive$) error stop 'stencil tracking failed'
        acc=acc+weight(q)*b%vec
       enddo
       dm(:,j)=acc/(12*h)
      enddo
      do j=1,6
       write(*,'(a,3i1,6es26.17)') 'RESULT fd',i,k,j,dm(j,:)
      enddo
     enddo
     do j=1,6
      write(*,'(a,2i1,6es26.17)') 'RESULT mat',i,j,mat(j,:)
     enddo
    enddo
    end program
    """
    LATTICE = """
    no_digested
    parameter[geometry]=open
    parameter[particle]=electron
    beginning[p0c]=1e9
    e: sbend,l=.8,g=.2,k1=.7,num_steps=8,fringe_type=none,exact_multipoles=off
    h: e,exact_multipoles=horizontally_pure
    v: e,exact_multipoles=vertically_pure
    lat: line=(e,h,v)
    use,lat
    """
    rows = records(fortran(PROGRAM, files={"lat.bmad": LATTICE}).log)
    defect = False
    for i, basis in enumerate(("off", "horizontally_pure", "vertically_pure"), 1):
        mat = np.array([rows[f"mat{i}{j}"] for j in range(1, 7)])
        fd = [np.array([rows[f"fd{i}{k}{j}"] for j in range(1, 7)]) for k in (1, 2)]
        if compare(f"{basis} central orbit invariance", rows[f"orbit{i}"], [0], atol=2e-15):
            raise RuntimeError("Matrix changed the central orbit")
        if compare(
            f"{basis} stencil convergence maximum", np.max(np.abs(fd[0] - fd[1])), 0, atol=2e-11
        ):
            raise RuntimeError("Particle stencils did not converge")
        mismatch = compare(
            f"{basis} maximum analytic/particle Jacobian residual",
            np.max(np.abs(mat - fd[1])),
            0,
            atol=2e-11,
        )
        compare(f"{basis} M21 (1/m), particle derivative", mat[1, 0], fd[1][1, 0], atol=2e-11)
        if i == 1 and mismatch:
            raise RuntimeError("Off-basis control failed")
        if i > 1:
            defect |= mismatch
    verdict(defect)

    ```

## Provenance

| Role | Model | Effort | Date | Code examined |
|---|---|---|---|---|
| Found | `gpt-6.1-sol` | xhigh | 2026-10-20 | 20261020-0 (`19b8afb46a40`), source, production, CPU, double, serial |
| Confirmed | `gpt-6-astra` | xhigh | 2026-10-21 | 20261020-0 (`19b8afb46a40`), source, production, CPU, double, serial |
| Confirmed | `gpt-6-astra` | xhigh | 2026-10-21 | 20261020-0 (`19b8afb46a40`), source, production, CPU, double, serial |

