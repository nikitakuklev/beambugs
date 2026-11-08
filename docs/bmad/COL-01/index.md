---
title: "COL-01: Uniform-charge deposition uses an uninitialized charge at 50,000 particles or more"
hide:
  - toc
---

# COL-01 — Uniform-charge deposition uses an uninitialized charge at 50,000 particles or more

| | |
|---|---|
| Code | [Bmad](../index.md) · Collective effects |
| Priority | P4 |
| Failure | Wrong physics |
| Evidence | Reproduced, independently rechecked |
| Last reproduced | 2026-11-05 on Bmad release [20261031-0](https://github.com/bmad-sim/bmad-ecosystem/tree/20261031-0) (`c4c629f`) |
| Source | [`deposit_particles`, bmad/space_charge/open_spacecharge_mod.f90:303–305](https://github.com/bmad-sim/bmad-ecosystem/blob/c4c629fc9071d3a56cb1ba5bc714baba65e9c5d3/bmad/space_charge/open_spacecharge_mod.f90#L303-L305)<br>[lines 315–318](https://github.com/bmad-sim/bmad-ecosystem/blob/c4c629fc9071d3a56cb1ba5bc714baba65e9c5d3/bmad/space_charge/open_spacecharge_mod.f90#L315-L318)<br>[lines 343–363](https://github.com/bmad-sim/bmad-ecosystem/blob/c4c629fc9071d3a56cb1ba5bc714baba65e9c5d3/bmad/space_charge/open_spacecharge_mod.f90#L343-L363)<br>[lines 134–153](https://github.com/bmad-sim/bmad-ecosystem/blob/c4c629fc9071d3a56cb1ba5bc714baba65e9c5d3/bmad/space_charge/open_spacecharge_mod.f90#L134-L153) |

Calling the space-charge helper `deposit_particles` with `total_charge` and at least 50,000 particles uses an uninitialized per-particle charge in its OpenMP region, corrupting the charge mesh even with one thread. The private charge is never assigned, although trilinear deposition must conserve the requested total charge. With total charge 1, the mesh sum is 1.000 at 49,999 particles and 0 at 50,000; an explicit per-particle charge array gives 1.000 at both counts. Bmad's space-charge and CSR tracking callers use explicit charges and avoid this branch.

Run `python COL-01.py` with [_bmad.py](../helpers/_bmad.py) in the same directory.

??? example "Reproducer: COL-01.py"

    [Download COL-01.py](COL-01.py)

    ```python
    """COL-01: Uniform-charge deposition uses an uninitialized charge at 50,000 particles or more

    Calling the space-charge helper `deposit_particles` with `total_charge` and at least 50,000
    particles uses an uninitialized per-particle charge in its OpenMP region, corrupting the charge
    mesh even with one thread. The private charge is never assigned, although trilinear deposition
    must conserve the requested total charge. With total charge 1, the mesh sum is 1.000 at 49,999
    particles and 0 at 50,000; an explicit per-particle charge array gives 1.000 at both counts.
    Bmad's space-charge and CSR tracking callers use explicit charges and avoid this branch.

    The script compiles a small Fortran program against the build that deposits a bunch of identical
    particles on an 8x8x8 mesh, once with `total_charge = 1` and once with the equivalent charge array
    `qa = 1/n`, at n = 49,999 (serial path) and n = 50,000 (OpenMP path), and compares the summed mesh
    charge with 1. The explicit-charge cases at both counts and the uniform case below the threshold
    are the controls. It runs with one OpenMP thread.
    """

    from _bmad import compare, fortran, verdict

    PROGRAM = """
    program deposit_probe
    use open_spacecharge_mod
    use, intrinsic :: iso_fortran_env, only: real64
    implicit none
    type(mesh3d_struct) m
    real(real64), allocatable :: x(:), y(:), z(:), q(:)
    integer n, k
    m%nhi = [8, 8, 8]
    do k = 1, 2
      n = 49998 + k
      allocate(x(n), y(n), z(n), q(n))
      ! All particles at one interior point of a fixed unit-cube mesh.
      x = 0.25_real64; y = 0.5_real64; z = 0.75_real64; q = 1.0_real64/n
      m%min = 0; m%max = 1; m%delta = 1.0_real64/7
      call deposit_particles(x, y, z, m, total_charge=1.0_real64, resize_mesh=.false.)
      print '(a,i0,1x,es25.16)', 'total_charge ', n, sum(m%rho)
      call deposit_particles(x, y, z, m, qa=q, resize_mesh=.false.)
      print '(a,i0,1x,es25.16)', 'qa ', n, sum(m%rho)
      deallocate(x, y, z, q)
    enddo
    end program
    """

    # Rounding of 50,000 eight-term trilinear sums stays below 1e-12 (observed 3.4e-13).
    ATOL = 1e-12

    log = fortran(PROGRAM).log
    sums = {}
    for line in log.splitlines():
        words = line.split()
        if len(words) == 3 and words[0] in ("total_charge", "qa"):
            sums[words[0], int(words[1])] = float(words[2])
    if len(sums) != 4:
        raise RuntimeError(f"Expected four deposited sums, got:\n{log}")

    controls = [
        compare(
            "qa = 1/n, n=49999 (serial path): sum(rho), expected total 1",
            sums["qa", 49999],
            1.0,
            atol=ATOL,
        ),
        compare(
            "qa = 1/n, n=50000 (OpenMP path): sum(rho), expected total 1",
            sums["qa", 50000],
            1.0,
            atol=ATOL,
        ),
        compare(
            "total_charge=1, n=49999 (serial path): sum(rho), expected total 1",
            sums["total_charge", 49999],
            1.0,
            atol=ATOL,
        ),
    ]
    if any(controls):
        raise RuntimeError("A control case failed; the comparison is not valid.")
    defect = compare(
        "total_charge=1, n=50000 (OpenMP path): sum(rho), expected total 1",
        sums["total_charge", 50000],
        1.0,
        atol=ATOL,
    )
    verdict(defect)

    ```

## Provenance

| Role | Model | Effort | Date | Code examined |
|---|---|---|---|---|
| Found | <span class="bb-nowrap">`gpt-6.1-sol`</span> | xhigh | <span class="bb-nowrap">2026-10-21</span> | 20261020-0 (`19b8afb46a40`), source, production, CPU, double, serial |
| Confirmed | <span class="bb-nowrap">`gpt-6-astra`</span> | xhigh | <span class="bb-nowrap">2026-10-21</span> | 20261020-0 (`19b8afb46a40`), source, production, CPU, double, serial |

