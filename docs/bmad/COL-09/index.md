---
title: "COL-09: S-based FFT space charge gives full kicks at both element endpoints"
hide:
  - toc
---

# COL-09 — S-based FFT space charge gives full kicks at both element endpoints

| | |
|---|---|
| Code | [Bmad](../index.md) · Collective effects |
| Priority | P1 |
| Failure | Wrong physics |
| Evidence | Probe-pinned |
| Last reproduced | 2026-11-05 on Bmad release [20261031-0](https://github.com/bmad-sim/bmad-ecosystem/tree/20261031-0) (`c4c629f`) |
| Source | [`track1_bunch_csr`, bmad/space_charge/csr_and_space_charge_mod.f90:228–229](https://github.com/bmad-sim/bmad-ecosystem/blob/c4c629fc9071d3a56cb1ba5bc714baba65e9c5d3/bmad/space_charge/csr_and_space_charge_mod.f90#L228-L229)<br>[line 297](https://github.com/bmad-sim/bmad-ecosystem/blob/c4c629fc9071d3a56cb1ba5bc714baba65e9c5d3/bmad/space_charge/csr_and_space_charge_mod.f90#L297)<br>[line 305](https://github.com/bmad-sim/bmad-ecosystem/blob/c4c629fc9071d3a56cb1ba5bc714baba65e9c5d3/bmad/space_charge/csr_and_space_charge_mod.f90#L305)<br>[lines 355–356](https://github.com/bmad-sim/bmad-ecosystem/blob/c4c629fc9071d3a56cb1ba5bc714baba65e9c5d3/bmad/space_charge/csr_and_space_charge_mod.f90#L355-L356)<br>[`apply_fft_3d_kicks`, line 1731](https://github.com/bmad-sim/bmad-ecosystem/blob/c4c629fc9071d3a56cb1ba5bc714baba65e9c5d3/bmad/space_charge/csr_and_space_charge_mod.f90#L1731)<br>[`track1_bunch_hom`, bmad/multiparticle/beam_mod.f90:319](https://github.com/bmad-sim/bmad-ecosystem/blob/c4c629fc9071d3a56cb1ba5bc714baba65e9c5d3/bmad/multiparticle/beam_mod.f90#L319)<br>[bmad/doc/param-structs.tex:487–493](https://github.com/bmad-sim/bmad-ecosystem/blob/c4c629fc9071d3a56cb1ba5bc714baba65e9c5d3/bmad/doc/param-structs.tex#L487-L493) |

In s-based tracking with `space_charge_method = fft_3d`, an element split into n space-charge steps receives n + 1 full-step kicks, including both endpoint kicks. For a straight element with a frozen field, their weights integrate over L(1 + 1/n) instead of the element length L. The slice and CSR solvers in the same loop halve the endpoint weights. For a low-charge bunch in a 1 mm drift, the one-step impulse is 4/3 of the two-step impulse and twice the force integrated over L. The excess is 100/n percent.

Run `python COL-09.py` with [_bmad.py](../helpers/_bmad.py) in the same directory.

??? example "Reproducer: COL-09.py"

    [Download COL-09.py](COL-09.py)

    ```python
    """COL-09: S-based FFT space charge gives full kicks at both element endpoints

    In s-based tracking with `space_charge_method = fft_3d`, an element split into n space-charge
    steps receives n + 1 full-step kicks, including both endpoint kicks. For a straight element with
    a frozen field, their weights integrate over L(1 + 1/n) instead of the element length L. The
    slice and CSR solvers in the same loop halve the endpoint weights. For a low-charge bunch in a 1
    mm drift, the one-step impulse is 4/3 of the two-step impulse and twice the force integrated
    over L. The excess is 100/n percent.

    The script compiles a Fortran program that tracks a 27-particle electron bunch of total charge
    1e-18 C, small enough that the bunch does not change its own field, through a 1 mm drift with
    `space_charge_method = fft_3d` at 1, 2 and 4 space-charge steps, one process per step count. It
    compares the first particle's px impulse between step counts, which must agree for a frozen field.
    Particle survival, reference momentum and charge are checked as controls in every run.
    """

    import numpy as np
    from _bmad import compare, fortran, verdict

    PROGRAM = r"""
    program fft_steps
      use bmad
      use csr_and_space_charge_mod
      implicit none
      type(lat_struct) lat
      type(bunch_struct) bunch
      type(coord_struct) centroid(0:1)
      integer i, j, k, n, nstep
      logical err
      read(*, *) nstep
      call bmad_parser('lat.bmad', lat)
      call init_coord(centroid(0), lat%ele(0), downstream_end$)
      call track1(centroid(0), lat%ele(1), lat%param, centroid(1))
      space_charge_com%n_bin = 8
      space_charge_com%space_charge_mesh_size = [16, 16, 16]
      call reallocate_bunch(bunch, 27)
      ! A 3x3x3 cube of electrons with 1 mm spacing, at rest in the beam frame.
      n = 0
      do i = -1, 1
        do j = -1, 1
          do k = -1, 1
            n = n + 1
            call init_coord(bunch%particle(n), [i*0.001_rp, 0.0_rp, j*0.001_rp, 0.0_rp, k*0.001_rp, 0.0_rp], &
              lat%ele(1), upstream_end$, electron$)
            bunch%particle(n)%charge = 1.0e-18_rp / 27
          enddo
        enddo
      enddo
      bunch%n_live = 27; bunch%charge_tot = 1.0e-18_rp; bunch%charge_live = bunch%charge_tot
      space_charge_com%ds_track_step = lat%ele(1)%value(l$) / nstep
      call track1_bunch_csr(bunch, lat%ele(1), centroid, err)
      if (err) error stop 'FFT tracking failed'
      write(*, '(a,i0,es25.16e3)') 'RESULT px ', nstep, bunch%particle(1)%vec(2)
      write(*, '(a,i0,3es25.16e3)') 'RESULT state ', nstep, real(count(bunch%particle%state == alive$), rp), &
        bunch%particle(1)%p0c, sum(bunch%particle%charge)
    end program
    """

    LATTICE = """parameter[particle] = electron
    parameter[p0c] = 1e7
    d: drift, l=0.001, space_charge_method=fft_3d
    l: line=(d)
    use, l
    """

    # The field changes by less than 1e-10 relative while the bunch crosses the drift, so a
    # correctly weighted impulse is independent of the step count to better than 1e-9.
    RTOL = 1e-9


    def track(steps: int) -> float:
        log = fortran(PROGRAM, files={"lat.bmad": LATTICE}, stdin=f"{steps}\n").log
        rows = {}
        for line in log.splitlines():
            words = line.split()
            if len(words) >= 4 and words[0] == "RESULT" and int(words[2]) == steps:
                rows[words[1]] = [float(w) for w in words[3:]]
        if set(rows) != {"px", "state"}:
            raise RuntimeError(f"Missing results for {steps} steps:\n{log}")
        # Controls: all 27 particles survive, reference momentum and total charge are unchanged.
        if compare(
            f"{steps} steps: live particles, p0c (eV), total charge (C)",
            rows["state"],
            [27, 1e7, 1e-18],
            atol=0,
            rtol=3e-15,
        ):
            raise RuntimeError("Bunch state control failed; the comparison is not valid.")
        return rows["px"][0]


    steps = np.array([1, 2, 4])
    impulse = np.array([track(int(n)) for n in steps])
    if not (impulse < 0).all():
        raise RuntimeError(f"Expected a nonzero space-charge impulse, got {impulse}")
    print(f"px impulse of particle 1 at n = {steps.tolist()} steps: {impulse.tolist()}")
    # Full-weight endpoint kicks sum to (n + 1) full steps, L(1 + 1/n) in total. Removing that
    # factor gives the length-L impulse; it agrees for every n because the field is frozen.
    print(f"impulse * n/(n + 1): {(impulse * steps / (steps + 1)).tolist()}")
    # A step-count comparison requires a frozen field. The nearly vanishing charge makes
    # particle motion negligible; check that the full-endpoint prediction accounts for all
    # three impulses before attributing their difference to endpoint weights. Correctly
    # weighted, equal impulses are also a valid outcome after an upstream fix.
    equal = np.allclose(impulse, impulse[0], atol=0, rtol=RTOL)
    scaled = impulse * steps / (steps + 1)
    if not equal and not np.allclose(scaled, scaled[0], atol=0, rtol=RTOL):
        raise RuntimeError("The impulses match neither frozen-field weighting: comparison invalid.")
    defect = compare(
        "px impulse ratio, 1 step / 2 steps; frozen field gives 1",
        impulse[0] / impulse[1],
        1.0,
        atol=0,
        rtol=RTOL,
    )
    defect |= compare(
        "px impulse ratio, 2 steps / 4 steps; frozen field gives 1",
        impulse[1] / impulse[2],
        1.0,
        atol=0,
        rtol=RTOL,
    )
    verdict(defect)

    ```

## Provenance

| Role | Model | Effort | Date | Code examined |
|---|---|---|---|---|
| Found | <span class="bb-nowrap">`gpt-6-astra`</span> | xhigh | <span class="bb-nowrap">2026-10-20</span> | 20261020-0 (`19b8afb46a40`), source, production, CPU, double, serial |
| Confirmed | <span class="bb-nowrap">`gpt-6.1-sol`</span> | xhigh | <span class="bb-nowrap">2026-10-20</span> | 20261020-0 (`19b8afb46a40`), source, production, CPU, double, serial |

