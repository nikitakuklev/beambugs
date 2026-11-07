---
title: "COL-75: Automatic FFT mesh bounds permit deposition beyond the density array"
hide:
  - toc
---

# COL-75 — Automatic FFT mesh bounds permit deposition beyond the density array

| | |
|---|---|
| Code | [Bmad](../index.md) · Collective effects |
| Priority | P3 |
| Severity | Crash or API hazard |
| Evidence | Reproduced, independently rechecked |
| Last reproduced | 2026-11-06 on Bmad release [20261031-0](https://github.com/bmad-sim/bmad-ecosystem/tree/20261031-0) (`c4c629f`) |
| Tags | <span class="bb-tag">partly-source-only</span> [?](../../methodology.md#tags) |
| Source | [FFT deposit/solve/gather contract](https://github.com/bmad-sim/bmad-ecosystem/blob/c4c629fc9071d3a56cb1ba5bc714baba65e9c5d3/bmad/doc/multiparticle.tex#L488-L502)<br>[deposit_particles](https://github.com/bmad-sim/bmad-ecosystem/blob/c4c629fc9071d3a56cb1ba5bc714baba65e9c5d3/bmad/space_charge/open_spacecharge_mod.f90#L136-L149)<br>[Tight padding, lazy containment and stored mesh geometry](https://github.com/bmad-sim/bmad-ecosystem/blob/c4c629fc9071d3a56cb1ba5bc714baba65e9c5d3/bmad/space_charge/open_spacecharge_mod.f90#L204-L243)<br>[The production density allocation](https://github.com/bmad-sim/bmad-ecosystem/blob/c4c629fc9071d3a56cb1ba5bc714baba65e9c5d3/bmad/space_charge/open_spacecharge_mod.f90#L264-L270)<br>[Serial cloud-in-cell indexing and all eight neighbor accesses](https://github.com/bmad-sim/bmad-ecosystem/blob/c4c629fc9071d3a56cb1ba5bc714baba65e9c5d3/bmad/space_charge/open_spacecharge_mod.f90#L365-L389)<br>[Batch gathering](https://github.com/bmad-sim/bmad-ecosystem/blob/c4c629fc9071d3a56cb1ba5bc714baba65e9c5d3/bmad/space_charge/open_spacecharge_mod.f90#L535-L542)<br>[Tracking growth/shrink defaults](https://github.com/bmad-sim/bmad-ecosystem/blob/c4c629fc9071d3a56cb1ba5bc714baba65e9c5d3/bmad/modules/bmad_struct.f90#L2198-L2200)<br>[Tao accepts both tuning names](https://github.com/bmad-sim/bmad-ecosystem/blob/c4c629fc9071d3a56cb1ba5bc714baba65e9c5d3/tao/code/tao_set_mod.f90#L686-L688)<br>[Time-based FFT/cathode deposition](https://github.com/bmad-sim/bmad-ecosystem/blob/c4c629fc9071d3a56cb1ba5bc714baba65e9c5d3/bmad/space_charge/space_charge_mod.f90#L115-L120)<br>[s-based FFT deposition and longitudinal recentering](https://github.com/bmad-sim/bmad-ecosystem/blob/c4c629fc9071d3a56cb1ba5bc714baba65e9c5d3/bmad/space_charge/csr_and_space_charge_mod.f90#L1692-L1708)<br>[CSR3D source preparation](https://github.com/bmad-sim/bmad-ecosystem/blob/c4c629fc9071d3a56cb1ba5bc714baba65e9c5d3/bmad/space_charge/csr_and_space_charge_mod.f90#L1987-L1988)<br>[The ≥50,000-particle OpenMP branch](https://github.com/bmad-sim/bmad-ecosystem/blob/c4c629fc9071d3a56cb1ba5bc714baba65e9c5d3/bmad/space_charge/open_spacecharge_mod.f90#L314-L319) |

Automatic FFT space-charge meshes can place a particle on the last grid node while cloud-in-cell deposition still accesses a neighbor beyond the density array. Tight fitting reaches this when padding rounds away: a 1.000 µm bunch near 1000 m on 129 longitudinal nodes accesses node 130. Default ten-percent padding also permits the access when a particle reaches a reused mesh's upper bound; the default 64-node dimension then accesses node 65. Bounds checking aborts, and AddressSanitizer detects a read beyond the allocation. Correct deposition must use valid nodes. The unchecked examples conserve charge, so the established failure is a memory-access hazard.

Run `python COL-75.py` with [_bmad.py](../helpers/_bmad.py) in the same directory.

??? example "Reproducer: COL-75.py"

    [Download COL-75.py](COL-75.py)

    ```python
    """COL-75: Automatic FFT mesh bounds permit deposition beyond the density array

    Automatic FFT space-charge meshes can place a particle on the last grid node while cloud-in-cell deposition still accesses a neighbor beyond the density array. Tight fitting reaches this when padding rounds away: a 1.000 µm bunch near 1000 m on 129 longitudinal nodes accesses node 130. Default ten-percent padding also permits the access when a particle reaches a reused mesh's upper bound; the default 64-node dimension then accesses node 65. Bounds checking aborts, and AddressSanitizer detects a read beyond the allocation. Correct deposition must use valid nodes. The unchecked examples conserve charge, so the established failure is a memory-access hazard.

    Setup: the build's GNU Fortran compiler with AddressSanitizer support (`libasan`); the script compiles diagnostic copies with `-fcheck=bounds` and `-fsanitize=address`.

    The script deposits an eight-corner, 1 pC box with positive extent using production automatic bounds, including two calls that reach a cached default upper boundary. It extracts the complete unchanged deposition subroutine into a temporary diagnostic module and checks both Fortran bounds and AddressSanitizer failures against valid endpoint stencils and conserved charge. All compilation and runtime scratch uses the shared runner.
    """

    import os
    import re
    from pathlib import Path

    import numpy as np
    from _bmad import RunFailure, _execute, _recipe, _scratch, fortran, source, verdict

    CLIENT = r"""
    program check_mesh
    use open_spacecharge_mod, only: mesh3d_struct, original_deposit => deposit_particles
    use bmad, only: space_charge_com
    {use_checked}
    implicit none
    type(mesh3d_struct) mesh
    real(8) x(8),y(8),z(8),q(8),shift,growth,shrink,pad
    integer i,j,k,n,mode,idx,nz
    read(*,*)shift,growth,shrink,mode,nz
    ! Preserve the actual common-block default bits, including default-real initializers.
    if(growth==0.1d0)growth=space_charge_com%mesh_growth_factor
    if(shrink==0.1d0)shrink=space_charge_com%mesh_shrink_factor
    mesh%nhi=space_charge_com%space_charge_mesh_size
    mesh%nhi(3)=nz
    n=0
    do k=0,1
    do j=0,1
    do i=0,1
     n=n+1;x(n)=(i-0.5d0)*2d-3;y(n)=(j-0.5d0)*2d-3
     z(n)=shift+k*1d-6;q(n)=1.25d-13
    enddo
    enddo
    enddo
    call {deposit}(x,y,z,mesh,qa=q,mesh_growth_factor=growth,mesh_shrink_factor=shrink)
    if(mode==1)then
     ! An expanding bunch remains inside the cached default mesh, at its upper boundary.
     where(z>shift)z=mesh%max(3)
     call {deposit}(x,y,z,mesh,qa=q,mesh_growth_factor=growth,mesh_shrink_factor=shrink)
    endif
    pad=mesh%max(3)-maxval(z)
    idx=floor((maxval(z)-mesh%min(3))*(1d0/mesh%delta(3))+1)
    write(*,'(a,3i8,8es26.17)')'RESULT ',mode,idx,ubound(mesh%rho,3), &
     mesh%min(3),mesh%max(3),mesh%delta(3),maxval(z),pad,sum(mesh%rho),sum(q), &
     1d0-((mesh%min(3)-maxval(z))+idx*mesh%delta(3))*(1d0/mesh%delta(3))
    end program
    """

    CASES = [
        ("center_tight", "0 0 0 0 129", False),
        ("translated_tight", "1000 0 0 0 129", True),
        ("translated_default", "1000 0.1 0.1 0 129", False),
        ("translated_growth_zero", "1000 0 0.1 0 129", False),
        ("cached_boundary_default", "0 0.1 0.1 1 129", True),
        ("default64_boundary", "0 0.1 0.1 1 64", True),
        ("default64_translated_tight", "1000 0 0 0 64", False),
    ]


    def result_row(log):
        rows = [line.split()[1:] for line in log.splitlines() if line.startswith("RESULT ")]
        if len(rows) != 1 or len(rows[0]) != 11:
            raise RuntimeError("Expected one complete production deposition record: " + log)
        return np.array(rows[0], dtype=float)


    def diagnostics(body, flags):
        # Only the enclosing module changes. The full production subroutine is byte-identical.
        module = (
            "module diagnostic_deposition\nuse open_spacecharge_mod, only: mesh3d_struct\n"
            "implicit none\ninteger, parameter :: dp=kind(1d0)\ncontains\n"
            + body + "\nend module\n"
        )
        code = module + CLIENT.format(
            use_checked="use diagnostic_deposition", deposit="deposit_particles"
        )
        compile_argv, link_argv = _recipe(False)
        with _scratch() as tmp:
            work = Path(tmp)
            (work / "probe.f90").write_text(code)
            _execute([*compile_argv, *flags], work, "", 290)
            _execute([*link_argv, *flags], work, "", 290)
            results = []
            for label, args, _unsafe in CASES:
                run = work / label
                run.mkdir()
                try:
                    log = _execute([str(work / "probe")], run, args + "\n", 290)
                    status = 0
                except RunFailure as error:
                    log, status = error.log, error.status
                results.append((label, status, log))
            return results


    def main():
        production = CLIENT.format(use_checked="", deposit="original_deposit")
        unsafe_results = []
        for label, args, unsafe in CASES:
            row = result_row(fortran(production, stdin=args + "\n").log)
            _mode, lower, upper, minimum, _maximum, delta, extent, pad, charge, assigned, ghost = row
            np.testing.assert_allclose(charge, assigned, rtol=2e-14, atol=0)
            # Binary64 operation order matches the actual lower-index expression.
            calculated = np.floor((extent - minimum) * (1.0 / delta) + 1)
            if lower != calculated:
                raise RuntimeError("Independent lower-index calculation disagrees")
            invalid = lower + 1 > upper
            unsafe_results.append(invalid)
            print(f"{label}: lower/neighbor={int(lower)}/{int(lower+1)}, allocated_upper={int(upper)}; "
                  f"expected lower<= {int(upper-1)}; padding={pad:.17g} m; ghost_weight={ghost:.10g}; "
                  f"Q={charge:.17g} expected_sum_qa={assigned:.17g} C")
            if invalid != unsafe:
                raise RuntimeError("Unexpected endpoint stencil")
            if label == "translated_tight":
                desired = 1e-6 * (extent - 1000) / (upper - 1)
                half_ulp = np.spacing(1000.0) / 2
                print(f"  intended_padding={desired:.17g} m; half_ulp_at_1000m={half_ulp:.17g} m")
        text = (source() / "bmad/space_charge/open_spacecharge_mod.f90").read_text()
        start = text.index("subroutine deposit_particles(")
        end = text.index("end subroutine deposit_particles", start) + len("end subroutine deposit_particles")
        body = text[start:end]
        os.environ["ASAN_OPTIONS"] = "detect_leaks=0"
        failures = []
        for kind, flags in [("bounds", ["-fcheck=bounds"]), ("asan", ["-fsanitize=address", "-g"])]:
            results = diagnostics(body, flags)
            for (label, status, log), (_, _args, unsafe) in zip(results, CASES, strict=True):
                if not unsafe:
                    if status:
                        raise RuntimeError(f"{kind} {label} control failed: {log}")
                    row = result_row(log)
                    np.testing.assert_allclose(row[-3], row[-2], rtol=2e-14, atol=0)
                    print(f"{kind} {label}: PASS; valid stencil and charge conserved")
                    continue
                if kind == "bounds":
                    match = re.search(r"Index '(\d+)' of dimension 3 of array 'mesh3d%rho' above upper bound of (\d+)", log)
                    expected_upper = 64 if label == "default64_boundary" else 129
                    if not status or not match or tuple(map(int, match.groups())) != (expected_upper+1, expected_upper):
                        raise RuntimeError(f"Unexpected bounds result: {log}")
                    print(f"bounds {label}: Index '{match[1]}' of dimension 3 above upper bound {match[2]}; "
                          "expected valid-neighbor deposition")
                else:
                    match = re.search(r"(\d+) bytes after (\d+)-byte region", log)
                    if not status or "heap-buffer-overflow" not in log or "READ of size 8" not in log or not match:
                        raise RuntimeError(f"Unexpected AddressSanitizer result: {log}")
                    print(f"asan {label}: heap-buffer-overflow, READ of size 8, {match[1]} bytes after "
                          f"{match[2]}-byte allocation; expected access inside allocation")
                failures.append(True)
        print("7 production cases conserve charge rtol=2e-14; 8 diagnostic controls pass; "
              "3 invalid stencils each fail both bounds and allocation checks")
        verdict(sum(unsafe_results) == 3 and len(failures) == 6)


    if __name__ == "__main__":
        main()

    ```

## Provenance

| Role | Model | Effort | Date | Code examined |
|---|---|---|---|---|
| Found | `gpt-6-astra` | xhigh | 2026-11-06 | 20261031-0 (`c4c629fc9071d3a56cb1ba5bc714baba65e9c5d3`), source, production, GNU Fortran 15.3.0, CPU, double, OpenMP one thread, MPI disabled, unchanged deposition diagnostics: bounds |
| Confirmed | `gpt-6.1-sol` | xhigh | 2026-11-06 | 20261031-0 (`c4c629fc9071d3a56cb1ba5bc714baba65e9c5d3`), source, production, GNU Fortran 15.3.0, CPU, double, OpenMP one thread, MPI disabled, unchanged deposition diagnostics: bounds and AddressSanitizer |

