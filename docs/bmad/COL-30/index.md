---
title: "COL-30: Touschek-background spread propagation discards its compression correction"
hide:
  - toc
---

# COL-30 — Touschek-background spread propagation discards its compression correction

| | |
|---|---|
| Code | [Bmad](../index.md) · Collective effects |
| Priority | P3 |
| Severity | Wrong physics |
| Evidence | Reproduced, independently rechecked |
| Last reproduced | 2026-11-05 on Bmad 20261031-0 (`c4c629f`) |
| Tags | <span class="bb-tag">partly-source-only</span> [?](../../methodology.md#tags) |
| Source | [touschek_background spread propagation, bsim/touschek_background/touschek_background.f90:475–516](https://github.com/bmad-sim/bmad-ecosystem/blob/c4c629fc9071d3a56cb1ba5bc714baba65e9c5d3/bsim/touschek_background/touschek_background.f90#L475-L516)<br>[beam-input path, lines 421–458](https://github.com/bmad-sim/bmad-ecosystem/blob/c4c629fc9071d3a56cb1ba5bc714baba65e9c5d3/bsim/touschek_background/touschek_background.f90#L421-L458)<br>[Touschek-background manual, “sigma_p.out”](https://github.com/bmad-sim/bmad-ecosystem/blob/c4c629fc9071d3a56cb1ba5bc714baba65e9c5d3/bsim/touschek_background/doc/touschek_background.tex#L306-L315)<br>[map coefficients, bmad/doc/elements.tex:5307–5375](https://github.com/bmad-sim/bmad-ecosystem/blob/c4c629fc9071d3a56cb1ba5bc714baba65e9c5d3/bmad/doc/elements.tex#L5307-L5375)<br>[identity defaults, coefficient replacement and length behavior, lines 5451–5493](https://github.com/bmad-sim/bmad-ecosystem/blob/c4c629fc9071d3a56cb1ba5bc714baba65e9c5d3/bmad/doc/elements.tex#L5451-L5493) |

With use_beam enabled in touschek_background, bunch compression measured by beam tracking is omitted from the propagated energy spread written to sigma_p.out. The propagation loop updates an unused scalar instead of the spread carried between elements. For a factor-two linear compression at 1 GeV with zero radiation excitation and disabled intrabeam scattering, the application reports a relative spread of 10⁻⁶ instead of 2 × 10⁻⁶, a 50% underestimate. The identity map gives the correct unchanged spread.

Download these shared helpers into the same directory as the reproducer scripts:

- [Download _bmad.py](../helpers/_bmad.py) · [View source](../helpers/index.md#_bmad)

??? example "Reproducer: COL-30.py"

    [Download COL-30.py](COL-30.py)

    ```python
    """COL-30: Touschek-background spread propagation discards its compression correction

    With use_beam enabled in touschek_background, bunch compression measured by beam tracking is omitted from the propagated energy spread written to sigma_p.out. The propagation loop updates an unused scalar instead of the spread carried between elements. For a factor-two linear compression at 1 GeV with zero radiation excitation and disabled intrabeam scattering, the application reports a relative spread of 10⁻⁶ instead of 2 × 10⁻⁶, a 50% underestimate. The identity map gives the correct unchanged spread.

    The script compiles the complete, unchanged application from BMAD_SOURCE against BMAD_DIST and the installed Open MPI, then runs two MPI ranks on the CPU selected with taskset. Fixed particles traverse an identity map, a longitudinal scaling map, and another identity map; analytic covariance transport and a separate beam-tracking program check the expected spread. Identity, expansion, and fixed-bunch-length cases isolate the compression correction, and all compiled programs are removed after execution.
    """

    from __future__ import annotations

    import io
    import itertools
    import os
    import re
    import shlex
    import shutil
    import tempfile
    from pathlib import Path

    import numpy as np
    from _bmad import TIMEOUT, _execute, _recipe, build, compare, source, verdict
    from numpy.typing import NDArray

    ORACLE = """\
    program check_beam
    use bmad
    use beam_mod
    implicit none
    type(lat_struct) lat
    type(coord_struct), allocatable :: orbit(:)
    type(beam_init_struct) init
    type(beam_struct) beam
    type(bunch_params_struct) params
    type(normal_modes_struct) mode
    type(rad_int_all_ele_struct) rad
    logical err
    integer i, cache
    call bmad_parser('lat.bmad', lat)
    call twiss_and_track(lat, orbit)
    init%position_file = 'beam.dat'
    call init_beam_distribution(lat%ele(0), lat%param, init, beam, err)
    if (err) error stop 'beam initialization failed'
    cache = -1
    call radiation_integrals(lat, orbit, mode, cache, 0, rad_int_by_ele=rad)
    write(*, '(a,es24.16)') 'INITIAL_LATTICE_SIGMA_Z ', lat%ele(0)%z%sigma
    do i = 0, lat%n_ele_track
      if (i > 0) then
        call track1_bunch(beam%bunch(1), lat%ele(i), err)
        if (err) error stop 'beam tracking failed'
      endif
      call calc_bunch_params(beam%bunch(1), params, err)
      if (err) error stop 'beam parameters failed'
      write(*, '(a,i3,4es24.16)') 'ORACLE ', i, lat%ele(i)%s, sqrt(params%sigma(5,5)), &
           sqrt(params%sigma(6,6)), rad%branch(0)%ele(i)%lin_i3_E7
    enddo
    end program
    """

    PARAMETERS = """\
    &parameters
     lat_file = 'lat.bmad'
     aperture_file = 'aperture.dat'
     n_turns = 1
     start_stop_type = 1
     s_prod_start = 0
     s_prod_end = -1
     s_lost_start = 0
     s_lost_end = -1
     do_ibs = .false.
     a_emittance = 3e-7
     b_emittance = 3e-7
     bunch_charge = 77e-12
     bunch_length = 0.002
     energy_spread_eV = 1000
     count_col_losses = 1
     test_collimator = 0.0025
     collimate = .false.
     N_test_part = 1
     N_data_points = 8
     use_beam = {use_beam}
     verbose = .true.
     beam_init%position_file = 'beam.dat'
    /
    """


    def require(condition: bool, message: str) -> None:
        if not condition:
            raise RuntimeError(message)


    def compile_programs(work: Path) -> None:
        """Use the pin's compiler/link recipe, adding its bsim library and installed MPI."""
        compile_argv, link_argv = _recipe(False)
        wrapper = shutil.which("mpifort")
        require(wrapper is not None, "Open MPI mpifort must be on PATH.")
        mpi_compile = shlex.split(_execute([str(wrapper), "--showme:compile"], work, "", TIMEOUT))
        mpi_link = shlex.split(_execute([str(wrapper), "--showme:link"], work, "", TIMEOUT))
        stems = (
            "touschek_background_mpi_mod",
            "touschek_background_mod",
            "special_collimate_lattice",
            "touschek_background",
        )
        for stem in stems:
            original = source() / "bsim/touschek_background" / f"{stem}.f90"
            shutil.copyfile(original, work / original.name)
            replacements = {"probe.f90": f"{stem}.f90", "probe.o": f"{stem}.o"}
            command = [replacements.get(arg, arg) for arg in compile_argv]
            _execute([*command, *mpi_compile], work, "", TIMEOUT)

        link = list(link_argv)
        index = link.index("probe.o")
        link[index : index + 1] = [f"{stem}.o" for stem in stems]
        index = next(i for i, arg in enumerate(link) if arg.endswith("/libbmad.a"))
        link.insert(index, str(build() / "production/lib/libbsim.a"))
        link[link.index("-o") + 1] = "touschek_background"
        _execute([*link, *mpi_link], work, "", TIMEOUT)

        (work / "probe.f90").write_text(ORACLE)
        _execute(compile_argv, work, "", TIMEOUT)
        _execute(link_argv, work, "", TIMEOUT)


    def inputs(case: Path, factor: float, use_beam: bool) -> None:
        case.mkdir()
        # The Taylor-element contract makes L affect s and time, not the phase-space map.
        # The longitudinal map diag(c, 1/c) is symplectic: Sigma_out = M Sigma_in M^T.
        # Thus sigma_z scales by c and sigma_pz by 1/c, with no radiation integrals.
        (case / "lat.bmad").write_text(
            "parameter[geometry] = open\n"
            "parameter[particle] = electron\n"
            "parameter[e_tot] = 1e9\n"
            "beginning[beta_a] = 10\n"
            "beginning[beta_b] = 10\n"
            "id: taylor, l = 1\n"
            f"scale: taylor, l = 1, {{5: {factor:.17g} | 5}}, {{6: {1 / factor:.17g} | 6}}\n"
            "lat: line = (id, scale, id)\n"
            "use, lat\n"
        )
        # Zero aperture is the application's documented-in-source skip flag. It still
        # completes the spread calculation, MPI broadcast, output and normal shutdown.
        (case / "aperture.dat").write_text("0 0 0\n1 0 0\n2 0 0\n3 0 0\n")
        (case / "params.in").write_text(PARAMETERS.format(use_beam=".true." if use_beam else ".false."))
        # All 64 sign combinations have zero mean and a diagonal population covariance.
        amplitudes = (1e-4, 1e-5, 1e-4, 1e-5, 0.002, 1e-6)
        lines = ["!ASCII::3", "0", "1", "64", "BEGIN_BUNCH", "electron", "77e-12", "0", "0"]
        for signs in itertools.product((-1, 1), repeat=6):
            lines.append(" ".join(f"{s * a:.17g}" for s, a in zip(signs, amplitudes, strict=True)))
        (case / "beam.dat").write_text("\n".join([*lines, "END_BUNCH", ""]))


    def check_oracle(work: Path, case: Path, factor: float) -> NDArray[np.float64]:
        log = _execute([str(work / "probe")], case, "", TIMEOUT)
        rows = [line.removeprefix("ORACLE ") for line in log.splitlines() if line.startswith("ORACLE ")]
        values = np.loadtxt(io.StringIO("\n".join(rows)), ndmin=2)
        require(values.shape == (5, 5), "Independent tracking must return all five lattice locations.")
        require(bool(np.isfinite(values).all()), "Independent tracking returned nonfinite values.")
        require(bool(np.array_equal(values[:, 0], np.arange(5))), "Incorrect oracle element order.")
        require(bool(np.array_equal(values[:, 1], [0, 1, 2, 3, 3])), "Incorrect oracle s positions.")
        require(bool(np.all(values[:, 4] == 0)), "Radiation excitation must be exactly zero.")
        scaling = np.array([1, 1, factor, factor, factor])
        expected = np.column_stack((0.002 * scaling, 1e-6 / scaling))
        require(
            bool(np.allclose(values[:, 2:4], expected, atol=0, rtol=2e-13)),
            "Independent bunch tracking disagrees with analytic covariance transport.",
        )
        return values


    def run_case(work: Path, label: str, factor: float, use_beam: bool, cpu: int) -> bool:
        case = work / label
        inputs(case, factor, use_beam)
        launcher = shutil.which("mpiexec")
        require(launcher is not None, "Open MPI mpiexec must be on PATH.")
        # MPI requires a master and a worker. Bind both explicitly because the MPI
        # launcher can change inherited affinity; each case runs in sequence.
        log = _execute(
            [
                str(launcher),
                "--bind-to",
                "none",
                "--oversubscribe",
                "-n",
                "2",
                "taskset",
                "-c",
                str(cpu),
                str(work / "touschek_background"),
                "params.in",
            ],
            case,
            "",
            TIMEOUT,
        )
        require("Energy spread calculations complete..." in log, "Spread calculation did not finish.")
        require("Total number of test particles generated:" in log, "Application did not finish.")
        require(not re.search(r"\b(?:ERROR|FATAL|NaN)\b", log), "Application reported an error.")
        details = (case / "touschek_track.details").read_text()
        require(
            re.search(r"Total number of test particles generated:\s+0\b", details) is not None,
            "Zero-aperture control unexpectedly generated scattering particles.",
        )
        observed = np.loadtxt(case / "sigma_p.out", ndmin=2)
        require(observed.shape == (5, 3), "Application must write all five lattice locations.")
        require(bool(np.isfinite(observed).all()), "Application output contains nonfinite values.")
        require(bool(np.array_equal(observed[:, 0], [0, 1, 2, 3, 3])), "Incorrect output s positions.")
        require(bool(np.all(observed[:, 2] == 1e9)), "Reference energy must stay at 1 GeV.")
        require(bool(np.all(observed[:2, 1] == 1e-6)), "Incoming spread control failed.")
        oracle = check_oracle(work, case, factor)
        if use_beam:
            times = np.array(re.findall(r"sigma_z/c:\s+([\d.Ee+-]+)\s+fs", log), dtype=float)
            # Verbose application lengths are printed to 0.001 fs.
            require(
                times.shape == (4,)
                and bool(np.allclose(times, oracle[1:, 2] * 1e15 / 299792458, atol=5.1e-4, rtol=0)),
                "Application bunch lengths disagree with independent tracking.",
            )
        expected = 1e-6 / factor if use_beam else 1e-6
        # ES14.4 output retains five significant figures. The independent full-precision
        # tracking above uses 2e-13; this comparison respects the application's file precision.
        mismatch = compare(
            f"{label}: sigma_p at and after map; expected from "
            + ("diag(c,1/c)" if use_beam else "fixed bunch length"),
            observed[2:, 1],
            np.full(3, expected),
            atol=0,
            rtol=5.1e-5,
        )
        print(
            f"{label}: independent tracked [sigma_z m, sigma_p]="
            f"[{oracle[2, 2]:.12g}, {oracle[2, 3]:.12g}]; radiation excitation=0"
        )
        return mismatch


    def main() -> None:
        cpus = os.sched_getaffinity(0)
        require(len(cpus) == 1, "Select one CPU with taskset before running this reproducer.")
        cpu = next(iter(cpus))
        parent = Path(tempfile.gettempdir()) / "bmad-scratch" / "gap-touschek"
        parent.mkdir(parents=True, exist_ok=True)
        with tempfile.TemporaryDirectory(prefix="repro-", dir=parent) as directory:
            work = Path(directory)
            compile_programs(work)
            require(not run_case(work, "identity", 1, True, cpu), "Identity control failed.")
            compression = run_case(work, "compression", 0.5, True, cpu)
            expansion = run_case(work, "expansion", 2, True, cpu)
            require(
                not run_case(work, "fixed-length", 0.5, False, cpu),
                "Fixed-bunch-length control failed.",
            )
        verdict(compression or expansion)


    if __name__ == "__main__":
        main()

    ```

## Provenance

| Role | Model | Effort | Date | Code examined |
|---|---|---|---|---|
| Found<br><small>source check</small> | `gpt-6.1-sol` | xhigh | 2026-10-21 | 20261020-0 (`19b8afb46a40`) |
| Found<br><small>source check</small> | `gpt-6-astra` | xhigh | 2026-10-21 | 20261020-0 (`19b8afb46a40`) |
| Confirmed<br><small>compression-dependent energy-spread propagation</small> | `gpt-6-astra` | xhigh | 2026-11-05 | 20261031-0 (`c4c629fc9071`), source, production, GNU Fortran 15.3.0, CPU, double, serial libraries, OpenMP=1; application Open MPI 5.0.10, 2 ranks on one CPU |

