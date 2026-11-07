---
title: "SCT-03: MPI Touschek generation crashes when tracking is disabled"
hide:
  - toc
---

# SCT-03 — MPI Touschek generation crashes when tracking is disabled

| | |
|---|---|
| Code | [Elegant](../index.md) · Scattering and pressure |
| Priority | P3 |
| Severity | Crash or API hazard |
| Evidence | Reproduced, independently rechecked |
| Last reproduced | 2026-11-04 on Elegant master (`0355dd6`, head build) |
| Source | [`lostParticle` declaration, touschekScatter.c:345](https://github.com/rtsoliday/elegant/blob/0355dd6aca12c861971fbf8f3ae78a0ded991734/src/touschekScatter.c#L345)<br>[tracking branch, touschekScatter.c:735–797](https://github.com/rtsoliday/elegant/blob/0355dd6aca12c861971fbf8f3ae78a0ded991734/src/touschekScatter.c#L735-L797)<br>[cleanup, touschekScatter.c:886–889](https://github.com/rtsoliday/elegant/blob/0355dd6aca12c861971fbf8f3ae78a0ded991734/src/touschekScatter.c#L886-L889)<br>[`free_czarray_2d`, mdblib/array.c:420–424](https://github.com/rtsoliday/SDDS/blob/a3f730a14c49ddd4f4d6347b42d60d34aa7cadc6/mdblib/array.c#L420-L424)<br>[namelist default, touschekScatter.nl:37](https://github.com/rtsoliday/elegant/blob/0355dd6aca12c861971fbf8f3ae78a0ded991734/src/touschekScatter.nl#L37)<br>[touschek_scatter, elegant.tex:7699](https://github.com/rtsoliday/elegant/blob/0355dd6aca12c861971fbf8f3ae78a0ded991734/doc/elegant.tex#L7699)<br>[`do_track`, elegant.tex:7809–7813](https://github.com/rtsoliday/elegant/blob/0355dd6aca12c861971fbf8f3ae78a0ded991734/doc/elegant.tex#L7809-L7813) |

When Pelegant, Elegant's MPI version, runs the touschek_scatter command with its default do_track = 0, which generates the scattered particles without tracking them, it crashes with a segmentation fault after generating the particles at the first TSCATTER element. Serial Elegant completes the same input. The cleanup frees the array of lost particles on the master rank, but that array exists only when tracking is enabled, and the freeing routine dereferences it. For three TSCATTER elements and 40 generated particles per element, serial Elegant writes three pages of 40 particles, and Pelegant with do_track = 1 does too, but Pelegant with do_track = 0 aborts.

Run `python SCT-03.py` with [_elegant.py](../helpers/_elegant.py) in the same directory.

??? example "Reproducer: SCT-03.py"

    [Download SCT-03.py](SCT-03.py)

    ```python
    """SCT-03: MPI Touschek generation crashes when tracking is disabled

    When Pelegant, Elegant's MPI version, runs the touschek_scatter command with its default do_track =
    0, which generates the scattered particles without tracking them, it crashes with a segmentation
    fault after generating the particles at the first TSCATTER element. Serial Elegant completes the
    same input. The cleanup frees the array of lost particles on the master rank, but that array exists
    only when tracking is enabled, and the freeing routine dereferences it. For three TSCATTER elements
    and 40 generated particles per element, serial Elegant writes three pages of 40 particles, and
    Pelegant with do_track = 1 does too, but Pelegant with do_track = 0 aborts.

    Setup: always runs 3-rank Pelegant (Open MPI `mpirun` on `PATH`) beside serial Elegant, because the
    defect occurs only under MPI.

    The script runs touschek_scatter on a three-element TSCATTER lattice with 40 generated particles per
    element, once in serial Elegant and twice in 3-rank Pelegant, with tracking disabled and enabled. It
    compares the pages of the scattered-particle file with the expected three pages of 40 and reports
    how a run that aborts ends. The serial run and the tracked parallel run are the controls: they show
    that the input is valid and that the crash follows from skipping tracking.
    """

    import sys

    from _elegant import RunFailure, modes, run, verdict

    PARTICLES = 40  # generated per TSCATTER site
    LATTICE = "T0: TSCATTER\nD0: DRIF,L=0.4\nT1: TSCATTER\nD1: DRIF,L=0.9\nT2: TSCATTER\nBL: LINE=(T0,D0,T1,D1,T2)\n"
    # Momentum aperture of +-0.003 at the three sites, in the table format touschek_scatter reads.
    APERTURE = """SDDS1
    &column name=s, type=double, units=m &end
    &column name=deltaPositive, type=double &end
    &column name=deltaNegative, type=double &end
    &column name=ElementName, type=string &end
    &column name=ElementType, type=string &end
    &column name=ElementOccurence, type=long &end
    &data mode=ascii, no_row_counts=1 &end
    0.0 0.003 -0.003 T0 TSCATTER 1
    0.4 0.003 -0.003 T1 TSCATTER 1
    1.3 0.003 -0.003 T2 TSCATTER 1
    """
    COMMANDS = """&run_setup lattice=lattice.lte, use_beamline=BL, p_central=800.0, random_number_seed=31317 &end
    &run_control n_passes=1 &end
    &twiss_output matched=0, beta_x=8, alpha_x=0.3, beta_y=5, alpha_y=-0.2 &end
    &touschek_scatter charge=1e-09, frequency=230609583.07692307, emit_x=2e-09, emit_y=3e-10,
     sigma_dp=0.0008, sigma_s=0.004, Momentum_Aperture="aperture.sdds", Momentum_Aperture_scale=1,
     verbosity=4, n_simulated={n}, ignored_portion=0, do_track={track}, bunch="scattered.sdds" &end
    """


    def crash_signature(log):
        """True when the log shows the established crash: SIGSEGV after the beam data is freed, before the rest."""
        return (
            "Terminated by SIGSEGV" in log
            and "Freeing beam data" in log
            and "Freeing other data" not in log
        )


    def generate(mode, track):
        """Rows per page of the scattered-particle file, or None after the established crash.

        Any other failure propagates and ends the script without a verdict.
        """
        files = {
            "lattice.lte": LATTICE,
            "aperture.sdds": APERTURE,
            "run.ele": COMMANDS.format(n=PARTICLES, track=track),
        }
        try:
            result = run(files, mode, outputs=("scattered.sdds",))
        except RunFailure as error:
            if mode == "mpi" and not track and crash_signature(error.log):
                return None
            raise
        bunch = result["scattered.sdds"]
        return [len(bunch[page, "x"]) for page in range(bunch.n_pages)]


    defect = False
    print("rows per page of the scattered-particle file, one page per TSCATTER site")
    for mode in ("serial", *modes(mpi_only=True)):
        for track in (0, 1):
            if mode == "serial" and track:
                continue
            observed = generate(mode, track)
            control = mode == "serial" or track
            label = f"{mode} do_track={track}" + (" (control)" if control else "")
            shown = (
                str(observed)
                if observed is not None
                else "[SIGSEGV after 'Freeing beam data', before 'Freeing other data']"
            )
            print(f"{label}: observed={shown} expected={[PARTICLES] * 3}")
            # A completed run must hold three pages of 40 particles; anything else is an unrelated
            # output failure. The failing run may already have written the first site's page.
            if observed is not None and observed != [PARTICLES] * 3:
                sys.exit(f"{label} completed with pages {observed}: the comparison is not valid")
            defect |= observed is None
    verdict(defect)

    ```

## Provenance

| Role | Model | Effort | Date | Code examined |
|---|---|---|---|---|
| Found | `gpt-6-astra` | xhigh | 2026-10-25 | master (`75a2e1a1a58b`), source with SDDS a60b9bbed7cc, CPU, 3-rank Pelegant |
| Confirmed | `gpt-6-astra` | high | 2026-10-25 | master (`75a2e1a1a58b`), source with SDDS a60b9bbed7cc, CPU, serial and 3-rank Pelegant |
| Confirmed | `gpt-6.1-sol` | high | 2026-10-25 | master (`75a2e1a1a58b`), source with SDDS a60b9bbed7cc, CPU, serial and 3-rank Pelegant |

