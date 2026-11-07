---
title: "IO-01: Pelegant fails to load a multi-page beam file when a later page leaves a worker without rows"
hide:
  - toc
---

# IO-01 — Pelegant fails to load a multi-page beam file when a later page leaves a worker without rows

| | |
|---|---|
| Code | [Elegant](../index.md) · Lattices, particles and I/O |
| Priority | P3 |
| Severity | Crash or API hazard |
| Evidence | Reproduced, independently rechecked |
| Last reproduced | 2026-11-04 on Elegant master (`0355dd6`, head build) |
| Source | [activity flag, sdds_beam.c:886](https://github.com/rtsoliday/elegant/blob/0355dd6aca12c861971fbf8f3ae78a0ded991734/src/sdds_beam.c#L886)<br>[activity assignment and column reads, sdds_beam.c:1092–1113](https://github.com/rtsoliday/elegant/blob/0355dd6aca12c861971fbf8f3ae78a0ded991734/src/sdds_beam.c#L1092-L1113)<br>[empty-column check, SDDS_extract.c:886–888](https://github.com/rtsoliday/SDDS/blob/a3f730a14c49ddd4f4d6347b42d60d34aa7cadc6/SDDSlib/SDDS_extract.c#L886-L888)<br>[row partition, SDDS_MPI_binary.c:2243–2251](https://github.com/rtsoliday/SDDS/blob/a3f730a14c49ddd4f4d6347b42d60d34aa7cadc6/SDDSlib/SDDS_MPI_binary.c#L2243-L2251)<br>[track_pages_separately, doc/elegant.tex:7454–7455](https://github.com/rtsoliday/elegant/blob/0355dd6aca12c861971fbf8f3ae78a0ded991734/doc/elegant.tex#L7454-L7455)<br>[unsupported cases, doc/Pelegant.tex:222–228](https://github.com/rtsoliday/elegant/blob/0355dd6aca12c861971fbf8f3ae78a0ded991734/doc/Pelegant.tex#L222-L228) |

When Pelegant, Elegant's MPI version, loads several particle-file pages with `track_pages_separately=0`, it should combine them into one beam. If a later page gives no rows to a worker that read an earlier page, the run stops with "Unable to get column--no rows left" before tracking. A five-particle file split into pages of four and one fails on three ranks (two workers), whereas serial Elegant and Pelegant with a one-plus-four split return all five particles unchanged.

Run `python IO-01.py` with [_elegant.py](../helpers/_elegant.py) in the same directory.

??? example "Reproducer: IO-01.py"

    [Download IO-01.py](IO-01.py)

    ```python
    """IO-01: Pelegant fails to load a multi-page beam file when a later page leaves a worker without rows

    When Pelegant, Elegant's MPI version, loads several particle-file pages with `track_pages_separately=0`, it should combine them into one beam. If a later page gives no rows to a worker that read an earlier page, the run stops with "Unable to get column--no rows left" before tracking. A five-particle file split into pages of four and one fails on three ranks (two workers), whereas serial Elegant and Pelegant with a one-plus-four split return all five particles unchanged.

    Setup: always runs 3-rank Pelegant (Open MPI `mpirun` on `PATH`) beside serial Elegant, because the defect occurs only under MPI.

    The script writes a binary SDDS beam file of five particles in two pages and tracks it through a zero-length drift, so that the output must reproduce the input exactly. The page sizes 4+1 trigger the fault, and the controls 1+4 and 3+2 hold the same particles but give every page after the first at least two rows. Serial Elegant is the control for the 4+1 case.
    """

    import sys
    from itertools import pairwise

    import numpy as np
    from _elegant import RunFailure, modes, particles, run, verdict

    N = 5
    X = 1e-4 * np.arange(1, N + 1)  # m, distinct per particle so that every row is checked
    IDS = np.arange(1, N + 1)
    # A zero-length drift leaves the coordinates unchanged, so the output equals the input exactly.
    EXPECTED = {"x": X, "xp": np.zeros(N), "y": np.zeros(N), "yp": np.zeros(N), "t": np.zeros(N)} | {
        "p": np.full(N, 3.0)
    }
    LATTICE = "D: EDRIFT,L=0\nBL: LINE=(D)\n"
    COMMANDS = (
        "&run_setup lattice=lattice.lte,use_beamline=BL,p_central=3,output=out.sdds,"
        "default_order=2,load_balancing_on=0 &end\n"
        "&run_control n_steps=1,n_passes=1 &end\n"
        "&sdds_beam input=beam.sdds,track_pages_separately=0 &end\n"
        "&track &end\n"
    )
    ERROR = "Unable to get column--no rows left"


    def pages_of(sizes):
        """The five particles split into consecutive pages of the given sizes."""
        edges = list(pairwise(np.cumsum([0, *sizes])))
        return [{"x": X[a:b], "p": np.full(len(X[a:b]), 3.0), "particleID": IDS[a:b]} for a, b in edges]


    def load(mode, pages):
        """(all six columns by particleID, "") read back from the output, or (None, error text) if the run stops."""
        try:
            result = run(
                {"lattice.lte": LATTICE, "run.ele": COMMANDS},
                mode,
                {"beam.sdds": pages_of(pages)},
            )
        except RunFailure as error:
            return None, ERROR if ERROR in error.log else error.log[-300:]
        if result["out.sdds"].n_pages != 1:
            sys.exit(f"{mode}: the output is not one page")
        return particles(result["out.sdds"]), ""


    def matches_input(found):
        """True when the output holds each input particle once with all six coordinates unchanged."""
        return (
            len(found["particleID"]) == N
            and np.array_equal(found["particleID"], IDS)
            and all(np.array_equal(found[name], EXPECTED[name]) for name in EXPECTED)
        )


    def compare(label, found, message):
        """Print the loaded (ID, x) rows or the error, next to the five rows of the input file."""
        rows = (
            "stopped: " + message
            if found is None
            else [(int(i), float(x)) for i, x in zip(found["particleID"], found["x"], strict=True)]
        )
        print(
            f"{label}: observed={rows} expected={[(int(i), float(x)) for i, x in zip(IDS, X, strict=True)]}"
        )


    defect = False
    for mode in ("serial", *modes(mpi_only=True)):
        for pages in ((4, 1), (1, 4), (3, 2)):
            found, message = load(mode, pages)
            label = f"{mode} pages {pages[0]}+{pages[1]}"
            compare(label, found, message)
            # All six coordinates of the output are compared with the input; the rows printed show x only.
            loaded = found is not None and matches_input(found)
            if mode == "serial" or pages != (4, 1):
                # Serial and the MPI controls must load all five particles unchanged.
                if not loaded:
                    sys.exit(f"{label}: control failed, the comparison is invalid ({message})")
            elif found is None:
                # Only the recognized error is the defect; any other failure is a failed run.
                if message != ERROR:
                    sys.exit(f"{label}: the run failed with another error ({message})")
                defect = True
            elif not loaded:
                sys.exit(f"{label}: the run succeeded but the output differs from the input")
    verdict(defect)

    ```

## Provenance

| Role | Model | Effort | Date | Code examined |
|---|---|---|---|---|
| Found | `gpt-6-astra` | xhigh | 2026-10-25 | master (`75a2e1a1a58b`), source with SDDS a60b9bbed7cc, CPU, 3-rank Pelegant |
| Confirmed | `gpt-6-astra` | high | 2026-10-25 | master (`75a2e1a1a58b`), source with SDDS a60b9bbed7cc, CPU, serial and 3-rank Pelegant |
| Confirmed | `gpt-6.1-sol` | high | 2026-10-25 | master (`75a2e1a1a58b`), source with SDDS a60b9bbed7cc, CPU, serial and 3-rank Pelegant |

