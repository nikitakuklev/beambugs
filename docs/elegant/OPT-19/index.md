---
title: "OPT-19: tracking_matrix_cleanup=1 crashes at second order"
hide:
  - toc
---

# OPT-19 — tracking_matrix_cleanup=1 crashes at second order

| | |
|---|---|
| Code | [Elegant](../index.md) · Optics |
| Priority | P4 |
| Failure | Crash or API hazard |
| Evidence | Reproduced, independently rechecked |
| Last reproduced | 2026-11-04 on Elegant commit [`7514db3`](https://github.com/rtsoliday/elegant/commit/7514db36f764), committed 2026-11-04 |
| Source | [`determineMatrixHigherOrder`, src/analyze.c:1528–1531](https://github.com/rtsoliday/elegant/blob/0355dd6aca12c861971fbf8f3ae78a0ded991734/src/analyze.c#L1528-L1531)<br>[frees Q and sets `M->Q = NULL`, src/matrix.c:439–443](https://github.com/rtsoliday/elegant/blob/0355dd6aca12c861971fbf8f3ae78a0ded991734/src/matrix.c#L439-L443)<br>[`remove_s_dependent_matrix_elements`, src/matrix.c:1271–1290](https://github.com/rtsoliday/elegant/blob/0355dd6aca12c861971fbf8f3ae78a0ded991734/src/matrix.c#L1271-L1290)<br>[`global_settings`, src/global_settings.nl:29](https://github.com/rtsoliday/elegant/blob/0355dd6aca12c861971fbf8f3ae78a0ded991734/src/global_settings.nl#L29) |

Setting `global_settings tracking_matrix_cleanup=1`, which removes spurious path-coordinate terms from element maps that Elegant derives by tracking, makes Elegant crash with a segmentation fault when the matrix order is 2, the `run_setup` default. The clean-up writes third-order coefficients after the third-order storage of the map has been freed. A beamline of two CCBENDs, whose maps are derived by tracking, ends with "Terminated by SIGSEGV" when its second-order matrix is computed; the same input at order 3, or with the clean-up off, finishes and writes R56 = 0.05613 m.

Run `python OPT-19.py` with [_elegant.py](../helpers/_elegant.py) in the same directory.

??? example "Reproducer: OPT-19.py"

    [Download OPT-19.py](OPT-19.py)

    ```python
    """OPT-19: tracking_matrix_cleanup=1 crashes at second order

    Setting `global_settings tracking_matrix_cleanup=1`, which removes spurious path-coordinate terms
    from element maps that Elegant derives by tracking, makes Elegant crash with a segmentation fault
    when the matrix order is 2, the `run_setup` default. The clean-up writes third-order coefficients
    after the third-order storage of the map has been freed. A beamline of two CCBENDs, whose maps are
    derived by tracking, ends with "Terminated by SIGSEGV" when its second-order matrix is computed; the
    same input at order 3, or with the clean-up off, finishes and writes R56 = 0.05613 m.

    The script computes the second-order matrix of two CCBENDs, whose maps Elegant derives by tracking,
    in serial Elegant with `global_settings tracking_matrix_cleanup=1` and `default_order=2`, and
    reports whether the run ends with a segmentation fault. The controls are the same input with the
    cleanup off and the same input at order 3; both must finish and write the matrix.
    """

    from _elegant import RunFailure, run, verdict

    LATTICE = "E: DRIF,L=0.8\nB: CCBEND,L=1.1,ANGLE=0.16,K1=0.1,K2=2,N_SLICES=32\nBL: LINE=(B,E,B)\n"


    def case(cleanup, order):
        """Return (finished, note) for one run."""
        commands = (
            f"&global_settings tracking_matrix_cleanup={cleanup} &end\n"
            f"&run_setup lattice=lattice.lte, use_beamline=BL, p_central_mev=3000,"
            f" default_order={order} &end\n"
            f"&matrix_output SDDS_output=matrix.sdds, SDDS_output_order={order} &end\n"
        )
        try:
            result = run({"lattice.lte": LATTICE, "run.ele": commands}, outputs=("matrix.sdds",))
        except RunFailure as failure:
            crashed = "SIGSEGV" in failure.log
            return False, f"exit status {failure.status}{', Terminated by SIGSEGV' if crashed else ''}"
        r56 = float(result["matrix.sdds"][0, "R56"][-1])
        return True, f"finished, R56={r56:.10g} m"


    defect = False
    for cleanup, order, control in ((0, 2, True), (1, 3, True), (1, 2, False)):
        finished, note = case(cleanup, order)
        label = "control, " if control else ""
        print(
            f"{label}tracking_matrix_cleanup={cleanup} default_order={order}: observed={note} expected=finished"
        )
        if control and not finished:
            raise SystemExit("A control run failed: invalid comparison.")
        defect |= not finished
    verdict(defect)

    ```

## Provenance

| Role | Model | Effort | Date | Code examined |
|---|---|---|---|---|
| Found | `claude-opus-5-5` | high | 2026-11-04 | master (`7514db36f764`), source with SDDS a3f730a14c49, CPU, serial |
| Confirmed | `gpt-6.1-sol` | xhigh | 2026-11-04 | master (`7514db36f764`), source with SDDS a3f730a14c49, CPU, serial |

