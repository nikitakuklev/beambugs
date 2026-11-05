---
title: "COL-32: insert_sceffects never stores verbosity, so the insertion notice reads uninitialized memory"
hide:
  - toc
---

# COL-32 — insert_sceffects never stores verbosity, so the insertion notice reads uninitialized memory

| | |
|---|---|
| Code | [Elegant](../index.md) · Collective effects |
| Priority | P4 |
| Severity | Crash or API hazard |
| Evidence | Reproduced, independently rechecked |
| Last reproduced | 2026-11-04 on Elegant master (`0355dd6`) |
| Source | [SC_SPEC, get_beamline.c:2327–2330](https://github.com/rtsoliday/elegant/blob/0355dd6aca12c861971fbf8f3ae78a0ded991734/src/get_beamline.c#L2327-L2330)<br>[addSCSpec, get_beamline.c:2339–2357](https://github.com/rtsoliday/elegant/blob/0355dd6aca12c861971fbf8f3ae78a0ded991734/src/get_beamline.c#L2339-L2357)<br>[notice, get_beamline.c:2399–2400](https://github.com/rtsoliday/elegant/blob/0355dd6aca12c861971fbf8f3ae78a0ded991734/src/get_beamline.c#L2399-L2400)<br>[verbosity, elegant.tex:4491](https://github.com/rtsoliday/elegant/blob/0355dd6aca12c861971fbf8f3ae78a0ded991734/doc/elegant.tex#L4491) |

insert_sceffects accepts verbosity but never copies it into the stored insertion specification, whose verbosity field, left uninitialized on the heap, then decides whether a notice is printed for each inserted SCMULT. With verbosity = 5 and one insertion, the run prints no notice, and Valgrind reports a conditional jump on an uninitialized value in the insertion routine, created by the allocation of the specification. The manual says larger nonzero values of verbosity request more printed detail, and the source prints the notice for positive values. The kicks are unaffected.

Download these shared helpers into the same directory as the reproducer scripts:

- [Download _elegant.py](../helpers/_elegant.py) · [View source](../helpers/index.md#_elegant)
- [Download _scinsert.py](../helpers/_scinsert.py) · [View source](../helpers/index.md#_scinsert)

??? example "Reproducer: COL-32.py"

    [Download COL-32.py](COL-32.py)

    ```python
    """COL-32: insert_sceffects never stores verbosity, so the insertion notice reads uninitialized memory

    insert_sceffects accepts verbosity but never copies it into the stored insertion specification,
    whose verbosity field, left uninitialized on the heap, then decides whether a notice is printed for
    each inserted SCMULT. With verbosity = 5 and one insertion, the run prints no notice, and Valgrind
    reports a conditional jump on an uninitialized value in the insertion routine, created by the
    allocation of the specification. The manual says larger nonzero values of verbosity request more
    printed detail, and the source prints the notice for positive values. The kicks are unaffected.

    The script runs serial Elegant under Valgrind (which must be on PATH) on a one-segment ring (a
    0.8 nC bunch of 243 particles and a 1 m EMATRIX quarter turn) with insert_sceffects name = "R",
    skip = 1 and verbosity = 5. Expected are one insertion notice and no use of uninitialized memory;
    the verdict is Valgrind's report of a conditional jump on an uninitialized value in insertSCMULT,
    originating in the allocation of the insertion specification. The same run without
    insert_sceffects is the control.
    """

    import os
    import re
    import shutil
    import subprocess
    import sys
    import tempfile
    from pathlib import Path

    from _elegant import verdict, write_beam
    from _scinsert import BEAM, LATTICE, TAIL, spec

    TIMEOUT = 600  # seconds per Valgrind run
    NOTICE = "Inserting SCMULT after R"
    # One Valgrind error block: a header line, then the call stack and origin ("at"/"by" lines and
    # the "Uninitialised value was created" paragraph of --track-origins).
    UNINITIALISED = re.compile(
        r"Conditional jump or move depends on uninitialised value\(s\)\n"
        r"((?:==\d+==\s+(?:at|by) .*\n)+)(?:==\d+==\s+Uninitialised value was created by (.*)\n"
        r"((?:==\d+==\s+(?:at|by) .*\n)+))?"
    )
    CASES = (
        ("insert_sceffects omitted (control)", "", 0, True),
        ('insert_sceffects name="R",skip=1,verbosity=5', spec(name="R", skip=1, verbosity=5), 1, False),
    )


    def checked_run(commands):
        """Serial Elegant under Valgrind: (exit status, log, [(first stack frame, origin frames)])."""
        if shutil.which("valgrind") is None or shutil.which("elegant") is None:
            sys.exit("valgrind and elegant must be on PATH.")
        if "RPN_DEFNS" not in os.environ:
            sys.exit("Set RPN_DEFNS to the defns.rpn file of SDDS (see the elegant installation).")
        env = dict(os.environ, OMP_NUM_THREADS="1")
        with tempfile.TemporaryDirectory(prefix="elegant-repro-") as tmp:
            work = Path(tmp)
            (work / "lattice.lte").write_text(LATTICE)
            (work / "run.ele").write_text("&global_settings echo_namelists=0 &end\n" + commands + TAIL)
            write_beam(work / "beam.sdds", BEAM)
            try:
                done = subprocess.run(
                    ["valgrind", "-q", "--error-limit=no", "--track-origins=yes", "elegant", "run.ele"],
                    cwd=work,
                    env=env,
                    capture_output=True,
                    text=True,
                    timeout=TIMEOUT,
                    check=False,
                )
            except subprocess.TimeoutExpired:
                sys.exit(f"valgrind elegant exceeded {TIMEOUT} s")
        log = done.stdout + done.stderr
        reports = [
            (stack.splitlines()[0].split(": ", 1)[1], origin_stack)
            for stack, _, origin_stack in UNINITIALISED.findall(log)
            if "insertSCMULT" in stack.splitlines()[0]
        ]
        return done.returncode, log, reports


    defect = False
    for label, commands, notices, control in CASES:
        status, log, reports = checked_run(commands)
        if status:
            sys.exit(f"serial {label}: exit status {status}:\n{log[-2000:]}")
        count = log.count(NOTICE)
        if reports:
            frames = [ln.split(": ", 1)[1] for ln in reports[0][1].splitlines() if "addSCSpec" in ln]
            origin = f"heap allocation in {frames[0]}" if frames else "unknown origin"
            shown = f"uninitialized conditional at {reports[0][0]}, from a {origin}"
        else:
            shown = "no uninitialized conditional in insertSCMULT"
        print(
            f"serial {label}: observed={count} insertion notice(s), {shown} "
            f"expected={notices} insertion notice(s), no uninitialized conditional"
        )
        if control and (reports or count != notices):
            sys.exit(f"serial {label}: the control fails")
        defect |= bool(reports) or count != notices
    verdict(defect)

    ```

## Provenance

| Role | Model | Effort | Date | Code examined |
|---|---|---|---|---|
| Found | `gpt-6-astra` | xhigh | 2026-11-04 | master (`0355dd6aca12`), source with SDDS a3f730a14c49, CPU, serial |
| Confirmed | `gpt-6.1-sol` | xhigh | 2026-11-04 | master (`0355dd6aca12`), source with SDDS a3f730a14c49, CPU, serial |

