---
title: "COL-51: LRWAKE inherits the preceding bunch count across independent new lattices"
hide:
  - toc
---

# COL-51 — LRWAKE inherits the preceding bunch count across independent new lattices

| | |
|---|---|
| Code | [Elegant](../index.md) · Collective effects |
| Priority | P4 |
| Severity | Crash or API hazard |
| Evidence | Reproduced, independently rechecked |
| Last reproduced | 2026-11-05 on Elegant master (`0355dd6`, committed 2026-11-03) |
| Source | [`track_through_lrwake`, lrwake.c:236–242](https://github.com/rtsoliday/elegant/blob/0355dd6aca12c861971fbf8f3ae78a0ded991734/src/lrwake.c#L236-L242)<br>[`track_through_lrwake`, lrwake.c:291–293](https://github.com/rtsoliday/elegant/blob/0355dd6aca12c861971fbf8f3ae78a0ded991734/src/lrwake.c#L291-L293)<br>[`index_bunch_assignments`, lrwake.c:17–25](https://github.com/rtsoliday/elegant/blob/0355dd6aca12c861971fbf8f3ae78a0ded991734/src/lrwake.c#L17-L25)<br>[`index_bunch_assignments`, lrwake.c:90–103](https://github.com/rtsoliday/elegant/blob/0355dd6aca12c861971fbf8f3ae78a0ded991734/src/lrwake.c#L90-L103)<br>[`set_up_lrwake`, lrwake.c:532–535](https://github.com/rtsoliday/elegant/blob/0355dd6aca12c861971fbf8f3ae78a0ded991734/src/lrwake.c#L532-L535)<br>[`set_up_lrwake`, lrwake.c:681–688](https://github.com/rtsoliday/elegant/blob/0355dd6aca12c861971fbf8f3ae78a0ded991734/src/lrwake.c#L681-L688) |

When successive `run_setup` commands load distinct lattices and new LRWAKE elements in one process, LRWAKE rejects a bunch count larger than that of its immediately preceding invocation. Each new element should allocate history for its own beam, and an identical two-bunch setup succeeds in a fresh process. With six particles per bunch and an identically zero wake, a one-bunch then two-bunch sequence completes only one of two runs; a two-bunch, one-bunch, two-bunch sequence completes only two of three. Both failures print “Error: number of bunches has increased.”, while repeated equal-count runs complete.

Run `python COL-51.py` with [_elegant.py](../helpers/_elegant.py) in the same directory.

??? example "Reproducer: COL-51.py"

    [Download COL-51.py](COL-51.py)

    ```python
    """COL-51: LRWAKE inherits the preceding bunch count across independent new lattices

    When successive `run_setup` commands load distinct lattices and new LRWAKE elements in one process, LRWAKE rejects a bunch count larger than that of its immediately preceding invocation. Each new element should allocate history for its own beam, and an identical two-bunch setup succeeds in a fresh process. With six particles per bunch and an identically zero wake, a one-bunch then two-bunch sequence completes only one of two runs; a two-bunch, one-bunch, two-bunch sequence completes only two of three. Both failures print “Error: number of bunches has increased.”, while repeated equal-count runs complete.

    The script runs five serial Elegant processes, each containing a prescribed sequence of distinct lattice, element and beam files. It uses consecutive particle IDs and IDSlotsPerBunch=6, checks completed output populations and unchanged momentum, and accepts only the specific bunch-count rejection as a defect. Multiple run setups deliberately share one native process because the defect concerns process-wide state; each sequence starts fresh.
    """

    import os
    import shutil
    import subprocess
    import tempfile
    from pathlib import Path

    import numpy as np
    import pysdds
    from _elegant import TIMEOUT, RunFailure, particles, verdict, write_beam

    ERROR = "Error: number of bunches has increased."
    WAKE = """SDDS1
    &column name=t,type=double,units=s &end
    &column name=W,type=double,units=V/C &end
    &data mode=ascii &end
    3
    0 0
    1e-9 0
    2e-9 0
    """


    def experiment(counts):
        if "RPN_DEFNS" not in os.environ:
            raise RuntimeError("Set RPN_DEFNS to the SDDS defns.rpn file")
        program = shutil.which("elegant")
        if program is None:
            raise RuntimeError("elegant must be on PATH")
        with tempfile.TemporaryDirectory(prefix="elegant-lrwake-") as tmp:
            folder = Path(tmp)
            (folder / "zero.sdds").write_text(WAKE)
            commands = []
            for i, count in enumerate(counts):
                # floor((particleID-1)/IDSlotsPerBunch) gives exactly count bunches.
                index = np.arange(6 * count)
                write_beam(
                    folder / f"beam{i}.sdds",
                    {
                        "x": np.full(len(index), 0.0002),
                        "y": np.full(len(index), -0.0001),
                        "t": (index // 6) * 0.8e-9 + (index % 6) * 1e-13,
                        "p": np.full(len(index), 800.0),
                        "particleID": index + 1,
                        "IDSlotsPerBunch": 6,
                    },
                )
                (folder / f"line{i}.lte").write_text(
                    f"C{i}: CHARGE,TOTAL=1.2e-10\n"
                    f'W{i}: LRWAKE,INPUTFILE="zero.sdds",TCOLUMN="t",WZCOLUMN="W",TURNS_TO_KEEP=2\n'
                    f"D{i}: DRIF,L={0.23 + i * 0.18:.17g}\nB{i}: LINE=(C{i},D{i},W{i})\n"
                )
                commands.append(f"""&run_setup lattice="line{i}.lte",use_beamline=B{i},
     p_central=800,output="final{i}.sdds" &end
    &run_control n_steps=1,n_passes=1 &end
    &sdds_beam input="beam{i}.sdds",use_bunched_mode=1 &end
    &track &end
    """)
            (folder / "run.ele").write_text("\n".join(commands))
            env = dict(os.environ, OMP_NUM_THREADS="1", OPENBLAS_NUM_THREADS="1", MKL_NUM_THREADS="1")
            result = subprocess.run(
                [program, "run.ele"],
                cwd=folder,
                env=env,
                text=True,
                capture_output=True,
                timeout=TIMEOUT,
                check=False,
            )
            log = result.stdout + result.stderr
            rejected = result.returncode == 1 and ERROR in log
            if result.returncode and not rejected:
                raise RunFailure(program, result.returncode, log)
            # Count actual completion markers, then validate the corresponding output
            # pages. A file created by run_setup alone does not prove tracking completed.
            completed = log.count("Post-tracking output completed.")
            max_change = 0.0
            # LRWAKE calls the energy updater even for zero voltage. Its p -> gamma
            # -> p round trip at p=800 returns one binary64 spacing below the input.
            gamma = np.sqrt(800.0**2 + 1)
            roundtrip = np.sqrt(gamma**2 - 1)
            for i in range(completed):
                out = particles(pysdds.read(folder / f"final{i}.sdds"))
                assert len(out["p"]) == 6 * counts[i], "Unexpected particle loss"
                np.testing.assert_allclose(out["p"], roundtrip, rtol=0, atol=np.spacing(800.0))
                max_change = max(max_change, float(np.max(np.abs(out["p"] - 800))))
            assert max_change <= np.spacing(800.0), "Zero-wake momentum control failed"
            if result.returncode == 0:
                assert completed == len(counts), "Successful exit without all output pages"
            print(
                f"serial new-run bunches={counts}: completed={completed} expected={len(counts)} "
                f"(independent element histories); status={result.returncode}; "
                f"bunch-count rejection={rejected}; max_abs_dp={max_change:g} expected=0"
            )
            if rejected:
                print(f"serial native diagnostic: {ERROR}")
            return rejected, completed


    def main():
        for counts in ([2], [1, 1], [2, 2]):
            rejected, completed = experiment(counts)
            assert not rejected and completed == len(counts), "Fresh/equal-count control failed"
        observed = [experiment(counts) for counts in ([1, 2], [2, 1, 2])]
        verdict(observed == [(True, 1), (True, 2)])


    if __name__ == "__main__":
        main()

    ```

## Provenance

| Role | Model | Effort | Date | Code examined |
|---|---|---|---|---|
| Found | `gpt-6-astra` | xhigh | 2026-11-05 | master (`0355dd6aca12`), source with SDDS a3f730a14c49, CPU, serial |
| Confirmed | `gpt-6.1-sol` | xhigh | 2026-11-05 | master (`0355dd6aca12`), source with SDDS a3f730a14c49, CPU, serial |

