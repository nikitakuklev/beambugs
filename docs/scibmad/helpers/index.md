---
title: "SciBmad shared helpers"
---

# SciBmad shared helpers

These modules support the [SciBmad reproducers](../index.md).
Each entry links all the helpers it needs, including those imported by another helper.
Save the linked helpers in the same directory as the downloaded reproducer scripts.

## `_scibmad.py` { #_scibmad }

[Download _scibmad.py](_scibmad.py)

??? example "Source: _scibmad.py"

    ```python
    """Launch embedded Julia reproducers with one numerical thread."""

    import os
    import subprocess
    from pathlib import Path


    def run_julia(program: str, *, cwd: Path | None = None) -> None:
        """Require a configured SciBmad launcher and propagate execution failures."""
        launcher = os.environ.get("SCIBMAD_JULIA")
        if not launcher:
            raise RuntimeError("Set SCIBMAD_JULIA to a Julia launcher that loads SciBmad")
        runner = Path(launcher).expanduser().resolve(strict=True)
        env = os.environ.copy()
        for name in (
            "OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS",
            "NUMEXPR_NUM_THREADS", "JULIA_NUM_THREADS", "JULIA_NUM_PRECOMPILE_TASKS",
            "NPROC", "PJRT_NPROC",
        ):
            env[name] = "1"
        subprocess.run(
            [str(runner), "--startup-file=no", "-e", program],
            cwd=cwd, env=env, check=True, timeout=290,
        )

    ```
