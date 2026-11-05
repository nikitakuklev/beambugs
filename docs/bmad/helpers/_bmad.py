"""Shared Bmad runner: isolated inputs, build-derived Fortran recipes, numerical verdicts.

BMAD_DIST names the build root containing production/{bin,lib,modules}; BMAD_SOURCE
names a source tree for source-reading reproducers. Native processes run serially with
one numerical thread. Scratch directories live under the system temp directory's
bmad-scratch/ and are removed on success, failure and timeout.
"""

from __future__ import annotations

import os
import re
import shlex
import signal
import subprocess
import tempfile
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path

import numpy as np
from numpy.typing import ArrayLike

TIMEOUT = 290  # seconds per compilation, link or native run
_THREADS = ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "NUMEXPR_NUM_THREADS")


class RunFailure(RuntimeError):
    """A failed process; status and the complete combined output remain available."""

    def __init__(self, program: str, status: int, log: str):
        super().__init__(f"{program} exited with status {status}:\n{log[-3000:]}")
        self.program, self.status, self.log = program, status, log


@dataclass
class Result:
    """Combined stdout/stderr and requested files as bytes (decode text with .decode())."""

    log: str
    outputs: dict[str, bytes]


def _root(variable: str) -> Path:
    value = os.environ.get(variable)
    if not value:
        raise RuntimeError(
            f"Set {variable} to the Bmad {'build' if variable == 'BMAD_DIST' else 'source'} root."
        )
    root = Path(value).expanduser().resolve()
    if not root.is_dir():
        raise RuntimeError(f"{variable} is not a directory: {root}")
    return root


def build() -> Path:
    """Resolve BMAD_DIST and require its production binaries, libraries and modules."""
    root = _root("BMAD_DIST")
    for name in ("bin", "lib", "modules"):
        if not (root / "production" / name).is_dir():
            raise RuntimeError(
                f"BMAD_DIST must name the build root; missing {root / 'production' / name}"
            )
    return root


def source() -> Path:
    """Resolve BMAD_SOURCE independently of the executable build."""
    root = _root("BMAD_SOURCE")
    if not (root / "bmad/doc/bmad.tex").is_file():
        raise RuntimeError(f"BMAD_SOURCE lacks bmad/doc/bmad.tex: {root}")
    return root


def _scratch() -> tempfile.TemporaryDirectory[str]:
    parent = Path(tempfile.gettempdir()) / "bmad-scratch"
    parent.mkdir(exist_ok=True)
    return tempfile.TemporaryDirectory(prefix="repro-", dir=parent)


def _path(work: Path, name: str) -> Path:
    if not name or Path(name).is_absolute() or ".." in Path(name).parts:
        raise ValueError(f"Expected a relative case file name: {name!r}")
    return work / name


def _execute(argv: Sequence[str], work: Path, stdin: str, timeout: float) -> str:
    env = dict(os.environ, **dict.fromkeys(_THREADS, "1"), OMP_DYNAMIC="FALSE")
    with subprocess.Popen(
        argv,
        cwd=work,
        env=env,
        stdin=subprocess.PIPE,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        start_new_session=True,
    ) as process:
        try:
            log, _ = process.communicate(stdin, timeout=timeout)
        except subprocess.TimeoutExpired:
            # Compilers spawn a linker; terminate the whole group before removing scratch.
            os.killpg(process.pid, signal.SIGKILL)
            log, _ = process.communicate()
            raise TimeoutError(f"{argv[0]} exceeded {timeout} s:\n{log[-3000:]}") from None
        if process.returncode:
            raise RunFailure(argv[0], process.returncode, log)
    return log


def _case(
    argv: Sequence[str],
    work: Path,
    files: Mapping[str, str | bytes] | None,
    outputs: Sequence[str],
    stdin: str,
    timeout: float,
) -> Result:
    work.mkdir()
    for name, content in (files or {}).items():
        path = _path(work, name)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(content.encode() if isinstance(content, str) else content)
    log = _execute(argv, work, stdin, timeout)
    return Result(log, {name: _path(work, name).read_bytes() for name in outputs})


def run(
    program: str,
    args: Sequence[str] = (),
    *,
    files: Mapping[str, str | bytes] | None = None,
    outputs: Sequence[str] = (),
    stdin: str = "",
    timeout: float = TIMEOUT,
) -> Result:
    """Run a named production/bin executable in fresh scratch; collect outputs before cleanup.

    Pass lattice, Tao command and namelist files inline. Nonzero exits raise RunFailure;
    timeouts raise TimeoutError; a missing requested output raises FileNotFoundError.
    Check completion markers and case-specific diagnostics even when the exit status is zero.
    """
    if Path(program).name != program:
        raise ValueError("program must be a name in BMAD_DIST/production/bin")
    executable = build() / "production/bin" / program
    if not executable.is_file() or not os.access(executable, os.X_OK):
        raise RuntimeError(f"Required executable missing or not executable: {executable}")
    with _scratch() as tmp:
        return _case([str(executable), *args], Path(tmp) / "run", files, outputs, stdin, timeout)


def _recipe(tao: bool) -> tuple[list[str], list[str]]:
    relative = (
        "tao/production/CMakeFiles/tao-exe.dir"
        if tao
        else "regression_tests/production/CMakeFiles/tracking_method_test-exe.dir"
    )
    recipe = build() / relative
    flags_text = (recipe / "flags.make").read_text()
    flags = []
    for key in ("Fortran_DEFINES", "Fortran_INCLUDES", "Fortran_FLAGS"):
        match = re.search(rf"^{key} = (.*)$", flags_text, re.MULTILINE)
        if match is None:
            raise RuntimeError(f"Missing {key} in {recipe / 'flags.make'}")
        flags.extend(shlex.split(match[1]))
    custom = re.findall(r"^# Custom flags: .*?_FLAGS = (.*)$", flags_text, re.MULTILINE)
    if len(custom) != 1:
        raise RuntimeError(f"Expected one source compile recipe in {recipe}")
    flags.extend(shlex.split(custom[0]))
    # The supplied GNU recipes use joined -J paths. Keep generated modules in scratch.
    if "-J" in flags:
        raise RuntimeError(f"Expected joined -J module path in {recipe}")
    flags = [flag for flag in flags if not flag.startswith("-J")]
    link = shlex.split((recipe / "link.txt").read_text())
    objects = [i for i, token in enumerate(link) if token.endswith(".o")]
    if not link or len(objects) != 1 or link.count("-o") != 1 or "@" in " ".join(link):
        raise RuntimeError(f"Expected a single-object executable link recipe in {recipe}")
    if not Path(link[0]).is_file():
        raise RuntimeError(f"Build compiler missing: {link[0]}")
    link[objects[0]] = "probe.o"
    link[link.index("-o") + 1] = "probe"
    # These diagnostics point into the original build; retain all actual link options in order.
    link = [
        token for token in link if not token.startswith(("-Wl,--dependency-file=", "-Wl,-Map="))
    ]
    return [link[0], *flags, "-J.", "-c", "probe.f90", "-o", "probe.o"], link


def fortran(
    program: str,
    args: Sequence[str] = (),
    *,
    tao: bool = False,
    files: Mapping[str, str | bytes] | None = None,
    outputs: Sequence[str] = (),
    stdin: str = "",
    timeout: float = TIMEOUT,
) -> Result:
    """Compile/link inline Fortran with the build's recipe, then run it in fresh scratch.

    The tracking_method_test recipe supplies Bmad/PTC libraries; tao=True selects Tao's
    own recipe for Tao modules. The compiler and every library come from flags.make/link.txt.
    The build must retain these CMake files and the compiler that produced its modules.
    """
    compile_argv, link_argv = _recipe(tao)
    with _scratch() as tmp:
        work = Path(tmp)
        (work / "probe.f90").write_text(program)
        _execute(compile_argv, work, "", timeout)
        _execute(link_argv, work, "", timeout)
        return _case([str(work / "probe"), *args], work / "run", files, outputs, stdin, timeout)


def compare(
    label: str,
    observed: ArrayLike,
    expected: ArrayLike,
    *,
    atol: float,
    rtol: float = 0.0,
) -> bool:
    """Print observed/expected values and return True for a discrepancy beyond the tolerances.

    Use a label that names the case, units and expectation. Shapes must match; empty arrays,
    nonfinite numbers and invalid tolerances raise ValueError. Crash/NaN claims need their
    own explicit checks. A passing control requires compare(...) to return False.
    """
    actual, target = np.asarray(observed, dtype=float), np.asarray(expected, dtype=float)
    if actual.shape != target.shape or not actual.size:
        raise ValueError(
            f"{label}: expected equal nonempty shapes, got {actual.shape}, {target.shape}"
        )
    if not (np.isfinite(actual).all() and np.isfinite(target).all()):
        raise ValueError(f"{label}: comparison requires finite values")
    if not (np.isfinite([atol, rtol]).all() and atol >= 0 and rtol >= 0):
        raise ValueError("Comparison tolerances must be finite and nonnegative")
    defect = not np.allclose(actual, target, atol=atol, rtol=rtol)
    print(
        f"{label}: observed={actual.tolist()} expected={target.tolist()} atol={atol:g} rtol={rtol:g}"
    )
    return defect


def verdict(defect: bool) -> None:
    """Print the final verdict after all required cases and controls have completed."""
    print("DEFECT REPRODUCED" if defect else "NOT REPRODUCED")
