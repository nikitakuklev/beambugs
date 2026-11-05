"""Strict process checks for the reviewed translator examples.

The diagnostic tracker deliberately lets Bmad report parser errors. Ordinary
controls must reject those diagnostics even if Bmad also returns finite records.
"""

import re

from _bmad import run
from _s3_24_translations import track as _diagnostic_track

ERROR = r"\[(?:ERROR|FATAL|ABORT)|ERROR IN "


def diagnostic_track(text, start=(0.001, 0, 0.002, 0, 0, 0), *, expected_error=None):
    """Allow only the named diagnostic; a successful future translation is valid."""
    result, log = _diagnostic_track(text, start=start, allow_diagnostics=True)
    messages = re.findall(
        r"(?ms)^(?:\[(?:ERROR|FATAL|ABORT)[^\n]*|ERROR IN [^\n]*).*?"
        r"(?=^\[|^ERROR IN |^PARSE_REJECTED|^FIRST |^ORBIT |^COMPLETED|\Z)",
        log,
    )
    if any(expected_error is None or expected_error not in message for message in messages):
        raise RuntimeError(log)
    if not result and not messages:
        raise RuntimeError(log)
    return result, log


def track(text, start=(0.001, 0, 0.002, 0, 0, 0)):
    """Require successful parsing/tracking with no fatal or error diagnostics."""
    result, log = diagnostic_track(text, start=start)
    if not result or re.search(ERROR, log):
        raise RuntimeError(log)
    return result, log


def export(text: str, kind: str) -> str:
    """Require a completed production export and reject parse/tracking errors."""
    suffix = {"madx": "madx", "mad8": "mad8", "sad": "sad", "elegant": "lte"}[kind]
    result = run(
        "bmad_to_mad_sad_elegant",
        [f"-{kind}", "case.bmad"],
        files={"case.bmad": text},
        outputs=[f"case.{suffix}"],
    )
    if "Written" not in result.log or re.search(ERROR, result.log):
        raise RuntimeError(result.log)
    return result.outputs[f"case.{suffix}"].decode()


def tao(text: str, commands: list[str]) -> dict[str, list[str]]:
    """Read Tao pipe records only after successful initialization and completion."""
    result = run(
        "tao",
        ["-noplot", "-noinit", "-lat", "case.bmad", "-startup", "run.tao"],
        files={"case.bmad": text, "run.tao": "\n".join([*commands, "quit", ""])},
    )
    if re.search(ERROR, result.log):
        raise RuntimeError(result.log)
    sections = {}
    current = None
    for line in result.log.splitlines():
        if line.startswith("Tao: "):
            current = sections.setdefault(line[5:].strip(), [])
        elif current is not None and line.strip():
            current.append(line)
    if "quit" not in sections or any(not sections.get(command) for command in commands):
        raise RuntimeError(result.log)
    if any("ERROR" in line for command in commands for line in sections[command]):
        raise RuntimeError(result.log)
    return sections
