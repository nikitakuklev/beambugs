"""API-23: MAD-X import documentation contains a malformed RFCAVITY sentence

The MAD-X lattice-import documentation contains a malformed sentence combining “is a thin kick model” with “becomes”. It should state that a finite-length MAD-X RFCAVITY becomes ExactDrift + ShortRF + ExactDrift at `min_model="exact"`. A 1.600 m cavity correctly imports as a 0.8000 m drift, a zero-length ShortRF kick, and another 0.8000 m drift, so only the explanatory prose requires repair.

Setup: `IMPACTX_SOURCE` set to a clone of BLAST-ImpactX/impactx at tag `26.10`, and the run pinned to one CPU (`taskset -c <n>`); the script reads the Python usage manual from the clone, and the shared `_slicing` helper refuses to run on more CPUs.

The script reads the Python usage manual from the clone and renders the RFCAVITY example with Docutils to check that the malformed sentence survives rendering. It imports inline finite-length and zero-length MAD-X cavities at all three model floors and compares their types and lengths with the documented drift-kick-drift construction. The shared helper caps numerical threads; temporary decks live beside this script and are removed on exit.
"""

import html
import importlib.metadata
import io
import os
import re
import sys
import tempfile
import warnings
from contextlib import redirect_stdout
from pathlib import Path

from _slicing import CPU
from docutils.core import publish_parts
from impactx.madx_to_impactx import read_lattice

BROKEN = "is a thin kick model, for example, becomes"
CORRECT = (
    "For example, a finite-length RFCAVITY becomes ExactDrift + ShortRF + ExactDrift "
    'at min_model="exact".'
)
FLOORS = {"linear": "Drift", "paraxial": "ChrDrift", "exact": "ExactDrift"}


def documentation():
    root = os.environ.get("IMPACTX_SOURCE")
    if not root:
        raise RuntimeError("Set IMPACTX_SOURCE to the ImpactX source checkout")
    lines = (Path(root) / "docs/source/usage/python.rst").read_text().splitlines()
    starts = [i for i, line in enumerate(lines) if "finite-length ``RFCAVITY``" in line]
    if len(starts) != 1:
        raise RuntimeError("Expected one finite-length RFCAVITY documentation example")
    start = starts[0]
    paragraph = []
    for line in lines[start:]:
        if not line.strip():
            break
        paragraph.append(line.strip())
        if line.rstrip().endswith("."):
            break
    source = " ".join(paragraph)
    fragment = publish_parts(source, writer_name="html5")["fragment"]
    rendered = html.unescape(re.sub(r"<[^>]+>", "", fragment)).strip()
    broken_source, broken_rendered = BROKEN in source, BROKEN in rendered
    if broken_source != broken_rendered:
        raise RuntimeError("Source and rendered-text checks disagree")
    print(
        f"DOC source line={start + 1} observed={source!r} expected={CORRECT!r} "
        "derivation=one subject and one connected verb phrase"
    )
    print(
        f"DOC rendered observed={rendered!r} expected={CORRECT!r} "
        "derivation=Docutils preserves the sentence's prose"
    )
    return broken_source


def conversions():
    with tempfile.TemporaryDirectory(dir=Path(__file__).resolve().parent, prefix="rf-") as folder:
        for length in (1.6, 0.0):
            # Finite MAD-X RFCAVITY: length L is split equally around a zero-length kick.
            # Explicit FREQ and LINE isolate the prose example from circumference/SEQUENCE paths.
            deck = (
                "beam,particle=electron,energy=2;"
                f"c:rfcavity,l={length},volt=0.75,freq=402.5,lag=0.25;"
                "r:line=(c);use,period=r;"
            )
            path = Path(folder) / "case.madx"
            path.write_text(deck)
            for floor, drift in FLOORS.items():
                with warnings.catch_warnings(record=True), redirect_stdout(io.StringIO()):
                    warnings.simplefilter("always")
                    lattice = read_lattice(str(path), min_model=floor)
                types = [type(element).__name__ for element in lattice]
                lengths = [element.ds for element in lattice]
                expected_types = [drift, "ShortRF", drift] if length else ["ShortRF"]
                expected_lengths = [length / 2, 0.0, length / 2] if length else [0.0]
                if types != expected_types:
                    raise RuntimeError(f"RF types changed: {types} != {expected_types}")
                residuals = [abs(a - b) for a, b in zip(lengths, expected_lengths, strict=True)]
                if not all(error <= 2e-14 for error in residuals):
                    raise RuntimeError(f"RF lengths changed: {lengths} != {expected_lengths}")
                residual = max(residuals)
                print(
                    f"RF L={length:.4f}m floor={floor} observed_types={types} "
                    f"expected_types={expected_types} observed_lengths_m={lengths} "
                    f"expected_lengths_m={expected_lengths} max_abs_residual_m={residual:.1e} "
                    "derivation=L/2+zero-length-kick+L/2; L=0 gives only the kick"
                )


def main():
    versions = {
        name: importlib.metadata.version(name) for name in ("impactx-noacc", "numpy", "docutils")
    }
    print(
        f"environment Python={sys.version.split()[0]} packages={versions} CPU={CPU} numerical_threads=1"
    )
    broken = documentation()
    conversions()
    print("DEFECT REPRODUCED" if broken else "NOT REPRODUCED")


if __name__ == "__main__":
    main()
