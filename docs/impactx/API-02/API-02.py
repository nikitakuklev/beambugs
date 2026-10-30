"""API-02: MAD-X ion import discards explicit mass and charge.

MAD-X permits ION to override the species defaults. The imported kinetic
energy must be total ENERGY minus the specified MASS, in consistent units.
"""

import contextlib
import io
import math
import os
import warnings
from collections.abc import Callable
from decimal import Decimal
from pathlib import Path
from tempfile import NamedTemporaryFile
from typing import cast

for variable in (
    "NPROC",
    "PJRT_NPROC",
    "OMP_NUM_THREADS",
    "MKL_NUM_THREADS",
    "OPENBLAS_NUM_THREADS",
    "NUMEXPR_NUM_THREADS",
):
    os.environ[variable] = "1"

from impactx import RefPart


def load(label, command):
    with (
        NamedTemporaryFile(mode="w", suffix=".madx", dir=Path(__file__).parent) as deck,
        warnings.catch_warnings(record=True),
        contextlib.redirect_stdout(io.StringIO()) as printed,
    ):
        deck.write(command + ";\n")
        deck.flush()
        warnings.simplefilter("always")
        reference = RefPart()
        # The shipped stub misdeclares this Python extension's bound method.
        load_file = cast(Callable[[str], RefPart], reference.load_file)
        load_file(deck.name)
    result = (reference.mass_MeV, reference.charge_qe, reference.kin_energy_MeV)
    print(
        f"{label}: mass_MeV={result[0]:.10f} charge_qe={result[1]:.0f} kinetic_MeV={result[2]:.10f}"
    )
    print(f"  policy={printed.getvalue().strip()}")
    return result


bad = False
for total_GeV in (Decimal(10), Decimal("12.5")):
    args = f"MASS=3.7273794, CHARGE=2, ENERGY={total_GeV}"
    ion_case = load("ION", f"BEAM, PARTICLE=ION, {args}")
    custom_case = load("ALPHA_CUSTOM", f"BEAM, PARTICLE=ALPHA_CUSTOM, {args}")
    expected = (3727.3794, 2.0, float(total_GeV * 1000 - Decimal("3727.3794")))
    assert all(math.isclose(a, b, rel_tol=0, abs_tol=2e-10) for a, b in zip(custom_case, expected))
    bad |= not all(math.isclose(a, b, rel_tol=0, abs_tol=2e-10) for a, b in zip(ion_case, expected))
    print(
        f"energy_GeV={total_GeV}: expected_mass_MeV={expected[0]:.10f} "
        f"expected_charge_qe={expected[1]:.0f} expected_kinetic_MeV={expected[2]:.10f}"
    )
print("DEFECT REPRODUCED" if bad else "NOT REPRODUCED")
