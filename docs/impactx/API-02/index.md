---
title: "API-02: MAD-X ion import discards explicit mass and charge"
hide:
  - toc
---

# API-02 — MAD-X ion import discards explicit mass and charge

| | |
|---|---|
| Code | [ImpactX](../index.md) · Python API |
| Priority | P2 |
| Failure | Wrong physics |
| Evidence | Reproduced, independently rechecked |
| Last reproduced | 2026-10-28 on ImpactX release [26.10](https://github.com/BLAST-ImpactX/impactx/tree/26.10) (`615921c`) |
| Pull request | [[gh:https://github.com/BLAST-ImpactX/impactx/pull/1727]], filed from this evidence |
| Source | [`madx_to_impactx.py#L1681-L1697`](https://github.com/BLAST-ImpactX/impactx/blob/b701a6e9d1f9e30cefb9a3e18e82ae7e02b9d943/src/python/impactx/madx_to_impactx.py#L1681-L1697) |

Importing a MAD-X `BEAM, PARTICLE=ION` with explicit `MASS` and `CHARGE` into ImpactX discards both in favor of a fixed default ion mass and charge, so the reference mass, charge, kinetic energy and rigidity are wrong before tracking starts. MAD-X documents independent mass and charge overrides for ions, and native MAD-X keeps them for this input. For an alpha particle, `BEAM, PARTICLE=ION, MASS=3.7273794, CHARGE=2, ENERGY=10` (GeV), ImpactX uses a mass of 939.6 MeV and charge 1, giving 9060 MeV kinetic energy instead of 6273 MeV. The importer prints that it ignores `MASS` and `CHARGE`; an unrecognized particle name keeps them.

Run `python API-02.py`.

??? example "Reproducer: API-02.py"

    [Download API-02.py](API-02.py)

    ```python
    """API-02: MAD-X ion import discards explicit mass and charge

    Importing a MAD-X `BEAM, PARTICLE=ION` with explicit `MASS` and `CHARGE` into ImpactX discards both
    in favor of a fixed default ion mass and charge, so the reference mass, charge, kinetic energy and
    rigidity are wrong before tracking starts. MAD-X documents independent mass and charge overrides for
    ions, and native MAD-X keeps them for this input. For an alpha particle, `BEAM, PARTICLE=ION,
    MASS=3.7273794, CHARGE=2, ENERGY=10` (GeV), ImpactX uses a mass of 939.6 MeV and charge 1, giving
    9060 MeV kinetic energy instead of 6273 MeV. The importer prints that it ignores `MASS` and
    `CHARGE`; an unrecognized particle name keeps them.

    The script loads `BEAM, PARTICLE=ION, MASS=3.7273794, CHARGE=2` with `ENERGY` = 10 and 12.5 GeV
    through `RefPart.load_file()` and compares the reference mass, charge and kinetic energy with the
    given mass, the given charge and `ENERGY` − `MASS`. The same deck with the unknown particle name
    `ALPHA_CUSTOM`, which takes the generic-species path, is the control and must match within 2×10⁻¹⁰
    MeV.
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

    ```

## Provenance

| Role | Model | Effort | Date | Code examined |
|---|---|---|---|---|
| Found | <span class="bb-nowrap">`gpt-6-astra`</span> | xhigh | <span class="bb-nowrap">2026-10-28</span> | 26.10 (`615921c6867c`), impactx-noacc wheel, CPU, double, serial |
| Confirmed | <span class="bb-nowrap">`gpt-6.1-sol`</span> | xhigh | <span class="bb-nowrap">2026-10-28</span> | 26.10 (`615921c6867c`), impactx-noacc wheel, CPU, double, serial |

