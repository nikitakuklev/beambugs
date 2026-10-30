---
title: "API-01: MAD-X PC leaves the reference energy at its default"
hide:
  - toc
---

# API-01 — MAD-X `PC` leaves the reference energy at its default

| | |
|---|---|
| Code | [ImpactX](../index.md) · Python API |
| Priority | P1 |
| Severity | Silent wrong result |
| Evidence | Reproduced, independently rechecked |
| Last reproduced | 2026-10-29 on development (`fa0a43b9bb07`), source, CPU, double, serial |
| Issue | [[gh:https://github.com/BLAST-ImpactX/impactx/issues/1726]], filed from this evidence |
| Pull request | [[gh:https://github.com/BLAST-ImpactX/impactx/pull/1727]], filed from this evidence |
| Source | [`MADXParser.py#L2243-L2266`](https://github.com/BLAST-ImpactX/impactx/blob/b701a6e9d1f9e30cefb9a3e18e82ae7e02b9d943/src/python/impactx/MADXParser.py#L2243-L2266)<br>[`getEtot()`](https://github.com/BLAST-ImpactX/impactx/blob/b701a6e9d1f9e30cefb9a3e18e82ae7e02b9d943/src/python/impactx/MADXParser.py#L2664-L2666) |

Importing a MAD-X lattice whose `BEAM` command gives the momentum `PC` instead of `ENERGY`, for example through `RefPart.load_file()`, leaves the reference total energy at the parser's 1 GeV default; `GAMMA`, `BETA` and `BRHO` are also ignored. MAD-X derives the total energy from these parameters, as `sqrt((pc)² + (mc²)²)` for `PC`, and native MAD-X 5.09.03 does so for the same input. A proton `BEAM, PC=2` is initialized at 61.73 MeV kinetic energy instead of 1271 MeV, so the reference velocity and rigidity are also wrong. Only the importer's general preview warning is printed, with no indication that `PC` was not used.

## Reproducer

Expected total energy is sqrt((pc)^2 + (mc^2)^2). Equivalent ENERGY input checks the reference importer independently of the PC parameter.

??? example "API-01.py"

    [Download API-01.py](API-01.py)

    ```python
    """API-01: MAD-X PC input leaves the reference energy at its default.

    Expected total energy is sqrt((pc)^2 + (mc^2)^2). Equivalent ENERGY input
    checks the reference importer independently of the PC parameter.
    """

    import contextlib
    import io
    import math
    import os
    import warnings
    from collections.abc import Callable
    from decimal import Decimal, localcontext
    from pathlib import Path
    from tempfile import NamedTemporaryFile
    from typing import Any, cast

    for variable in (
        "NPROC",
        "PJRT_NPROC",
        "OMP_NUM_THREADS",
        "MKL_NUM_THREADS",
        "OPENBLAS_NUM_THREADS",
        "NUMEXPR_NUM_THREADS",
    ):
        os.environ[variable] = "1"

    import impactx.MADXParser as parser_module
    from impactx import RefPart


    def load(label, command):
        with (
            NamedTemporaryFile(mode="w", suffix=".madx", dir=Path(__file__).parent) as deck,
            warnings.catch_warnings(record=True) as captured,
            contextlib.redirect_stdout(io.StringIO()),
        ):
            deck.write(command + ";\n")
            deck.flush()
            warnings.simplefilter("always")
            parser = parser_module.MADXParser()
            parser.parse(deck.name)
            reference = RefPart()
            # The shipped stub misdeclares this Python extension's bound method.
            load_file = cast(Callable[[str], RefPart], reference.load_file)
            load_file(deck.name)
        print(
            f"{label}: kinetic_MeV={reference.kin_energy_MeV:.12f} "
            f"parser_pc_GeV={cast(Any, parser).context.beam.pc:.16g} "
            f"parser_total_GeV={parser.getEtot():.16g}"
        )
        print(f"  warning_categories={[type(w.message).__name__ for w in captured]}")
        return reference.kin_energy_MeV


    bad = False
    # Species defaults use these exact SI constants in the converter.
    with localcontext() as context:
        context.prec = 60
        rest = (
            Decimal("1.67262192369e-27")
            * Decimal(299792458) ** 2
            / Decimal("1.602176634e-19")
            / Decimal("1e6")
        )
        for pc in (Decimal(2000), Decimal(3250)):
            total = (pc * pc + rest * rest).sqrt()
            energy_case = load("ENERGY", f"BEAM, PARTICLE=PROTON, ENERGY={total / 1000}")
            momentum_case = load("PC", f"BEAM, PARTICLE=PROTON, PC={pc / 1000}")
            expected = float(total - rest)
            print(
                f"pc_MeV={pc}: expected_kinetic_MeV={expected:.12f} "
                f"observed={momentum_case:.12f} error={momentum_case - expected:.12f}"
            )
            assert math.isclose(energy_case, expected, rel_tol=0, abs_tol=2e-10)
            bad |= not math.isclose(momentum_case, expected, rel_tol=1e-12, abs_tol=2e-10)
    print("DEFECT REPRODUCED" if bad else "NOT REPRODUCED")

    ```

## Provenance

| Role | Model | Effort | Date | Code examined |
|---|---|---|---|---|
| Found | `gpt-6-astra` | xhigh | 2026-10-28 | 26.10 (`615921c6867c`), impactx-noacc wheel, CPU, double, serial |
| Confirmed | `gpt-6.1-sol` | xhigh | 2026-10-28 | 26.10 (`615921c6867c`), impactx-noacc wheel, CPU, double, serial |

