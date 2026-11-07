---
title: "TAO-20: Documented normal.dhdj datum dereferences an unpopulated Taylor series"
hide:
  - toc
---

# TAO-20 — Documented normal.dhdj datum dereferences an unpopulated Taylor series

| | |
|---|---|
| Code | [Bmad](../index.md) · Tao |
| Priority | P3 |
| Failure | Crash or API hazard |
| Evidence | Probe-pinned |
| Last reproduced | 2026-11-05 on Bmad release [20261031-0](https://github.com/bmad-sim/bmad-ecosystem/tree/20261031-0) (`c4c629f`) |
| Source | [tao_ptc_normal_form, tao/code/tao_ptc_normal_form.f90:69–74](https://github.com/bmad-sim/bmad-ecosystem/blob/c4c629fc9071d3a56cb1ba5bc714baba65e9c5d3/tao/code/tao_ptc_normal_form.f90#L69-L74)<br>[normal.dhdj selection, tao/code/tao_evaluate_a_datum.f90:1859–1864](https://github.com/bmad-sim/bmad-ecosystem/blob/c4c629fc9071d3a56cb1ba5bc714baba65e9c5d3/tao/code/tao_evaluate_a_datum.f90#L1859-L1864)<br>[coefficient dispatch, tao/code/tao_evaluate_a_datum.f90:1902–1910](https://github.com/bmad-sim/bmad-ecosystem/blob/c4c629fc9071d3a56cb1ba5bc714baba65e9c5d3/tao/code/tao_evaluate_a_datum.f90#L1902-L1910)<br>[taylor_coef, bmad/modules/taylor_mod.f90:95–110](https://github.com/bmad-sim/bmad-ecosystem/blob/c4c629fc9071d3a56cb1ba5bc714baba65e9c5d3/bmad/modules/taylor_mod.f90#L95-L110)<br>[Data Types, normal.<type>, normal.type, tao/doc/data.tex:1446–1462](https://github.com/bmad-sim/bmad-ecosystem/blob/c4c629fc9071d3a56cb1ba5bc714baba65e9c5d3/tao/doc/data.tex#L1446-L1462) |

Requesting the documented normal.dhdj.1.000001 coefficient in Tao crashes with a segmentation fault on a stable alternating-quadrupole ring after its one-turn map has been calculated. The routine that would populate this Taylor series is commented out, but the datum path still asks for a coefficient from it. The query should return the coefficient or an explicit unavailable result.

Run `python TAO-20.py` with [_bmad.py](../helpers/_bmad.py) and [_tao_contracts_normal_form.py](../helpers/_tao_contracts_normal_form.py) in the same directory.

??? example "Reproducer: TAO-20.py"

    [Download TAO-20.py](TAO-20.py)

    ```python
    """TAO-20: Documented normal.dhdj datum dereferences an unpopulated Taylor series

    Requesting the documented normal.dhdj.1.000001 coefficient in Tao crashes with a segmentation fault on a stable alternating-quadrupole ring after its one-turn map has been calculated. The routine that would populate this Taylor series is commented out, but the datum path still asks for a coefficient from it. The query should return the coefficient or an explicit unavailable result.

    The script initializes a stable FODO ring, enables its one-turn map, and first verifies a successful completion-marker control. A fresh process then makes the documented query; only SIGSEGV immediately after that query, with clean preceding setup, counts as the reproduced crash.
    """

    import signal

    from _bmad import RunFailure, verdict
    from _tao_contracts_normal_form import ERROR, FODO, evaluated, tao

    base = FODO.replace(
        "fringe_type=none",
        "fringe_type=none, ptc_integration_type=drift_kick, num_steps=32, integrator_order=6",
    )
    commands = ["set universe 1 one_turn_map_calc = T", "pipe evaluate 314159"]
    control = tao(base, commands)
    assert evaluated(control.log, "314159") == 314159
    expression = "lat::normal.dhdj.1.000001"
    try:
        result = tao(base, [*commands, "pipe evaluate " + expression], check=False)
    except RunFailure as exc:
        before, marker, after = exc.log.partition("Tao: pipe evaluate " + expression + "\n")
        assert (
            exc.status == -signal.SIGSEGV
            and marker
            and after.startswith("\nProgram received signal SIGSEGV")
        ), exc.log
        assert not ERROR.search(before), before
        assert evaluated(before, "314159") == 314159
        print(
            "normal.dhdj.1.000001: SIGSEGV immediately after query with successful normal-form setup; expected coefficient or explicit unavailable result"
        )
        verdict(True)
    else:
        assert "Tao: quit\n" in result.log, result.log
        print(
            "normal.dhdj.1.000001: command completed without SIGSEGV; expected coefficient or explicit unavailable result"
        )
        verdict(False)

    ```

## Provenance

| Role | Model | Effort | Date | Code examined |
|---|---|---|---|---|
| Found | `gpt-6-astra` | xhigh | 2026-10-20 | 20261020-0 (`19b8afb46a40`), source, production, CPU, double, serial |
| Confirmed | `gpt-6.1-sol` | xhigh | 2026-10-20 | 20261020-0 (`19b8afb46a40`), source, production, CPU, double, serial |

