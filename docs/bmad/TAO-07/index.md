---
title: "TAO-07: Four a-mode chromatic pipe selectors read b-mode fields"
hide:
  - toc
---

# TAO-07 — Four a-mode chromatic pipe selectors read b-mode fields

| | |
|---|---|
| Code | [Bmad](../index.md) · Tao |
| Priority | P2 |
| Severity | Silent wrong result |
| Evidence | Reproduced, independently rechecked |
| Last reproduced | 2026-11-05 on Bmad release [20261031-0](https://github.com/bmad-sim/bmad-ecosystem/tree/20261031-0) (`c4c629f`) |
| Tags | <span class="bb-tag">partly-source-only</span> [?](../../methodology.md#tags) |
| Source | [`ele_param_value`, tao/code/tao_pipe_cmd.f90:9317–9324](https://github.com/bmad-sim/bmad-ecosystem/blob/c4c629fc9071d3a56cb1ba5bc714baba65e9c5d3/tao/code/tao_pipe_cmd.f90#L9317-L9324)<br>[Tao manual, “pipe ele:param” (`p:ele:param`), tao/doc/pipe-interface-commands.tex:974–984](https://github.com/bmad-sim/bmad-ecosystem/blob/c4c629fc9071d3a56cb1ba5bc714baba65e9c5d3/tao/doc/pipe-interface-commands.tex#L974-L984) |

The a-mode chromatic derivative selectors in Tao’s pipe interface return b-mode values. With a/b beta derivatives set to 11/22 and alpha derivatives set to 33/44, lat_list reports [22, 44] for the a mode, and ele:param returns 22 for its beta derivative. The ele:twiss response correctly returns the a-mode values 11 and 33. The four a-mode beta, alpha, dispersion, and dispersion-slope selectors must read their matching a-mode fields.

Run `python TAO-07.py` with [_bmad.py](../helpers/_bmad.py) and [_tao_optimization_io.py](../helpers/_tao_optimization_io.py) in the same directory.

??? example "Reproducer: TAO-07.py"

    [Download TAO-07.py](TAO-07.py)

    ```python
    """TAO-07: Four a-mode chromatic pipe selectors read b-mode fields

    The a-mode chromatic derivative selectors in Tao’s pipe interface return b-mode values. With a/b
    beta derivatives set to 11/22 and alpha derivatives set to 33/44, lat_list reports [22, 44] for
    the a mode, and ele:param returns 22 for its beta derivative. The ele:twiss response correctly
    returns the a-mode values 11 and 33. The four a-mode beta, alpha, dispersion, and
    dispersion-slope selectors must read their matching a-mode fields.

    The script initializes distinct a- and b-mode beta and alpha derivatives, then queries lat_list,
    ele:param, and ele:twiss. It uses ele:twiss and the lattice initial conditions as controls.
    """

    from _bmad import compare, verdict
    from _tao_optimization_io import ABS_INIT, LINE, block, control, numbers, parameters, tao

    commands = [
        "pipe lat_list beginning ele.a.dbeta_dpz,ele.a.dalpha_dpz",
        "pipe ele:param beginning ele.a.dbeta_dpz",
        "pipe ele:twiss beginning",
    ]
    result = tao(commands, lattice=LINE, init=ABS_INIT)
    twiss = parameters(block(result.log, commands[2]))
    control(
        "ele:twiss a-mode beta/alpha derivatives",
        [float(twiss["dbeta_dpz_a"]), float(twiss["dalpha_dpz_a"])],
        [11, 33],
        atol=5e-12,
    )
    defect = compare(
        "lat_list a-mode beta/alpha derivatives",
        numbers(block(result.log, commands[0])).ravel(),
        [11, 33],
        atol=5e-12,
    )
    param = parameters(block(result.log, commands[1]))
    defect |= compare(
        "ele:param a-mode beta derivative", float(param["ele.a.dbeta_dpz"]), 11, atol=5e-12
    )
    verdict(defect)

    ```

## Provenance

| Role | Model | Effort | Date | Code examined |
|---|---|---|---|---|
| Found | `gpt-6.1-sol` | xhigh | 2026-10-21 | 20261020-0 (`19b8afb46a40`), source, production, CPU, double, serial |
| Confirmed | `gpt-6-astra` | xhigh | 2026-10-21 | 20261020-0 (`19b8afb46a40`), source, production, CPU, double, serial |

