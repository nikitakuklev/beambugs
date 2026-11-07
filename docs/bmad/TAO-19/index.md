---
title: "TAO-19: ele:param tags matrix and vector arrays as scalar REAL"
hide:
  - toc
---

# TAO-19 — ele:param tags matrix and vector arrays as scalar REAL

| | |
|---|---|
| Code | [Bmad](../index.md) · Tao |
| Priority | P4 |
| Severity | Crash or API hazard |
| Evidence | Reproduced, independently rechecked |
| Last reproduced | 2026-11-05 on Bmad 20261031-0 (`c4c629f`, head build) |
| Source | [pipe type definitions, tao/code/tao_pipe_cmd.f90:15–21](https://github.com/bmad-sim/bmad-ecosystem/blob/c4c629fc9071d3a56cb1ba5bc714baba65e9c5d3/tao/code/tao_pipe_cmd.f90#L15-L21)<br>[ele:param dispatch and formatting, tao/code/tao_pipe_cmd.f90:3621–3644](https://github.com/bmad-sim/bmad-ecosystem/blob/c4c629fc9071d3a56cb1ba5bc714baba65e9c5d3/tao/code/tao_pipe_cmd.f90#L3621-L3644)<br>[re_str, tao/code/tao_pipe_cmd.f90:9036–9047](https://github.com/bmad-sim/bmad-ecosystem/blob/c4c629fc9071d3a56cb1ba5bc714baba65e9c5d3/tao/code/tao_pipe_cmd.f90#L9036-L9047) |

Tao’s pipe ele:param command labels matrix and vector payloads with the scalar REAL tag. The ele.mat6, ele.vec0, and ele.c_mat selectors return 36, 6, and 4 values respectively, while the protocol defines REAL_ARR for real arrays. A decoder that consumes one value for the declared scalar type discards the remaining payload. An analytic drift verifies the payload and ordering, and a scalar length query correctly returns REAL with one value.

Run `python TAO-19.py` with [_bmad.py](../helpers/_bmad.py) and [_tao_contracts_normal_form.py](../helpers/_tao_contracts_normal_form.py) in the same directory.

??? example "Reproducer: TAO-19.py"

    [Download TAO-19.py](TAO-19.py)

    ```python
    """TAO-19: ele:param tags matrix and vector arrays as scalar REAL

    Tao’s pipe ele:param command labels matrix and vector payloads with the scalar REAL tag. The ele.mat6, ele.vec0, and ele.c_mat selectors return 36, 6, and 4 values respectively, while the protocol defines REAL_ARR for real arrays. A decoder that consumes one value for the declared scalar type discards the remaining payload. An analytic drift verifies the payload and ordering, and a scalar length query correctly returns REAL with one value.

    The script queries the three array selectors and a scalar length in a drift lattice. It checks the matrix and vector values independently, then compares each type tag with the protocol’s scalar/array distinction.
    """

    import numpy as np
    from _bmad import compare, verdict
    from _tao_contracts_normal_form import LATTICE, block, record, tao

    # Zero starting transverse momentum makes the drift's matrix analytic.
    result = tao(LATTICE, [f"pipe ele:param E ele.{key}" for key in ("mat6", "vec0", "c_mat", "l")])
    matrix = np.eye(6)
    matrix[0, 1] = matrix[2, 3] = 1.7
    matrix[4, 5] = 1.7 * 510998.95069**2 / (1e12 + 510998.95069**2)
    bad = False
    for name, expected in (("mat6", matrix.ravel()), ("vec0", np.zeros(6)), ("c_mat", np.zeros(4))):
        row = record(block(result.log, f"pipe ele:param E ele.{name}"), "ele." + name)
        values = np.array(row[3:], dtype=float)
        # ele:param renders its payload with re_str(value,8).
        assert not compare(f"{name} payload control", values, expected, atol=5e-8)
        print(
            f"ele.{name}: type={row[1]}, values={len(values)}; expected REAL_ARR by pipe type contract"
        )
        bad |= row[1] != "REAL_ARR"
    row = record(block(result.log, "pipe ele:param E ele.l"), "ele.l")
    assert row[1] == "REAL" and len(row) == 4 and float(row[3]) == 1.7
    print("scalar length control: REAL, one value; expected REAL")
    verdict(bad)

    ```

## Provenance

| Role | Model | Effort | Date | Code examined |
|---|---|---|---|---|
| Found | `gpt-6-astra` | xhigh | 2026-10-20 | 20261020-0 (`19b8afb46a40`), source, production, CPU, double, serial |
| Confirmed | `gpt-6-astra` | xhigh | 2026-10-21 | 20261020-0 (`19b8afb46a40`), source, production, CPU, double, serial |
| Confirmed | `gpt-6.1-sol` | xhigh | 2026-11-05 | 20261031-0 (`c4c629fc9071`), source, production, GNU Fortran 15.3.0, CPU, double, serial, OpenMP=1 |

