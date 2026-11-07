---
title: "TAO-14: call -no_calc saves flags without disabling lattice calculation"
hide:
  - toc
---

# TAO-14 — call -no_calc saves flags without disabling lattice calculation

| | |
|---|---|
| Code | [Bmad](../index.md) · Tao |
| Priority | P4 |
| Failure | Silent wrong result |
| Evidence | Probe-pinned |
| Last reproduced | 2026-11-05 on Bmad release [20261031-0](https://github.com/bmad-sim/bmad-ecosystem/tree/20261031-0) (`c4c629f`) |
| Source | [tao_call_cmd, tao/code/tao_call_cmd.f90:43–53](https://github.com/bmad-sim/bmad-ecosystem/blob/c4c629fc9071d3a56cb1ba5bc714baba65e9c5d3/tao/code/tao_call_cmd.f90#L43-L53)<br>[saved flags, tao/code/tao_call_cmd.f90:111–113](https://github.com/bmad-sim/bmad-ecosystem/blob/c4c629fc9071d3a56cb1ba5bc714baba65e9c5d3/tao/code/tao_call_cmd.f90#L111-L113)<br>[tao_close_command_file, tao/code/tao_close_command_file.f90:22–24](https://github.com/bmad-sim/bmad-ecosystem/blob/c4c629fc9071d3a56cb1ba5bc714baba65e9c5d3/tao/code/tao_close_command_file.f90#L22-L24)<br>[call, s:call, tao/doc/command-list.tex:161–164](https://github.com/bmad-sim/bmad-ecosystem/blob/c4c629fc9071d3a56cb1ba5bc714baba65e9c5d3/tao/doc/command-list.tex#L161-L164) |

Calling a Tao command file with -no_calc still recalculates the lattice after edits inside the file. The manual promises to disable calculation until the file finishes and then restore its initial state. After a drift changes from 1.7 m to 2.5 m, an interior orbit query already returns x = 0.05101 m instead of the deferred 0.03501 m. An analytic drift map and the final orbit query establish that the update happens too early.

Run `python TAO-14.py` with [_bmad.py](../helpers/_bmad.py) and [_tao_contracts_normal_form.py](../helpers/_tao_contracts_normal_form.py) in the same directory.

??? example "Reproducer: TAO-14.py"

    [Download TAO-14.py](TAO-14.py)

    ```python
    """TAO-14: call -no_calc saves flags without disabling lattice calculation

    Calling a Tao command file with -no_calc still recalculates the lattice after edits inside the file. The manual promises to disable calculation until the file finishes and then restore its initial state. After a drift changes from 1.7 m to 2.5 m, an interior orbit query already returns x = 0.05101 m instead of the deferred 0.03501 m. An analytic drift map and the final orbit query establish that the update happens too early.

    The script changes a drift length inside a called file and queries the orbit both inside and after the call. It compares the horizontal position with the exact field-free map at the old and new lengths.
    """

    import numpy as np
    from _bmad import compare, verdict
    from _tao_contracts_normal_form import LATTICE, scalar, tao

    result = tao(
        LATTICE + "particle_start[px] = 0.02\n",
        ["call -no_calc edits.tao", "pipe -write outside.txt ele:orbit E"],
        files={"edits.tao": "set element E l = 2.5\npipe -write inside.txt ele:orbit E\n"},
        outputs=["inside.txt", "outside.txt"],
    )
    # With normalized px=0.02 and delta=py=0, x(L)=x0+L*px/sqrt(1-px**2).
    expected = [0.001 + length * 0.02 / np.sqrt(1 - 0.02**2) for length in (1.7, 2.5)]
    inside, outside = [
        scalar(result.outputs[name].decode(), "x") for name in ("inside.txt", "outside.txt")
    ]
    assert not compare("final orbit x (m), updated length 2.5 m", outside, expected[1], atol=5e-16)
    verdict(compare("interior orbit x (m), deferred length 1.7 m", inside, expected[0], atol=5e-16))

    ```

## Provenance

| Role | Model | Effort | Date | Code examined |
|---|---|---|---|---|
| Found | `gpt-6-astra` | xhigh | 2026-10-20 | 20261020-0 (`19b8afb46a40`), source, production, CPU, double, serial |
| Confirmed | `gpt-6.1-sol` | xhigh | 2026-10-20 | 20261020-0 (`19b8afb46a40`), source, production, CPU, double, serial |

