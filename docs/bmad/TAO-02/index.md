---
title: "TAO-02: Geodesic LM omits variable-penalty weights"
hide:
  - toc
---

# TAO-02 — Geodesic LM omits variable-penalty weights

| | |
|---|---|
| Code | [Bmad](../index.md) · Tao |
| Priority | P2 |
| Severity | Silent wrong result |
| Evidence | Reproduced, independently rechecked |
| Last reproduced | 2026-11-05 on Bmad 20261031-0 (`c4c629f`, head build) |
| Tags | <span class="bb-tag">partly-source-only</span> [?](../../methodology.md#tags) |
| Source | [`tao_geo_lm_func`, tao/code/tao_geodesic_lm_optimizer_mod.f90:148–163](https://github.com/bmad-sim/bmad-ecosystem/blob/c4c629fc9071d3a56cb1ba5bc714baba65e9c5d3/tao/code/tao_geodesic_lm_optimizer_mod.f90#L148-L163)<br>[`jacobian`, tao/code/tao_geodesic_lm_optimizer_mod.f90:204–231](https://github.com/bmad-sim/bmad-ecosystem/blob/c4c629fc9071d3a56cb1ba5bc714baba65e9c5d3/tao/code/tao_geodesic_lm_optimizer_mod.f90#L204-L231)<br>[“Optimization Overview”, Eq. `m1`, tao/doc/optimization.tex:50–65](https://github.com/bmad-sim/bmad-ecosystem/blob/c4c629fc9071d3a56cb1ba5bc714baba65e9c5d3/tao/doc/optimization.tex#L50-L65) |

Geodesic Levenberg–Marquardt optimization ignores the weights of variable penalties while applying the weights of data constraints. For a quadrupole strength y with objective y² + 100(y − 0.1)², it returns y = 0.05000 and merit 0.2525 instead of y = 0.09901 and merit 0.009901. At y = −0.2000 m⁻², its residual norm squared is 0.1300 while Tao reports merit 9.040. SVD reaches the analytic weighted optimum on the same one-variable problem.

Run `python TAO-02.py` with [_bmad.py](../helpers/_bmad.py) and [_tao_optimization_io.py](../helpers/_tao_optimization_io.py) in the same directory.

??? example "Reproducer: TAO-02.py"

    [Download TAO-02.py](TAO-02.py)

    ```python
    """TAO-02: Geodesic LM omits variable-penalty weights

    Geodesic Levenberg–Marquardt optimization ignores the weights of variable penalties while
    applying the weights of data constraints. For a quadrupole strength y with objective y² + 100(y
    − 0.1)², it returns y = 0.05000 and merit 0.2525 instead of y = 0.09901 and merit 0.009901. At
    y = −0.2000 m⁻², its residual norm squared is 0.1300 while Tao reports merit 9.040. SVD
    reaches the analytic weighted optimum on the same one-variable problem.

    The script compiles a probe of the geodesic residual vector and compares it with the weighted
    objective. Separate Tao processes run geodesic LM and an SVD control against the analytic minimum.
    """

    import re

    import numpy as np
    from _bmad import compare, fortran, verdict
    from _tao_optimization_io import LINE, WEIGHT_INIT, block, control, diagnostics, numbers, tao

    PROGRAM = """program review_geo_probe
    use tao_geodesic_lm_optimizer_mod, only: tao_geo_lm_func
    use tao_interface
    implicit none
    real(rp), allocatable :: a(:), y(:)
    real(rp) merit
    call tao_top_level(command = '-init case.init -noplot -nostartup')
    call tao_get_opt_vars(var_value = a)
    allocate(y(2))
    call tao_geo_lm_func(2, size(a), a, y)
    merit = tao_merit()
    write(*,'(a,2es25.16)') 'Geodesic residual vector: ', y
    write(*,'(a,es25.16)') 'Squared residual norm:    ', sum(y**2)
    write(*,'(a,es25.16)') 'Tao merit:               ', merit
    write(*,'(a,es25.16)') 'Expected penalty row:    ', 10.0_rp * (a(1) - 0.1_rp)
    write(*,'(a)') 'PROBE COMPLETE'
    end program
    """
    result = fortran(PROGRAM, tao=True, files={"case.init": WEIGHT_INIT, "line.bmad": LINE})
    if diagnostics(result.log) or "PROBE COMPLETE" not in result.log:
        raise RuntimeError(result.log)


    def row(label):
        match = re.search(rf"^{re.escape(label)}\s*(.*)$", result.log, re.MULTILINE)
        if match is None:
            raise RuntimeError(result.log)
        return np.array(match[1].split(), dtype=float)


    control("Tao initial merit, .2^2 + 100*(-.2-.1)^2", row("Tao merit:"), [9.04], atol=1e-13)
    control(
        "weighted variable residual, sqrt(100)*(-.2-.1)",
        row("Expected penalty row:"),
        [-3.0],
        atol=1e-14,
    )
    defect = compare(
        "geodesic residual vector", row("Geodesic residual vector:"), [-3.0, -0.2], atol=1e-14
    )
    defect |= compare(
        "geodesic squared norm versus Tao merit",
        row("Squared residual norm:"),
        row("Tao merit:"),
        atol=1e-13,
    )
    # d/dy [y^2 + 100*(y-.1)^2] = 202*y - 20.
    optimum = 10 / 101
    minimum = 1 / 101
    for optimizer in ("geodesic_lm", "svd"):
        commands = [
            f"set global optimizer = {optimizer}",
            "run",
            "pipe lat_list q2 ele.k1",
            "pipe merit",
        ]
        result = tao(commands, lattice=LINE, init=WEIGHT_INIT)
        got = numbers(block(result.log, commands[-2])).item()
        merit = float(block(result.log, commands[-1]))
        control(f"{optimizer} reported merit", merit, got**2 + 100 * (got - 0.1) ** 2, atol=2e-12)
        if optimizer == "svd":
            control("SVD weighted optimum k1 (m^-2)", got, optimum, atol=1e-11)
            control("SVD minimum merit", merit, minimum, atol=1e-13)
        else:
            defect |= compare("geodesic fitted k1 (m^-2)", got, optimum, atol=1e-11)
            defect |= compare("geodesic minimum merit", merit, minimum, atol=1e-13)
    verdict(defect)

    ```

## Provenance

| Role | Model | Effort | Date | Code examined |
|---|---|---|---|---|
| Found | `gpt-6.1-sol` | xhigh | 2026-10-21 | 20261020-0 (`19b8afb46a40`), source, production, CPU, double, serial |
| Confirmed | `gpt-6-astra` | xhigh | 2026-10-21 | 20261020-0 (`19b8afb46a40`), source, production, CPU, double, serial |

