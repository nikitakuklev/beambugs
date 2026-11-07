---
title: "TAO-01: Filtered variable penalties differentiate the wrong variable"
hide:
  - toc
---

# TAO-01 — Filtered variable penalties differentiate the wrong variable

| | |
|---|---|
| Code | [Bmad](../index.md) · Tao |
| Priority | P2 |
| Failure | Silent wrong result |
| Evidence | Reproduced, independently rechecked |
| Last reproduced | 2026-11-05 on Bmad release [20261031-0](https://github.com/bmad-sim/bmad-ecosystem/tree/20261031-0) (`c4c629f`) |
| Tags | <span class="bb-tag">partly-source-only</span> [?](../../methodology.md#tags) |
| Source | [`tao_svd_func`, tao/code/tao_svd_optimizer_mod.f90:172–177](https://github.com/bmad-sim/bmad-ecosystem/blob/c4c629fc9071d3a56cb1ba5bc714baba65e9c5d3/tao/code/tao_svd_optimizer_mod.f90#L172-L177)<br>[`tao_mrq_func`, tao/code/tao_lm_optimizer_mod.f90:213–217](https://github.com/bmad-sim/bmad-ecosystem/blob/c4c629fc9071d3a56cb1ba5bc714baba65e9c5d3/tao/code/tao_lm_optimizer_mod.f90#L213-L217)<br>[geodesic `jacobian`, tao/code/tao_geodesic_lm_optimizer_mod.f90:204–210](https://github.com/bmad-sim/bmad-ecosystem/blob/c4c629fc9071d3a56cb1ba5bc714baba65e9c5d3/tao/code/tao_geodesic_lm_optimizer_mod.f90#L204-L210)<br>[`tao_get_opt_vars`, tao/code/tao_get_opt_vars.f90:44–72](https://github.com/bmad-sim/bmad-ecosystem/blob/c4c629fc9071d3a56cb1ba5bc714baba65e9c5d3/tao/code/tao_get_opt_vars.f90#L44-L72)<br>[“Optimization Overview”, Eq. `m1`, tao/doc/optimization.tex:50–65](https://github.com/bmad-sim/bmad-ecosystem/blob/c4c629fc9071d3a56cb1ba5bc714baba65e9c5d3/tao/doc/optimization.tex#L50-L65) |

When a zero-weight optimization variable precedes a variable with a nonzero penalty, Tao assigns the penalty derivative to the wrong variable. The SVD and Levenberg–Marquardt penalty row is [1, 0] where central differences give [0, 1]. For quadrupole strengths x and y with objective x² + y² + (y − 0.1)², one SVD cycle returns (0.3000, 0) with merit 0.1000 instead of the exact minimum at (0, 0.05000) with merit 0.005000. Reordering the same variables so the penalized one comes first gives the correct derivatives and minimum.

Run `python TAO-01.py` with [_bmad.py](../helpers/_bmad.py) and [_tao_optimization_io.py](../helpers/_tao_optimization_io.py) in the same directory.

??? example "Reproducer: TAO-01.py"

    [Download TAO-01.py](TAO-01.py)

    ```python
    """TAO-01: Filtered variable penalties differentiate the wrong variable

    When a zero-weight optimization variable precedes a variable with a nonzero penalty, Tao assigns
    the penalty derivative to the wrong variable. The SVD and Levenberg–Marquardt penalty row is
    [1, 0] where central differences give [0, 1]. For quadrupole strengths x and y with objective
    x² + y² + (y − 0.1)², one SVD cycle returns (0.3000, 0) with merit 0.1000 instead of the exact
    minimum at (0, 0.05000) with merit 0.005000. Reordering the same variables so the penalized one
    comes first gives the correct derivatives and minimum.

    The script compares both optimizer Jacobians with central differences, then compares one SVD cycle
    with the analytic least-squares solution. Reordering the same variables is a control for both the
    derivatives and the fit.
    """

    import re

    import numpy as np
    from _bmad import compare, fortran, verdict
    from _tao_optimization_io import LINE, LINEAR_INIT, block, control, diagnostics, numbers, tao

    PROGRAM = """program review_jacobian_probe
    use tao_svd_optimizer_mod
    use tao_lm_optimizer_mod, only: tao_mrq_func
    implicit none
    real(rp), allocatable :: a(:), yp(:), ym(:), yy(:), jac(:,:), work_jac(:,:)
    real(rp) h
    integer status, j
    logical err

    call tao_top_level(command = '-init case.init -noplot -nostartup')
    call tao_dmodel_dvar_calc(.true., err)
    if (err) stop 1
    call tao_get_opt_vars(var_value = a)
    if (size(a) /= 2) stop 2
    allocate(yp(3), ym(3), yy(3), jac(3,size(a)), work_jac(3,size(a)))
    h = 1e-6_rp
    call tao_svd_func(a, yy, jac, status)
    if (status /= 0) stop 3
    write(*,'(a,2es25.16)') 'SVD penalty Jacobian: ', jac(1,:)
    call tao_mrq_func(a, yy, jac, status)
    if (status /= 0) stop 4
    write(*,'(a,2es25.16)') 'LM penalty Jacobian:  ', jac(1,:)
    do j = 1,size(a)
      a(j) = a(j) + h
      call tao_svd_func(a, yp, work_jac, status)
      if (status /= 0) stop 5
      a(j) = a(j) - 2*h
      call tao_svd_func(a, ym, work_jac, status)
      if (status /= 0) stop 6
      a(j) = a(j) + h
      write(*,'(a,i0,a,es25.16)') 'Finite difference, column ', j, ': ', (yp(1)-ym(1))/(2*h)
    enddo
    call tao_set_opt_vars(a)
    write(*,'(a)') 'PROBE COMPLETE'
    end program
    """
    result = fortran(PROGRAM, tao=True, files={"case.init": LINEAR_INIT, "line.bmad": LINE})
    if diagnostics(result.log) or "PROBE COMPLETE" not in result.log:
        raise RuntimeError(result.log)


    def row(label):
        match = re.search(rf"^{re.escape(label)}\s*(.*)$", result.log, re.MULTILINE)
        if match is None:
            raise RuntimeError(result.log)
        return np.array(match[1].split(), dtype=float)


    fd = np.concatenate([row(f"Finite difference, column {i}:") for i in (1, 2)])
    control("penalty central differences, derivative of y-0.1", fd, [0, 1], atol=5e-11)
    defect = compare("SVD penalty Jacobian", row("SVD penalty Jacobian:"), fd, atol=5e-11)
    defect |= compare("LM penalty Jacobian", row("LM penalty Jacobian:"), fd, atol=5e-11)
    commands = ["run", "pipe lat_list q* ele.k1", "pipe merit"]
    result = tao(commands, lattice=LINE, init=LINEAR_INIT)
    observed = numbers(block(result.log, commands[1])).ravel()
    # d/dx = 2x = 0; d/dy = 2y + 2(y-.1) = 0.
    expected = np.array([0.0, 0.05])
    defect |= compare("SVD fitted k1 (m^-2), analytic stationary point", observed, expected, atol=1e-10)
    merit = float(block(result.log, commands[2]))
    control(
        "reported merit, evaluated objective",
        merit,
        observed @ observed + (observed[1] - 0.1) ** 2,
        atol=1e-12,
    )
    defect |= compare("SVD minimum merit", merit, 0.005, atol=1e-12)

    # Only reorder the variable definitions: the lattice, data, weights and objective
    # stay identical. The penalty now occupies the first active-variable column.
    reordered_init = (
        LINEAR_INIT[: LINEAR_INIT.index("&tao_var")]
        + """&tao_var
      v1_var%name = 'v'
      default_step = 1e-5
      default_universe = '1'
      default_attribute = 'k1'
      ix_min_var = 1
      ix_max_var = 2
      var(1)%ele_name = 'q2'
      var(1)%merit_type = 'target'
      var(1)%meas = 0.1
      var(1)%weight = 1
      var(2)%ele_name = 'q1'
      var(2)%weight = 0
    /
    """
    )
    result = fortran(PROGRAM, tao=True, files={"case.init": reordered_init, "line.bmad": LINE})
    if diagnostics(result.log) or "PROBE COMPLETE" not in result.log:
        raise RuntimeError(result.log)
    control_fd = np.concatenate([row(f"Finite difference, column {i}:") for i in (1, 2)])
    control("reordered central differences, derivative in (y,x) order", control_fd, [1, 0], atol=5e-11)
    for optimizer in ("SVD", "LM"):
        control(
            f"reordered {optimizer} penalty Jacobian",
            row(f"{optimizer} penalty Jacobian:"),
            control_fd,
            atol=5e-11,
        )
    result = tao(commands, lattice=LINE, init=reordered_init)
    control(
        "reordered SVD fitted k1 (m^-2), same objective minimum",
        numbers(block(result.log, commands[1])).ravel(),
        expected,
        atol=1e-10,
    )
    control("reordered SVD minimum merit", float(block(result.log, commands[2])), 0.005, atol=1e-12)
    verdict(defect)

    ```

## Provenance

| Role | Model | Effort | Date | Code examined |
|---|---|---|---|---|
| Found | `gpt-6.1-sol` | xhigh | 2026-10-21 | 20261020-0 (`19b8afb46a40`), source, production, CPU, double, serial |
| Confirmed | `gpt-6-astra` | xhigh | 2026-10-21 | 20261020-0 (`19b8afb46a40`), source, production, CPU, double, serial |

