"""OPT-11: Variable weights distort physical maximum steps

Giving a matching variable a `weight` other than one changes its effective `max_step` in the
variable's own units. The solver works in weighted coordinates, so converting back to the variable
must convert the maximum step too, leaving the accepted physical bound independent of weight. For a
dimensionless variable starting at 3, targeting 20 and limited to `max_step` = 0.2, one accepted
`Line.match()` iteration changes it by 2.000 with weight 10 and by 0.02000 with weight 0.1, instead
of 0.2000. Weighted variables can therefore exceed the requested bound or be needlessly restricted;
unit weight is correct, and a single variable suffices to trigger the error.

The script takes one accepted `Line.match()` iteration on an empty line with a single variable, read
back as the target by a tracking-free action, for weights w = 0.1, 1 and 10. Solver coordinates x
relate to the variable k by k = w·x, so the bound in solver units must be `max_step`/w; the script
checks that conversion and shows that the physical change is w × 0.2 instead. Unit weight is the
control.
"""

from _matching import optimizer
from _prebuilt import identity, np

identity()
print("EXECUTION Python only; no tracking kernels invoked")
for weight in (0.1, 1.0, 10.0):
    line, opt = optimizer({"k": 3.0}, {"k": 20.0}, {"k": 0.2}, weights={"k": weight})
    opt.step(1)
    physical_change = line["k"] - 3.0
    print(
        "weight",
        weight,
        "physical_change",
        f"{physical_change:.17g}",
        "max_step",
        0.2,
        "solver_change",
        opt.solver._last_xstep,
    )
    np.testing.assert_allclose(physical_change, weight * 0.2, rtol=0, atol=1e-14)
    if weight > 1:
        assert physical_change > 0.2 + 1e-12
    # Independent unit conversion: k=w*x implies |dx| <= max_step/w.
    np.testing.assert_allclose(opt._err._x_to_knobs([0.2 / weight]), [0.2], rtol=1e-15, atol=0)
print("DEFECT REPRODUCED")
