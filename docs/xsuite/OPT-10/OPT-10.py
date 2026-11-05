"""OPT-10: Sequential step clipping violates earlier variable bounds

`Line.match()` can accept an iteration that exceeds an earlier variable's `Vary.max_step`, because
clipping a later variable rescales the already clipped step vector and enlarges it. A common scale
factor must satisfy every bound at once and not depend on variable order. For dimensionless
variables a and b starting at 0, with targets 6 and 12 and maximum steps 0.2 and 0.8, the accepted
changes are 0.4000 and 0.8000 instead of the correctly scaled 0.2000 and 0.4000: a moves twice its
allowed amount, while reversing the variable order gives the correct result. It occurs when several
variables are limited at once, even with unit weights, and in an accepted iteration, not only in a
trial evaluation.

The script sets up `Line.match()` on an empty line with two unit-weight variables that a
tracking-free action reads back as targets. It compares the internal clipping of the requested step
(6, 12) and one accepted iteration with the common scale factor min(0.2/6, 0.8/12) = 1/30, which
gives changes 0.2 and 0.4. Both variable orders run; the order (b, a) is the control.
"""

from _matching import optimizer
from _prebuilt import identity, np

identity()
print("EXECUTION Python only; no tracking kernels invoked")
for order in (("a", "b"), ("b", "a")):
    line, opt = optimizer(
        {"a": 0.0, "b": 0.0}, {"a": 6.0, "b": 12.0}, {"a": 0.2, "b": 0.8}, order=order
    )
    # A valid common rescale is min_i(max_step_i / abs(step_i)) = 1/30.
    requested = np.array([{"a": 6.0, "b": 12.0}[n] for n in order])
    clipped = opt._err._clip_to_max_steps(requested)
    expected = requested / 30
    print("order", order, "direct_clip", clipped, "expected common-scale clip", expected)
    opt.step(1)
    actual = np.array([line["a"], line["b"]])
    print("order", order, "accepted a,b", actual, "allowed", [0.2, 0.8])
    if order == ("a", "b"):
        np.testing.assert_allclose(clipped, [0.4, 0.8], rtol=1e-15, atol=0)
        np.testing.assert_allclose(actual, [0.4, 0.8], rtol=0, atol=2e-12)
        assert actual[0] > 0.2 + 1e-12
    else:
        np.testing.assert_allclose(clipped, expected, rtol=1e-15, atol=0)
        np.testing.assert_allclose(actual, [0.2, 0.4], rtol=0, atol=2e-12)
print("DEFECT REPRODUCED")
