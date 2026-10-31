"""Tracking-free matching setup shared by the step-limit and knob reproducers.

`ReadKnobs` is an action that returns the current values of named line variables, so targets can be
set on the variables themselves. `optimizer` builds a public `Line.match()` (not yet solved) on an
empty line without a tracking kernel, with given starting values, goals, maximum steps, optional
weights and variable order.
"""

import xdeps as xd
from _prebuilt import xt


class ReadKnobs(xd.Action):
    def __init__(self, line, names):
        self.line = line
        self.names = names

    def run(self):
        return {name: self.line[name] for name in self.names}


def optimizer(initial, goals, maxima, weights=None, order=None):
    line = xt.Line(elements=[])
    line.build_tracker(compile=False)  # Actions read variables; no tracking kernel is needed.
    for name, value in initial.items():
        line[name] = value
    names = tuple(initial)
    action = ReadKnobs(line, names)
    opt = line.match(
        solve=False,
        verbose=False,
        show_call_counter=False,
        vary=[
            xt.Vary(
                name,
                step=1e-4,
                max_step=maxima[name],
                weight=1.0 if weights is None else weights[name],
            )
            for name in (names if order is None else order)
        ],
        targets=[xt.Target(name, goals[name], tol=1e-11, action=action) for name in names],
    )
    return line, opt
