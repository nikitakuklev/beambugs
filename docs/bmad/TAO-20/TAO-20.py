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
