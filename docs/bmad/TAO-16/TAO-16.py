"""TAO-16: Startup and batch-command failures exit with process status zero

Starting Tao with an invalid command-line option or a missing explicit initialization file produces error diagnostics, executes none of the requested commands, and returns status zero. Invalid pipe and set commands also return zero after reporting errors. An unknown top-level command aborts its command file and returns the same success status. Automation needs an unsuccessful exit status to distinguish these failures from a completed run.

The script runs a successful completion-marker control, two startup failures, and four command failures in separate processes. It checks each specific diagnostic and whether the command file reaches its completion marker, then compares the exit status with the required batch failure status.
"""

from _bmad import RunFailure, verdict
from _tao_contracts_normal_form import ERROR, LATTICE, evaluated, tao

control = tao(LATTICE, ["pipe evaluate 314159"])
assert evaluated(control.log, "314159") == 314159
print("successful command file: status=0, completion=True; expected success")
cases = [
    (["-not_a_tao_option"], [], "BAD COMMAND LINE ARGUMENT", False),
    (["-init_file", "missing.init"], [], "TAO INITIALIZATION FILE NOT FOUND", False),
    (None, ["pipe no_such_subcommand"], "is not recognized", True),
    (None, ["pipe ele:mat6 1"], "INVALID", True),
    (None, ["set bmad_com no_such_parameter = T"], "BAD COMPONENT OR NUMBER", True),
    (None, ["this_is_not_a_command"], "ABORTING COMMAND FILE", False),
]
bad = False
for options, commands, diagnostic, continues in cases:
    try:
        result = tao(LATTICE, [*commands, "pipe evaluate 314159"], options=options, check=False)
        status, log = 0, result.log
    except RunFailure as exc:
        status, log = exc.status, exc.log
    assert diagnostic in log and ERROR.search(log), log
    completed = "Tao: pipe evaluate 314159\n" in log
    if status == 0:
        assert completed == continues, log
        if completed:
            assert evaluated(log, "314159") == 314159
    print(
        f"{diagnostic}: status={status}, completion={completed}; expected nonzero batch error status"
    )
    bad |= status == 0
verdict(bad)
