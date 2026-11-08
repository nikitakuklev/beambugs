"""API-11: `orbitFinalize()` with its default message raises `TypeError`

`orbit.utils.orbitFinalize(message=None)` concatenates the message to a string on rank 0, so calling it with the default argument, with `None` or with a non-string raises `TypeError` before it prints `STOP` and raises `SystemExit(1)`. A caller that relies on the helper to stop the run, or that catches `SystemExit` to handle a stop, sees an unrelated exception instead. With a string message the helper prints it and stops with `SystemExit(1)`.

The script calls `orbitFinalize()`, `orbitFinalize(None)` and `orbitFinalize(5)`, catches whatever they raise, and records whether `STOP` was printed. The control `orbitFinalize("msg")` must print `STOP` and raise `SystemExit(1)`.
"""

import contextlib
import io
import os

for _v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ.setdefault(_v, "1")

from orbit.utils import orbitFinalize


def call(*args):
    out = io.StringIO()
    with contextlib.redirect_stdout(out), contextlib.redirect_stderr(io.StringIO()):
        try:
            orbitFinalize(*args)
            exc = "nothing raised"
        except (SystemExit, TypeError) as e:  # SystemExit is the helper's stop
            exc = f"{type(e).__name__}({e})"
    return exc, "STOP" in out.getvalue()


# The helper's body: print the message on rank 0, print STOP, sys.exit(1).
reproduced = False
for label, args in (
    ("orbitFinalize()", ()),
    ("orbitFinalize(None)", (None,)),
    ("orbitFinalize(5)", (5,)),
):
    exc, stop = call(*args)
    print(
        f"{label}: observed {exc}, STOP printed {stop}; expected SystemExit(1), STOP printed True"
    )
    reproduced |= not exc.startswith("SystemExit")
exc, stop = call("msg")
print(
    f"control orbitFinalize('msg'): observed {exc}, STOP printed {stop}; expected SystemExit(1), STOP printed True"
)

print("DEFECT REPRODUCED" if reproduced else "NOT REPRODUCED")
