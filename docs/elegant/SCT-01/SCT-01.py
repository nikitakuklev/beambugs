"""SCT-01: Pressure-region diagnostic reads an invalid sample index

With IONEFFECTS' default pressure averaging, a lattice interval lying entirely before or after the
gas-pressure samples triggers an error diagnostic that reads position index −1. Evaluating the
message reads 8 bytes before the position array instead of safely reporting the coverage problem.
With samples at 2 m and 3 m and an interval from 0 to 1 m, the run stops with a value that is not a
sample position in the message, and a memory checker confirms an invalid read of size 8. Users whose
pressure profiles miss an element's averaging interval are affected.

The script writes a one-element IONEFFECTS lattice spanning 0 to 1 m and runs serial Elegant under
Valgrind with four pressure profiles: two that reach the interval (samples at 0, 0.5 and 1 m, and
only at 0.2 and 0.8 m), which are controls that must complete without an invalid read, one sampled
only beyond the interval at 2 m and 3 m, and one sampled only before it at −3 m to −1 m. The verdict
is an invalid read in computeAverageGasPressures during a run that ends with the pressure-region
coverage error.
"""

import os
import re
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path

from _elegant import verdict

LATTICE = "Q: CHARGE,TOTAL=1e-9\nD: DRIF,L=1\nI: IONEFFECTS\nL: LINE=(Q,D,I)\n"
# The single IONEFFECTS element spans s in [0, 1] m, the whole lattice.
COMMANDS = """&run_setup lattice=lattice.lte, use_beamline=L, p_central_mev=100, random_number_seed=31317 &end
&run_control n_steps=1, n_passes=1 &end
&ion_effects pressure_profile=pressure.sdds, ion_properties=ions.sdds, macro_ions=10,
 ion_span[0]=0.01, ion_span[1]=0.01, field_calculation_method=gaussian, verbosity=11 &end
&bunched_beam n_particles_per_bunch=10, emit_x=1e-9, emit_y=1e-9, beta_x=1, beta_y=1,
 sigma_s=1e-3, sigma_dp=1e-4 &end
&track &end
"""
IONS = """SDDS1
&column name=IonName, type=string &end
&column name=Mass, units=AMU, type=double &end
&column name=ChargeState, type=long &end
&column name=SourceName, type=string &end
&column name=CrossSection, units=Mb, type=double &end
&data mode=ascii, no_row_counts=1 &end
H2+ 2.016 1 H2 1.0
"""
PRESSURE = """SDDS1
&parameter name=Temperature, units=K, type=double &end
&column name=s, units=m, type=double &end
&column name=H2, units=Torr, type=double &end
&data mode=ascii, no_row_counts=1 &end
293
{rows}"""
# Pressure samples that must be uniformly spaced. The first two profiles reach the interval and
# are controls: the second has samples inside the interval without spanning it, which passes.
CASES = (
    ("profile 0 to 1 m covers the element (control)", (0.0, 0.5, 1.0), True),
    ("profile 0.2 to 0.8 m lies inside the element (control)", (0.2, 0.8), True),
    ("profile 2 to 3 m lies beyond the element", (2.0, 3.0), False),
    ("profile -3 to -1 m lies before the element", (-3.0, -2.0, -1.0), False),
)
# computeAverageGasPressures prints "(iStart=>s[iStart], iEnd=>s[iEnd])" when either index is -1;
# the failure it reports is recognized by its opening words, whatever follows.
COVERAGE_ERROR = "Failed to find indices corresponding to pressure region"
# One Valgrind error block: a header line, then the call stack ("at"/"by" lines).
INVALID_READ = re.compile(r"Invalid read of size (\d+)\n((?:==\d+==\s+(?:at|by) .*\n)+)")
TIMEOUT = 600  # seconds per Valgrind run


def inputs(samples):
    rows = "".join(f"{s!r} 1e-9\n" for s in samples)
    return {
        "lattice.lte": LATTICE,
        "run.ele": COMMANDS,
        "ions.sdds": IONS,
        "pressure.sdds": PRESSURE.format(rows=rows),
    }


def checked_run(samples):
    """Serial Elegant under Valgrind: (exit status, log, invalid reads inside computeAverageGasPressures)."""
    if shutil.which("valgrind") is None or shutil.which("elegant") is None:
        sys.exit("valgrind and elegant must be on PATH.")
    if "RPN_DEFNS" not in os.environ:
        sys.exit("Set RPN_DEFNS to the defns.rpn file of SDDS (see the elegant installation).")
    env = dict(os.environ, OMP_NUM_THREADS="1")
    with tempfile.TemporaryDirectory(prefix="elegant-repro-") as tmp:
        for name, text in inputs(samples).items():
            (Path(tmp) / name).write_text(text)
        try:
            done = subprocess.run(
                ["valgrind", "-q", "--error-limit=no", "elegant", "run.ele"],
                cwd=tmp,
                env=env,
                capture_output=True,
                text=True,
                timeout=TIMEOUT,
                check=False,
            )
        except subprocess.TimeoutExpired:
            sys.exit(f"valgrind elegant exceeded {TIMEOUT} s")
    log = done.stdout + done.stderr
    reads = [
        (size, stack.splitlines()[0].split(": ", 1)[1])
        for size, stack in INVALID_READ.findall(log)
        if "computeAverageGasPressures" in stack
    ]
    return done.returncode, log, reads


defect = False
print(
    "invalid reads in computeAverageGasPressures for the pressure region [0, 1] m, serial elegant"
)
for label, samples, control in CASES:
    status, log, reads = checked_run(samples)
    covered = COVERAGE_ERROR in log
    if control:
        if status or covered or reads:
            sys.exit(
                f"serial {label}: the run fails or reads outside the array: not a valid control"
            )
        print(f"serial {label}: observed=run completes, no invalid read expected=same")
        continue
    # An uncovered profile may only fail with the intended coverage error.
    if status != 0 and not covered:
        sys.exit(f"serial {label}: exit status {status} without the coverage error:\n{log[-2000:]}")
    shown = f"invalid read of size {reads[0][0]} at {reads[0][1]}" if reads else "no invalid read"
    print(f"serial {label}: observed={shown} expected=a coverage error with no invalid read")
    defect |= bool(reads)
verdict(defect)
