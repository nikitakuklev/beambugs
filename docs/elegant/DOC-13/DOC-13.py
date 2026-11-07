"""DOC-13: SHRFDF phase parameters are labelled Hz but consumed as radians

SHRFDF labels PHASE0 through PHASE9 in Hz although they are angular offsets in radians. Each phase is added directly to the harmonic argument before taking its sine or cosine. Independent scans of all ten modes show 2π periodicity and a crest at π/2; an input of 90 gives 0.8940 of crest rather than the degree-based crest. The generated parameter dictionary repeats the incorrect unit, while PERIOD_PHASE is correctly labelled RAD. Tracking already uses radians consistently.

Generate the native parameter dictionary and scan every harmonic at four input phases with on-axis zero-time rays. Compare crest-normalized canonical kicks with sin(phase), including a 2π shift and input 90, and check PERIOD_PHASE as a correctly labelled control.
"""

import re
import tempfile
from pathlib import Path

from _harmonic_rf import kick
from _slicing import close

# isort: split
import _elegant as native
import numpy as np


def dictionary():
    # The generated ASCII dictionary has heterogeneous pages; inspect its SHRFDF block directly.
    with tempfile.TemporaryDirectory(prefix="harmonic-dictionary-") as tmp:
        path = Path(tmp) / "dictionary.sdds"
        native.run(
            {
                "line.lte": "O:MARK\nBL:LINE=(O)\n",
                "run.ele": (
                    "&run_setup lattice=line.lte,use_beamline=BL,p_central=3200 &end\n"
                    "&run_control n_steps=1 &end\n"
                    f'&print_dictionary filename="{path}",SDDS_form=1 &end\n'
                ),
            },
            outputs=(),
        )
        block = (
            path.read_text()
            .split("!***** element type SHRFDF:", 1)[1]
            .split("!***** element type", 1)[0]
        )
        return dict(re.findall(r'^"([^"\n]+)" "([^"\n]*)"', block, flags=re.MULTILINE))


def main():
    units = dictionary()
    if units["PERIOD_PHASE"] != "RAD":
        raise RuntimeError("PERIOD_PHASE control is not labelled RAD")
    faults = []
    phases = [np.pi / 2, -0.43, -0.43 + 2 * np.pi, 90.0]
    for mode in range(10):
        unit = units[f"PHASE{mode}"]
        if unit not in ("HZ", "RAD"):
            raise RuntimeError(f"Unexpected PHASE{mode} unit: {unit}")
        values = [float(kick(mode, p, np.zeros((1, 6)))[0]) for p in phases]
        ratios = np.array(values) / values[0]
        expected = np.sin(phases)
        close(ratios, expected, 2e-15, "radian sine law")
        faults.append(unit == "HZ")
        print(
            f"PHASE{mode}: dictionary={unit}, expected=RAD; normalized kicks={ratios.tolist()}, expected={expected.tolist()} (sin of input in radians)"
        )
    print("PERIOD_PHASE: dictionary=RAD, expected=RAD (angular-offset control)")
    native.verdict(any(faults))


if __name__ == "__main__":
    main()
