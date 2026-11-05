"""TRK-44: Manual constants tables disagree with executable constants

Copying constants from the constants table or programming example in the Bmad manual gives values different from the module they describe. The table disagrees on electron and proton masses, electron radius and electron anomaly. The programming example gives different masses, radius and elementary charge; its charge exceeds the compiled value by 7.843 parts per million. Reading the manual passages and extracting the compiled module values establishes the discrepancy.

The script extracts the named numeric literals from both manuals under BMAD_SOURCE and compiles
a constants reader against BMAD_DIST. It compares the documented values with the compiled values
and uses the common speed of light as a passing control.
"""

import re

from _bmad import compare, source, verdict
from _tracking_contracts_materials import constants

values = constants()["constants"]
lat = (source() / "bmad/doc/lattice-file.tex").read_text()
program = (source() / "bmad/doc/misc-programming.tex").read_text()


def table_value(name):
    matches = re.findall(rf"^\s*{name}\s*&\s*\$([^$]+)\$", lat, re.MULTILINE)
    if len(matches) != 1:
        raise ValueError(f"Expected one manual table value for {name}")
    match = re.fullmatch(r"([\d.]+)d?\s*(?:\\pow\{(-?\d+)\})?", matches[0].strip())
    if match is None:
        raise ValueError(matches[0])
    return float(match[1]) * 10 ** int(match[2] or 0)


def example_value(name):
    matches = re.findall(rf"^\s*{name}\s*=\s*([\d.]+d[-+]?\d+)", program, re.MULTILINE)
    if len(matches) != 1:
        raise ValueError(f"Expected one programming example value for {name}")
    return float(matches[0].replace("d", "e"))


if compare("lattice speed-of-light control (m/s)", table_value("c_light"), values[5], atol=0):
    raise AssertionError("Speed-of-light control failed")
defects = []
for name, index in [("m_electron", 0), ("m_proton", 1), ("r_e", 2), ("anom_moment_electron", 3)]:
    defects.append(
        compare(
            f"lattice {name}, compiled value expected",
            table_value(name),
            values[index],
            atol=0,
            rtol=2e-16,
        )
    )
for name, index in [("m_electron", 0), ("m_proton", 1), ("r_e", 2), ("e_charge", 4)]:
    defects.append(
        compare(
            f"programming example {name}, compiled value expected",
            example_value(name),
            values[index],
            atol=0,
            rtol=2e-16,
        )
    )
verdict(any(defects))
