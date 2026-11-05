"""TAO-15: A later lattice-file option cancels explicit -noinit

Starting Tao with -noinit before -lattice_file allows an ambient initialization file to change the run. Reversing these options suppresses that file as documented. An ambient chromaticity step of 0.007000 is loaded in the first order, while the reversed order retains the compiled default of 0.0001000. The parser explicitly clears the suppression flag when it encounters the lattice filename.

The script supplies an ambient tao.init with a distinctive chromaticity step and runs both option orders in fresh processes. It checks the reversed-order default as a control and compares the resulting global parameter.
"""

from _bmad import compare, verdict
from _tao_contracts_normal_form import LATTICE, block, scalar, tao

values = []
for options in (["-noinit", "-lattice_file", "lat.bmad"], ["-lattice_file", "lat.bmad", "-noinit"]):
    result = tao(
        LATTICE,
        ["pipe global"],
        options=options,
        files={"tao.init": "&tao_params\n global%delta_e_chrom = 0.007\n/\n"},
    )
    values.append(scalar(block(result.log, "pipe global"), "delta_e_chrom"))
assert not compare("control, -noinit last: delta_e_chrom", values[1], 0.0001, atol=0)
verdict(
    compare(
        "-noinit first: delta_e_chrom; explicit suppression in either order",
        values[0],
        values[1],
        atol=0,
    )
)
