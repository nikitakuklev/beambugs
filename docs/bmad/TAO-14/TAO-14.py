"""TAO-14: call -no_calc saves flags without disabling lattice calculation

Calling a Tao command file with -no_calc still recalculates the lattice after edits inside the file. The manual promises to disable calculation until the file finishes and then restore its initial state. After a drift changes from 1.7 m to 2.5 m, an interior orbit query already returns x = 0.05101 m instead of the deferred 0.03501 m. An analytic drift map and the final orbit query establish that the update happens too early.

The script changes a drift length inside a called file and queries the orbit both inside and after the call. It compares the horizontal position with the exact field-free map at the old and new lengths.
"""

import numpy as np
from _bmad import compare, verdict
from _tao_contracts_normal_form import LATTICE, scalar, tao

result = tao(
    LATTICE + "particle_start[px] = 0.02\n",
    ["call -no_calc edits.tao", "pipe -write outside.txt ele:orbit E"],
    files={"edits.tao": "set element E l = 2.5\npipe -write inside.txt ele:orbit E\n"},
    outputs=["inside.txt", "outside.txt"],
)
# With normalized px=0.02 and delta=py=0, x(L)=x0+L*px/sqrt(1-px**2).
expected = [0.001 + length * 0.02 / np.sqrt(1 - 0.02**2) for length in (1.7, 2.5)]
inside, outside = [
    scalar(result.outputs[name].decode(), "x") for name in ("inside.txt", "outside.txt")
]
assert not compare("final orbit x (m), updated length 2.5 m", outside, expected[1], atol=5e-16)
verdict(compare("interior orbit x (m), deferred length 1.7 m", inside, expected[0], atol=5e-16))
