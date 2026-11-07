"""COL-20: The CSR method list omits the experimental three-dimensional selector

When users consult the CSR method list, it presents off and 1_dim as the possible settings but omits steady_state_3d. The parser accepts steady_state_3d, and the mesh-parameter description refers to it. Its separate tracking wrapper is explicitly marked experimental and nonoperational and restricts collective kicks to nonzero-length sector bends. The method list should explain that selector and its status, or identify itself as a list of supported methods only.

The script parses an sbend using steady_state_3d and checks the resulting method enum. It compares
that accepted selector with the manual method list and the wrapper's experimental declaration.
"""

from _bmad import verdict
from _collective_contracts import parse_lattice, rows, section, source_text

accepted, log = parse_lattice("sbend, l=1, g=0.1, csr_method=steady_state_3d")
if not accepted:
    raise RuntimeError(log)
method = rows(log)["csr_method"][0]
if method != 3:
    raise RuntimeError("Parser did not store the steady_state_3d enum")
manual = section(
    source_text("bmad/doc/methods.tex"), "The possible settings for", "The possible settings of"
)
wrapper = source_text("bmad/space_charge/csr_and_space_charge_mod.f90")
experimental = "EXPERIMENTAL. NOT CURRENTLY OPERATIONAL!" in wrapper
listed = "steady_state_3d" in manual.lower()
print(
    f"steady_state_3d: parser_accepts={accepted}, enum={method:g}, manual_lists={listed}; expected selector documented"
)
print(f"wrapper source declares experimental/nonoperational={experimental}")
verdict(accepted and not listed)
