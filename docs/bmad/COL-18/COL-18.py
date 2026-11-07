"""COL-18: CSR step-control documentation names two nonexistent attributes

When users set the element step size using either spelling in the CSR parameter description, the lattice parser rejects the input. The manual names csr_ds_track and csr_track_step, while the accepted attribute is csr_ds_step. Both CSR wrappers also print the stale name CSR_TRACK_STEP in their missing-step diagnostic. The documented names and diagnostics should identify csr_ds_step and retain the global ds_track_step fallback.

The script parses an sbend with csr_ds_step as a successful control, then tries each documented
spelling in a fresh parser process. It checks the explicit bad-attribute diagnostic and the manual
text from BMAD_SOURCE.
"""

from _bmad import verdict
from _collective_contracts import parse_lattice, source_text

manual = source_text("bmad/doc/param-structs.tex")
if not parse_lattice("sbend, l=1, g=0.1, csr_ds_step=0.01")[0]:
    raise RuntimeError("Registered csr_ds_step control failed")
defect = False
for name in ["csr_ds_track", "csr_track_step"]:
    accepted, log = parse_lattice(f"sbend, l=1, g=0.1, {name}=0.01")
    if not accepted and "BAD ATTRIBUTE NAME:" not in log:
        raise RuntimeError(log)
    documented = "\\vn{" + name + "}" in manual
    print(
        f"{name}: documented={documented}, parser_accepts={accepted}; expected documented input accepted"
    )
    defect |= documented and not accepted
verdict(defect)
