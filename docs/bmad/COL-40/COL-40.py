"""COL-40: The beam–beam manual names a nonexistent Twiss-reference attribute

When users copy the beam–beam manual's s_beta_ref attribute to specify the strong-beam Twiss reference position, the parser rejects it. The accepted attribute is s_twiss_ref, which the field-size calculation and other manual prose already use. A beam–beam element with s_twiss_ref parses successfully, while the otherwise identical element with s_beta_ref fails. Both stale names in the manual should be replaced with s_twiss_ref.

The script parses otherwise identical beam–beam elements with the accepted and documented attribute
names. It requires a successful control and the explicit bad-attribute error for s_beta_ref, then
checks that the stale name remains in the manual.
"""

from _bmad import verdict
from _collective_contracts import parse_lattice, source_text

base = "beambeam, sig_x=0.001, sig_y=0.002, sig_z=0.01, beta_a_strong=1, beta_b_strong=1"
if not parse_lattice(base + ", s_twiss_ref=0.1")[0]:
    raise RuntimeError("Registered s_twiss_ref control failed")
accepted, log = parse_lattice(base + ", s_beta_ref=0.1")
if not accepted and "BAD ATTRIBUTE NAME:" not in log:
    raise RuntimeError(log)
manual = source_text("bmad/doc/elements.tex")
documented = "s_beta_ref" in manual.lower()
print(
    f"s_beta_ref: documented={documented}, parser_accepts={accepted}; expected documented input accepted"
)
print("s_twiss_ref: parser_accepts=True (control)")
verdict(documented and not accepted)
