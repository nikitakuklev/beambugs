"""LAT-19: The manual contradicts the additive initialization time offset

Initializing an ordinary moving-particle bunch with use_particle_start enabled adds t_offset to the time specified by particle_start. The manual first says the offset always applies, then says it is ignored in this mode. An electron at 1 MeV/c with starting time 2 ns and offset 3 ns is initialized at 5 ns, with longitudinal coordinate −0.5339 m determined by the initial 2 ns. The contradictory paragraph should describe the additive time convention.

The script initializes one particle from an inline lattice and compares its time, speed and longitudinal coordinate with the finite-beta expectation. BMAD_SOURCE supplies the contradictory manual statements.
"""

import numpy as np
from _bmad import compare, fortran, verdict
from _s3_12_particle_contracts import TIME, document_claim, rows

lat = "parameter[particle]=electron\nparameter[p0c]=1e6\nparticle_start[t]=2e-9\nd: drift,l=1\nl: line=(d)\nuse,l\n"
row = rows(fortran(TIME, files={"lat.bmad": lat}).log)["time"]
beta = 1e6 / np.hypot(1e6, row[3])
if compare(
    "time (s), z (m), beta; additive offset and z=-beta*c*2ns",
    row[:3],
    [5e-9, -beta * 299792458 * 2e-9, beta],
    atol=0,
    rtol=3e-15,
):
    raise RuntimeError("Additive-time control failed")
verdict(
    document_claim(
        "bmad/doc/beam-init.tex",
        r"the setting of \vn{beam_init%t_offset} will be ignored in favor of \vn{particle_start[t]}",
    )
)
