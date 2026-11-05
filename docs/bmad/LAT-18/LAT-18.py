"""LAT-18: The documented Gaussian default differs from both source defaults

Omitting the Gaussian-converter setting selects Ziggurat in both the beam-initialization structure and the random-state structure. The manual instead lists exact as the default and gives only exact and quick as choices. Runtime initialization of both structures exposes the disagreement directly. The documented default and supported converter list should include Ziggurat.

The script compiles a small program that reports the untouched beam and random-state defaults from the built modules. It also generates a finite Gaussian sample using the default converter. BMAD_SOURCE supplies the documented default.
"""

import numpy as np
from _bmad import fortran, verdict
from _s3_12_particle_contracts import document_claim

program = """
program defaults
use bmad
use random_mod
implicit none
type(beam_init_struct) bi
type(random_state_struct) rs
real(rp) sample(4)
print *, 'BEAM_DEFAULT ', trim(bi%random_gauss_converter)
print *, 'STATE_DEFAULT ', trim(ran_gauss_converter_name(rs%gauss_converter))
call ran_seed_put(314159)
call ran_gauss(sample)
write(*,'(a,4es26.17e3)') 'SAMPLE ', sample
end program
"""
log = fortran(program).log
print(log.strip())
assert "BEAM_DEFAULT ziggurat" in log and "STATE_DEFAULT ziggurat" in log
sample = next(line.split()[1:] for line in log.splitlines() if line.startswith("SAMPLE "))
sample = np.array(sample, dtype=float)
assert sample.shape == (4,) and np.isfinite(sample).all()
verdict(
    document_claim(
        "bmad/doc/beam-init.tex", 'character random_gauss_converter ! "exact" (default) or "quick".'
    )
)
