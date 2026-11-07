"""COL-05: A low live-particle count leaves stale space-charge fields

When fewer than two eligible particles remain in a time-based space-charge calculation, the field
routine returns without clearing its electric and magnetic outputs. With cathode images off, the
skipped calculation should leave zero fields, but an earlier field can reach subsequent tracking.
After seven of eight particles are lost, the survivor retains an electric field of (−5818, −5818,
−2215) V/m. A fresh one-particle call also preserves arbitrary electric and magnetic sentinel
values, locating an unwritten-output path.

The script computes fields for eight particles and calls the routine again after seven are lost,
with cathode images off. A separate fresh program supplies arbitrary electric and magnetic sentinels
to a one-particle call.
"""

import numpy as np
from _bmad import compare, fortran, verdict
from _collective_fields import rows

PROGRAM = r"""
program field_probe
use space_charge_mod
implicit none
type(bunch_struct) b
type(branch_struct) br
type(em_field_struct) f(8)
logical image
integer i
allocate(b%particle(8),br%ele(0:1))
space_charge_com%space_charge_mesh_size=[8,8,8]
image=.false.
do i=1,8
 b%particle(i)%vec(1)=merge(-0.001_rp,0.001_rp,btest(i-1,0))
 b%particle(i)%vec(3)=merge(-0.001_rp,0.001_rp,btest(i-1,1))
 b%particle(i)%s=0.01_rp+merge(-0.001_rp,0.001_rp,btest(i-1,2))
 b%particle(i)%species=electron$
 b%particle(i)%beta=0.8_rp
 b%particle(i)%charge=1e-12_rp
 b%particle(i)%ix_ele=1
 b%particle(i)%state=alive$
enddo
call sc_field_calc(b,br,image,1.0_rp,f)
print '(a,3es24.15)', 'RESULT eight ', f(1)%E
b%particle(2:8)%state=lost$
call sc_field_calc(b,br,image,1.0_rp,f)
print '(a,3es24.15)', 'RESULT one ', f(1)%E
print '(a,3es24.15)', 'RESULT lost ', f(2)%E
end program
"""

SENTINEL = r"""
program sentinel
use space_charge_mod
implicit none
type(bunch_struct) b
type(branch_struct) br
type(em_field_struct) f(1)
logical image
allocate(b%particle(1),br%ele(0:1))
b%particle(1)%state=alive$
b%particle(1)%species=electron$
b%particle(1)%beta=.8_rp
b%particle(1)%ix_ele=1
image=.false.
f(1)%E=[11.0_rp,22.0_rp,33.0_rp]
f(1)%B=[44.0_rp,55.0_rp,66.0_rp]
call sc_field_calc(b,br,image,1.0_rp,f)
print '(a,6es24.15)', 'RESULT sentinel ',f(1)%E,f(1)%B
end program
"""


data = rows(fortran(PROGRAM).log)
if not np.isfinite(data["eight"]).all() or np.max(abs(data["eight"])) < 1000:
    raise RuntimeError("Eight-particle nonzero-field control failed")
print(f"Eight-particle control E (V/m): {data['eight'].tolist()}")
defect = compare("survivor E (V/m), skipped image-free solve", data["one"], np.zeros(3), atol=1e-12)
defect |= compare("lost-particle E (V/m), skipped solve", data["lost"], np.zeros(3), atol=1e-12)
fresh = rows(fortran(SENTINEL).log)["sentinel"]
defect |= compare("fresh-call E (V/m) and B (T), zero-field contract", fresh, np.zeros(6), atol=0)
verdict(defect)
