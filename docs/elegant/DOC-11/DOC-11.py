"""DOC-11: Manual orders tune_shift_with_amplitude after twiss_output, where Elegant rejects or ignores it

The Elegant manual says `tune_shift_with_amplitude` must follow `twiss_output`. Given in that order
with Twiss output scheduled for each step (`output_at_each_step=1`), Elegant stops with "you must
give tune_shift_with_amplitude before twiss_output"; after an immediate `twiss_output`, Elegant has
already written the Twiss file, so no tune shift is computed and dnux/dAx and the other tune-shift
columns stay 0. The working order is `tune_shift_with_amplitude` before `twiss_output`. For eight
FODO cells with a thin octupole of integrated strength K₃L, that order gives dnux/dAx = 1.411 m⁻¹,
within 0.02% of the first-order detuning K₃L βx²/(32π), while the documented order gives an error or
0.

The script reads the sequence statement of the `tune_shift_with_amplitude` section from
`doc/elegant.tex` of the Elegant source checkout named by `ELEGANT_SOURCE` (required) and exits if
it is missing or unrecognized. It runs serial Elegant in tracking mode on eight FODO cells with one
thin octupole, first with `tune_shift_with_amplitude` before a scheduled `twiss_output` (the
control, compared with first-order octupole detuning), then in the documented order with scheduled
and with immediate Twiss output, and compares their dnux/dAx, dnux/dAy, dnuy/dAx and dnuy/dAy with
the control's.
"""

import os
import re
import sys
from pathlib import Path

import numpy as np
from _elegant import RunFailure, run, verdict

K3L = 3.0  # m^-3, integrated octupole strength of the thin MULT at the start of the ring
LATTICE = f"""F: QUAD,L=0.25,K1=1.1,ORDER=1
G: QUAD,L=0.25,K1=-1.0,ORDER=1
D: DRIF,L=0.7
E: DRIF,L=0.9
O: MULT,ORDER=3,KNL={K3L!r}
C: LINE=(F,D,G,E)
A: LINE=(O,8*C)
"""
SETUP = "&run_setup lattice=lattice.lte, use_beamline=A, p_central_mev=2400, default_order=3 &end\n"
TSWA = (
    "&tune_shift_with_amplitude turns=4096, x0=1e-6, y0=1e-6, x1=0.002, y1=0.002, grid_size=4,"
    " order=1, tune_output=tunes.sdds &end\n"
)
ACTION = "&run_control &end\n&bunched_beam &end\n&track &end\n"
COLUMNS = ("dnux/dAx", "dnux/dAy", "dnuy/dAx", "dnuy/dAy")


def twiss(scheduled):
    """twiss_output; output_at_each_step=1 schedules the computation at the next action."""
    return (
        "&twiss_output filename=twiss.sdds, matched=1, concat_order=3, "
        f"output_at_each_step={int(scheduled)} &end\n"
    )


source = os.environ.get("ELEGANT_SOURCE")
if not source:
    sys.exit("ELEGANT_SOURCE must name the Elegant source checkout.")
manual = Path(source) / "doc" / "elegant.tex"
if not manual.is_file():
    sys.exit(f"{manual} does not exist.")
text = manual.read_text()
section = text[text.index(r"\subsection{tune\_shift\_with\_amplitude") :]
section = section[: section.index(r"\subsection", 1)]
sequence = re.findall(r"\\item sequence: must (follow|precede) \\verb\|twiss_output\|\.", section)
if len(sequence) != 1:
    sys.exit(
        f"The manual's sequence statement for tune_shift_with_amplitude is unrecognized: {sequence}"
    )
documented_first = "twiss" if sequence[0] == "follow" else "tswa"
print(f"manual: tune_shift_with_amplitude must {sequence[0]} twiss_output")


def tswa_run(order, scheduled):
    """Derivatives written to the Twiss file, Twiss parameters at the start, and grid rows; None if
    Elegant stops with an error (then also the error line)."""
    body = TSWA + twiss(scheduled) if order == "tswa" else twiss(scheduled) + TSWA
    try:
        out = run({"lattice.lte": LATTICE, "run.ele": SETUP + body + ACTION},
                  outputs=("twiss.sdds", "tunes.sdds"))  # fmt: skip
    except RunFailure as failure:
        errors = [line for line in failure.log.splitlines() if "error" in line.lower()]
        return None, f"exit {failure.status}: {errors[-1] if errors else 'no error line'}"
    tw, tunes = out["twiss.sdds"], out["tunes.sdds"]
    par = {p.name: p.data[0] for p in tw.parameters}
    rows = len(tunes[0, "x"]) if tunes.n_pages else 0
    start = {k: float(tw[0, k][0]) for k in ("betax", "betay")}
    return dict(dnu={k: float(par[k]) for k in COLUMNS}, rows=rows, **start), "exit 0"


# Control: the source order with scheduled Twiss output. First-order perturbation theory for the
# thin octupole, H = K3L (x^4 - 6 x^2 y^2 + y^4)/24 (MULT kick -K3L/6 Re (x+iy)^3), averaged over
# the betatron phases with x = sqrt(2 Jx bx) cos(phi), gives dnux/dJx = K3L bx^2/(16 pi),
# dnux/dJy = dnuy/dJx = -K3L bx by/(8 pi), dnuy/dJy = K3L by^2/(16 pi). Elegant's amplitude is the
# Courant-Snyder invariant A = 2J, so each dnu/dA is half of dnu/dJ.
control, status = tswa_run("tswa", scheduled=True)
if control is None:
    sys.exit(f"control: the source order failed ({status})")
bx, by = control["betax"], control["betay"]
theory = {
    "dnux/dAx": K3L * bx**2 / (32 * np.pi),
    "dnux/dAy": -K3L * bx * by / (16 * np.pi),
    "dnuy/dAx": -K3L * bx * by / (16 * np.pi),
    "dnuy/dAy": K3L * by**2 / (32 * np.pi),
}
print(f"control, tune_shift_with_amplitude before scheduled twiss_output: {status}, {control['rows']} "
      f"grid points; betax={bx:.10g} m, betay={by:.10g} m at the octupole")  # fmt: skip
for k in COLUMNS:
    print(
        f"  {k}: observed={control['dnu'][k]:.6f} m^-1 first-order octupole theory={theory[k]:.6f} m^-1"
    )
if not all(np.isclose(control["dnu"][k], theory[k], rtol=0.01, atol=0) for k in COLUMNS):
    sys.exit("control: the source-order tune shifts do not match the octupole detuning")

defect = False
for scheduled in (True, False):
    label = "scheduled" if scheduled else "immediate"
    result, status = tswa_run(documented_first, scheduled)
    if result is None:
        print(f"documented order, {label} twiss_output: observed={status} "
              f"expected=exit 0 with dnux/dAx={control['dnu']['dnux/dAx']:.6f} m^-1")  # fmt: skip
        defect = True
        continue
    got = result["dnu"]
    print(f"documented order, {label} twiss_output: {status}, {result['rows']} grid points; "
          f"dnux/dAx observed={got['dnux/dAx']:.6f} expected={control['dnu']['dnux/dAx']:.6f} m^-1, "
          f"dnuy/dAy observed={got['dnuy/dAy']:.6f} expected={control['dnu']['dnuy/dAy']:.6f} m^-1")  # fmt: skip
    defect |= not all(np.isclose(got[k], control["dnu"][k], rtol=1e-6, atol=0) for k in COLUMNS)
print(
    "comparison: control within 1% of first-order theory; the documented order must run and "
    "write the control's tune shifts (rtol=1e-6)"
)
verdict(defect)
