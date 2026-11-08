"""PAR-14: MAD-X treats a thin MULTIPOLE's KNL[0] as a thin bend of the reference, and MAKETHIN writes bends in that form

Setup: needs cpymad (MAD-X 5), which the PyORBIT environment does not provide; run it with a Python that has cpymad.

The script tracks the sequences of the main reproducer in MAD-X: the 1 m drift, thin multipole with KNL = {0.001} and 1 m drift at the design momentum and with dE = 1 MeV, its SURVEY, the KICKER control and the eight-cell thin-lens ring. MAD-X leaves the design particle on axis behind the multipole and after one turn of the ring, and bends the survey by −K0L. It then slices a sector bend into two with MAKETHIN and prints the elements SAVE writes for the slices, MULTIPOLEs whose KNL[0] is the slice's bend angle.
"""

import math
import os
import tempfile

from cpymad.madx import Madx

EK, MASS = 1.0, 0.93827208816
E = EK + MASS
P0 = math.sqrt(E * E - MASS * MASS)
LINE = "d: drift, l=1.0;\nseq: sequence, l=2.0, refer=entry;\nel, at=1.0;\nendsequence;\n"


def run(deck, seq, dE):
    m = Madx(stdout=False)
    m.input(f"beam, particle=proton, energy={E!r};\n{deck}use, sequence={seq};")
    m.input(f"track, onepass=true, onetable=true; start, pt={dE / P0!r}; run, turns=1; endtrack;")
    x = float(m.table.trackone.x[-1])
    theta = float(m.survey().theta[-1])
    m.quit()
    return x, theta


for label, deck in (
    ("multipole knl={0.001}", "el: multipole, knl={0.001};\n"),
    ("control, kicker hkick=-0.001", "el: kicker, hkick=-0.001;\n"),
):
    for dE in (0.0, 1e-3):
        x, theta = run(deck + LINE, "seq", dE)
        print(
            f"MAD-X [{label}, dE = {dE * 1e3:g} MeV] x at 2 m {x:+.6e} m; survey angle at 2 m {theta:+.6e} rad"
        )

th = 2.0 * math.pi / 16
places = [
    f"{n}, at={10.0 * c + at};"
    for c in range(8)
    for n, at in (("qf", 0.0), ("bd", 2.5), ("qd", 5.0), ("bd", 7.5))
]
ring = (
    f"qf: multipole, knl={{0, 0.2}};\nqd: multipole, knl={{0, -0.2}};\nbd: multipole, knl={{{th!r}}};\n"
    "ring: sequence, l=80.0, refer=entry;\n" + "\n".join(places) + "\nendsequence;\n"
)
x, theta = run(ring, "ring", 0.0)
print(
    f"MAD-X [thin-lens ring, design particle] x after one turn {x:+.6e} m; survey angle {theta:+.6e} rad (-2*pi = {-2 * math.pi:+.6e})"
)

m = Madx(stdout=False)
m.input(
    f"beam, particle=proton, energy={E!r};\nb: sbend, l=2.0, angle=0.1;\n"
    "seq: sequence, l=4.0, refer=centre;\nb, at=2.0;\nendsequence;\nuse, sequence=seq;\n"
    "select, flag=makethin, class=sbend, slice=2;\nmakethin, sequence=seq;\nuse, sequence=seq;"
)
with tempfile.TemporaryDirectory() as d:
    path = os.path.join(d, "thin.seq")
    m.input(f'save, sequence=seq, file="{path}";')
    with open(path) as f:
        print(
            "MAD-X MAKETHIN, two slices, of 'b: sbend, l=2.0, angle=0.1' saves:",
            [ln.strip() for ln in f if "multipole" in ln],
        )
m.quit()
