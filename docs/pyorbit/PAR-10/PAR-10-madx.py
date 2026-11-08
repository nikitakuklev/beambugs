"""PAR-10: MAD-X reads the exponent literals as written, and SAVE writes small strengths with an exponent

Setup: needs cpymad (MAD-X 5), which the PyORBIT environment does not provide; run it with a Python that has cpymad.

The script evaluates the attribute literals of the main reproducer in MAD-X. It then builds the 16-cell FODO ring with eight kicker correctors in MAD-X, saves it with SAVE, checks that the saved file is the SAVE-form deck the main reproducer reads, prints its corrector lines, and computes the one-pass orbit from the origin with TWISS. MAD-X gives each literal its written value and x = 1.156 mm, y = −0.9718 mm at the end of the pass.
"""

import decimal
import os
import tempfile

from cpymad.madx import Madx

KICKS = {
    "ch1": 3.7e-05,
    "ch2": -1.25e-05,
    "ch3": 2e-05,
    "ch4": -4.4e-05,
    "cv1": 1.5e-05,
    "cv2": -2.75e-05,
    "cv3": 6.1e-06,
    "cv4": -3.3e-05,
}


def ring_deck(fixed_point):
    """The SAVE-form deck of the main reproducer (same code)."""
    parents = {
        "qf": "qf: quadrupole,l:=lq ,k1:=kqf ;",
        "qd": "qd: quadrupole,l:=lq ,k1:=kqd ;",
        "mb": "mb: sbend,l:=lb ,angle:=2 * pi / ( 2 * ncell ) ;",
        "ch": "ch: kicker,l:= 0;",
        "cv": "cv: kicker,l:= 0;",
    }

    def lit(v):
        s = f"{v:.10g}"
        return format(decimal.Decimal(s), "f") if fixed_point else s

    defs = ["lq = 0.5;", "kqf = 0.2851;", "lb = 2;", "ncell = 16;", "kqd = -0.2851;"]
    seq = ["ring: sequence, l = 160;"]
    for i in range(16):
        s0 = 10 * i
        cell = [
            (0.25, f"qf{i}: qf"),
            (2.5, f"mb{i}a: mb"),
            (5.0, f"qd{i}: qd"),
            (7.5, f"mb{i}b: mb"),
        ]
        if i % 4 == 0:
            k = i // 4 + 1
            cell += [
                (0.75, f"ch{k}: ch,hkick:= {lit(KICKS[f'ch{k}'])}"),
                (5.75, f"cv{k}: cv,vkick:= {lit(KICKS[f'cv{k}'])}"),
            ]
        for at, text in sorted(cell):
            name, rest = text.split(": ")
            parent, _, attr = rest.partition(",")
            if parent in parents:
                defs.append(parents.pop(parent))
            defs.append(f"{text};")
            seq.append(f"{name}: {parent}, at = {at + s0:g}" + (f",{attr};" if attr else ";"))
    return "\n".join(defs + seq + ["endsequence;"]) + "\n"


TAIL = "seq: sequence, l=10, refer=centre;\n{name}, at=2;\nendsequence;\n"
LITERALS = (
    ("quadrupole k1=1.5e-3", "q: quadrupole, l=1, k1=1.5e-3;\n", "q", "k1"),
    ("quadrupole k1=0.25e-3", "q: quadrupole, l=1, k1=0.25e-3;\n", "q", "k1"),
    ("quadrupole k1:=1.2345678901e-05", "q: quadrupole, l=1, k1:=1.2345678901e-05;\n", "q", "k1"),
    (
        "SAVE form, kicker hkick:= -1.25e-05",
        "c: kicker, l:= 0;\nc1: c,hkick:= -1.25e-05;\n",
        "c1",
        "hkick",
    ),
    (
        "SAVE form, kicker vkick:= 3.7e-05",
        "c: kicker, l:= 0;\nc1: c,vkick:= 3.7e-05;\n",
        "c1",
        "vkick",
    ),
    ("sbend angle=1.5e-3", "b: sbend, l=1, angle=1.5e-3;\n", "b", "angle"),
)
for label, definition, name, attr in LITERALS:
    m = Madx(stdout=False)
    m.input("beam;\n" + definition + TAIL.format(name=name))
    m.use("seq")
    print(f"MAD-X [{label}] {attr} = {float(m.elements[name][attr]):.11g}")
    m.quit()

m = Madx(stdout=False)
m.input("""
beam, particle=proton, energy=1.938272088;  ! total energy: kinetic 1 GeV
ncell = 16; lb = 2; lq = 0.5;
kqf = 0.2851; kqd = -0.2851;
qf: quadrupole, l:=lq, k1:=kqf;
qd: quadrupole, l:=lq, k1:=kqd;
mb: sbend, l:=lb, angle:=2*pi/(2*ncell);
ch: kicker, l=0;
cv: kicker, l=0;
""")
places = []
for i in range(16):
    s0 = 10.0 * i
    places += [
        (s0 + 0.25, f"qf{i}: qf, at={s0 + 0.25:.4f};"),
        (s0 + 2.5, f"mb{i}a: mb, at={s0 + 2.5:.4f};"),
        (s0 + 5.0, f"qd{i}: qd, at={s0 + 5.0:.4f};"),
        (s0 + 7.5, f"mb{i}b: mb, at={s0 + 7.5:.4f};"),
    ]
    if i % 4 == 0:
        k = i // 4 + 1
        places += [
            (s0 + 0.75, f"ch{k}: ch, at={s0 + 0.75:.4f};"),
            (s0 + 5.75, f"cv{k}: cv, at={s0 + 5.75:.4f};"),
        ]
m.input(
    "\n".join(
        ["ring: sequence, l=160, refer=centre;"] + [t for _, t in sorted(places)] + ["endsequence;"]
    )
)
for name, v in KICKS.items():
    m.input(f"{name}->{'hkick' if name.startswith('ch') else 'vkick'} = {v!r};")
m.use("ring")
with tempfile.TemporaryDirectory() as d:
    path = os.path.join(d, "ring.seq")
    m.input(f'save, sequence=ring, file="{path}";')
    with open(path) as f:
        saved = f.read()
print("MAD-X SAVE output equals the main reproducer's SAVE-form deck:", saved == ring_deck(False))
print(
    "MAD-X SAVE corrector definitions:",
    [ln for ln in saved.splitlines() if "kick" in ln and "at =" not in ln],
)
tw = m.twiss(sequence="ring", betx=10, bety=10, x=0, px=0, y=0, py=0)  # one pass from the origin
print(f"MAD-X one pass x {float(tw.x[-1]) * 1e3:.4f} mm, y {float(tw.y[-1]) * 1e3:.4f} mm")
m.quit()
