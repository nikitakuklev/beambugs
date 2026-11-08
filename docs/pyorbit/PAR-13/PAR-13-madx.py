"""PAR-13: MAD-X applies K1S, K2S and K3S

Setup: needs cpymad (MAD-X 5), which the PyORBIT environment does not provide; run it with a Python that has cpymad.

The script places each element of the main reproducer in the same 3 m line and computes the exit orbit from x = 1 mm, y = 2 mm with TWISS, which matches the main reproducer's expected values. It also shows that MAD-X ignores K1S on an SBEND, the case the reader leaves out correctly.
"""

from cpymad.madx import Madx

E = 1.0 + 0.93827208816  # total energy of a 1 GeV proton
CASES = (
    ("quadrupole L=1 K1S=0.5", "quadrupole, l=1.0, k1s=0.5"),
    ("quadrupole L=1 K1=0.5 K1S=0.2", "quadrupole, l=1.0, k1=0.5, k1s=0.2"),
    ("sextupole L=0.5 K2S=6", "sextupole, l=0.5, k2s=6.0"),
    ("octupole L=0.5 K3S=60", "octupole, l=0.5, k3s=60.0"),
    ("quadrupole L=1 K1=0.5", "quadrupole, l=1.0, k1=0.5"),
    ("sbend L=1 ANGLE=0.1 K1S=0.4", "sbend, l=1.0, angle=0.1, k1s=0.4"),
    ("sbend L=1 ANGLE=0.1", "sbend, l=1.0, angle=0.1"),
)
for label, element in CASES:
    m = Madx(stdout=False)
    m.input(
        f"beam, particle=proton, energy={E!r};\ne1: {element};\nseq: sequence, l=3, refer=centre;\ne1, at=1.5;\nendsequence;\nuse, sequence=seq;"
    )
    tw = m.twiss(betx=10, bety=10, x=1e-3, px=0, y=2e-3, py=0)
    exit_ = ", ".join(f"{float(tw[k][-1]):+.6e}" for k in ("x", "px", "y", "py"))
    print(f"MAD-X [{label}] exit (x, px, y, py) ({exit_})")
    m.quit()
