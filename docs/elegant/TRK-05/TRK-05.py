"""TRK-05: Zero-voltage MODRF and RAMPRF advance the arrival time by L instead of the slanted path length

With VOLT = 0, MODRF and RAMPRF move particles along straight lines but advance the longitudinal
coordinate by cavity length L, omitting the extra path L(√(1 + x′² + y′²) − 1) for transverse slopes
x′ and y′. A nonzero nominal voltage with exactly zero effective field gives the full slanted-path
time, as an exact-drift control confirms. For L = 1.7 m and electron momentum 2 mₑc with x′ = 0.03
and y′ = −0.02, arrival is at 6.340 ns instead of 6.344 ns, 4.120 ps early. Switching these cavities
off changes arrival times and downstream RF phases for particles with nonzero transverse slopes.

The script tracks particles with three slope pairs through a 1.7 m MODRF and a 1.7 m RAMPRF with
VOLT = 0, and compares their arrival times with the time of flight over the slanted path at constant
speed. Controls: the on-axis pair, which has no extra path, a MODRF and a RAMPRF with a nonzero VOLT whose effective field is
exactly zero (AMOFFSET = -1, or a zero waveform), and an EDRIFT.
"""

import sys

import numpy as np
from _elegant import modes, particles, run, verdict

C = 299792458.0  # m/s, c_mks of SDDS
P0 = 2.0  # particle momentum, units of m_e c
LENGTH = 1.7  # m
# Slopes (xp, yp): on axis, the pair of the summary, and a larger pair; four copies of each pair
# so that every MPI worker has some.
SLOPES = np.repeat([[0.0, 0.0], [0.03, -0.02], [0.1, 0.04]], 4, axis=0)
BEAM = {"xp": SLOPES[:, 0], "yp": SLOPES[:, 1], "p": np.full(12, P0)}
RTOL = 1e-13  # the exact cases agree to 3e-16
# Constant waveform factors for RAMPRF, as an ASCII SDDS table.
WAVE = "SDDS1\n&column name=t, type=double, units=s &end\n&column name=v, type=double &end\n&data mode=ascii, no_row_counts=1 &end\n-1 {v}\n1 {v}\n"
CAVITY = 'FREQ=1e8,PHASE=0,FIDUCIAL="light"'
LATTICES = {
    "EDRIFT": f"E: EDRIFT,L={LENGTH!r}\n",
    "MODRF, VOLT=0": f"E: MODRF,L={LENGTH!r},VOLT=0,{CAVITY}\n",
    "RAMPRF, VOLT=0": f'E: RAMPRF,L={LENGTH!r},VOLT=0,{CAVITY},VOLT_WAVEFORM="wave.sdds=t+v"\n',
    "MODRF, VOLT=10000, AMOFFSET=-1": f"E: MODRF,L={LENGTH!r},VOLT=10000,{CAVITY},AMOFFSET=-1\n",
    "RAMPRF, VOLT=10000, zero waveform": f'E: RAMPRF,L={LENGTH!r},VOLT=10000,{CAVITY},VOLT_WAVEFORM="wave.sdds=t+v"\n',
}


def arrival_times(mode, name):
    """Arrival time (s) of the three slope pairs after the element; particles start at t = 0."""
    lattice = LATTICES[name] + "BL: LINE=(E)\n"
    wave = WAVE.format(v=0 if "zero waveform" in name else 1)
    commands = (
        f"&run_setup lattice=lattice.lte, use_beamline=BL, p_central={P0!r}, "
        "output=out.sdds &end\n"
        "&run_control n_steps=1, n_passes=1 &end\n"
        "&sdds_beam input=beam.sdds &end\n"
        "&track &end\n"
    )
    files = {"lattice.lte": lattice, "run.ele": commands, "wave.sdds": wave}
    out = particles(run(files, mode, {"beam.sdds": BEAM})["out.sdds"])
    if len(out["t"]) != 12:
        sys.exit(f"{mode} {name}: {12 - len(out['t'])} particles lost")
    if not np.allclose(out["p"], P0, rtol=1e-14, atol=0):
        sys.exit(f"{mode} {name}: the momentum changed")
    # The expected flight time assumes the field-free straight ray: slopes unchanged and
    # x = L xp, y = L yp from the origin. Anything else invalidates the comparison.
    straight = (out["x"], out["xp"], out["y"], out["yp"])
    ray = (LENGTH * SLOPES[:, 0], SLOPES[:, 0], LENGTH * SLOPES[:, 1], SLOPES[:, 1])
    if not all(np.allclose(o, e, rtol=0, atol=1e-14) for o, e in zip(straight, ray, strict=True)):
        sys.exit(f"{mode} {name}: the transverse coordinates are not the straight ray")
    times = out["t"].reshape(3, 4)
    if not (times == times[:, :1]).all():
        sys.exit(f"{mode} {name}: identical particles arrived at different times")
    return times[:, 0]


def compare(label, observed, expected):
    """Print one case; return True when observed and expected disagree."""
    bad = not np.allclose(observed, expected, rtol=RTOL, atol=0)
    fmt = lambda a: "[" + ", ".join(f"{v:.16g}" for v in a) + "]"
    print(f"{label}: observed={fmt(observed)} expected={fmt(expected)}")
    return bad


# Constant speed over the slanted path: t = L sqrt(1 + xp^2 + yp^2) / (beta c), with
# beta = p/sqrt(1 + p^2) for p in units of m_e c.
path = LENGTH * np.sqrt(1 + SLOPES[::4, 0] ** 2 + SLOPES[::4, 1] ** 2)
expected = path / (P0 / np.sqrt(1 + P0**2) * C)
defect = False
print("columns: arrival time (s) of the slope pairs (0, 0), (0.03, -0.02), (0.1, 0.04)")
for mode in modes():
    for name in LATTICES:
        observed = arrival_times(mode, name)
        # On axis the slanted path is L: the first pair must agree for every element.
        if not np.isclose(observed[0], expected[0], rtol=RTOL, atol=0):
            sys.exit(f"{mode} {name}: the on-axis control disagrees, the comparison is not valid.")
        bad = compare(f"{mode} {name}", observed, expected)
        if "VOLT=0" in name:
            defect |= bad
        elif bad:
            sys.exit(f"{mode} {name}: the control disagrees with the slanted path.")
print(f"comparison: rtol={RTOL:g}")
verdict(defect)
