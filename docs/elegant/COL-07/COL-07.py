"""COL-07: WAKE cache substitutes the first requested column pair

When two WAKE elements in one Elegant run select different wake columns (WCOLUMN) or time columns (TCOLUMN) from the same table file, the second element silently reuses the first element's columns, because the table is cached by file name alone. For a table with kernels A and B = 3A, the earliest particles of a 12-electron, 2 nC bunch should lose the sum of both kernels, 8000 eV; the shared file gives 4000 eV (A applied twice) or 12000 eV (B applied twice), depending on element order, which is an error of −50% or +50%. A byte-identical copy of the table under another file name gives the correct result.

The script tracks twelve electrons (four arrival times, three particles each) through two
zero-length WAKE elements that read a table holding two wake columns, B = 3A, and two time
columns, and measures each group's energy loss. It compares the loss with the causal particle-pair
sum of the two requested kernels. The control gives the second element a byte-identical copy of
the table under another file name, which restores the requested kernel in every case.
"""

import sys

import numpy as np
from _elegant import modes, particles, run, verdict

ELECTRON_EV = 0.51099906e6  # SDDS constants.h me_mev, in eV
CHARGE, COUNT, P = 2e-9, 12, 10.0
GROUP_TIMES = np.array([0, 8, 24, 48]) * 1e-12  # s, three particles per group
TIMES = np.repeat(GROUP_TIMES, 3)
ROWS = 100
K = np.arange(ROWS)
TABLE = {
    "t": K * 1e-12,  # s
    "t2": 2 * K * 1e-12,
    "A": 4e12 * np.exp(-K / 17),  # V/C
    "B": 3 * 4e12 * np.exp(-K / 17),
}
UNITS = {"t": "s", "t2": "s", "A": "V/C", "B": "V/C"}
ATOL_CONTROL = 1e-7  # eV; floor from subtracting gamma values near 10.05
MIN_ERROR = 100.0  # eV; the defect shifts the losses by thousands of eV


def sdds_table():
    """The wake table as an ASCII SDDS page."""
    head = "SDDS1\n" + "".join(
        f"&column name={name}, type=double, units={UNITS[name]} &end\n" for name in TABLE
    )
    rows = "".join(
        " ".join(repr(float(TABLE[name][i])) for name in TABLE) + "\n" for i in range(ROWS)
    )
    return head + f"&data mode=ascii &end\n! page number 1\n{ROWS}\n" + rows


def expected_loss(first, second):
    """Energy lost by each group, eV, if the elements apply their requested kernels.

    The wake energy loss of a particle is q sum_j W(t_i - t_j) over the particles at or ahead of
    it (t_j <= t_i), with q the macroparticle charge in C and W in V/C, giving eV. The kernel of
    an element is the table column W(tcolumn): zero-length elements leave arrival times
    unchanged, so the losses of the two elements add. Delays fall on table nodes.
    """
    q = CHARGE / COUNT
    loss = np.zeros(len(GROUP_TIMES))
    for tcol, wcol in (first, second):
        for i, ti in enumerate(GROUP_TIMES):
            for tj in GROUP_TIMES[GROUP_TIMES <= ti]:
                loss[i] += 3 * q * np.interp(ti - tj, TABLE[tcol], TABLE[wcol])
    return loss


def tracked_loss(mode, first, second, separate):
    """Energy lost by each group in eV, from the momentum change of the tracked beam."""
    names = ("wake.sdds", "wake2.sdds" if separate else "wake.sdds")
    lattice = f"Q: CHARGE,TOTAL={CHARGE!r}\n"
    for n, ((tcol, wcol), file) in enumerate(zip((first, second), names)):
        lattice += (
            f'W{n}: WAKE,INPUTFILE="{file}",TCOLUMN="{tcol}",WCOLUMN="{wcol}",'
            "N_BINS=0,SMOOTHING=0,INTERPOLATE=0\n"
        )
    lattice += "BL: LINE=(Q,W0,W1)\n"
    commands = (
        "&global_settings echo_namelists=0 &end\n"
        f"&run_setup lattice=lattice.lte,use_beamline=BL,p_central={P!r},output=out.sdds &end\n"
        "&run_control n_steps=1,n_passes=1 &end\n"
        "&sdds_beam input=beam.sdds &end\n"
        "&track &end\n"
    )
    files = {"lattice.lte": lattice, "run.ele": commands, "wake.sdds": sdds_table()}
    if separate:
        files["wake2.sdds"] = sdds_table()  # same bytes, other file name
    beam = {"t": TIMES, "p": np.full(COUNT, P), "particleID": np.arange(1, COUNT + 1)}
    out = particles(run(files, mode, {"beam.sdds": beam})["out.sdds"])
    if not np.array_equal(out["particleID"], np.arange(1, COUNT + 1)):
        sys.exit(f"{mode}: particles lost")
    if not np.allclose(out["t"], TIMES, rtol=0, atol=1e-21):
        sys.exit(f"{mode}: arrival times changed")
    gain = np.sqrt(1 + out["p"] ** 2) - np.sqrt(1 + P**2)
    return -ELECTRON_EV * gain.reshape(-1, 3).mean(axis=1)


def compare(mode, label, observed, expected):
    """Print one case and return its largest deviation from the pair sum, eV.

    Nonfinite or mismatched arrays are a failed run: NaN would otherwise slip past the thresholds.
    """
    if observed.shape != expected.shape or not (
        np.all(np.isfinite(observed)) and np.all(np.isfinite(expected))
    ):
        sys.exit(f"{mode} {label}: invalid observed or expected losses")
    error = np.max(np.abs(observed - expected))
    print(
        f"{mode} {label}: observed={np.array2string(observed, separator=', ', precision=10)} "
        f"expected={np.array2string(expected, separator=', ', precision=10)} "
        f"max deviation={error:.10g} eV"
    )
    return error


CASES = (
    ("W columns A then B", ("t", "A"), ("t", "B")),
    ("W columns B then A", ("t", "B"), ("t", "A")),
    ("T columns t then t2, W column A", ("t", "A"), ("t2", "A")),
)
defect = False
print("energy loss in eV of the four groups (arrival times 0, 8, 24, 48 ps)")
for mode in modes():
    for label, first, second in CASES:
        expected = expected_loss(first, second)
        control = compare(
            mode,
            f"{label}, separate files",
            tracked_loss(mode, first, second, separate=True),
            expected,
        )
        if control > ATOL_CONTROL:
            sys.exit(f"{mode}: the separate-file control is off by {control:g} eV")
        error = compare(
            mode,
            f"{label}, shared file",
            tracked_loss(mode, first, second, separate=False),
            expected,
        )
        defect |= error > MIN_ERROR
print(
    f"control tolerance {ATOL_CONTROL:g} eV; defect if a shared-file loss is off by > {MIN_ERROR:g} eV"
)
verdict(defect)
