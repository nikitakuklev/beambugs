"""TRK-31: A custom particle keeps the anomalous magnetic moment of the previously selected species

Selecting a custom particle with change_particle sets its mass and charge but not its anomalous
magnetic moment a, which keeps the value of the species selected before, or the electron's in a
fresh run. Spin tracking of the same custom particle therefore depends on earlier change_particle
commands. For a custom particle with the electron's mass and charge at momentum 200 mₑc, an on-axis
spin (1, 0, 0) through a 0.65 m SOLE with KS = 0.9000 m⁻¹ ends at sᵧ = −0.5528 when custom is the
only selection, but at −0.9980 after a proton selection, the Thomas–BMT result for the proton's a =
1.793. No input sets a for a custom particle.

The script runs the same custom particle, lattice and beam after four selection histories (none,
proton, muon, and proton then electron) and requires identical exit spins, with the custom-only run
as the reference. The reference must equal the analytic precession with the electron moment and the
proton-electron-custom history must restore it (controls); each departure is matched to the
analytic precession with the earlier species' moment.
"""

import sys

import numpy as np
from _elegant import modes, verdict
from _sole import A_ELECTRON, A_MUON, A_PROTON, axial_rotation, compare, on_axis_angle, track

LENGTH, P0, KS = 0.65, 200.0, 0.9
ELEMENT = f"SOLE,L={LENGTH!r},KS={KS!r}"
CUSTOM = "&change_particle name=custom, mass_ratio=1, charge_ratio=1 &end\n"
ATOL = 1e-13
HISTORIES = {  # label: change_particle commands before the final custom selection
    "custom only": [],
    "proton, then custom": ["proton"],
    "muon, then custom": ["muon"],
    "proton, electron, then custom": ["proton", "electron"],
}


def final_spin(mode, earlier):
    before = "".join(f"&change_particle name={name} &end\n" for name in earlier) + CUSTOM
    out, _ = track(mode, ELEMENT, P0, {"spx": 1.0}, before=before)
    return np.array([out["spx"], out["spy"], out["spz"]])


def bmt(anomaly):
    """On-axis Thomas-BMT spin from (1, 0, 0) for anomalous moment a (see _sole.on_axis_angle).

    The angle -KS L (1 + a) depends on the species only through a, because KS is given.
    """
    return axial_rotation(on_axis_angle(KS, LENGTH, anomaly))


defect = False
for mode in modes():
    spins = {label: final_spin(mode, earlier) for label, earlier in HISTORIES.items()}
    baseline = spins["custom only"]
    # The final inputs (species definition, lattice, beam) are identical in every history, so the
    # spin must be the same; the custom-only run is the reference.
    if not compare(
        f"{mode} custom only (reference) vs BMT with the electron a",
        baseline,
        bmt(A_ELECTRON),
        atol=ATOL,
    ):
        sys.exit("The custom-only reference is not the electron-moment rotation: unexpected.")
    if not compare(
        f"{mode} proton, electron, then custom (control)",
        spins["proton, electron, then custom"],
        baseline,
        atol=ATOL,
    ):
        sys.exit("Reselecting the electron did not restore the reference: the control fails.")
    for label, anomaly in (("proton, then custom", A_PROTON), ("muon, then custom", A_MUON)):
        defect |= not compare(f"{mode} {label}", spins[label], baseline, atol=ATOL)
        # The departure is the rotation with the earlier species' anomalous moment.
        species = label.split(",")[0]
        if not compare(
            f"{mode} {label} vs BMT with the {species} a={anomaly:g}",
            spins[label],
            bmt(anomaly),
            atol=ATOL,
        ):
            sys.exit(f"{label}: the spin is not the rotation with the {species} moment")
print(f"comparison: atol={ATOL:g}")
verdict(defect)
