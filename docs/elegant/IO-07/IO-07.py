"""IO-07: Pelegant WATCH converts every species' kinetic energy with the electron mass

In Pelegant, WATCH centroid and parameter output converts every species' mean kinetic energy using the electron rest energy. For a proton distribution with dimensionless momenta 0.75, 1.5, 2.75, and 5.5 in equal proportions, it reports 0.9670 MeV instead of 1776 MeV, an underestimate by a factor of 1836. The expected mean follows from K=(γ−1)mc², where γ is the Lorentz factor and m the proton mass. ID selection retains the error. Serial proton output, named electron and positron controls, momenta, and drift times are correct.

Setup: always runs 3-rank Pelegant (Open MPI `mpirun` on `PATH`) beside serial Elegant, because the defect occurs only under MPI.

Twenty fixed particles cross a 0.75 m exact drift before centroid, parameter and ID-selected WATCH outputs. A 70-digit relativistic oracle uses the selected species mass, with electron, positron and serial controls and independent momentum and flight-time checks.
"""

from _elegant import modes, verdict
from _watch_energy import ME, MMU, MP, check


def main():
    defects = []
    for mode in ("serial", *modes(mpi_only=True)):
        for label, species, mass, tabulated in (
            ("electron", "electron", ME, True),
            ("positron", "positron", ME, True),
            ("proton", "proton", MP, False),
            ("muon", "muon", MMU, False),
            ("custom100_q1", "custom,mass_ratio=100,charge_ratio=-1", 100 * ME, False),
        ):
            defects.extend(check(mode, label, species, mass, tabulated=tabulated))
    verdict(all(defects))


if __name__ == "__main__":
    main()
