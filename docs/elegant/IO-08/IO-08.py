"""IO-08: Serial WATCH divides custom-species kinetic energy by charge magnitude

For a custom species whose charge magnitude differs from one elementary charge, serial WATCH reports mean kinetic energy divided by that charge magnitude in elementary-charge units. A species with mass 100 electron masses and charge +2e reports 48.35 MeV instead of 96.70 MeV for the tested momentum distribution; charge −2e gives the same error, while +0.5e doubles the result. The expected kinetic energy follows from mass and momentum and is independent of charge. Both centroid and parameter modes fail; unit-charge output and particle drift times are correct.

Twenty fixed particles with mass 100 electron masses cross a 0.75 m exact drift for charges +e, +2e, −2e and +0.5e in separate serial runs. The script compares both WATCH energy modes and an ID-selected output with a charge-independent 70-digit relativistic oracle and checks the precise 1/|q/e| error. Momentum, IDs and flight time provide controls;
"""

from _elegant import verdict
from _watch_energy import ME, check


def main():
    defects = []
    for label, ratio in (("q+1", -1), ("q+2", -2), ("q-2", 2), ("q+0.5", -0.5)):
        # change_particle charge_ratio is relative to the electron's negative charge.
        defects.extend(check("serial", label, f"custom,mass_ratio=100,charge_ratio={ratio}",
                             100 * ME, charge=abs(ratio)))
    verdict(all(defects))


if __name__ == "__main__":
    main()
