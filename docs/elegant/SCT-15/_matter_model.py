"""Shared setup for the MATTER energy-loss reproducers: any material, momentum and options.

`track` runs one MATTER element on a beam of electrons that all start on axis at one momentum.
`model` evaluates the event rates and slice count with the formulas and constants of the MATTER
source, so a reproducer can predict the counts its own model implies. `gamma_loss` recovers the
energy loss of each electron from the stored momenta without cancellation.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from _elegant import particles, run

# SDDS constants.h and the MATTER source (matter.c), as Elegant uses them.
PI = 3.141592653589793
RE = 2.81794092e-15  # classical electron radius (m)
E_MKS, H_MKS, C_MKS = 1.60217733e-19, 6.6260755e-34, 2.99792458e8
EPSILON0 = 1e7 / (4 * PI * C_MKS**2)
K_BOLTZMANN = 1.380658e-23  # J/K
N_AVOGADRO = 6.02214129e23
AMU_SOURCE = 1.6605e-27  # matter.c's AMU macro (kg), used in the single-scattering rates
M_PROTON = 1.6726485e-27  # matter.c's mp_mks (kg)
U_KG = 1e-3 / N_AVOGADRO  # atomic mass unit implied by radiationLength's N_A/A per gram
ALPHA_FS = E_MKS**2 / (4 * PI * EPSILON0 * H_MKS / (2 * PI) * C_MKS)  # radiationLength's alpha
Y0 = 1e-8  # lower cutoff of the bremsstrahlung and recoil fractional-transfer samplers
# Mean photons per radiation length above Y0: (4/3)[-ln y0 - (1 - y0) + 3/8 (1 - y0^2)] (matter.c:125)
BREMS_PER_X0 = 4 / 3 * (-np.log(Y0) - (1 - Y0) + 3 / 8 * (1 - Y0**2))


def radiation_length(Z: int, A: float, rho: float) -> float:
    """Radiation length in m by matter.c's radiationLength (Tsai, PDG): Z, A in AMU, rho in kg/m^3."""
    a2 = (ALPHA_FS * Z) ** 2
    f = a2 * (1 / (1 + a2) + 0.20206 + a2 * (-0.0369 + a2 * (0.0083 - a2 * 0.002)))
    light = {1: (5.31, 6.144), 2: (4.79, 5.621), 3: (4.74, 5.805), 4: (4.71, 5.924)}
    lrad, lradp = light.get(Z, (np.log(184.15 * Z ** (-1 / 3)), np.log(1194 * Z ** (-2 / 3))))
    per_atom = 4 * ALPHA_FS * RE**2 * (Z * Z * (lrad - f) + Z * lradp)  # m^2
    return 1 / (per_atom * rho * N_AVOGADRO / (A * 1e-3))


@dataclass(frozen=True)
class Rates:
    """Mean events per electron over the whole element, and the slice count, as matter.c sets them."""

    elastic: float  # probScatter
    brems: float  # probBSScatter
    recoil: float  # probERScatter
    sections: int  # sections0


def model(Z: int, A: float, rho: float, length: float, p0: float, plimit: float = 0.05) -> Rates:
    """Rates of matter.c:118-130 for reference momentum p0 (m_e c) and density rho (kg/m^3)."""
    beta = p0 / np.sqrt(p0**2 + 1)
    k1 = 4 * Z * (Z + 1) * (RE / (beta * p0)) ** 2
    k2 = (Z ** (1 / 3) / 137.036 / p0) ** 2
    n_atom = rho / (AMU_SOURCE * A)
    elastic = n_atom * length * k1 * PI**3 / (k2**2 + k2 * PI**2)
    brems = length / radiation_length(Z, A, rho) * BREMS_PER_X0
    recoil = length * n_atom * 2 * PI * Z * RE**2 / p0 * (1 / Y0 - 1)
    return Rates(elastic, brems, recoil, int(elastic / plimit + 1))


def beam(count: int, p: float) -> dict[str, np.ndarray]:
    """`count` electrons on axis with zero slopes at momentum p (m_e c), numbered from 1."""
    zeros = np.zeros(count)
    return {"x": zeros, "xp": zeros, "y": zeros, "yp": zeros, "t": zeros, "p": np.full(count, p),
            "particleID": np.arange(1, count + 1)}  # fmt: skip


def track(mode: str, element: str, p0: float, seed: int, count: int) -> dict[str, np.ndarray]:
    """Track `count` electrons at p0 through `M: MATTER,<element>`; return the columns by particleID.

    The output must hold every input particle (none may stop in the material).
    """
    lattice = f"M: MATTER,{element}\nBL: LINE=(M)\n"
    commands = (
        f"&run_setup lattice=lattice.lte, use_beamline=BL, p_central={p0!r}, "
        f"output=out.sdds, random_number_seed={seed} &end\n"
        "&run_control n_steps=1, n_passes=1 &end\n"
        "&sdds_beam input=beam.sdds &end\n"
        "&track &end\n"
    )
    columns = beam(count, p0)
    result = run({"lattice.lte": lattice, "run.ele": commands}, mode, {"beam.sdds": columns})
    out = particles(result["out.sdds"])
    if not np.array_equal(out["particleID"], columns["particleID"]):
        raise SystemExit(f"{mode}: the output particles differ from the input ones (losses?)")
    return out


def gamma_loss(p_in: float, p_out: np.ndarray) -> np.ndarray:
    """gamma_in - gamma_out from momenta in m_e c: (p_in - p_out)(p_in + p_out)/(gamma_in + gamma_out)."""
    g_in, g_out = np.sqrt(1 + p_in**2), np.sqrt(1 + p_out**2)
    return (p_in - p_out) * (p_in + p_out) / (g_in + g_out)
