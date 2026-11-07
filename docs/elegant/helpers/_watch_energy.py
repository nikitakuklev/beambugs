"""WATCH energy fixtures and independent relativistic oracles for IO-07 and IO-08."""

import math
from decimal import Decimal, localcontext

import numpy as np
from _diagnostics import conversion_defect, run, scalar
from _elegant import particles

C = Decimal(299792458)
E = Decimal("1.60217733e-19")
ME = Decimal("9.1093897e-31")
MP = Decimal("1.6726485e-27")
MMU = Decimal("1.88353109e-28")
ME_MEV = Decimal("0.51099906")
PS = [0.75, 1.5, 2.75, 5.5] * 5
LENGTH = Decimal("0.75")
RTOL = 1e-14
LATTICE = """D: EDRIFT,L=0.75
C: WATCH,FILENAME="centroids.sdds",MODE="centroids"
P: WATCH,FILENAME="parameters.sdds",MODE="parameters"
S: WATCH,FILENAME="selected.sdds",MODE="centroids",START_PID=1,END_PID=7
W: WATCH,FILENAME="coordinates.sdds",MODE="coordinates"
BL: LINE=(D,C,P,S,W)
"""


def energy(ps, mass, tabulated=False):
    """Mean K in MeV from E² − (pc)² = (mc²)², evaluated at 70-digit precision."""
    with localcontext() as ctx:
        ctx.prec = 70
        rest = ME_MEV if tabulated else mass * C**2 / E / Decimal(10)**6
        # SDDS p = beta*gamma, so gamma = sqrt(1+p²); average energies, not momenta.
        return float(sum(((1 + Decimal(str(p))**2).sqrt() - 1) * rest for p in ps) / len(ps))


def check(mode, label, species, mass, charge=1, tabulated=False):
    """Check both energy modes, ID selection, momenta and exact-drift flight times."""
    commands = f"""&change_particle name={species} &end
&run_setup lattice=lattice.lte,use_beamline=BL,p_central=2,output=out.sdds,random_number_seed=24681 &end
&run_control n_passes=1 &end
&sdds_beam input=beam.sdds,use_bunched_mode=1 &end
&track &end
"""
    result = run(
        {"lattice.lte": LATTICE, "run.ele": commands}, mode,
        {"beam.sdds": {"p": np.array(PS), "particleID": np.arange(1, 21)}},
        ("centroids.sdds", "parameters.sdds", "selected.sdds", "coordinates.sdds", "out.sdds"),
    )
    defects = []
    values = []
    for name, ps in (("centroids.sdds", PS), ("parameters.sdds", PS), ("selected.sdds", PS[:7])):
        data = result[name]
        actual = scalar(data, "KAverage", column=True)
        expected = energy(ps, mass, tabulated)
        schema = next(c for c in data.columns if c.name == "KAverage")
        if schema.nm["units"] != "MeV" or schema.nm["description"] != "Mean kinetic energy":
            raise RuntimeError("The KAverage schema changed")
        p_mean = scalar(data, "pAverage", column=True)
        if not math.isclose(p_mean, math.fsum(ps) / len(ps), rel_tol=5e-15, abs_tol=0):
            raise RuntimeError("Mean-momentum control failed")
        affected = (mode == "mpi" and not tabulated) or (mode == "serial" and charge != 1)
        if affected:
            faulty = energy(ps, ME, True) if mode == "mpi" else expected / charge
            defects.append(conversion_defect(actual, expected, faulty, rtol=RTOL))
        elif not math.isclose(actual, expected, rel_tol=RTOL, abs_tol=0):
            raise RuntimeError("Species energy control failed")
        values.append(actual)
    with localcontext() as ctx:
        ctx.prec = 70
        # Zero slopes and zero initial time give t = L*gamma/(p*c).
        times = np.array([float(LENGTH * (1 + Decimal(str(p))**2).sqrt()
                                / (Decimal(str(p)) * C)) for p in PS])
    max_time_error = 0.0
    for name in ("coordinates.sdds", "out.sdds"):
        data = particles(result[name])
        if not np.array_equal(data["particleID"], np.arange(1, 21)):
            raise RuntimeError("Particle-ID control failed")
        if not np.array_equal(data["p"], PS):
            raise RuntimeError("Particle-momentum control failed")
        if not np.allclose(data["t"], times, rtol=5e-15, atol=0):
            raise RuntimeError("Exact-drift time control failed")
        max_time_error = max(max_time_error, float(np.max(np.abs(data["t"] / times - 1))))
    print(
        f"{mode}/{label}: KAverage_MeV centroids={values[0]:.17g} parameters={values[1]:.17g} "
        f"expected={energy(PS, mass, tabulated):.17g}; IDs1-7={values[2]:.17g} "
        f"expected={energy(PS[:7], mass, tabulated):.17g} (mean((sqrt(1+p²)-1)*mc²)); "
        f"IDs,p exact; max drift relative error={max_time_error:.3g}"
    )
    return defects
