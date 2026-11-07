"""Analytic reference-change fixtures for transfer, covariance and W checks.

The reference species is a proton, with canonical momenta normalized to local
p0c. Conventional updates preserve zeta/beta0, rather than absolute time.
"""

from _prebuilt import CTX, np, xt

P0 = 650e6
MASS = xt.PROTON_MASS_EV
W0 = np.eye(6)
for j, beta, alpha in ((0, 2.7, -0.4), (2, 1.6, 0.2), (4, 3.2, 0.0)):
    W0[j : j + 2, j : j + 2] = [[np.sqrt(beta), 0], [-alpha / np.sqrt(beta), 1 / np.sqrt(beta)]]
NEMITT = np.array([1.7e-6, 0.9e-6])
# Canonical geometric emittance is epsilon_n * mass0 / p0c in eV units.
SIGMA0 = W0[:4, :4] @ np.diag(np.repeat(NEMITT * MASS / P0, 2)) @ W0[:4, :4].T


def cases():
    """Yield native changes and their exact canonical linear transport."""
    energy = np.hypot(P0, MASS)
    for kind in ("absolute", "increment", "segment", "accelerating_segment", "cavity"):
        for ratio in (1.0, 1.8, 0.8):
            if kind == "cavity" and ratio != 1:
                continue
            p = xt.Particles(_context=CTX, p0c=P0, mass0=MASS, zeta=0.013)
            if kind == "absolute":
                element = xt.ReferenceEnergyChange(p0c=P0 * ratio)
            elif kind == "increment":
                element = xt.ReferenceEnergyIncrease(Delta_p0c=P0 * (ratio - 1))
            elif kind in ("segment", "accelerating_segment"):
                gain = np.hypot(P0 * ratio, MASS) - energy
                element = xt.LineSegmentMap(
                    energy_ref_increment=gain,
                    energy_increment=gain if kind == "accelerating_segment" else 0,
                )
            else:
                element = xt.Cavity(voltage=100e6, frequency=0, phase=np.pi / 2)
            line = xt.Line(elements={"change": element, "end": xt.Marker()})
            line.particle_ref = p.copy()
            line.build_tracker(_context=CTX)
            t = (P0 * ratio / np.hypot(P0 * ratio, MASS)) / (P0 / energy)
            # Pure rebasing conserves physical energy and transverse momentum.
            expected = np.diag([1.0, 1 / ratio, 1.0, 1 / ratio, t, 1 / (ratio * t)])
            if kind in ("segment", "accelerating_segment"):
                # LineSegmentMap smooths the transverse normalization equally
                # between position and momentum, unlike a standalone rebase.
                expected[:4, :4] = np.eye(4) / np.sqrt(ratio)
            yield kind, ratio, t, line, p, expected


def twiss_modes(kind):
    # Ordinary cavities suppress the physical energy kick in 4D Twiss.
    return ("6d",) if kind == "cavity" else ("4d", "6d")


def twiss(line, p, mode):
    init = xt.TwissInit(particle_on_co=p.copy(), W_matrix=W0.copy(), element_name="change")
    return line.twiss(init=init, method=mode, chrom=False)
