"""TRK-10: Ring RF nodes raise the reference energy without rescaling z, so an accelerating ring shrinks the longitudinal emittance by β₀,initial/β₀,final

PyORBIT's longitudinal coordinate is z = −β₀cΔt, with β₀ the reference speed and Δt a particle's arrival time after the reference particle. The ring RF nodes that raise the reference energy (`RingRFTEAPOT`, the `orbit.rf_cavities` harmonic nodes driven by a synchronous phase or by a rigidity ramp, and `Frequency_RFNode`) change β₀ but leave z unchanged, so every particle's Δt is multiplied by β₀,in/β₀,out at each pass; the linac RF gaps apply the required z → z·β₀,out/β₀,in. Over a ramp the (z, dE) area is conserved instead of the (Δt, dE) area: accelerating protons from 50 to 200 MeV in a 100 m ring shrinks the rms (Δt, dE) emittance to 0.5547 of its initial value, β₀,initial/β₀,final, where a lab-frame tracker keeps it at 1.000.

The script first passes one particle 0.1 m ahead of a 0.4 GeV proton reference through each reference-raising ring path and through the linac `BaseRfGap` (control) and compares its arrival time before and after the event, which must be continuous. It then ramps 64 particles on a small matched ellipse for 3000 turns through a 100 m drift ring with `SyncPhaseDep_Harmonic_RFNode` and compares the rms (Δt, dE) emittance with an independent tracker written in (Δt, dE); the same ring with a stationary bucket is the control.
"""

import math
import os

for _v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ.setdefault(_v, "1")

from orbit.core import teapot_base as TPB
from orbit.core.bunch import Bunch
from orbit.core.linac import BaseRfGap
from orbit.rf_cavities import RFNode

C_LIGHT = 2.99792458e8
MP = 0.938272013  # PyORBIT default proton mass, GeV
CIRC = 100.0  # ring length, m


def new_bunch(kin, parts):
    b = Bunch()
    b.mass(MP)
    b.charge(1.0)
    b.getSyncParticle().kinEnergy(kin)
    for p in parts:
        b.addParticle(*p)
    return b


def arrival_time(b, i):
    # z = -beta0 c dt, so dt = -z/(beta0 c) with the beta0 in force now; dt > 0 means later than the reference
    return -b.z(i) / (b.getSyncParticle().beta() * C_LIGHT)


def sync_phase_dict(v, phi_deg):
    # constant voltage and RF phase; SyncPhase = 0, so the reference gains q V sin(phi_deg) per pass
    return {
        "gammaTrans": 1.0e10,
        "RFHNum": 1,
        "n_tuple": 1,
        "time": (0.0, 1.0),
        "SyncPhase": (0.0, 0.0),
        "RFVoltage": (v, v),
        "RFPhase": (phi_deg, phi_deg),
    }


# ---------------------------------------------------------------- single events
K0, Z0 = 0.4, 0.1  # GeV, m
V, PHI = 1.0e-3, 30.0  # GV, deg: reference gain q V sin(30 deg) = 0.5 MeV
defects, controls_ok = 0, True


def event(label, apply, control=False):
    global defects, controls_ok
    b = new_bunch(K0, [(0.0, 0.0, 0.0, 0.0, Z0, 0.0)])
    b_in, t_in = b.getSyncParticle().beta(), arrival_time(b, 0)
    apply(b)
    b_out, t_out = b.getSyncParticle().beta(), arrival_time(b, 0)
    # A reference change at a fixed event does not move the particle in time: dt is continuous, which needs
    # z_out = z_in * beta_out/beta_in.
    off = abs(t_out - t_in) > 1e-12 * abs(t_in)
    if control:
        controls_ok &= not off
    else:
        defects += off
    print(
        f"[{label}] beta0 {b_in:.10f} -> {b_out:.10f}; z observed {b.z(0):.12e} m, expected z_in*beta_out/beta_in "
        f"{Z0 * b_out / b_in:.12e} m; dt observed {t_out:.9e} s, expected {t_in:.9e} s (continuous); "
        f"relative change {(t_out - t_in) / t_in:+.3e}{' (control)' if control else ''}"
    )


def sync_phase_node(b):
    # the constructor raises the reference of the bunch it is given once, so it gets a scratch bunch
    node = RFNode.SyncPhaseDep_Harmonic_RFNode(
        2 * math.pi / CIRC, sync_phase_dict(V, PHI), new_bunch(K0, []), 0.0, "rf"
    )
    node.trackBunch(b)


def brho_node(b):
    # rigidity 3.19 T m, slightly above the 0.4 GeV reference's 3.1795 T m: the node sets the reference to it
    acc = {
        "gammaTrans": 1.0e10,
        "RFHNum": 1,
        "n_tuple": 1,
        "time": (0.0, 1.0),
        "BRho": (3.19, 3.19),
        "RFVoltage": (V, V),
        "RFPhase": (0.0, 0.0),
    }
    RFNode.BRhoDep_Harmonic_RFNode(
        2 * math.pi / CIRC, acc, new_bunch(K0, []), 0.0, "rf"
    ).trackBunch(b)


event("TPB.RingRF (RingRFTEAPOT), h=1", lambda b: TPB.RingRF(b, CIRC, 1, V, math.radians(PHI), 1))
event("SyncPhaseDep_Harmonic_RFNode", sync_phase_node)
event("BRhoDep_Harmonic_RFNode", brho_node)
event(
    "Frequency_RFNode 1 MHz",
    lambda b: RFNode.Frequency_RFNode(1.0e6, V, -60.0, 0.0, "f").trackBunch(b),
)
event(
    "linac BaseRfGap 402.5 MHz",
    lambda b: BaseRfGap().trackBunch(b, 402.5e6, V, math.radians(-60.0)),
    control=True,
)

# ---------------------------------------------------------------- accelerating ramp
KI, NTURN, VR, AZ, NP = 0.05, 3000, 2.0e-4, 1.0, 64
PHIR = math.degrees(math.asin(0.25))  # 50 keV per turn: 50 -> 200 MeV in 3000 turns


def ring_ensemble(kin):
    # 64 points on the matched small-amplitude ellipse of the linear map z += a dE, dE -= b z
    sp = new_bunch(kin, []).getSyncParticle()
    a = CIRC / (sp.momentum() * sp.beta() * sp.gamma() ** 2)
    b = VR * 2 * math.pi / CIRC * math.cos(math.radians(PHIR))
    ae = AZ * math.sqrt(b / a)
    return [
        (AZ * math.cos(2 * math.pi * k / NP), ae * math.sin(2 * math.pi * k / NP))
        for k in range(NP)
    ]


def emittance(t, e):
    n = len(t)
    mt, me = sum(t) / n, sum(e) / n
    ctt = sum((x - mt) ** 2 for x in t) / n
    cee = sum((x - me) ** 2 for x in e) / n
    cte = sum((x - mt) * (y - me) for x, y in zip(t, e)) / n
    return math.sqrt(ctt * cee - cte * cte)


def pyorbit_ramp(phi_deg):
    b = new_bunch(KI, [(0.0, 0.0, 0.0, 0.0, z, de) for z, de in ring_ensemble(KI)])
    node = RFNode.SyncPhaseDep_Harmonic_RFNode(
        2 * math.pi / CIRC, sync_phase_dict(VR, phi_deg), new_bunch(KI, []), 0.0, "rf"
    )
    eps0 = emittance([arrival_time(b, i) for i in range(NP)], [b.dE(i) for i in range(NP)])
    for _ in range(NTURN):
        TPB.drift(b, CIRC)
        node.trackBunch(b)
    eps1 = emittance([arrival_time(b, i) for i in range(NP)], [b.dE(i) for i in range(NP)])
    return eps1 / eps0, b.getSyncParticle().kinEnergy(), b.getSyncParticle().beta()


def kin2beta(k):
    return math.sqrt(k * (k + 2 * MP)) / (k + MP)


def independent_ramp(phi_deg):
    # Tracker in (dt, dE) with no z: exact flight-time difference over the drift ring,
    # dt += (C/c)(1/beta(K0 + dE) - 1/beta0), then the kick of Harmonic_Cav's phase law
    # q V (sin(h omega0 dt + phi) - sin phi), omega0 = 2 pi beta0 c/C, and the reference gain q V sin phi.
    k0 = KI
    beta0 = kin2beta(k0)
    pts = ring_ensemble(KI)
    t = [-z / (beta0 * C_LIGHT) for z, _ in pts]
    e = [de for _, de in pts]
    eps0 = emittance(t, e)
    s = math.sin(math.radians(phi_deg))
    for _ in range(NTURN):
        beta0 = kin2beta(k0)
        t = [ti + CIRC / C_LIGHT * (1.0 / kin2beta(k0 + ei) - 1.0 / beta0) for ti, ei in zip(t, e)]
        om = 2 * math.pi * beta0 * C_LIGHT / CIRC
        e = [ei + VR * (math.sin(om * ti + math.radians(phi_deg)) - s) for ti, ei in zip(t, e)]
        k0 += VR * s
    return emittance(t, e) / eps0, k0


for phi, control in ((PHIR, False), (0.0, True)):
    rp, kf, bf = pyorbit_ramp(phi)
    rc, kfc = independent_ramp(phi)
    bi = kin2beta(KI)
    print(
        f"[ramp phi_s={phi:.4f} deg, {NTURN} turns{', control' if control else ''}] K {KI:.3f} -> {kf:.6f} GeV "
        f"(independent {kfc:.6f}); rms (dt, dE) emittance final/initial observed {rp:.6f}, expected {rc:.6f} "
        f"(independent (dt, dE) tracker); beta0_i/beta0_f = {bi / bf:.6f}"
    )
    if control:
        controls_ok &= abs(rp - rc) < 1e-4
    else:
        defects += abs(rp - rc) > 1e-2

print("DEFECT REPRODUCED" if defects == 5 and controls_ok else "NOT REPRODUCED")
