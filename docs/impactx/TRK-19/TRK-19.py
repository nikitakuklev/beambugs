"""TRK-19: `ExactCFbend` reverses the skew-decapole horizontal kick

`ExactCFbend` applies the skew-decapole (`k_skew[4]`) horizontal kick with the wrong sign: it is the
negative of the derivative of the documented vector potential (Zolkin 2017, Table XI), while the
vertical kick is correct. The map is therefore not symplectic, and an independent integration of the
curvilinear Hamiltonian reproduces ImpactX to 1.3e-14 only with this sign reversed. For a proton at
γ = 3.25 in a 0.15 m bend with curvature 2 m⁻¹ and skew strength 160 m⁻⁵, a particle at x = 0.08 m,
y = −0.05 m exits with `px` off by 1.321e-4, and the Jacobian M violates the symplectic condition
MᵀJM = J by 6.795e-4. Spin tracking (experimental in ImpactX) uses the same wrong field, with
spin-component errors up to 7.899e-4. The error grows with strength and amplitude.

The script tracks 25 protons, a centre point and a fourth-order finite-difference stencil around it,
through the bend with integrator orders 2, 4 and 6, and measures the full-element Jacobian M. It
checks MᵀJM = J and compares the centre exit with a DOP853 integration of the curvilinear
Hamiltonian, with the correct and with the reversed horizontal kick; it also verifies symbolically
that the code's kicks are the negative radial and the positive vertical derivative of the documented
potential. Pure-bend, normal-decapole, skew-octupole and skew-sextupole cases are controls and must
be symplectic.
"""

import os

for name in [
    "OMP_NUM_THREADS",
    "OPENBLAS_NUM_THREADS",
    "MKL_NUM_THREADS",
    "NUMEXPR_NUM_THREADS",
    "NPROC",
    "PJRT_NPROC",
]:
    os.environ[name] = "1"
os.environ["PYTHONDONTWRITEBYTECODE"] = "1"
os.sched_setaffinity(0, {min(os.sched_getaffinity(0))})

import importlib.metadata
import sys
from typing import Any

import numpy as np
from impactx import Config, ImpactX, ImpactXParIter

PHASE = ["position_x", "momentum_x", "position_y", "momentum_y", "position_t", "momentum_t"]
SPIN = ["spin_x", "spin_y", "spin_z"]


class Beam:
    def __init__(self, count):
        config = Config.to_dict()
        assert not any(config[k] for k in ("have_mpi", "have_gpu", "have_omp"))
        assert config["precision"] == config["precision_particles"] == "DOUBLE"
        self.count = count
        self.sim = ImpactX()
        self.sim.tiny_profiler = False
        self.sim.memory_profiler = False
        self.sim.verbose = 0
        self.sim.particle_shape = 2
        self.sim.space_charge = "off"
        self.sim.csr = False
        self.sim.isr = False
        self.sim.isr_on_ref_part = False
        self.sim.diagnostics = False
        self.sim.slice_step_diagnostics = False
        self.sim.spin = False
        self.sim.init_grids()
        self.ref = self.sim.beam.ref
        self.ref.set_species("proton").set_kin_energy_MeV(self.ref.mass_MeV)
        zeros = np.zeros(count)
        self.sim.beam.add_n_particles(
            zeros,
            zeros,
            np.linspace(-0.001, 0.001, count),
            zeros,
            zeros,
            zeros,
            qm=self.ref.qm_ratio_SI,
            w=np.ones(count),
        )

    def views(self):
        # A particle iterator is a cursor. Access its tile before advancing it.
        real = None
        for pti in ImpactXParIter(self.sim.beam, level=0):
            real = pti.soa().to_numpy(copy=False).real
            assert len(real[PHASE[0]]) == self.count
        if real is None:
            raise RuntimeError("Beam unexpectedly has no tile")
        return real

    def run(
        self, element, phase, spins=None, species="proton", gamma=2.0, anomaly=None, spin=False
    ):
        self.ref.reset()
        self.ref.set_species(species)
        self.ref.set_kin_energy_MeV((gamma - 1) * self.ref.mass_MeV)
        if anomaly is not None:
            self.ref.set_gyromagnetic_anomaly(anomaly)
        self.ref.x = self.ref.y = self.ref.z = self.ref.t = self.ref.s = self.ref.sedge = 0.0
        arrays = self.views()
        arrays["qm"][:] = self.ref.qm_ratio_SI
        for i, key in enumerate(PHASE):
            arrays[key][:] = phase[:, i]
        if spins is not None:
            for i, key in enumerate(SPIN):
                arrays[key][:] = spins[:, i]
        self.sim.spin = spin
        self.sim.lattice.clear()
        self.sim.lattice.append(element)
        reference_pt = self.ref.pt
        self.sim.track_particles()
        assert self.ref.pt == reference_pt
        arrays = self.views()
        orbital = np.column_stack([arrays[key].copy() for key in PHASE])
        spins_out = np.column_stack([arrays[key].copy() for key in SPIN])
        return orbital, spins_out

    def close(self):
        self.sim.finalize()


def environment():
    from amrex import space3d as amr

    print(
        f"Python {sys.version.split()[0]} numpy={np.__version__} "
        f"scipy={importlib.metadata.version('scipy')} "
        f"sympy={importlib.metadata.version('sympy')} "
        f"impactx-noacc={importlib.metadata.version('impactx-noacc')} "
        f"pyamrex={amr.__version__}",
        flush=True,
    )
    print("build", Config.to_dict(), flush=True)


import sympy as sp
from impactx import elements
from scipy.integrate import solve_ivp

r, v = sp.symbols("r v", positive=True)
# Zolkin (2017), Table XI: rho*A_theta for the skew S-decapole,
# divided by its coefficient. v denotes the dimensionless vertical coordinate.
potential = (
    v
    / 120
    * (
        -sp.Rational(15, 8) * (5 * r**4 - 4 * r**2 - 1)
        + 5 * (r**2 - 1) * v**2
        + v**4
        + 5 * (3 + sp.Rational(3, 2) * r**2 - 2 * v**2) * r**2 * sp.log(r)
    )
)
fx_expr = sp.diff(potential, r)
fy_expr = sp.diff(potential, v)
fx = sp.lambdify((r, v), fx_expr, "numpy")
fy = sp.lambdify((r, v), fy_expr, "numpy")
# Independently transcribed only for the exact-algebra source comparison.
source_x = v * r / 12 * (3 * (-1 + r * r) + (2 * v * v - 3 * (1 + r * r)) * sp.log(r))
source_y = (
    3
    + 8 * v**4
    + 12 * r * r
    - 15 * r**4
    + 24 * v * v * (-1 + r * r)
    + 12 * r * r * (2 - 4 * v * v + r * r) * sp.log(r)
) / 192
assert sp.simplify(source_x + fx_expr) == 0
assert sp.simplify(source_y - fy_expr) == 0
curl = sp.factor(sp.diff(source_x, v) - sp.diff(source_y, r))
assert sp.simplify(curl + 2 * sp.diff(fx_expr, v)) == 0


def oracle(initial, length, h, strength, gamma, source_sign=False, tight=False):
    # H = -rho*pz + (rho^2-1)/2 - k*P(rho,h*y)/h^5 - pt/beta.
    # Its gradients include the reference-time subtraction and constant rigidity.
    beta = np.sqrt(1 - gamma**-2)

    def rhs(_s, q):
        x, px, y, py, _t, pt = q
        rho, vert = 1 + h * x, h * y
        pz = np.sqrt(1 - 2 * pt / beta + pt * pt - px * px - py * py)
        kick_x = strength / h**4 * fx(rho, vert)
        kick_y = strength / h**4 * fy(rho, vert)
        if source_sign:
            kick_x = -kick_x
        return [
            rho * px / pz,
            h * (pz - rho) + kick_x,
            rho * py / pz,
            kick_y,
            rho * (1 / beta - pt) / pz - 1 / beta,
            0.0,
        ]

    sol = solve_ivp(
        rhs,
        (0, length),
        initial,
        method="DOP853",
        rtol=3e-14 if tight else 2e-12,
        atol=2e-16 if tight else 2e-14,
    )
    assert sol.success, sol.message
    return sol.y[:, -1]


def stencil(center, spacing):
    return np.vstack(
        [center]
        + [
            center + multiple * spacing * np.eye(6)[axis]
            for axis in range(6)
            for multiple in (-2, -1, 1, 2)
        ]
    )


def jacobian(output, spacing):
    return np.column_stack(
        [
            (output[1 + 4 * i] - 8 * output[2 + 4 * i] + 8 * output[3 + 4 * i] - output[4 + 4 * i])
            / (12 * spacing)
            for i in range(6)
        ]
    )


def main():
    environment()
    print("Table-XI potential:", potential)
    print("source_x = -dP/dr; source_y = dP/dv: EXACT")
    print("dimensionless source curl:", curl)
    print(
        "isolated-kick curl at x=0.12, y=0.08, h=1, k4=50, L=0.2:",
        (10 * curl.subs({r: sp.Rational(28, 25), v: sp.Rational(2, 25)})).evalf(45),
    )
    x, y, h = sp.symbols("x y h", real=True)
    print("source Fx leading h->0:", sp.limit(source_x.subs({r: 1 + h * x, v: h * y}) / h**4, h, 0))
    print("oracle Fx leading h->0:", sp.limit(fx_expr.subs({r: 1 + h * x, v: h * y}) / h**4, h, 0))
    center = np.array([0.08, 0.015, -0.05, 0.004, 0.0007, -0.002])
    gamma, curvature, strength, length = 3.25, 2.0, 160.0, 0.15
    form = np.kron(np.eye(3), np.array([[0.0, 1.0], [-1.0, 0.0]]))
    records: list[dict[str, Any]] = []
    beam = Beam(25)
    try:
        cases = [
            ("skew2", 2, 8, 4, strength, curvature, length, center),
            ("skew4", 4, 8, 4, strength, curvature, length, center),
            ("skew6", 6, 8, 4, strength, curvature, length, center),
            ("skew6_fine", 6, 32, 4, strength, curvature, length, center),
            ("half_strength", 6, 16, 4, strength / 2, curvature, length, center),
            ("negative_strength", 6, 16, 4, -strength, curvature, length, center),
            ("half_length", 6, 16, 4, strength, curvature, length / 2, center),
            ("reverse_curvature", 6, 16, 4, strength, -curvature, length, center),
            (
                "positive_y",
                6,
                16,
                4,
                strength,
                curvature,
                length,
                center * np.array([1, 1, -1, -1, 1, 1]),
            ),
            ("pure_bend", 6, 16, 4, 0.0, curvature, length, center),
            ("normal_decapole", 6, 16, 4, strength, curvature, length, center),
            ("skew_octupole", 6, 16, 3, 10.0, curvature, length, center),
            ("skew_sextupole", 6, 16, 2, 2.0, curvature, length, center),
        ]
        for name, order, steps, multipole_index, k, hh, ll, initial in cases:
            for spacing in [2e-4, 6e-5, 2e-5] if name == "skew6_fine" else [6e-5]:
                normal, skew = [hh, 0.0, 0.0, 0.0, 0.0], [0.0] * 5
                if name == "normal_decapole":
                    normal[multipole_index] = k
                else:
                    skew[multipole_index] = k
                element = elements.ExactCFbend(
                    ds=ll, k_normal=normal, k_skew=skew, int_order=order, mapsteps=steps
                )
                q, _ = beam.run(element, stencil(initial, spacing), gamma=gamma)
                matrix = jacobian(q, spacing)
                residue = matrix.T @ form @ matrix - form
                rec = dict(
                    case=name,
                    spacing=spacing,
                    order=order,
                    steps=steps,
                    k=k,
                    h=hh,
                    length=ll,
                    symplectic_error=float(np.max(np.abs(residue))),
                    initial=initial.tolist(),
                    final=q[0].tolist(),
                )
                assert np.isfinite(q).all()
                if multipole_index == 4 and name != "normal_decapole":
                    correct = oracle(initial, ll, hh, k, gamma)
                    faulty = oracle(initial, ll, hh, k, gamma, source_sign=True)
                    refined = oracle(initial, ll, hh, k, gamma, tight=True)
                    rec.update(
                        oracle_error=float(np.max(np.abs(q[0] - correct))),
                        oracle_px_error=float(q[0, 1] - correct[1]),
                        source_ode_error=float(np.max(np.abs(q[0] - faulty))),
                        oracle_refinement=float(np.max(np.abs(correct - refined))),
                    )
                records.append(rec)
                print(
                    f"{name} spacing={spacing:g} order={order} steps={steps} "
                    f"symplectic_error={rec['symplectic_error']:.12g} "
                    + (
                        f"oracle_error={rec['oracle_error']:.12g} "
                        f"px_error={rec['oracle_px_error']:.12g} "
                        f"source_ode_error={rec['source_ode_error']:.12g} "
                        f"oracle_refinement={rec['oracle_refinement']:.12g}"
                        if "oracle_error" in rec
                        else ""
                    ),
                    flush=True,
                )
    finally:
        beam.close()
    refined = [row for row in records if row["case"] == "skew6_fine"]
    for row in refined:
        assert 6.7e-4 < row["symplectic_error"] < 6.9e-4
        assert 1.32e-4 < row["oracle_px_error"] < 1.33e-4
        assert row["source_ode_error"] < 5e-13
        assert row["oracle_refinement"] < 5e-13
    assert np.ptp([row["symplectic_error"] for row in refined]) < 2e-10
    for row in records[-4:]:
        assert row["symplectic_error"] < 1e-9
    assert records[-4]["oracle_error"] < 5e-13
    print("DEFECT REPRODUCED", flush=True)


if __name__ == "__main__":
    main()
