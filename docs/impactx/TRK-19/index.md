---
title: "TRK-19: ExactCFbend reverses the skew-decapole horizontal kick"
hide:
  - toc
---

# TRK-19 — `ExactCFbend` reverses the skew-decapole horizontal kick

| | |
|---|---|
| Code | [ImpactX](../index.md) · Tracking |
| Priority | P3 (spin P4) |
| Severity | Wrong physics |
| Evidence | Reproduced, independently rechecked |
| Last reproduced | 2026-11-05 on ImpactX commit [`f9c2a82`](https://github.com/BLAST-ImpactX/impactx/commit/f9c2a82e24b7), committed 2026-11-05 |
| Tags | <span class="bb-tag">experimental</span> [?](../../methodology.md#tags) |
| Source | [`ExactCFbend.H#L379-L389`](https://github.com/BLAST-ImpactX/impactx/blob/b701a6e9d1f9e30cefb9a3e18e82ae7e02b9d943/src/elements/ExactCFbend.H#L379-L389)<br>[manual](https://github.com/BLAST-ImpactX/impactx/blob/615921c6867c895cf3a8fcf1d12991079d0f7d75/docs/source/usage/python.rst#L2073-L2099)<br>[`ExactCFbend.H#L506-L538`](https://github.com/BLAST-ImpactX/impactx/blob/615921c6867c895cf3a8fcf1d12991079d0f7d75/src/elements/ExactCFbend.H#L506-L538)<br>[spin contract](https://github.com/BLAST-ImpactX/impactx/blob/615921c6867c895cf3a8fcf1d12991079d0f7d75/docs/source/usage/parameters.rst#L2792-L2800) |

`ExactCFbend` applies the skew-decapole (`k_skew[4]`) horizontal kick with the wrong sign: it is the negative of the derivative of the documented vector potential (Zolkin 2017, Table XI), while the vertical kick is correct. The map is therefore not symplectic, and an independent integration of the curvilinear Hamiltonian reproduces ImpactX to 1.3e-14 only with the sign reversed. For a proton at γ = 3.25 in a 0.15 m bend with curvature 2 m⁻¹ and skew strength 160 m⁻⁵, a particle at x = 0.08 m, y = −0.05 m exits with `px` off by 1.321e-4, and the Jacobian M violates MᵀJM = J by 6.795e-4. Spin tracking (experimental) uses the same wrong field, with spin errors up to 7.899e-4.

Run `python TRK-19.py`.

??? example "Reproducer: TRK-19.py"

    [Download TRK-19.py](TRK-19.py)

    ```python
    """TRK-19: `ExactCFbend` reverses the skew-decapole horizontal kick

    `ExactCFbend` applies the skew-decapole (`k_skew[4]`) horizontal kick with the wrong sign: it is the
    negative of the derivative of the documented vector potential (Zolkin 2017, Table XI), while the
    vertical kick is correct. The map is therefore not symplectic, and an independent integration of the
    curvilinear Hamiltonian reproduces ImpactX to 1.3e-14 only with the sign reversed. For a proton at γ
    = 3.25 in a 0.15 m bend with curvature 2 m⁻¹ and skew strength 160 m⁻⁵, a particle at x = 0.08 m, y
    = −0.05 m exits with `px` off by 1.321e-4, and the Jacobian M violates MᵀJM = J by 6.795e-4. Spin
    tracking (experimental) uses the same wrong field, with spin errors up to 7.899e-4.

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

    ```

??? example "Reproducer: TRK-19-spin.py"

    [Download TRK-19-spin.py](TRK-19-spin.py)

    ```python
    """TRK-19: the same skew-decapole field error in spin tracking

    With spin tracking enabled, the script tracks 12 protons that share the principal fixture's orbit
    and carry the three Cartesian unit spins through the same `ExactCFbend`, in 20 cases that each run
    in a fresh single-threaded child process: coarser and finer steps, slicing, integrator orders 2 and
    4, and a second fixture with negative curvature whose strength, length, energy, anomaly, field unit
    and species are varied, plus a pure dipole, an on-axis particle and spin tracking disabled. It
    compares the exit spins with a Cartesian Lorentz and Thomas–BMT integration to the exit plane in
    three field models: the physical field reconstructed from Maxwell's equations, the physical field
    along the implemented orbit, and the implemented field, which must reproduce ImpactX. A moving-frame
    integration of the documented potential cross-checks the physical result, and the pure-dipole and
    on-axis cases must agree with it.
    """

    import argparse
    import importlib.metadata
    import json
    import math
    import os
    import subprocess
    import sys
    from pathlib import Path
    from typing import Any

    for key in (
        "NPROC",
        "PJRT_NPROC",
        "OMP_NUM_THREADS",
        "OPENBLAS_NUM_THREADS",
        "MKL_NUM_THREADS",
        "NUMEXPR_NUM_THREADS",
    ):
        os.environ[key] = "1"
    os.environ["PYTHONDONTWRITEBYTECODE"] = "1"
    os.sched_setaffinity(0, {min(os.sched_getaffinity(0))})

    import numpy as np
    import sympy as sp
    from scipy.integrate import solve_ivp

    BASE: dict[str, Any] = dict(
        gamma=3.25,
        h=2.0,
        k=160.0,
        length=0.15,
        state=[0.08, 0.015, -0.05, 0.004, 0.0007, -0.002],
        steps=64,
        order=6,
        slices=1,
        unit=0,
        species="proton",
        anomaly=None,
        spin=True,
    )
    OTHER: dict[str, Any] = dict(
        gamma=2.4,
        h=-1.6,
        k=320.0,
        length=0.21,
        state=[-0.065, -0.023, 0.043, -0.011, 0.0014, 0.018],
    )
    CASES: dict[str, dict[str, Any]] = {
        "anchor": {},
        "anchor_coarse": dict(steps=8),
        "anchor_fine": dict(steps=128),
        "anchor_sliced": dict(steps=16, slices=4),
        "order2": dict(order=2, steps=256),
        "order4": dict(order=4, steps=64),
        "other": OTHER,
        "other_half_k": dict(OTHER, k=160.0),
        "other_negative_k": dict(OTHER, k=-320.0),
        "other_half_length": dict(OTHER, length=0.105),
        "other_gamma": dict(OTHER, gamma=5.2),
        "other_negative_anomaly": dict(OTHER, anomaly=-0.7),
        "other_tiny_anomaly": dict(OTHER, anomaly=1e-9),
        "other_unit1": dict(OTHER, unit=1),
        "electron_unit1": dict(OTHER, species="electron", unit=1),
        "other_spin_off": dict(OTHER, spin=False),
        "dipole": dict(OTHER, k=0.0),
        "axis": dict(OTHER, state=[0.0, 0.0, 0.0, 0.0, 0.0014, 0.0]),
        "small_length": dict(OTHER, length=0.001),
        "small_length_half": dict(OTHER, length=0.0005),
    }


    def field_functions():
        """Construct a vacuum transverse field from its midplane radial component.

        The midplane datum follows by differentiating rho*A_theta in Zolkin,
        Table XI, skew n=5, with respect to its dimensionless vertical argument.
        Maxwell divergence and curl then determine the off-midplane coefficients.
        This construction supplies the vertical field independently of map2/map3.
        """
        r, v = sp.symbols("r v", positive=True)
        a0 = (
            -3 * (5 * r**4 - 4 * r**2 - 1) / 8 + sp.Rational(3, 2) * (2 + r**2) * r**2 * sp.log(r)
        ) / (24 * r)
        b1 = sp.simplify(-sp.diff(a0, r) - a0 / r)
        a2 = sp.simplify(sp.diff(b1, r) / 2)
        b3 = sp.simplify((-sp.diff(a2, r) - a2 / r) / 3)
        a4 = sp.simplify(sp.diff(b3, r) / 4)
        assert sp.simplify(sp.diff(a4, r) + a4 / r) == 0
        bx = a0 + a2 * v**2 + a4 * v**4
        by = b1 * v + b3 * v**3
        source_bx = (
            3
            + 8 * v**4
            + 12 * r**2
            - 15 * r**4
            + 24 * v**2 * (-1 + r**2)
            + 12 * r**2 * (2 - 4 * v**2 + r**2) * sp.log(r)
        ) / (192 * r)
        source_by = -v * (3 * (-1 + r**2) + (2 * v**2 - 3 * (1 + r**2)) * sp.log(r)) / 12
        assert sp.simplify(bx - source_bx) == 0
        assert sp.simplify(by + source_by) == 0
        assert sp.simplify(sp.diff(bx, r) + bx / r + sp.diff(by, v)) == 0
        assert sp.simplify(sp.diff(by, r) - sp.diff(bx, v)) == 0
        # Cross-check the independently reconstructed field against the full Table XI.
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
        assert sp.simplify(bx - sp.diff(potential, v) / r) == 0
        assert sp.simplify(by + sp.diff(potential, r) / r) == 0
        div_bad = sp.simplify(sp.diff(bx, r) + bx / r + sp.diff(source_by, v))
        evaluate = sp.lambdify((r, v), (bx, by), modules="numpy", cse=True)
        return evaluate, str(div_bad)


    def run_wheel(name):
        # Only this child imports ImpactX; each child owns exactly one simulation.
        import impactx
        from impactx import ImpactX, ImpactXParIter, elements

        case = BASE | CASES[name]
        config = impactx.Config.to_dict()
        assert all(not config[key] for key in ("have_mpi", "have_omp", "have_gpu"))
        assert config["precision"] == config["precision_particles"] == "DOUBLE"
        sim = ImpactX()
        sim.particle_shape = 2
        sim.space_charge = "off"
        sim.csr = False
        sim.isr = False
        sim.spin = case["spin"]
        sim.diagnostics = False
        sim.slice_step_diagnostics = False
        sim.memory_profiler = False
        sim.tiny_profiler = False
        sim.verbose = 0
        try:
            sim.init_grids()
            ref = sim.beam.ref
            ref.set_species(case["species"])
            ref.set_kin_energy_MeV((case["gamma"] - 1) * ref.mass_MeV)
            if case["anomaly"] is not None:
                ref.set_gyromagnetic_anomaly(case["anomaly"])
            case["anomaly"] = ref.gyromagnetic_anomaly
            case["gamma"] = -ref.pt
            n = 12
            initial = np.tile(case["state"], (n, 1))
            initial[:, 4] += np.linspace(-0.0023, 0.0023, n)
            spin = np.tile(np.eye(3), (4, 1))
            sim.beam.add_n_particles(
                *[initial[:, j].copy() for j in [0, 2, 4, 1, 3, 5]],
                qm=ref.qm_ratio_SI,
                w=np.ones(n),
            )
            offset = 0
            for tile in ImpactXParIter(sim.beam, level=0):
                arrays = tile.soa().to_numpy(copy=False).real
                count = len(arrays["position_x"])
                for j, key in enumerate(["spin_x", "spin_y", "spin_z"]):
                    arrays[key][:] = spin[offset : offset + count, j]
                offset += count
            assert offset == n
            scale = ref.rigidity_Tm if case["unit"] == 1 else 1.0
            sim.lattice.append(
                elements.ExactCFbend(
                    ds=case["length"],
                    k_normal=[case["h"] * scale, 0.0, 0.0, 0.0, 0.0],
                    k_skew=[0.0, 0.0, 0.0, 0.0, case["k"] * scale],
                    unit=case["unit"],
                    mapsteps=case["steps"],
                    nslice=case["slices"],
                    int_order=case["order"],
                )
            )
            sim.track_particles()
            # Distinct arrival coordinates retain the input order in this static field.
            frame = sim.beam.to_df(local=True).sort_values("position_t")
            coordinates = frame[
                ["position_x", "momentum_x", "position_y", "momentum_y", "position_t", "momentum_t"]
            ].to_numpy()
            spins = frame[["spin_x", "spin_y", "spin_z"]].to_numpy()
            assert coordinates.shape == (n, 6) and spins.shape == (n, 3)
            assert np.isfinite(coordinates).all() and np.isfinite(spins).all()
            assert ref.gamma == case["gamma"]
            record = dict(
                case=case,
                initial=initial.tolist(),
                initial_spins=spin.tolist(),
                final=coordinates.tolist(),
                spins=spins.tolist(),
                reference_rigidity=ref.rigidity_Tm,
                runtime=dict(
                    python=sys.version.split()[0],
                    impactx=impactx.__version__,
                    wheel=importlib.metadata.version("impactx-noacc"),
                    pyamrex=impactx.amr.__version__,
                    numpy=np.__version__,
                    module=impactx.__file__,
                    config=config,
                    scipy=importlib.metadata.version("scipy"),
                    sympy=sp.__version__,
                ),
            )
        finally:
            sim.finalize()
        print("RESULT " + json.dumps(record))


    def lab_oracle(case, evaluate, model, resolution=1):
        """Integrate in Cartesian lab coordinates, with ct as independent variable.

        Stop at the final sector plane using an azimuth event. Canonical transverse
        momenta equal kinetic ones in this gauge, since only A_s is nonzero.
        The rotation R_y(-h*s) converts the moving basis to fixed Cartesian axes.
        """
        h, k, length = case["h"], case["k"], case["length"]
        x, px, y, py, t, pt = case["state"]
        b0 = math.sqrt(case["gamma"] ** 2 - 1)
        beta0 = b0 / case["gamma"]
        gamma = case["gamma"] - b0 * pt
        pz = math.sqrt((gamma**2 - 1) / b0**2 - px**2 - py**2)
        anomaly = case["anomaly"]
        z0 = np.r_[[x, y, 0.0], b0 * np.array([px, py, pz]), np.eye(3).ravel()]
        min_rho = [math.inf]
        min_ps = [math.inf]

        def geometry(z):
            angle = math.atan2(h * z[2], 1 + h * z[0])
            rho = math.hypot(1 + h * z[0], h * z[2])
            c, sn = math.cos(angle), math.sin(angle)
            rotation = np.array([[c, 0, -sn], [0, 1, 0], [sn, 0, c]])
            return angle, rho, rotation

        def rhs(_ct, z):
            _angle, rho, rotation = geometry(z)
            fx, fy = evaluate(np.longdouble(rho), np.longdouble(h * z[1]))
            skew = k / h**4 * np.array([fx, fy], dtype=float)
            physical = b0 * np.array([skew[0], h + skew[1], 0.0])
            implemented = b0 * np.array([skew[0], h - skew[1], 0.0])
            orbit_b = rotation @ (physical if model == "physical" else implemented)
            spin_b = rotation @ (implemented if model == "implemented" else physical)
            momentum = z[3:6]
            min_rho[0] = min(min_rho[0], rho)
            min_ps[0] = min(min_ps[0], (rotation.T @ momentum)[2] / b0)
            omega = -(anomaly + 1 / gamma) * spin_b + (
                anomaly / (gamma * (gamma + 1)) * np.dot(momentum, spin_b) * momentum
            )
            rotation_rate = np.array(
                [[0, -omega[2], omega[1]], [omega[2], 0, -omega[0]], [-omega[1], omega[0], 0]]
            )
            spin_rate = rotation_rate @ z[6:].reshape(3, 3)
            return np.r_[momentum / gamma, np.cross(momentum / gamma, orbit_b), spin_rate.ravel()]

        class ExitPlane:
            terminal = True
            direction = 1

            def __call__(self, _ct, z):
                return geometry(z)[0] / h - length

        bound = 2 * length / beta0 * (1 + abs(h * x))
        solution = solve_ivp(
            rhs,
            (0, bound),
            z0,
            method="RK45",
            events=ExitPlane(),
            rtol=2e-13,
            atol=2e-15,
            max_step=bound / (800 * resolution),
        )
        assert solution.success and len(solution.t_events[0]) == 1, solution.message
        z = solution.y_events[0][0]
        final_ct = solution.t_events[0][0]
        _angle, rho, rotation = geometry(z)
        local_p = rotation.T @ z[3:6] / b0
        output = [(rho - 1) / h, local_p[0], z[1], local_p[1], t + final_ct - length / beta0, pt]
        spins = (rotation.T @ z[6:].reshape(3, 3)).T
        return (
            np.array(output),
            spins,
            dict(min_rho=min_rho[0], min_ps=min_ps[0], evaluations=solution.nfev),
        )


    def discovery_oracle(case):
        """Independent potential derivatives and fixed-s moving-frame DOP853 check."""
        gamma0, anomaly = case["gamma"], case["anomaly"]
        bg = math.sqrt(gamma0**2 - 1)
        beta = bg / gamma0
        length, h, strength = case["length"], case["h"], case["k"]
        phase = np.tile(case["state"], (3, 1))
        spin = np.eye(3)
        r, v = sp.symbols("r v", positive=True)
        # Zolkin Table XI, rho*A_theta for a skew S-decapole, as retained in TRK-19.
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
        fx = sp.lambdify((r, v), sp.diff(potential, r), "numpy")
        fy = sp.lambdify((r, v), sp.diff(potential, v), "numpy")

        def rhs(_s, flat):
            z = flat.reshape((-1, 9))
            x, px, y, py, _t, pt = z[:, :6].T
            rho, vert = 1 + h * x, h * y
            momentum = np.sqrt(1 - 2 * pt / beta + pt**2)
            ps = np.sqrt(momentum**2 - px**2 - py**2)
            force_x = strength / h**4 * fx(rho, vert)
            force_y = strength / h**4 * fy(rho, vert)
            out = np.zeros_like(z)
            out[:, 0] = rho * px / ps
            out[:, 1] = h * (ps - rho) + force_x
            out[:, 2] = rho * py / ps
            out[:, 3] = force_y
            out[:, 4] = rho * (1 / beta - pt) / ps - 1 / beta
            # B/(B*rho_ref) follows curl(A); normalization by q/(m*c) adds bg.
            b = np.column_stack([force_y / rho, h + (-force_x) / rho, np.zeros(len(x))]) * bg
            unit = np.column_stack([px, py, ps]) / momentum[:, None]
            gamma = gamma0 - bg * pt
            # Thomas-BMT per metre, with the reference frame's +h*y rotation.
            omega = (
                -rho[:, None]
                / (bg * ps[:, None])
                * (
                    (1 + anomaly * gamma[:, None]) * b
                    - anomaly * (gamma[:, None] - 1) * np.sum(unit * b, axis=1)[:, None] * unit
                )
            )
            omega[:, 1] += h
            out[:, 6:] = np.cross(omega, z[:, 6:])
            return out.ravel()

        solution = solve_ivp(
            rhs,
            (0, length),
            np.column_stack([phase, spin]).ravel(),
            method="DOP853",
            rtol=2.3e-14,
            atol=2e-16,
        )
        assert solution.success
        return solution.y[:, -1].reshape((-1, 9))


    def cross_case_controls(records, results, evaluate):
        """Check units, slicing, disabled spin and the local BMT field difference."""
        rows = {row["name"]: row for row in results}

        def wheel(name):
            return records[name]

        for name, other in [("other_unit1", "other"), ("other_spin_off", "other")]:
            residual = float(np.max(np.abs(np.array(wheel(name)["final"]) - wheel(other)["final"])))
            print(name, "cross_case_orbit_error", residual)
            assert residual < 1e-13
        unit_error = float(
            np.max(np.abs(np.array(wheel("other_unit1")["spins"]) - wheel("other")["spins"]))
        )
        print("unit_conversion_spin_error", unit_error)
        assert unit_error < 2e-13
        slice_error = float(
            np.max(np.abs(np.array(wheel("anchor_sliced")["spins"]) - wheel("anchor")["spins"]))
        )
        print("slice_spin_error", slice_error)
        assert slice_error == 0

        case = rows["small_length"]["case"]
        x, px, y, py, _t, pt = case["state"]
        b0 = math.sqrt(case["gamma"] ** 2 - 1)
        gamma = case["gamma"] - b0 * pt
        ps = math.sqrt((gamma**2 - 1) / b0**2 - px**2 - py**2)
        momentum = b0 * np.array([px, py, ps])
        rho = 1 + case["h"] * x
        _bx, by = evaluate(np.longdouble(rho), np.longdouble(case["h"] * y))
        by = float(by)
        delta_b = np.array([0, -2 * b0 * case["k"] / case["h"] ** 4 * by, 0])
        anomaly = case["anomaly"]
        delta_omega = (
            (
                -(anomaly + 1 / gamma) * delta_b
                + anomaly / (gamma * (gamma + 1)) * np.dot(momentum, delta_b) * momentum
            )
            * rho
            * gamma
            / (b0 * ps)
        )
        generator = np.cross(np.broadcast_to(delta_omega, (3, 3)), np.eye(3))
        deltas = []
        for name in ["small_length", "small_length_half"]:
            actual = np.array(wheel(name)["spins"])[:3]
            expected = np.array(rows[name]["oracles"]["corrected_spin"]["spins"])
            deltas.append(actual - expected)
            print(
                name,
                "spin_error_per_m",
                float(np.max(np.abs(actual - expected))) / rows[name]["case"]["length"],
            )
        extrapolated = (4 * deltas[1] - deltas[0]) / case["length"]
        print("initial_delta_omega_per_m", delta_omega.tolist())
        print("extrapolated_generator_error_per_m", float(np.max(np.abs(extrapolated - generator))))
        assert np.max(np.abs(extrapolated - generator)) < 2e-7

        domain = [
            row["oracles"][model]
            for row in rows.values()
            for model in ["physical", "corrected_spin", "implemented"]
        ]
        print("all_oracle_domain_min_rho", min(o["min_rho"] for o in domain))
        print("all_oracle_domain_min_ps", min(o["min_ps"] for o in domain))
        print("maximum_spin_norm_error", max(row["spin_norm_error"] for row in rows.values()))
        print("maximum_spin_repeat_error", max(row["replicate_spin_error"] for row in rows.values()))


    def main():
        parser = argparse.ArgumentParser()
        parser.add_argument("--case", choices=list(CASES))
        args = parser.parse_args()
        if args.case:
            run_wheel(args.case)
            return
        evaluate, div_bad = field_functions()
        print(
            "Maxwell reconstruction: correct Bx, reversed skew By; symbolic divergence/curl checks PASS",
            flush=True,
        )
        print("Dimensionless implemented divergence:", div_bad, flush=True)
        results = []
        records = {}
        for name in CASES:
            process = subprocess.run(
                [sys.executable, str(Path(__file__).resolve()), "--case", name],
                capture_output=True,
                text=True,
                timeout=290,
                check=False,
            )
            if process.returncode:
                raise RuntimeError(process.stdout + process.stderr)
            lines = [
                line.removeprefix("RESULT ")
                for line in process.stdout.splitlines()
                if line.startswith("RESULT ")
            ]
            assert len(lines) == 1, process.stdout + process.stderr
            record = json.loads(lines[0])
            records[name] = record
            if name == "anchor":
                print("Runtime " + json.dumps(record["runtime"], sort_keys=True), flush=True)
            case = record["case"]
            actual = np.array(record["final"])[5]
            # Row 5 has the center state's arrival time plus its known particle offset.
            actual[4] -= np.array(record["initial"])[5, 4] - case["state"][4]
            actual_spins = np.array(record["spins"])[:3]
            oracles: dict[str, dict[str, Any]] = {}
            for model in ["physical", "corrected_spin", "implemented"]:
                coordinates, spins, domain = lab_oracle(case, evaluate, model)
                oracles[model] = dict(
                    coordinates=coordinates.tolist(),
                    spins=spins.tolist(),
                    orbit_error=float(np.max(np.abs(actual - coordinates))),
                    spin_error=float(np.max(np.abs(actual_spins - spins))),
                    **domain,
                )
            if name in {"anchor", "other"}:
                _coordinates, refined, _domain = lab_oracle(
                    case, evaluate, "corrected_spin", resolution=2
                )
                refinement = float(np.max(np.abs(refined - oracles["corrected_spin"]["spins"])))
            else:
                refinement = None
            item: dict[str, Any] = dict(
                name=name,
                case=case,
                oracles=oracles,
                oracle_refinement=refinement,
                spin_norm_error=float(np.max(np.abs(np.linalg.norm(record["spins"], axis=1) - 1))),
                replicate_spin_error=float(
                    np.max(np.abs(np.array(record["spins"]) - np.tile(actual_spins, (4, 1))))
                ),
                spin_preservation_error=float(
                    np.max(np.abs(np.array(record["spins"]) - record["initial_spins"]))
                ),
            )
            assert item["spin_norm_error"] < 2e-12
            assert item["replicate_spin_error"] < 1e-13
            if case["spin"] and case["order"] == 6:
                assert oracles["implemented"]["spin_error"] < 2e-11
                assert oracles["implemented"]["orbit_error"] < 2e-12
            if name in {"dipole", "axis"}:
                assert oracles["physical"]["spin_error"] < 2e-12
            if name == "anchor":
                assert oracles["physical"]["spin_error"] > 7e-4
                assert oracles["corrected_spin"]["spin_error"] > 8e-4
            if not case["spin"]:
                assert item["spin_preservation_error"] == 0
            assert all(o["min_rho"] > 0 and o["min_ps"] > 0 for o in oracles.values())
            if refinement is not None:
                assert refinement < 2e-13
            if name == "anchor":
                expected = discovery_oracle(case)
                crosscheck = float(np.max(np.abs(expected[:, 6:] - oracles["physical"]["spins"])))
                assert crosscheck < 2e-13
                print(f"potential_DOP853_vs_Maxwell_RK45_spin={crosscheck:.15g}", flush=True)
                print(
                    "physical_initial_z",
                    oracles["physical"]["spins"][2],
                    "observed",
                    actual_spins[2].tolist(),
                    flush=True,
                )
            results.append(item)
            p, c, i = [oracles[m] for m in ["physical", "corrected_spin", "implemented"]]
            if case["spin"]:
                print(
                    f"{name}: physical_spin={p['spin_error']:.15g} corrected_spin={c['spin_error']:.15g} implemented_spin={i['spin_error']:.15g} implemented_orbit={i['orbit_error']:.15g} refinement={refinement}",
                    flush=True,
                )
            else:
                print(
                    f"{name}: spin_preservation={item['spin_preservation_error']} implemented_orbit={i['orbit_error']:.15g}",
                    flush=True,
                )
        cross_case_controls(records, results, evaluate)
        print("SERIAL CASES COMPLETE", len(results), flush=True)
        print("Spin tracking is experimental and unsupported.")
        print("DEFECT REPRODUCED", flush=True)


    if __name__ == "__main__":
        main()

    ```

## Provenance

| Role | Model | Effort | Date | Code examined |
|---|---|---|---|---|
| Found | `gpt-6-astra` | xhigh | 2026-10-29 | 26.10 (`615921c6867c`), impactx-noacc wheel, CPU, double, serial |
| Found<br><small>spin-field coefficient observation</small> | `gpt-6.1-sol` | xhigh | 2026-10-29 | 26.10 (`615921c6867c`), impactx-noacc wheel, CPU, double, serial |
| Found<br><small>spin extension</small> | `gpt-6-astra` | xhigh | 2026-10-29 | 26.10 (`615921c6867c`), impactx-noacc wheel, CPU, double, serial |
| Confirmed | `gpt-6.1-sol` | xhigh | 2026-10-29 | 26.10 (`615921c6867c`), impactx-noacc wheel, CPU, double, serial |
| Confirmed<br><small>spin extension</small> | `gpt-6.1-sol` | xhigh | 2026-10-29 | 26.10 (`615921c6867c`), impactx-noacc wheel, CPU, double, serial |

