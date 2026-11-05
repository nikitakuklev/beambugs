---
title: "TRK-20: CFbend spin coupling is singular at coincident frequencies"
hide:
  - toc
---

# TRK-20 — `CFbend` spin coupling is singular at coincident frequencies

| | |
|---|---|
| Code | [ImpactX](../index.md) · Tracking |
| Priority | P4 |
| Severity | Wrong physics |
| Evidence | Reproduced, independently rechecked |
| Last reproduced | 2026-10-29 on ImpactX development (`fa0a43b`) |
| Tags | <span class="bb-tag">experimental</span> [?](../../methodology.md#tags) |
| Source | [`CFbend.H#L171-L184`](https://github.com/BLAST-ImpactX/impactx/blob/b701a6e9d1f9e30cefb9a3e18e82ae7e02b9d943/src/elements/CFbend.H#L171-L184)<br>[spin push](https://github.com/BLAST-ImpactX/impactx/blob/b701a6e9d1f9e30cefb9a3e18e82ae7e02b9d943/src/elements/CFbend.H#L327-L345)<br>[manual](https://github.com/BLAST-ImpactX/impactx/blob/615921c6867c895cf3a8fcf1d12991079d0f7d75/docs/source/usage/parameters.rst#L2792-L2800) |

With spin tracking (experimental), `CFbend` divides its spin coupling by k + (Gγh)², for gradient k, magnetic anomaly G and curvature h = 1/`rc`, which vanishes when the vertical focusing frequency √(−k) equals |Gγh|. The Thomas–BMT solution stays finite there; on the design orbit it is a rotation by GγhL about the vertical axis over length L. For electrons at γ = 0.5/G = 431.2 in a `CFbend` with `rc` = 1 m, `k` = −0.25 m⁻² and L = 0.31 m, all spin components are NaN while the orbit is correct. Nearby the spins are finite but wrong: at γ = 2 with G = 0.25 and the gradient detuned by a relative 1e-15, the error is 7.450e-5.

??? example "Reproducer: TRK-20.py"

    [Download TRK-20.py](TRK-20.py)

    ```python
    """TRK-20: `CFbend` spin coupling is singular at coincident frequencies

    With spin tracking (experimental), `CFbend` divides its spin coupling by k + (Gγh)², for gradient
    k, magnetic anomaly G and curvature h = 1/`rc`, which vanishes when the vertical focusing frequency
    √(−k) equals |Gγh|. The Thomas–BMT solution stays finite there; on the design orbit it is a
    rotation by GγhL about the vertical axis over length L. For electrons at γ = 0.5/G = 431.2 in a
    `CFbend` with `rc` = 1 m, `k` = −0.25 m⁻² and L = 0.31 m, all spin components are NaN while the
    orbit is correct. Nearby the spins are finite but wrong: at γ = 2 with G = 0.25 and the gradient
    detuned by a relative 1e-15, the error is 7.450e-5.

    The script tracks 12 on-axis electrons carrying the three Cartesian unit spins through `CFbend` at
    the exact coincidence (one and nine slices, reversed curvature, spin tracking off), at nominal
    coincidences for a low-γ electron and a proton, and through `ExactCFbend` as a control, and compares
    the exit spins with the design rotation. With the anomaly set to G = 0.25 at γ = 2, nine of the
    particles are offset to y = 2 mm and `py` = −0.001 and tracked at relative detunings from ±1e-4 to
    0, against a quadrature of the linearized Thomas–BMT generator in the design spin frame. The orbit
    is checked against the linear vertical map in every case.
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


    import json
    import math

    from impactx import elements
    from scipy.integrate import quad
    from scipy.spatial.transform import Rotation


    def oracle(phase, spins, length, curvature, k, gamma, anomaly):
        """Interaction-picture integral for the documented linearized spin map.

        With x=px=pt=0, vertical orbit is harmonic and the linear BMT
        perturbations are Omega_x=-(1+G*gamma)*k*y,
        Omega_z=G*(gamma-1)*h*py. Rotate them into the design spin frame,
        integrate and exponentiate, then apply the design rotation.
        """
        assert k < 0
        omega = np.sqrt(-k)
        a = anomaly * curvature * gamma
        design = Rotation.from_rotvec([0.0, -a * length, 0.0])
        output = []
        for q, spin in zip(phase, spins, strict=True):
            assert np.array_equal(q[[0, 1, 5]], np.zeros(3))

            def integrand(s, component, q=q):
                y = q[2] * np.cos(omega * s) + q[3] * np.sin(omega * s) / omega
                py = -q[2] * omega * np.sin(omega * s) + q[3] * np.cos(omega * s)
                ox = -(1 + anomaly * gamma) * k * y
                oz = anomaly * (gamma - 1) * curvature * py
                if component == 0:
                    return np.cos(a * s) * ox + np.sin(a * s) * oz
                return -np.sin(a * s) * ox + np.cos(a * s) * oz

            ax = quad(integrand, 0, length, args=(0,), epsabs=1e-16)[0]
            az = quad(integrand, 0, length, args=(2,), epsabs=1e-16)[0]
            output.append(design.apply(Rotation.from_rotvec([ax, 0.0, az]).apply(spin)))
        return np.array(output)


    def main():
        environment()
        beam = Beam(12)
        records: list[dict[str, Any]] = []
        try:
            spins = np.tile(np.eye(3), (4, 1))
            phase = np.zeros((12, 6))
            phase[:, 4] = np.linspace(-0.002, 0.002, 12)
            beam.ref.set_species("electron")
            electron_g = beam.ref.gyromagnetic_anomaly
            cases = [
                (
                    "electron_resonance",
                    "electron",
                    0.5 / electron_g,
                    None,
                    1.0,
                    0.31,
                    1,
                    0.0,
                    False,
                    False,
                ),
                (
                    "electron_sliced",
                    "electron",
                    0.5 / electron_g,
                    None,
                    1.0,
                    0.31,
                    9,
                    0.0,
                    False,
                    False,
                ),
                (
                    "electron_reverse",
                    "electron",
                    0.5 / electron_g,
                    None,
                    -0.5,
                    0.31,
                    1,
                    0.0,
                    False,
                    False,
                ),
                ("electron_low_gamma", "electron", 3.0, None, 2.0, 0.31, 1, 0.0, False, False),
                ("proton_resonance", "proton", 2.0, None, 0.5, 0.07, 1, 0.0, False, False),
                ("spin_off", "electron", 0.5 / electron_g, None, 1.0, 0.31, 1, 0.0, False, True),
                ("exact_control", "electron", 0.5 / electron_g, None, 1.0, 0.31, 1, 0.0, True, False),
            ]
            # Exact binary G and gamma isolate the denominator without tuning a
            # built-in anomaly. Nonzero y,py exercise its finite limiting integral.
            for detuning in [1e-4, 1e-8, 1e-12, 1e-15, 0.0, -1e-12, -1e-4]:
                cases.append(
                    (
                        f"detuning_{detuning}",
                        "electron",
                        2.0,
                        0.25,
                        1.0,
                        0.37,
                        1,
                        detuning,
                        False,
                        False,
                    )
                )
            for name, species, gamma, override, hh, length, slices, detuning, exact, disabled in cases:
                beam.ref.set_species(species).set_kin_energy_MeV((gamma - 1) * beam.ref.mass_MeV)
                if override is not None:
                    beam.ref.set_gyromagnetic_anomaly(override)
                gg = beam.ref.gyromagnetic_anomaly
                actual_gamma = beam.ref.gamma
                a = gg * hh * actual_gamma
                k = -(a * a) * (1 + detuning)
                input_phase = phase.copy()
                if name.startswith("detuning"):
                    input_phase[3:, 2] = 0.002
                    input_phase[3:, 3] = -0.001
                expected = oracle(input_phase, spins, length, hh, k, actual_gamma, gg)
                if override == 0.25 and detuning == 0.0:
                    # At omega_y=a, integrate cos^2, sin^2 and sin*cos directly.
                    cc = length / 2 + np.sin(2 * a * length) / (4 * a)
                    ss = length / 2 - np.sin(2 * a * length) / (4 * a)
                    cs = (1 - np.cos(2 * a * length)) / (4 * a)
                    c1, b = 1 + gg * actual_gamma, gg * (actual_gamma - 1) * hh
                    y0, py0 = input_phase[3, 2:4]
                    ax = y0 * (c1 * a * a * cc - b * a * ss) + py0 * (c1 * a + b) * cs
                    az = -y0 * (c1 * a * a + b * a) * cs + py0 * (-c1 * a * ss + b * cc)
                    closed = Rotation.from_rotvec([0.0, -a * length, 0.0]).apply(
                        Rotation.from_rotvec([ax, 0.0, az]).apply(spins[3])
                    )
                    print(
                        "RESONANT_CLOSED_FORM",
                        json.dumps(
                            dict(
                                ax=ax,
                                az=az,
                                quadrature_error=float(np.max(np.abs(closed - expected[3]))),
                            )
                        ),
                    )
                    assert np.max(np.abs(closed - expected[3])) < 1e-15
                if exact:
                    element = elements.ExactCFbend(
                        ds=length,
                        k_normal=[hh, k],
                        k_skew=[0.0, 0.0],
                        int_order=6,
                        mapsteps=64,
                        nslice=slices,
                    )
                else:
                    element = elements.CFbend(ds=length, rc=1 / hh, k=k, nslice=slices)
                q, s = beam.run(
                    element,
                    input_phase,
                    spins,
                    species=species,
                    gamma=gamma,
                    anomaly=override,
                    spin=not disabled,
                )
                finite = np.isfinite(s).all(axis=1)
                assert finite.all() or np.isnan(s).all()
                orb_expected = input_phase.copy()
                w = np.sqrt(-k)
                orb_expected[:, 2] = (
                    np.cos(w * length) * input_phase[:, 2] + np.sin(w * length) / w * input_phase[:, 3]
                )
                orb_expected[:, 3] = (
                    -w * np.sin(w * length) * input_phase[:, 2] + np.cos(w * length) * input_phase[:, 3]
                )
                target = spins if disabled else expected
                rec = dict(
                    case=name,
                    gamma=actual_gamma,
                    G=gg,
                    h=hh,
                    length=length,
                    slices=slices,
                    k=k,
                    denominator_python=k + a * a,
                    denominator_fma=math.fma(a, a, k),
                    gx=k + hh * hh,
                    gy=-k,
                    finite_spins=int(finite.sum()),
                    spin_error=float(np.max(np.abs(s - target))) if finite.all() else None,
                    orbit_error=float(np.max(np.abs(q - orb_expected))),
                    expected_first=target[0].tolist(),
                    observed_first=s[0].tolist(),
                )
                print(
                    f"{name}: finite_spins={rec['finite_spins']}/12 "
                    f"spin_error={rec['spin_error']} orbit_error={rec['orbit_error']:.12g} "
                    f"D_python={rec['denominator_python']:.12g} D_fma={rec['denominator_fma']:.12g}",
                    flush=True,
                )
                if name == "electron_resonance":
                    print(
                        "expected_initial_x",
                        rec["expected_first"],
                        "observed",
                        rec["observed_first"],
                        flush=True,
                    )
                    print(
                        f"gamma={actual_gamma:.17g} G={gg:.17g} h={hh:g} k={k:g} L={length:g} gx={rec['gx']:g} gy={rec['gy']:g}",
                        flush=True,
                    )
                records.append(rec)
            assert all(row["orbit_error"] < 2e-15 for row in records)
            assert records[0]["finite_spins"] == 0
            assert records[1]["finite_spins"] == 0
            assert records[2]["finite_spins"] == 0
            assert records[5]["spin_error"] == 0
            assert records[6]["spin_error"] < 1e-13
            assert records[7]["spin_error"] < 1e-13
            assert records[10]["spin_error"] > 1e-5
            assert records[11]["finite_spins"] == 0
        finally:
            beam.close()
        print("Spin tracking is experimental and unsupported.", flush=True)
        print("DEFECT REPRODUCED", flush=True)


    if __name__ == "__main__":
        main()

    ```

## Provenance

| Role | Model | Effort | Date | Code examined |
|---|---|---|---|---|
| Found | `gpt-6-astra` | xhigh | 2026-10-29 | 26.10 (`615921c6867c`), impactx-noacc wheel, CPU, double, serial |
| Confirmed | `gpt-6.1-sol` | xhigh | 2026-10-29 | 26.10 (`615921c6867c`), impactx-noacc wheel, CPU, double, serial |

