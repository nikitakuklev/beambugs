---
title: "TRK-17: SoftSolenoid omits transverse fringe-field spin precession"
hide:
  - toc
---

# TRK-17 — `SoftSolenoid` omits transverse fringe-field spin precession

| | |
|---|---|
| Code | [ImpactX](../index.md) · Tracking |
| Priority | P4 |
| Severity | Wrong physics |
| Evidence | Reproduced, independently rechecked |
| Last reproduced | 2026-10-29 on ImpactX development (`fa0a43b`) |
| Tags | <span class="bb-tag">experimental</span> [?](../../methodology.md#tags) |
| Source | [`SoftSol.H#L418-L462`](https://github.com/BLAST-ImpactX/impactx/blob/b701a6e9d1f9e30cefb9a3e18e82ae7e02b9d943/src/elements/SoftSol.H#L418-L462)<br>[`map3()`](https://github.com/BLAST-ImpactX/impactx/blob/b701a6e9d1f9e30cefb9a3e18e82ae7e02b9d943/src/elements/SoftSol.H#L477-L544)<br>[spin push](https://github.com/BLAST-ImpactX/impactx/blob/b701a6e9d1f9e30cefb9a3e18e82ae7e02b9d943/src/elements/SoftSol.H#L243-L295)<br>[soft-element contract](https://github.com/BLAST-ImpactX/impactx/blob/615921c6867c895cf3a8fcf1d12991079d0f7d75/docs/source/theory/softedge_elements.rst#L17-L29)<br>[spin contract](https://github.com/BLAST-ImpactX/impactx/blob/615921c6867c895cf3a8fcf1d12991079d0f7d75/docs/source/usage/parameters.rst#L2792-L2800) |

With spin tracking enabled (experimental in ImpactX), `SoftSolenoid` omits the spin precession caused by the radial fringe field of a varying longitudinal field, because its spin map discards the field derivative. Maxwell's equations give the off-axis transverse fields −x·b′/2 and −y·b′/2 for b the longitudinal field divided by the rigidity Bρ; an independent Thomas–BMT integration with these fields disagrees with ImpactX while the orbits agree, and at vanishing magnetic anomaly ImpactX's spin does not follow the velocity, as it must. For a proton at γ = 3.3 offset by 20 μm in an asymmetric 0.7 m field profile, the transverse spin error at the exit is 1.548e-4. On-axis precession is correct; off-axis particles are affected.

??? example "Reproducer: TRK-17.py"

    [Download TRK-17.py](TRK-17.py)

    ```python
    """TRK-17: `SoftSolenoid` omits transverse fringe-field spin precession

    With spin tracking enabled (experimental in ImpactX), `SoftSolenoid` omits the spin precession
    caused by the radial fringe field of a varying longitudinal field, because its spin map discards the
    field derivative. Maxwell's equations give the off-axis transverse fields −x·b′/2 and −y·b′/2 for b
    the longitudinal field divided by the rigidity Bρ; an independent Thomas–BMT integration with these
    fields disagrees with ImpactX while the orbits agree, and at vanishing magnetic anomaly ImpactX's
    spin does not follow the velocity, as it must. For a proton at γ = 3.3 offset by 20 μm in an
    asymmetric 0.7 m field profile, the transverse spin error at the exit is 1.548e-4. On-axis
    precession is correct; off-axis particles are affected.

    The script tracks 16 particles with horizontal or vertical offsets, zero transverse momentum and
    spin (0, 0, 1) through `SoftSolenoid` profiles whose field and slope vanish at both ends, b = 2
    sin²(πz) over 1 m and an asymmetric sin⁴ profile over 0.7 m. It fits the exit transverse spin and
    kinetic momentum per unit offset and compares them with a fourth-order Runge–Kutta solution of the
    Larmor-frame orbit combined with an integral identity for the Thomas–BMT equation. Controls cover
    zero field, both planes, field reversal, small and negative anomalies, smaller offsets and fields,
    the SI field unit with four slices, and on-axis precession, which must agree.
    """

    import os

    for name in (
        "OMP_NUM_THREADS",
        "OPENBLAS_NUM_THREADS",
        "MKL_NUM_THREADS",
        "NUMEXPR_NUM_THREADS",
        "NPROC",
        "PJRT_NPROC",
    ):
        os.environ[name] = "1"

    os.sched_setaffinity(0, {min(os.sched_getaffinity(0))})

    import importlib.metadata
    import math
    import sys

    import numpy as np
    from impactx import Config, ImpactX, RefPart, elements

    ORBIT = [
        "position_x",
        "position_y",
        "position_t",
        "momentum_x",
        "momentum_y",
        "momentum_t",
    ]
    SPIN = ["spin_x", "spin_y", "spin_z"]


    class Runtime:
        def __init__(self):
            print(
                f"Python {sys.version.split()[0]} numpy={np.__version__} "
                f"scipy={importlib.metadata.version('scipy')} "
                f"impactx-noacc={importlib.metadata.version('impactx-noacc')}"
            )
            print("build", Config.to_dict())
            config = Config.to_dict()
            assert not any(config[key] for key in ("have_mpi", "have_gpu", "have_omp"))
            assert config["precision"] == config["precision_particles"] == "DOUBLE"
            self.sim = ImpactX()
            self.sim.tiny_profiler = False
            self.sim.memory_profiler = False
            self.sim.particle_shape = 1
            self.sim.space_charge = "off"
            self.sim.csr = False
            self.sim.isr = False
            self.sim.diagnostics = False
            self.sim.slice_step_diagnostics = False
            self.sim.verbose = 0
            self.sim.init_grids()

        def run(self, lattice, phase, *, gamma=2.0, species="proton", anomaly=None, spin=None):
            beam = self.sim.beam
            beam.clear_particles()
            ref = beam.ref
            ref.reset()
            ref.set_species(species)
            ref.set_kin_energy_MeV((gamma - 1) * ref.mass_MeV)
            if anomaly is not None:
                ref.set_gyromagnetic_anomaly(anomaly)
            self.sim.spin = spin is not None
            phase = np.asarray(phase)
            args = dict(zip(("x", "y", "t", "px", "py", "pt"), phase.T.copy()))
            args.update(qm=ref.qm_ratio_SI, w=np.ones(len(phase)))
            if spin is not None:
                spin = np.broadcast_to(spin, (len(phase), 3))
                args.update(dict(zip(("sx", "sy", "sz"), spin.T.copy())))
            beam.add_n_particles(**args)
            self.sim.lattice.clear()
            self.sim.lattice.extend(lattice)
            self.sim.track_particles()
            # These probes have pt=0 (or a common thin kick) and distinct arrival
            # times; sorting by t preserves the input correspondence across tiles.
            frame = beam.to_df(local=True).sort_values("position_t")
            assert len(frame) == len(phase)
            return frame[ORBIT].to_numpy(), frame[SPIN].to_numpy(), ref.copy()

        def close(self):
            self.sim.finalize()


    def phase_space():
        result = np.zeros((16, 6))
        result[:, 2] = np.linspace(-0.003, 0.003, len(result))
        return result


    PROFILES = {
        "sin2": (1.0, [1.0, 0.5], [0.0, 0.0]),
        # sin^4(pi*z/L)*(1 + 0.35*sin(2*pi*z/L)), centered Fourier convention.
        "asymmetric": (0.7, [0.75, 0.5, 0.125, 0.0], [0.0, -0.109375, -0.0875, -0.021875]),
    }


    def field_and_integral(z, length, scale, cosines, sines):
        """b=Bz/(B*rho) and I(z)=integral_0^z b(w) dw, analytically."""
        angle = 2 * math.pi * (z / length - 0.5)
        b = cosines[0] / 2
        integral = cosines[0] * z / 2
        for j in range(1, len(cosines)):
            k = 2 * math.pi * j / length
            b += cosines[j] * math.cos(j * angle) + sines[j] * math.sin(j * angle)
            integral += cosines[j] * math.sin(j * angle) / k
            integral -= sines[j] * (math.cos(j * angle) - (-1) ** j) / k
        return scale * b, scale * integral


    def larmor_oracle(profile, scale, gamma, anomaly, steps):
        """Unit horizontal offset, field-free endpoints, initial u=0 and S=z.

        q=x+i*y=e^(-i*I/2) Q and p=e^(-i*I/2) P, with
        Q'=P, P'=-b^2*Q/4. Kinetic u=e^(-i*I/2)*(P-i*b*Q/2).
        If J=integral b*u*exp(i*(1+G)*I) dz, BMT integration by parts gives
        s=(1+G*gamma)*u - i*(1+G)*G*gamma*exp(-i*(1+G)*I)*J.
        Keeping only the longitudinal-field pitch term gives s_without_fringe.
        """
        length, cosines, sines = PROFILES[profile]

        def rhs(z, state):
            q, p = state[:2]
            b, integral = field_and_integral(z, length, scale, cosines, sines)
            u = np.exp(-0.5j * integral) * (p - 0.5j * b * q)
            dj = b * u * np.exp(1j * (1 + anomaly) * integral)
            return np.array([p, -0.25 * b * b * q, dj.real, dj.imag])

        state = np.array([1.0, 0.0, 0.0, 0.0])
        h = length / steps
        for i in range(steps):
            z = i * h
            k1 = rhs(z, state)
            k2 = rhs(z + h / 2, state + h * k1 / 2)
            k3 = rhs(z + h / 2, state + h * k2 / 2)
            k4 = rhs(z + h, state + h * k3)
            state += h * (k1 + 2 * k2 + 2 * k3 + k4) / 6
        _, integral = field_and_integral(length, length, scale, cosines, sines)
        u = np.exp(-0.5j * integral) * state[1]
        j = state[2] + 1j * state[3]
        weighted_j = np.exp(-1j * (1 + anomaly) * integral) * j
        spin = (1 + anomaly * gamma) * u - 1j * (1 + anomaly) * anomaly * gamma * weighted_j
        without_fringe = -1j * anomaly * (gamma - 1) * weighted_j
        return np.array([u, spin, without_fringe])


    def components(z):
        return [float(z.real), float(z.imag)]


    cache = {}


    def converged_oracle(profile, scale, gamma, anomaly):
        key = (profile, scale, gamma, anomaly)
        if key not in cache:
            coarse = larmor_oracle(*key, 4096)
            fine = larmor_oracle(*key, 8192)
            error = float(np.max(abs(fine - coarse)))
            assert error < 2e-13
            cache[key] = fine
            print("ORACLE_REFINEMENT", key, error, flush=True)
        return cache[key]


    r = Runtime()
    try:
        for label, profile, scale, gamma, species, anomaly, steps, slices, amplitude, unit in (
            ("proton_512", "sin2", 2.0, 2.0, "proton", None, 512, 1, 1e-6, 0),
            ("proton_2048", "sin2", 2.0, 2.0, "proton", None, 2048, 1, 1e-6, 0),
            ("proton_8192", "sin2", 2.0, 2.0, "proton", None, 8192, 1, 1e-7, 0),
            ("explicit_G_proton_400", "sin2", 2.0, 2.0, "proton", 1.79284734463, 400, 1, 1e-6, 0),
            ("explicit_G_proton_6400", "sin2", 2.0, 2.0, "proton", 1.79284734463, 6400, 1, 1e-7, 0),
            ("electron", "sin2", 2.0, 2.0, "electron", None, 2048, 1, 1e-6, 0),
            (
                "explicit_G_electron_400",
                "sin2",
                2.0,
                2.0,
                "electron",
                0.00115965218073,
                400,
                1,
                1e-6,
                0,
            ),
            ("small_G", "sin2", 2.0, 2.0, "proton", 1e-9, 2048, 1, 2e-5, 0),
            ("negative_G", "sin2", 2.0, 7.0, "proton", -1e-9, 2048, 1, 2e-5, 0),
            ("asym_2048", "asymmetric", -2.3, 3.3, "proton", None, 2048, 1, 2e-5, 0),
            ("asym_8192", "asymmetric", -2.3, 3.3, "proton", None, 8192, 1, 2e-5, 0),
            ("asym_small_offset", "asymmetric", -2.3, 3.3, "proton", None, 8192, 1, 2e-6, 0),
            ("asym_SI_sliced", "asymmetric", -2.3, 3.3, "proton", None, 2048, 4, 2e-6, 1),
            ("asym_small_G", "asymmetric", -2.3, 3.3, "proton", 1e-9, 2048, 4, 2e-5, 1),
            ("asym_field_flip", "asymmetric", 2.3, 3.3, "proton", None, 8192, 1, 2e-6, 0),
            ("weak_field", "asymmetric", -0.23, 3.3, "proton", None, 2048, 1, 2e-5, 0),
            ("no_field", "asymmetric", 0.0, 3.3, "proton", None, 512, 1, 2e-5, 0),
        ):
            length, cosines, sines = PROFILES[profile]
            ref = RefPart().set_species(species)
            ref.set_kin_energy_MeV((gamma - 1) * ref.mass_MeV)
            if anomaly is None:
                anomaly = ref.gyromagnetic_anomaly
            want_u, want_s, no_fringe = converged_oracle(profile, scale, gamma, anomaly)
            if abs(anomaly) < 1e-8:
                assert abs(want_s - want_u) < 1e-7
            phase = phase_space()
            offsets = amplitude * np.array([-1, -0.75, -0.5, -0.25, 0.25, 0.5, 0.75, 1])
            phase[:8, 0] = offsets
            phase[8:, 1] = offsets
            q0 = phase[:, 0] + 1j * phase[:, 1]
            el = elements.SoftSolenoid(
                ds=length,
                bscale=scale if unit == 0 else scale * ref.rigidity_Tm,
                cos_coefficients=cosines,
                sin_coefficients=sines,
                unit=unit,
                mapsteps=steps,
                nslice=slices,
            )
            out, spin, _ = r.run(
                [el], phase, gamma=gamma, species=species, anomaly=anomaly, spin=(0, 0, 1)
            )
            kinetic = out[:, 3] + 1j * out[:, 4]
            transverse_spin = spin[:, 0] + 1j * spin[:, 1]
            got_u = np.vdot(q0, kinetic) / np.vdot(q0, q0)
            got_s = np.vdot(q0, transverse_spin) / np.vdot(q0, q0)
            norm_error = float(np.max(abs(np.linalg.norm(spin, axis=1) - 1)))
            assert norm_error < 5e-14
            assert abs(got_u - want_u) < 4e-7
            # Local field and reference curvature both vanish at these endpoints.
            assert abs(field_and_integral(0, length, scale, cosines, sines)[0]) < 1e-15
            assert abs(field_and_integral(length, length, scale, cosines, sines)[0]) < 1e-15
            if scale != 0:
                assert abs(got_s - want_s) > 1e-3
            else:
                np.testing.assert_array_equal(spin, np.broadcast_to([0, 0, 1], spin.shape))
            row = dict(
                label=label,
                G=anomaly,
                gamma=gamma,
                amplitude=amplitude,
                unit=unit,
                nslice=slices,
                steps_per_slice=steps,
                got_spin=components(got_s),
                BMT_spin=components(want_s),
                no_fringe_spin=components(no_fringe),
                got_kinetic=components(got_u),
                expected_kinetic=components(want_u),
                orbital_error=float(abs(got_u - want_u)),
                spin_error=float(abs(got_s - want_s)),
                spin_norm_error=norm_error,
                linearity_residual=float(np.max(abs(transverse_spin - got_s * q0))),
            )
            print(
                f"{label} G={anomaly:.14g} gamma={gamma:g} offset={amplitude:g} "
                f"steps={steps} slices={slices} unit={unit} "
                f"spin=({got_s.real:.12g},{got_s.imag:.12g}) "
                f"BMT=({want_s.real:.12g},{want_s.imag:.12g}) "
                f"orbit_error={row['orbital_error']:.12g} "
                f"spin_error={row['spin_error']:.12g} "
                f"no_fringe_error={abs(got_s - no_fringe):.12g}",
                flush=True,
            )

        phase = phase_space()
        length, cosines, sines = PROFILES["asymmetric"]
        el = elements.SoftSolenoid(
            ds=length,
            bscale=-2.3,
            cos_coefficients=cosines,
            sin_coefficients=sines,
            mapsteps=2048,
        )
        _, spin, ref = r.run([el], phase, gamma=3.3, spin=(1, 0, 0))
        integral = field_and_integral(length, length, -2.3, cosines, sines)[1]
        angle = -(1 + ref.gyromagnetic_anomaly) * integral
        want = np.array([math.cos(angle), math.sin(angle), 0])
        error = float(np.max(abs(spin - want)))
        assert error < 2e-12
        print("ON_AXIS_CONTROL", spin[0].tolist(), "expected", want.tolist(), "error", error)
        print("Manual declares spin tracking unsupported")
        print("DEFECT REPRODUCED", flush=True)
    finally:
        r.close()

    ```

## Provenance

| Role | Model | Effort | Date | Code examined |
|---|---|---|---|---|
| Found | `gpt-6-astra` | xhigh | 2026-10-28 | 26.10 (`615921c6867c`), impactx-noacc wheel, CPU, double, serial |
| Confirmed | `gpt-6.1-sol` | xhigh | 2026-10-28 | 26.10 (`615921c6867c`), impactx-noacc wheel, CPU, double, serial |

