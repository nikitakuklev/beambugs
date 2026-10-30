---
title: "TRK-16: Rolled ThinDipole, Sbend and ExactSbend omit vertical steering"
hide:
  - toc
---

# TRK-16 — Rolled `ThinDipole`, `Sbend` and `ExactSbend` omit vertical steering

| | |
|---|---|
| Code | [ImpactX](../index.md) · Tracking |
| Priority | P1 |
| Severity | Wrong physics (rotation-error reading) |
| Evidence | Reproduced, independently rechecked |
| Last reproduced | 2026-10-29 on development (`fa0a43b9bb07`), source, CPU, double, serial |
| Tags | <span class="bb-tag">contract-unsettled</span> [?](../../methodology.md#tags) |
| Issue | [[gh:https://github.com/BLAST-ImpactX/impactx/issues/1725]], filed from this evidence |
| Source | [`ThinDipole.H#L154-L193`](https://github.com/BLAST-ImpactX/impactx/blob/b701a6e9d1f9e30cefb9a3e18e82ae7e02b9d943/src/elements/ThinDipole.H#L154-L193)<br>[`Sbend.H#L178-L284`](https://github.com/BLAST-ImpactX/impactx/blob/b701a6e9d1f9e30cefb9a3e18e82ae7e02b9d943/src/elements/Sbend.H#L178-L284)<br>[`ExactSbend.H#L166-L305`](https://github.com/BLAST-ImpactX/impactx/blob/b701a6e9d1f9e30cefb9a3e18e82ae7e02b9d943/src/elements/ExactSbend.H#L166-L305)<br>[alignment wrapper](https://github.com/BLAST-ImpactX/impactx/blob/b701a6e9d1f9e30cefb9a3e18e82ae7e02b9d943/src/elements/mixin/beamoptic.H#L440-L507)<br>[`Sbend`](https://github.com/BLAST-ImpactX/impactx/blob/615921c6867c895cf3a8fcf1d12991079d0f7d75/docs/source/usage/parameters.rst#L1855-L1859)<br>[`ExactSbend`](https://github.com/BLAST-ImpactX/impactx/blob/615921c6867c895cf3a8fcf1d12991079d0f7d75/docs/source/usage/parameters.rst#L1910-L1914)<br>[`ThinDipole`](https://github.com/BLAST-ImpactX/impactx/blob/615921c6867c895cf3a8fcf1d12991079d0f7d75/docs/source/usage/parameters.rst#L2325-L2329)<br>[reference-trajectory contract](https://github.com/BLAST-ImpactX/impactx/blob/615921c6867c895cf3a8fcf1d12991079d0f7d75/docs/source/theory/concepts.rst#L9-L20) |

Setting `rotation` (a roll about the beam axis) on a `ThinDipole`, `Sbend` or `ExactSbend` produces no vertical steering. Under the manual's rotation-error convention, a bend of angle θ rolled by α must give the small-angle kick `py = −θ·sin α`, and ImpactX gives `py = 0`. A 3 mrad bend rolled by 20 mrad misses a kick of −6.000e-5, about 60 μm after a 1 m drift. Examples and the MAD-X importer also use `rotation` as a design roll, under which the reference orbit must bend vertically; under both readings the orbit stays in the horizontal plane.

## Reproducer

Requires scipy.

Lorentz impulse and independent lab-frame circular-orbit exit-plane checks.

One serial, double-precision simulation per process; no output files.

??? example "TRK-16.py"

    [Download TRK-16.py](TRK-16.py)

    ```python
    """TRK-16: Rolled bends omit the leading vertical steering.

    Lorentz impulse and independent lab-frame circular-orbit exit-plane checks.

    One serial, double-precision simulation per process; no output files.
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
    from scipy.optimize import newton

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


    def circular_oracle(theta, alpha, radius):
        """Uniform tilted B; intersect its helix with the ideal exit plane.

        qB/P0 = (-sin(alpha), cos(alpha), 0)/radius. For initial n=(0,0,1),
        n(u)=(-cos(alpha)*sin(u), -sin(alpha)*sin(u), cos(u)).
        The nominal exit plane has normal (-sin(theta),0,cos(theta)).
        """
        st, ct = math.sin(theta), math.cos(theta)
        ca, sa = math.cos(alpha), math.sin(alpha)
        u = newton(
            lambda u: ca * st * 2 * math.sin(u / 2) ** 2 + ct * math.sin(u) - st,
            theta,
            fprime=lambda u: ca * st * math.sin(u) + ct * math.cos(u),
            tol=2e-15,
        )
        d = st * ca
        closed_form = math.atan2(d, ct) + math.asin(st * (1 - ca) / math.hypot(ct, d))
        np.testing.assert_allclose(u, closed_form, rtol=2e-15, atol=0)
        return -sa * math.sin(u), -2 * radius * sa * math.sin(u / 2) ** 2


    r = Runtime()
    try:
        for model in ("ThinDipole", "Sbend", "ExactSbend"):
            for theta, alpha, radius, slices, gamma in (
                (0.003, 0.02, 0.8, 1, 3.5),
                (0.003, -0.02, 0.8, 4, 3.5),
                (0.003, 0.04, 0.8, 1, 3.5),
                (0.006, 0.02, 1.6, 4, 1.2),
                (-0.003, 0.02, -0.8, 7, 7.0),
                (0.003, 0.0, 0.8, 1, 3.5),
            ):
                rotation = math.degrees(alpha)
                if model == "ThinDipole":
                    el = elements.ThinDipole(theta=math.degrees(theta), rc=radius, rotation=rotation)
                elif model == "Sbend":
                    el = elements.Sbend(ds=radius * theta, rc=radius, nslice=slices, rotation=rotation)
                else:
                    el = elements.ExactSbend(
                        ds=radius * theta,
                        phi=math.degrees(theta),
                        nslice=slices,
                        rotation=rotation,
                    )
                out, _, ref = r.run([el], phase_space(), gamma=gamma)
                py, y = circular_oracle(theta, alpha, radius)
                assert np.max(abs(out[:, 4])) == 0
                assert ref.py == ref.y == 0
                row = dict(
                    model=model,
                    theta=theta,
                    alpha=alpha,
                    radius=radius,
                    nslice=el.nslice,
                    gamma=gamma,
                    got_py=float(out[0, 4]),
                    thin_py=-theta * math.sin(alpha),
                    circular_py=py,
                    circular_y=y,
                    got_y=float(out[0, 1]),
                    ref_py=ref.py,
                )
                print(
                    f"{model} theta={theta:g} alpha={alpha:g} radius={radius:g} "
                    f"nslice={el.nslice} gamma={gamma:g} py={row['got_py']:.15g} "
                    f"impulse={row['thin_py']:.15g} circular={py:.15g} ref_py={ref.py:g}",
                    flush=True,
                )

        # Exercise the explicit SI-field branch and a negative-charge reference.
        ref0 = RefPart().set_species("electron")
        ref0.set_kin_energy_MeV(4 * ref0.mass_MeV)
        el = elements.ExactSbend(
            ds=0.0024,
            phi=math.degrees(0.003),
            B=ref0.rigidity_Tm / 0.8,
            rotation=math.degrees(0.02),
            nslice=3,
        )
        out, _, ref = r.run([el], phase_space(), species="electron", gamma=5)
        assert np.max(abs(out[:, 4])) == 0
        print("SI_FIELD_ELECTRON", out[0, [1, 4]].tolist(), "B_T", el.B, flush=True)

        # Shared alignment sign control; this element retains its constant impulse.
        out, _, _ = r.run(
            [elements.Kicker(xkick=-0.003, ykick=0, rotation=math.degrees(0.02))],
            phase_space(),
        )
        expected = -0.003 * math.sin(0.02)
        np.testing.assert_allclose(out[:, 4], expected, rtol=2e-15, atol=0)
        print("KICKER_CONTROL", out[0, 4], "expected", expected, flush=True)

        # A nonzero energy deviation proves that roll is active for deviation terms.
        phase = phase_space()
        phase[:, 5] = -0.002
        out, _, ref = r.run(
            [elements.ThinDipole(theta=math.degrees(0.003), rc=0.8, rotation=math.degrees(0.02))],
            phase,
            gamma=3.5,
        )
        f = math.sqrt(1 + 0.004 / ref.beta + 0.002**2) - 1
        expected = 0.003 * f * math.sin(0.02)
        np.testing.assert_allclose(out[:, 4], expected, rtol=1e-12, atol=1e-19)
        print("CHROMATIC_ROLL_CONTROL", out[0, 4], "expected", expected, flush=True)
        print("DEFECT REPRODUCED", flush=True)
    finally:
        r.close()

    ```

## Provenance

| Role | Model | Effort | Date | Code examined |
|---|---|---|---|---|
| Found | `gpt-6-astra` | xhigh | 2026-10-28 | 26.10 (`615921c6867c`), impactx-noacc wheel, CPU, double, serial |
| Confirmed | `gpt-6.1-sol` | xhigh | 2026-10-28 | 26.10 (`615921c6867c`), impactx-noacc wheel, CPU, double, serial |

