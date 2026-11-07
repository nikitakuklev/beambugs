"""TRK-16: Rolled `ThinDipole`, `Sbend`, `ExactSbend` and pure-dipole `ExactCFbend` omit vertical steering

Setting `rotation`, a roll α about the beam axis, on `ThinDipole`, `Sbend`, `ExactSbend`, or
pure-dipole `ExactCFbend` omits vertical steering. Under the documented rotation-error contract, a
bend of angle θ should give normalized vertical momentum py ≈ −θ sin α; a 3.000 mrad bend rolled by
20.00 mrad gives zero instead of −6.000e−5. An independent Lorentz integration confirms this
leading effect. Under the design-roll interpretation used by examples, the reference should bend
vertically, yet its vertical position and momentum stay zero.

The script tracks 16 deterministic on-axis particles through signed bends, rolls, energies and
slices, comparing vertical momentum with the Lorentz impulse and exact tilted-circle exit.
Pure-dipole ExactCFbend cases use matched ExactSbend controls, both coefficient units and
integration orders 2/4/6; zero-roll and off-axis controls check active alignment. A rolled Kicker,
an off-energy ThinDipole and an electron in an explicit ExactSbend field check the shared sign and
field conventions."""

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
from typing import Any

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


def steering_missing(out, ref, expected_py):
    """Neither the particles nor the reference acquire the expected vertical direction.

    Particle py is normalized by P0; the reference direction is ref.py/(beta0*gamma0).
    """
    observed = max(np.max(abs(out[:, 4])), abs(ref.py / ref.beta_gamma))
    return bool(observed < 0.01 * abs(expected_py))


# One flag per rolled case: True when its leading vertical steering is absent.
missing = []
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
            if alpha:
                missing.append(steering_missing(out, ref, py))
            if not alpha or missing[-1]:
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

    # A pure normal dipole excludes the independent skew-dipole and higher-
    # multipole faults. Reference momenta use mc; particle momenta use P0.
    base: dict[str, Any] = dict(
        radius=0.8,
        length=0.0024,
        alpha=0.02,
        gamma=3.5,
        species="proton",
        unit=0,
        nslice=1,
        order=2,
        steps=10,
        dx=0.0,
    )
    for label, changes in (
        ("baseline", {}),
        ("zero_roll", dict(alpha=0.0)),
        ("negative_roll", dict(alpha=-0.02)),
        ("negative_curvature", dict(radius=-0.8)),
        ("low_gamma", dict(gamma=1.2)),
        ("electron_SI", dict(species="electron", gamma=7.0, unit=1)),
        ("order4_slices4", dict(order=4, steps=3, nslice=4)),
        ("order6_slices7", dict(order=6, steps=5, nslice=7)),
        ("single_step", dict(steps=1)),
        ("small_dx", dict(dx=1e-16)),
        ("long_radius", dict(radius=100.0, length=0.3)),
        ("half_bend", dict(length=0.0012)),
        ("half_roll", dict(alpha=0.01)),
    ):
        cfg: dict[str, Any] = base | changes
        radius, length, alpha = cfg["radius"], cfg["length"], cfg["alpha"]
        theta = length / radius
        ref0 = RefPart().set_species(cfg["species"])
        ref0.set_kin_energy_MeV((cfg["gamma"] - 1) * ref0.mass_MeV)
        coefficient = (ref0.rigidity_Tm if cfg["unit"] == 1 else 1.0) / radius
        common: dict[str, Any] = dict(
            ds=length, nslice=cfg["nslice"], rotation=math.degrees(alpha), dx=cfg["dx"]
        )
        bend = elements.ExactCFbend(
            **common,
            k_normal=[coefficient],
            k_skew=[0.0],
            unit=cfg["unit"],
            int_order=cfg["order"],
            mapsteps=cfg["steps"],
        )
        control = elements.ExactSbend(**common, phi=math.degrees(theta))
        out, _, ref = r.run([bend], phase_space(), species=cfg["species"], gamma=cfg["gamma"])
        matched, _, matched_ref = r.run(
            [control], phase_space(), species=cfg["species"], gamma=cfg["gamma"]
        )
        expected_py, expected_y = circular_oracle(theta, alpha, radius)
        if alpha:
            assert abs(expected_py) > 1e-5
            missing.append(steering_missing(out, ref, expected_py))
        # Zero roll is a control; a rolled case missing its steering must be
        # the unsteered sector map of ExactSbend with a nominal reference.
        if not alpha or missing[-1]:
            np.testing.assert_allclose(out, matched, rtol=0, atol=2e-14)
            assert np.max(abs(out[:, [0, 1, 3, 4]])) < 2e-14
            assert np.max(abs(out[:, 4])) < 2e-18
            assert np.max(abs(out[:, 1])) < 2e-16
            assert ref.py == ref.y == matched_ref.py == matched_ref.y == 0
            # The nominal reference arc stays horizontal under the error-roll
            # contract. The design-roll alternative requires its vertical rotation.
            nominal = [-2 * radius * math.sin(theta / 2) ** 2, 0, radius * math.sin(theta)]
            np.testing.assert_allclose([ref.x, ref.y, ref.z], nominal, rtol=0, atol=2e-14)
            np.testing.assert_allclose(
                np.array([ref.px, ref.py, ref.pz]) / ref.beta_gamma,
                [-math.sin(theta), 0, math.cos(theta)],
                rtol=0,
                atol=2e-14,
            )
        print(
            f"ExactCFbend {label} py={out[0, 4]:.17g} circular_py={expected_py:.17g} "
            f"y={out[0, 1]:.17g} circular_y={expected_y:.17g} "
            f"ref_ny={ref.py / ref.beta_gamma:.17g} design_ny={-math.sin(alpha) * math.sin(theta):.17g} "
            f"matched_py={matched[0, 4]:.17g} match_max={np.max(abs(out - matched)):.3g} "
            "derivation=tilted_circle_at_nominal_exit;matched_ExactSbend",
            flush=True,
        )

    # A nonzero input deviation makes alignment's roll mixing observable even
    # though neither pure-dipole map supplies the missing constant orbit term.
    phase = phase_space()
    phase[:, 0] = 1e-4
    offaxis = []
    for alpha in (0.02, 0.0):
        common = dict(ds=0.0024, rotation=math.degrees(alpha))
        out, _, _ = r.run(
            [elements.ExactCFbend(**common, k_normal=[1.25], k_skew=[0.0])], phase, gamma=3.5
        )
        matched, _, _ = r.run(
            [elements.ExactSbend(**common, phi=math.degrees(0.003))], phase, gamma=3.5
        )
        if not alpha or all(missing):
            np.testing.assert_allclose(out, matched, rtol=0, atol=2e-14)
        offaxis.append(out[0, 4])
        print(
            f"EXACTCFBEND_OFFAXIS_CONTROL alpha={alpha:g} py={out[0, 4]:.17g} "
            f"matched_py={matched[0, 4]:.17g} derivation=matched_ExactSbend",
            flush=True,
        )
    assert abs(offaxis[0]) > 1e-10
    assert offaxis[1] == 0

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
    missing.append(steering_missing(out, ref, 0.003 * math.sin(0.02)))
    if missing[-1]:
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
    print(f"ROLLED_CASES_WITHOUT_STEERING {sum(missing)}/{len(missing)}", flush=True)
    print("DEFECT REPRODUCED" if any(missing) else "NOT REPRODUCED", flush=True)
finally:
    r.close()
