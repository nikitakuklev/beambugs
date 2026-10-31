"""COL-04: `Gauss2p5D` underweights the initial quadrature interval

`Gauss2p5D` space charge computes its transverse forces with a quadrature that treats the initial
interval [0, δ], δ = `gauss_taylor_delta`, separately, and divides that interval's contribution by
three a second time when it completes the Simpson sum. The forces therefore carry a bias that does
not vanish as the number of quadrature points grows. Direct integration of the Gaussian Coulomb
convolution, checked against Gauss's law for a round beam, gives the correct forces. With δ = 0.01,
4001 points and a test particle at (x, y) = (σx, 0.6σy), aspect ratio σy/σx = 0.25 gives relative
errors of −5.736×10⁻⁵ in px and −1.648×10⁻⁵ in py; aspect ratio 4 gives −2.312×10⁻⁴ and −8.103×10⁻⁴.
At the default 101 points, ordinary Simpson discretization error adds to this bias.

The script builds deterministic sources from eight-point elliptical rings in equal-charge
longitudinal bins, so that the model's measured widths are exact, and records the transverse kicks
of one `Gauss2p5D` kick on zero-weight test particles. It compares them with adaptive quadrature of
the Coulomb convolution (checked by Gauss's law for the round beam) and with a prediction that
applies the extra factor of one third to the initial interval, for a round beam and aspect ratios
0.25 and 4 over a scan of δ and quadrature points. A case with δ = 2×10⁻⁶, where the bias is
negligible, and a proton case with every length tripled serve as controls.
"""

import os

for variable in (
    "OMP_NUM_THREADS",
    "OPENBLAS_NUM_THREADS",
    "MKL_NUM_THREADS",
    "NUMEXPR_NUM_THREADS",
    "NPROC",
    "PJRT_NPROC",
):
    os.environ[variable] = "1"
os.sched_setaffinity(0, {min(os.sched_getaffinity(0))})

import importlib.metadata
import json
import math
import sys

import amrex.space3d as amr
import impactx
import numpy as np
from impactx import Config, ImpactX, RefPart, elements
from scipy.constants import c
from scipy.integrate import quad


def simulation(**options):
    print(
        f"Python {sys.version.split()[0]} numpy={np.__version__} scipy={importlib.metadata.version('scipy')}"
    )
    config = Config.to_dict()
    assert all(not config[k] for k in ("have_mpi", "have_omp", "have_gpu"))
    assert config["precision"] == config["precision_particles"] == "DOUBLE"
    print(
        json.dumps(
            {
                "impactx": impactx.__version__,
                "impactx_file": impactx.__file__,
                "impactx-noacc": importlib.metadata.version("impactx-noacc"),
                "pyamrex": amr.__version__,
                "config": config,
            }
        )
    )
    sim = ImpactX()
    settings = dict(
        particle_shape=1,
        diagnostics=False,
        slice_step_diagnostics=False,
        tiny_profiler=False,
        memory_profiler=False,
        csr=False,
        isr=False,
        spin=False,
        strang_split=False,
        verbose=0,
    )
    settings.update(options)
    for key, value in settings.items():
        setattr(sim, key, value)
    try:
        sim.init_grids()
    except BaseException:
        sim.finalize()
        raise
    return sim


def insert(sim, ref, positions, weights):
    phase = np.zeros((len(positions), 6))
    phase[:, :3] = positions
    sim.beam.set_ref_particle(ref)
    sim.beam.add_n_particles(*phase.T.copy(), qm=ref.qm_ratio_SI, w=weights)


def momenta(sim, count):
    frame = sim.beam.to_df(local=True).reset_index(drop=True).sort_values("idcpu")
    return frame[["momentum_x", "momentum_y", "momentum_t"]].to_numpy()[-count:]


def case(sim, aspect, scale=1.0, species="positron", gamma=7.0):
    sx, sy = 0.0007 * scale, 0.0007 * scale * aspect
    extent, bins = 0.48 * scale, 25
    length, charge = 0.027 * scale, 3e-11
    ref = RefPart().set_species(species)
    ref.set_kin_energy_MeV((gamma - 1) * ref.mass_MeV)
    # An eight-point ellipse has the exact prescribed weighted central widths.
    angles = 2 * np.pi * np.arange(8) / 8
    ring = np.c_[math.sqrt(2) * sx * np.cos(angles), math.sqrt(2) * sy * np.sin(angles)]
    source = np.array(
        [
            [x, y, -extent / 2 + (j + 0.5) * extent / (bins - 1)]
            for j in range(bins - 1)
            for x, y in ring
        ]
    )
    probes_xy = np.array([[1, 0], [0, 1], [1, 0.6], [3, 1.5], [-1, -0.6]]) * [sx, sy]
    probes = np.c_[probes_xy, np.full(len(probes_xy), 0.003 * scale)]
    positions = np.vstack([source, [0, 0, -extent / 2], [0, 0, extent / 2], probes])
    weights = np.r_[
        np.full(len(source), charge / abs(ref.charge) / len(source)), np.zeros(2 + len(probes))
    ]
    # With lambda_ct=Q/extent and p0=beta*gamma*m*c, the Coulomb coefficient
    # is k*q*lambda_ct*L/(m*c^2*(beta*gamma)^3). Use the kernel's k=c^2*1e-7
    # to separate its quadrature error from the small SI-constant convention gap.
    coefficient = (
        c**2
        * 1e-7
        * abs(ref.charge)
        * charge
        * length
        / (extent * ref.mass * c**2 * ref.beta_gamma**3)
    )
    oracle = np.zeros((len(probes), 2))
    for row, (x, y) in enumerate(probes_xy):
        # u=s/sigma_x^2; this integral follows from the Gaussian Coulomb
        # convolution, with physical denominators sigma_i^2+s.
        for component in range(2):

            def integrand(u, x=x, y=y, component=component):
                ax, ay = 1 + u, aspect**2 + u
                exponent = -0.5 * ((x / sx) ** 2 / ax + (y / sx) ** 2 / ay)
                return math.exp(exponent) / (math.sqrt(ax * ay) * (ax if component == 0 else ay))

            value, error = quad(integrand, 0, np.inf, epsabs=2e-13, epsrel=2e-13)
            assert error < 2e-11
            oracle[row, component] = coefficient * (x if component == 0 else y) / sx**2 * value
    if aspect == 1:
        # Cylindrical Gauss law is a closed-form check on the Coulomb quadrature.
        radius2 = np.sum(probes_xy**2, axis=1)
        closed = (
            coefficient
            * 2
            * (-np.expm1(-radius2 / (2 * sx * sx)))[:, None]
            * probes_xy
            / radius2[:, None]
        )
        valid = oracle != 0
        closed_error = np.max(np.abs((oracle[valid] - closed[valid]) / closed[valid]))
        assert closed_error < 2e-14
        print(f"round_Gauss_law_oracle_residual={closed_error:.12g}")
    sim.lattice.clear()
    sim.lattice.append(elements.Drift(ds=length))
    print(
        f"particles={len(positions)} long_bunch_aspect={math.hypot(sx, sy) / (ref.beta_gamma * np.std(source[:, 2])):.12g}"
    )
    results = []
    for delta, nint in [
        (0.01, 101),
        (0.02, 1001),
        (0.02, 4001),
        (0.01, 4001),
        (0.005, 4001),
        (2e-6, 4001),
    ]:
        sim.space_charge_gauss_taylor_delta = delta
        sim.space_charge_gauss_nint = nint
        sim.beam.clear_particles()
        insert(sim, ref, positions, weights)
        moments = sim.beam.beam_moments()
        assert abs(moments["sig_x"] / sx - 1) < 2e-14
        assert abs(moments["sig_y"] / sy - 1) < 2e-14
        sim.track_particles()
        assert sim.beam.ref.gamma == ref.gamma
        measured = momenta(sim, len(probes))
        assert np.isfinite(measured).all()
        assert np.all(measured[:, 2] == 0)
        observed = measured[:, :2]
        assert np.max(np.abs(observed[2] + observed[4])) < np.max(np.abs(observed)) * 2e-14
        valid = oracle != 0
        relative = np.zeros_like(oracle)
        relative[valid] = observed[valid] / oracle[valid] - 1
        # Secondary source check: integrate the omitted interval independently.
        predicted = np.zeros_like(oracle)
        corrected = np.zeros_like(oracle)
        inverse_aspect = sx / sy
        for row, (x, y) in enumerate(probes_xy):
            for component in range(2):

                def tail(u, x=x, y=y, component=component, aspect=inverse_aspect):
                    a = aspect**2 + (1 - aspect**2) * u * u
                    e = math.exp(-u * u * ((x / sx) ** 2 + (y / sy) ** 2 / a) / 2)
                    return u * e / (math.sqrt(a) * (1 if component == 0 else a))

                exact_low = quad(tail, 0, delta, epsabs=1e-15)[0]
                trapezoid = delta * tail(delta) / 2
                multiplier = (
                    coefficient
                    * 2
                    * inverse_aspect
                    * (x / sx if component == 0 else y / sy)
                    / (sx if component == 0 else sy)
                )
                predicted[row, component] = oracle[row, component] + multiplier * (
                    trapezoid / 3 - exact_low
                )
                corrected[row, component] = oracle[row, component] + multiplier * (
                    trapezoid - exact_low
                )
        source_mismatch = float(
            np.max(np.abs((observed[valid] - predicted[valid]) / oracle[valid]))
        )
        entry = dict(
            delta=delta,
            nint=nint,
            observed=observed.tolist(),
            oracle=oracle.tolist(),
            relative=relative.tolist(),
            source_mismatch=source_mismatch,
            corrected_relative=((corrected[valid] - oracle[valid]) / oracle[valid]).tolist(),
        )
        results.append(entry)
        print(
            f"aspect={aspect:g} species={species} scale={scale:g} gamma={gamma:g} "
            f"delta={delta:g} nint={nint} px={observed[2, 0]:.16g} py={observed[2, 1]:.16g} "
            f"oracle_px={oracle[2, 0]:.16g} oracle_py={oracle[2, 1]:.16g} "
            f"rel_px={relative[2, 0]:.12g} rel_py={relative[2, 1]:.12g} "
            f"source_residual={source_mismatch:.12g} "
            f"restored_weight_max_error={np.max(np.abs((corrected[valid] - oracle[valid]) / oracle[valid])):.12g}"
        )
        if nint == 4001:
            assert source_mismatch < 2e-10
        if delta == 2e-6:
            assert np.max(np.abs(relative)) < 4e-9
    result = dict(
        settings=dict(aspect=aspect, scale=scale, species=species, gamma=gamma),
        sx=sx,
        sy=sy,
        gamma=ref.gamma,
        particles=len(positions),
        probes=probes.tolist(),
        cases=results,
    )
    return result


def main():
    sim = simulation(
        space_charge="Gauss2p5D",
        space_charge_gauss_charge_z_bins=25,
        space_charge_apply_longitudinal_kick=False,
    )
    try:
        runs = [case(sim, aspect) for aspect in (1.0, 0.25, 4.0)]
        scaled = case(sim, 0.25, scale=3.0, species="proton", gamma=19.0)
        scaling_error = max(
            float(np.max(np.abs(np.asarray(a["relative"]) - b["relative"])))
            for a, b in zip(runs[1]["cases"], scaled["cases"], strict=True)
        )
        assert scaling_error < 2e-14
        print(f"length_mass_energy_control_relative_difference={scaling_error:.12g}")
        reproduced = all(abs(run["cases"][3]["relative"][2][0]) > 3e-5 for run in runs)
        print("DEFECT REPRODUCED" if reproduced else "NOT REPRODUCED")
        if not reproduced:
            raise AssertionError("Expected the initial-interval quadrature bias")
    finally:
        sim.finalize()


if __name__ == "__main__":
    main()
