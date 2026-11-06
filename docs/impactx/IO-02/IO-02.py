"""IO-02: Phase-space charge histograms ignore particle weights

The phase-space plot from `plot_phasespace()` labels its charge histograms Q [C/bin] but counts
macroparticles instead of summing their charges q_ref·w, where q_ref is the reference-particle
charge and w the number of real particles a macroparticle represents. The displayed values are
therefore counts, and bins whose particles carry different weights also have wrong relative charges.
For a 1.550 nC test beam, a 2×2 x–px histogram shows 16 in every bin instead of the charges from
8.613×10⁻¹¹ to 8.613×10⁻¹⁰ C that a direct sum of q_ref·w gives, and splitting each macroparticle
into two half-weight copies doubles every bin while the beam is unchanged. The numerical charge and
moment diagnostics are correct.

The script builds a deterministic weighted product beam at γ = 7 and checks its total charge and emittances against analytic values. It compares the installed two-bin phase-space histograms with charges summed directly over known sign categories. Repeating with each particle split into two half-weight copies must preserve the physical bin charges.
"""

import os

for key in (
    "OMP_NUM_THREADS",
    "OPENBLAS_NUM_THREADS",
    "MKL_NUM_THREADS",
    "NUMEXPR_NUM_THREADS",
    "NPROC",
    "PJRT_NPROC",
):
    os.environ[key] = "1"
from pathlib import Path
from tempfile import TemporaryDirectory

plot_config = TemporaryDirectory(prefix=".IO-02-mpl-", dir=Path(__file__).resolve().parent)
os.environ["MPLCONFIGDIR"] = plot_config.name
os.sched_setaffinity(0, {min(os.sched_getaffinity(0))})

from importlib import import_module
from importlib.metadata import version
from itertools import product

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from impactx import ImpactX

module = import_module("impactx.extensions.ImpactXParticleContainer")
bad = False


# Each independent +/- coordinate has probability ratio 1:r. Analytic stddev:
# sigma = 2 a sqrt(r)/(1+r). All coordinate covariances are zero.
ratios = np.array([2.0, 5.0, 3.0, 7.0, 11.0, 13.0])
amplitudes = np.array([0.8e-3, 0.3e-3, 0.6e-3, 0.2e-3, 1.5e-3, 4e-3])
bits = np.array(list(product([0, 1], repeat=6)))
u = (2 * bits - 1) * amplitudes
weights = 1e5 * np.prod(ratios**bits, axis=1)
sigmas = 2 * amplitudes * np.sqrt(ratios) / (1 + ratios)
emittances = sigmas[::2] * sigmas[1::2]
gamma = 7.0
bg = np.sqrt(gamma**2 - 1)

sim = ImpactX()
sim.particle_shape = 1
sim.space_charge = False
sim.diagnostics = False
sim.slice_step_diagnostics = False
sim.verbose = 0
sim.init_grids()

try:
    print(
        "impactx-noacc",
        version("impactx-noacc"),
        "numpy",
        version("numpy"),
        "matplotlib",
        version("matplotlib"),
    )
    print("Analytic product beam: 64 vertices; splitting each macroparticle preserves all physics")
    plotters = [("installed plotter", module.ix_pc_plot_mpl_phasespace)]
    for split in [1, 2]:
        sim.beam.clear_particles()
        ref = sim.beam.ref
        ref.reset()
        ref.set_species("proton").set_kin_energy_MeV((gamma - 1) * ref.mass_MeV)
        samples = np.repeat(u, split, axis=0)
        sample_weights = np.repeat(weights / split, split)
        sim.beam.add_n_particles(
            **dict(zip(("x", "px", "y", "py", "t", "pt"), samples.T.copy())),
            qm=ref.qm_ratio_SI,
            w=sample_weights,
        )
        moments = sim.beam.beam_moments()
        charge = ref.charge * 1e5 * np.prod(1 + ratios)
        np.testing.assert_allclose(moments["charge_C"], charge, rtol=3e-15, atol=0)
        for j, plane in enumerate("xyt"):
            np.testing.assert_allclose(
                moments["emittance_" + plane], emittances[j], rtol=3e-14, atol=0
            )
            np.testing.assert_allclose(
                moments["emittance_" + plane + "n"], emittances[j] * bg, rtol=3e-14, atol=0
            )
        for label, plotter in plotters:
            fig = plotter(sim.beam, num_bins=2)
            try:
                panels = [axis for axis in fig.axes if axis.images]
                assert len(panels) == 3
                print(f"--- {label}; split={split}; charge={charge:.15g} C ---")
                for j, (plane, panel) in enumerate(zip("xyt", panels)):
                    displayed = np.ma.filled(panel.images[0].get_array(), 0)
                    # Direct sum over known sign categories, independent of histogram2d.
                    expected = np.zeros((2, 2))
                    for b, population in zip(bits, weights):
                        expected[b[2 * j + 1], b[2 * j]] += ref.charge * population
                    np.testing.assert_allclose(expected.sum(), charge, rtol=3e-15, atol=0)
                    shape_matches = np.allclose(
                        expected / expected.sum(), displayed / displayed.sum(), rtol=1e-10, atol=0
                    )
                    bad |= not (
                        shape_matches and np.allclose(displayed, expected, rtol=3e-15, atol=0)
                    )
                    assert panel.images[0].colorbar.ax.get_ylabel() in (
                        r"$Q$ [C/bin]",
                        r"$|Q|$ [C/bin]",
                    )
                    print(
                        f"{plane}: displayed bins={displayed.tolist()} sum={displayed.sum():g}; physical bins={expected.tolist()}"
                    )
            finally:
                plt.close(fig)
finally:
    sim.finalize()
    plot_config.cleanup()
print("DEFECT REPRODUCED" if bad else "NOT REPRODUCED")
