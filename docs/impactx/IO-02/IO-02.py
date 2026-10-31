"""IO-02: Phase-space charge histograms ignore particle weights

The phase-space plot from `plot_phasespace()` labels its charge histograms Q [C/bin] but counts
macroparticles instead of summing their charges q_ref·w, where q_ref is the reference-particle
charge and w the number of real particles a macroparticle represents. The displayed values are
therefore counts, and bins whose particles carry different weights also have wrong relative charges.
For a 1.550 nC test beam, a 2×2 x–px histogram shows 16 in every bin instead of the charges from
8.613×10⁻¹¹ to 8.613×10⁻¹⁰ C that a direct sum of q_ref·w gives, and splitting each macroparticle
into two half-weight copies doubles every bin while the beam is unchanged. The numerical charge and
moment diagnostics are correct.

The script builds a deterministic beam of 64 weighted particles at the vertices of a box in (x, px,
y, py, t, pt) for a γ = 7 proton reference, checks its total charge and emittances against analytic
values, and draws the plot with two bins per axis using the installed `plot_phasespace()` function
and the unmodified function from the ImpactX development branch. It compares each panel's displayed
bins with charges summed directly as q_ref·w over the known sign categories. Repeating with each
particle split into two half-weight copies, which leaves the beam unchanged, doubles the displayed
bins. Set IMPACTX_SOURCE to a clone of the ImpactX repository.
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

import ast
import inspect
import subprocess
from importlib import import_module
from importlib.metadata import version
from itertools import product

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from impactx import ImpactX
from impactx.impactx_pybind import CoordSystem

REPO = os.environ["IMPACTX_SOURCE"]
PATH = "src/python/impactx/extensions/ImpactXParticleContainer.py"
module = import_module("impactx.extensions.ImpactXParticleContainer")
pinned = subprocess.check_output(["git", "-C", REPO, "show", f"615921c6867c:{PATH}"], text=True)
assert inspect.getsource(module) == pinned


def development_plot():
    source = subprocess.check_output(
        ["git", "-C", REPO, "show", f"b701a6e9d1f9e30cefb9a3e18e82ae7e02b9d943:{PATH}"], text=True
    )
    functions: list[ast.stmt] = [
        node
        for node in ast.parse(source).body
        if isinstance(node, ast.FunctionDef) and node.name == "ix_pc_plot_mpl_phasespace"
    ]
    assert len(functions) == 1
    namespace = {"CoordSystem": CoordSystem}
    exec(  # noqa: S102 -- execute the inspected upstream function without editing its body
        compile(
            ast.Module(body=functions, type_ignores=[]),
            "<unmodified development plotting function>",
            "exec",
        ),
        namespace,
    )
    return namespace["ix_pc_plot_mpl_phasespace"]


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
    plotters = [
        ("pinned 26.10", module.ix_pc_plot_mpl_phasespace),
        ("development function / 26.10 core", development_plot()),
    ]
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
                    np.testing.assert_array_equal(displayed, np.full((2, 2), 16 * split))
                    np.testing.assert_allclose(expected.sum(), charge, rtol=3e-15, atol=0)
                    assert not np.allclose(
                        expected / expected.sum(), displayed / displayed.sum(), rtol=1e-10, atol=0
                    )
                    assert panel.images[0].colorbar.ax.get_ylabel() == r"$Q$ [C/bin]"
                    print(
                        f"{plane}: displayed bins={displayed.tolist()} sum={displayed.sum():g}; physical bins={expected.tolist()}"
                    )
            finally:
                plt.close(fig)
finally:
    sim.finalize()
    plot_config.cleanup()
print("DEFECT REPRODUCED")
