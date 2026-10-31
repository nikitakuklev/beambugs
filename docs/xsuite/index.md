---
title: Xsuite
hide:
  - toc
---

# Xsuite

Verified defects in [Xsuite](https://xsuite.readthedocs.io) ([source](https://github.com/xsuite/xsuite)).
Each entry links to its reproducer, the models that found and confirmed it, and any upstream issue or pull request.
Machine-readable: [defects.json](defects.json).
Tags are explained in [Methodology](../methodology.md#tags).

## Running the reproducers

Install Xsuite 0.66.3 (`pip install xsuite==0.66.3`; the reproducers ran with xtrack
0.118.2, xfields 0.27.5, xpart 0.23.23, xobjects 0.7.2, xdeps 0.10.21 and xcoll
0.12.5) and build its prebuilt CPU kernels with `xsuite-prebuild regenerate`.
Download a reproducer together with the `_*.py` helpers shown beside it and run
`python <ID>.py`, one script per process.

Some reproducers also need: cpymad, madng-tpsa, scipy.

## Defects

<div class="bb-filters" data-column="Area"></div>

| ID | Defect | Area | Priority | Severity | Last reproduced | Issue | PR |
|---|---|---|---|---|---|---|---|
| [TRK-01](TRK-01/index.md) | Solenoid body ignores per-particle charge-to-mass scaling | Tracking | P2 | Wrong physics | 2026-10-26 (`3cd8bfc`, main) |  |  |
| [TRK-02](TRK-02/index.md) | Tracking optimization drops a full bend's multipole fringe | Tracking | P2 | Wrong physics | 2026-10-26 (`3cd8bfc`, main) |  | [[gh:https://github.com/xsuite/xtrack/pull/911]] |
| [TRK-03](TRK-03/index.md) | RF-derived linear segment map produces NaNs for electrons | Tracking | P3 | Wrong physics | 2026-10-26 (`3cd8bfc`, main) |  |  |
| [TRK-04](TRK-04/index.md) | Optimizing RF slices drops roll and phase taper | Tracking | P1 | Wrong physics | 2026-10-26 (`3cd8bfc`, main) |  |  |
| [TRK-05](TRK-05/index.md) | Sliced thick Multipole repeats the full reference bend angle | Tracking | P2 | Wrong physics | 2026-10-26 (`3cd8bfc`, main) |  |  |
| [TRK-06](TRK-06/index.md) | Zero linear chromaticity shifts the quadratic coefficient into the linear slot | Tracking | P2 | Wrong physics | 2026-10-26 (`3cd8bfc`, main) |  |  |
| [TRK-07](TRK-07/index.md) | Optimization disables radiation in curved-body RBend slices | Tracking | P1 | Wrong physics | 2026-10-26 (`3cd8bfc`, main) |  |  |
| [TRK-08](TRK-08/index.md) | SplineBoris omits species factors in physical charge and momentum | Tracking | P3 | Wrong physics | 2026-10-26 (`3cd8bfc`, main) |  |  |
| [TRK-09](TRK-09/index.md) | Solenoid spin tracking uses the wrong kinetic momentum | Tracking | P2 | Wrong physics | 2026-10-30 (`ad5e808`; not on main) |  | [[gh:https://github.com/xsuite/xtrack/pull/923]] |
| [TRK-10](TRK-10/index.md) | Linear dipole-edge finite-gap focusing has wrong species scaling | Tracking | P3 | Wrong physics | 2026-10-26 (`3cd8bfc`, main) |  |  |
| [TRK-11](TRK-11/index.md) | Elens has inverted species scaling and extra momentum dependence | Tracking | P3 | Wrong physics | 2026-10-29 (`f09e0fb`, main) |  |  |
| [TRK-12](TRK-12/index.md) | Wire kicks omit particle charge-to-mass scaling | Tracking | P3 | Wrong physics | 2026-10-29 (`f09e0fb`, main) |  |  |
| [TRK-13](TRK-13/index.md) | Exciter activates before its stated starting time | Tracking | P3 | Wrong physics | 2026-10-29 (`f09e0fb`, main) |  |  |
| [TRK-14](TRK-14/index.md) | Elens residual kicks fail forward/backward tracking | Tracking | P3 | Wrong physics | 2026-10-29 (`f09e0fb`, main) |  |  |
| [TRK-15](TRK-15/index.md) | Zero horizontal focusing loses accelerated-path terms | Tracking | P2 | Wrong physics | 2026-10-29 (`f09e0fb`, main) |  |  |
| [TRK-16](TRK-16/index.md) | Quantum photon rate omits reference charge squared | Tracking | P3 | Wrong physics | 2026-10-29 (`f09e0fb`, main) |  |  |
| [TRK-17](TRK-17/index.md) | Off-origin polygon apertures retain particles outside their boundary <span class="bb-tag">partly-source-only</span> | Tracking | P2 | Wrong physics | 2026-10-29 (`f09e0fb`, main) |  |  |
| [OPT-01](OPT-01/index.md) | Native Taylor maps silently omit RFMultipole and CrabCavity kicks <span class="bb-tag">experimental</span> | Optics and matching | P4 | Wrong physics | 2026-10-26 (`3cd8bfc`, main) |  |  |
| [OPT-02](OPT-02/index.md) | Native Taylor maps ignore an absolute-time cavity's RF clock <span class="bb-tag">experimental</span> <span class="bb-tag">known-limitation</span> | Optics and matching | P4 | Silent wrong result | 2026-10-26 (`3cd8bfc`, main) |  |  |
| [OPT-03](OPT-03/index.md) | Solenoid Taylor map has zero strength sensitivity at zero field <span class="bb-tag">experimental</span> | Optics and matching | P4 | Silent wrong result | 2026-10-26 (`3cd8bfc`, main) |  |  |
| [OPT-04](OPT-04/index.md) | Finite-difference transfer matrices reset the supplied RF clock | Optics and matching | P2 | Silent wrong result | 2026-10-26 (`3cd8bfc`, main) |  |  |
| [OPT-05](OPT-05/index.md) | 4D Twiss retains RF energy kicks from LineSegmentMap | Optics and matching | P2 | Silent wrong result | 2026-10-26 (`3cd8bfc`, main) |  |  |
| [OPT-06](OPT-06/index.md) | A subrange's second-order chromaticity depends on its absolute s position | Optics and matching | P2 | Silent wrong result | 2026-10-26 (`3cd8bfc`, main) |  |  |
| [OPT-07](OPT-07/index.md) | Periodic subrange slip factor uses the full parent line length | Optics and matching | P2 | Silent wrong result | 2026-10-26 (`3cd8bfc`, main) |  |  |
| [OPT-08](OPT-08/index.md) | Extracted Twiss init loses a downstream reference-momentum change | Optics and matching | P2 | Silent wrong result | 2026-10-26 (`3cd8bfc`, main) |  |  |
| [OPT-09](OPT-09/index.md) | Sliced decelerating cavities are counted as synchrotron-radiation loss | Optics and matching | P2 | Silent wrong result | 2026-10-26 (`3cd8bfc`, main) |  |  |
| [OPT-10](OPT-10/index.md) | Sequential step clipping violates earlier variable bounds | Optics and matching | P4 | Silent wrong result | 2026-10-29 (`6484795`, main) |  |  |
| [OPT-11](OPT-11/index.md) | Variable weights distort physical maximum steps | Optics and matching | P4 | Silent wrong result | 2026-10-29 (`6484795`, main) |  |  |
| [OPT-12](OPT-12/index.md) | Knob construction applies absolute limits to additive corrections | Optics and matching | P2 | Silent wrong result | 2026-10-29 (`f09e0fb`, main) |  |  |
| [OPT-13](OPT-13/index.md) | Reversed Twiss tables reconstruct incorrect transverse transfer maps | Optics and matching | P2 | Silent wrong result | 2026-10-29 (`f09e0fb`, main) |  |  |
| [OPT-14](OPT-14/index.md) | Full radiation equilibrium covariance omits accumulated diffusion | Optics and matching | P3 | Silent wrong result | 2026-10-29 (`f09e0fb`, main) |  |  |
| [OPT-15](OPT-15/index.md) | Closest tune approach is inflated across an integer boundary | Optics and matching | P3 | Silent wrong result | 2026-10-29 (`f09e0fb`, main) |  |  |
| [COL-01](COL-01/index.md) | q-Gaussian normalization overflows near the Gaussian limit | Collective effects | P3 | Wrong physics | 2026-10-26 (`223b4fa`, main) |  |  |
| [COL-02](COL-02/index.md) | Displaced Gaussian space charge integrates the wrong longitudinal-kick ray | Collective effects | P2 | Wrong physics | 2026-10-26 (`223b4fa`, main) |  |  |
| [COL-03](COL-03/index.md) | Particle-to-mesh deposition omits source charge ratios <span class="bb-tag">known-limitation</span> | Collective effects | P3 | Wrong physics | 2026-10-26 (`223b4fa`, main) |  |  |
| [COL-04](COL-04/index.md) | CollectiveMonitor slice emittance omits the covariance square | Collective effects | P4 | Silent wrong result | 2026-10-26 (`223b4fa`, main) |  |  |
| [COL-05](COL-05/index.md) | Strong-strong 2D beam-beam moments include lost particles | Collective effects | P2 | Wrong physics | 2026-10-26 (`223b4fa`, main) |  |  |
| [COL-06](COL-06/index.md) | 2.5D FFT space charge: the longitudinal kick of a nonuniform bunch violates geometric similarity <span class="bb-tag">contract-unsettled</span> | Collective effects | P2 | Wrong physics | 2026-10-29 (`3f06c87`, main) |  |  |
| [COL-07](COL-07/index.md) | Loss-map periodic aperture cells have negative lengths | Collective effects | P3 | Silent wrong result | 2026-10-29 (`64e13f5`, main) |  |  |
| [COL-08](COL-08/index.md) | Loss-map initial energy double-counts preceding batches | Collective effects | P4 | Silent wrong result | 2026-10-29 (`64e13f5`, main) |  |  |
| [COL-09](COL-09/index.md) | Noninterpolated aperture losses are omitted from total energy | Collective effects | P4 | Silent wrong result | 2026-10-29 (`64e13f5`, main) |  |  |
| [COL-10](COL-10/index.md) | Noninterpolated loss maps discard coordinate reversal | Collective effects | P3 | Silent wrong result | 2026-10-29 (`64e13f5`, main) |  |  |
| [COL-11](COL-11/index.md) | Interpolated electron lens omits the transit-speed factor | Collective effects | P3 | Wrong physics | 2026-10-29 (`3f06c87`, main) |  |  |
| [COL-12](COL-12/index.md) | ElectronCloud applies inconsistent reference-charge factors | Collective effects | P3 | Wrong physics | 2026-10-29 (`3f06c87`, main) |  |  |
| [COL-13](COL-13/index.md) | Everest deposited-energy accumulators omit macroparticle weights <span class="bb-tag">partly-source-only</span> | Collective effects | P3 | Silent wrong result | 2026-10-29 (`64e13f5`, main) |  |  |
| [COL-14](COL-14/index.md) | Loss-map primary energy uses secondary deposits | Collective effects | P3 | Silent wrong result | 2026-10-29 (`64e13f5`, main) |  |  |
| [COL-15](COL-15/index.md) | Loss-map export drops deposits without absorbed particles | Collective effects | P3 | Silent wrong result | 2026-10-29 (`64e13f5`, main) |  |  |
| [COL-16](COL-16/index.md) | Touschek Monte Carlo diagnostic omits sampled angular width <span class="bb-tag">partly-source-only</span> | Collective effects | P4 | Silent wrong result | 2026-10-29 (`3f06c87`, main) |  |  |
| [COL-17](COL-17/index.md) | All-lost tracking calls freeze wake history | Collective effects | P3 | Wrong physics | 2026-10-29 (`3f06c87`, main) |  |  |
| [COL-18](COL-18/index.md) | Round-beam variance derivatives use an absolute near-axis cutoff <span class="bb-tag">partly-source-only</span> | Collective effects | P3 | Wrong physics | 2026-10-29 (`3f06c87`, main) |  |  |
| [COL-19](COL-19/index.md) | B&M applies an extra beta factor to momentum dispersion | Collective effects | P2 | Wrong physics | 2026-10-29 (`3f06c87`, main) |  |  |
| [COL-20](COL-20/index.md) | Zero horizontal dispersion invariant creates a NaN B&M rate | Collective effects | P3 | Crash or API hazard | 2026-10-29 (`3f06c87`, main) |  |  |
| [IO-01](IO-01/index.md) | Single-harmonic longitudinal generation uses proton mass for other species | Lattices, particles and I/O | P2 | Wrong physics | 2026-10-26 (`2633e52`, main) |  |  |
| [IO-03](IO-03/index.md) | Checkpoint loading restores RNG words onto reordered particle coordinates | Lattices, particles and I/O | P4 | Silent wrong result | 2026-10-26 (`3cd8bfc`, main) |  |  |
| [IO-04](IO-04/index.md) | Renaming a MAD-X deferred variable freezes its contribution to imported dependants | Lattices, particles and I/O | P3 | Silent wrong result | 2026-10-26 (`3cd8bfc`, main) |  |  |
| [IO-05](IO-05/index.md) | MAD-X bend roll errors are imported as design tilts | Lattices, particles and I/O | P1 | Wrong physics | 2026-10-26 (`3cd8bfc`, main) |  |  |
| [IO-06](IO-06/index.md) | Negative-charge reference rigidity fails a constructor round trip | Lattices, particles and I/O | P3 | Silent wrong result | 2026-10-26 (`3cd8bfc`, main) |  |  |
| [IO-07](IO-07/index.md) | Manager.copy disconnects containers from dependency tasks | Lattices, particles and I/O | P4 | Silent wrong result | 2026-10-29 (`6484795`, main) |  |  |
| [IO-08](IO-08/index.md) | Accelerating RF contour emittance uses symmetric turning points | Lattices, particles and I/O | P4 | Silent wrong result | 2026-10-29 (`2633e52`, main) |  |  |
| [IO-09](IO-09/index.md) | RF phase setter leaves cached bucket geometry stale | Lattices, particles and I/O | P4 | Silent wrong result | 2026-10-29 (`2633e52`, main) |  |  |
| [IO-10](IO-10/index.md) | Linear longitudinal matching omits position-momentum correlation <span class="bb-tag">partly-source-only</span> | Lattices, particles and I/O | P3 | Wrong physics | 2026-10-29 (`2633e52`, main) |  |  |
| [CTX-01](CTX-01/index.md) | CPU linked-array views bypass dependent-state updates | Contexts and kernels | P1 | Wrong physics | 2026-10-29 (`dfe34ae`, main) |  |  |
| [CTX-02](CTX-02/index.md) | Copied nested fields disconnect from parent storage | Contexts and kernels | P1 | Wrong physics | 2026-10-29 (`dfe34ae`, main) |  |  |
