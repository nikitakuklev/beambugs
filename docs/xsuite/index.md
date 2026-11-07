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

1. Install Xsuite 0.66.3 (`pip install xsuite==0.66.3`); the reproducers ran with xtrack
   0.118.2, xfields 0.27.5, xpart 0.23.23, xobjects 0.7.2, xdeps 0.10.21 and xcoll 0.12.5.
2. Build its prebuilt CPU kernels with `xsuite-prebuild regenerate`. Scripts that use the
   `_prebuilt` helper forbid kernel compilation and run only these kernels.
3. Download a reproducer and the `_*.py` helpers linked on its entry page from the shared
   [helpers directory](helpers/index.md), and save the helpers beside the script.
4. Run `python <ID>.py`, one script per process.

A script that needs more than this states it in a `Setup:` paragraph of its description.

Some reproducers also need: cpymad, madng-tpsa, mpmath, scipy.

## Defects

<div class="bb-filters" data-column="Area"></div>

| ID | Defect | Area | Priority | Severity | Last reproduced | Issue | PR |
|---|---|---|---|---|---|---|---|
| [TRK-04](TRK-04/index.md) | Optimizing RF slices drops roll and phase taper | Tracking | P1 | Wrong physics | 2026-10-26 (`3cd8bfc`, head) |  |  |
| [TRK-07](TRK-07/index.md) | Optimization disables radiation in curved-body RBend slices | Tracking | P1 | Wrong physics | 2026-10-26 (`3cd8bfc`, head) |  |  |
| [IO-05](IO-05/index.md) | MAD-X bend roll errors are imported as design tilts | Lattices, particles and I/O | P1 | Wrong physics | 2026-10-26 (`3cd8bfc`, head) |  |  |
| [CTX-01](CTX-01/index.md) | CPU linked-array views bypass dependent-state updates | Contexts and kernels | P1 | Wrong physics | 2026-10-29 (`dfe34ae`, head) |  |  |
| [CTX-02](CTX-02/index.md) | Copied nested fields disconnect from parent storage | Contexts and kernels | P1 | Wrong physics | 2026-10-29 (`dfe34ae`, head) |  |  |
| [TRK-01](TRK-01/index.md) | Solenoid body ignores per-particle charge-to-mass scaling | Tracking | P2 | Wrong physics | 2026-10-26 (`3cd8bfc`, head) |  |  |
| [TRK-02](TRK-02/index.md) | Tracking optimization drops a full bend's multipole fringe | Tracking | P2 | Wrong physics | 2026-10-26 (`3cd8bfc`, head) |  | [[gh:https://github.com/xsuite/xtrack/pull/911]] |
| [TRK-05](TRK-05/index.md) | Sliced thick Multipole repeats the full reference bend angle | Tracking | P2 | Wrong physics | 2026-10-26 (`3cd8bfc`, head) |  |  |
| [TRK-06](TRK-06/index.md) | Zero linear chromaticity shifts the quadratic coefficient into the linear slot | Tracking | P2 | Wrong physics | 2026-10-26 (`3cd8bfc`, head) |  |  |
| [TRK-09](TRK-09/index.md) | Solenoid spin tracking uses the wrong kinetic momentum | Tracking | P2 | Wrong physics | 2026-10-26 (`ad5e808`; not on head 2026-10-29) |  | [[gh:https://github.com/xsuite/xtrack/pull/923]] |
| [TRK-15](TRK-15/index.md) | Zero horizontal focusing loses accelerated-path terms | Tracking | P2 | Wrong physics | 2026-10-29 (`f09e0fb`, head) |  |  |
| [TRK-17](TRK-17/index.md) | Off-origin polygon apertures retain particles outside their boundary <span class="bb-tag">partly-source-only</span> | Tracking | P2 | Wrong physics | 2026-10-29 (`f09e0fb`, head) |  |  |
| [TRK-18](TRK-18/index.md) | Radiation tracking omits the field wedges of angled dipole faces <span class="bb-tag">partly-source-only</span> | Tracking | P2 | Wrong physics | 2026-11-07 (`f09e0fb`, head) |  |  |
| [TRK-19](TRK-19/index.md) | Spin precession omits tracked charge-to-mass scaling <span class="bb-tag">partly-source-only</span> | Tracking | P2 | Wrong physics | 2026-11-07 (`f09e0fb`, head) |  |  |
| [TRK-23](TRK-23/index.md) | Optimized thin bends scale geometric arrival shifts by charge-to-mass ratio | Tracking | P2 | Wrong physics | 2026-11-07 (`f09e0fb`, head) |  |  |
| [TRK-24](TRK-24/index.md) | Thin-element simplification discards quadrupole and dipole spin transport | Tracking | P2 | Wrong physics | 2026-11-07 (`f09e0fb`, head) |  |  |
| [TRK-26](TRK-26/index.md) | Backtracking silently skips spin transport and its rejection <span class="bb-tag">partly-source-only</span> | Tracking | P2 | Wrong physics | 2026-11-07 (`f09e0fb`, head) | [[gh:https://github.com/xsuite/xsuite/issues/647]] |  |
| [TRK-27](TRK-27/index.md) | Slicing helper names overwrite existing physical elements <span class="bb-tag">partly-source-only</span> | Tracking | P2 | Wrong physics | 2026-11-07 (`f09e0fb`, head) |  |  |
| [TRK-28](TRK-28/index.md) | Absolute-time RF phase loses particle arrival offsets at long clocks <span class="bb-tag">partly-source-only</span> | Tracking | P2 | Wrong physics | 2026-11-07 (`f09e0fb`, head) |  |  |
| [TRK-29](TRK-29/index.md) | Absolute-time RF clock accumulates rounding once per turn <span class="bb-tag">partly-source-only</span> | Tracking | P2 | Wrong physics | 2026-11-07 (`f09e0fb`, head) |  |  |
| [OPT-16](OPT-16/index.md) | Radiation integrals omit the dipole-face term of I₄ | Optics and matching | P2 | Wrong physics | 2026-11-07 (`f09e0fb`, head) |  |  |
| [OPT-23](OPT-23/index.md) | Periodic spin Twiss omits the longitudinal fixed-point constraint | Optics and matching | P2 | Wrong physics | 2026-11-06 (`ad5e808`; not on head 2026-11-06) |  | [[gh:https://github.com/xsuite/xtrack/pull/923]] |
| [COL-02](COL-02/index.md) | Displaced Gaussian space charge integrates the wrong longitudinal-kick ray | Collective effects | P2 | Wrong physics | 2026-10-26 (`223b4fa`, head) |  |  |
| [COL-05](COL-05/index.md) | Strong-strong 2D beam-beam moments include lost particles | Collective effects | P2 | Wrong physics | 2026-10-26 (`223b4fa`, head) |  |  |
| [COL-06](COL-06/index.md) | 2.5D FFT space charge: the longitudinal kick of a nonuniform bunch violates geometric similarity <span class="bb-tag">contract-unsettled</span> | Collective effects | P2 | Wrong physics | 2026-10-29 (`3f06c87`, head) |  |  |
| [COL-19](COL-19/index.md) | B&M applies an extra beta factor to momentum dispersion | Collective effects | P2 | Wrong physics | 2026-10-29 (`3f06c87`, head) |  |  |
| [COL-23](COL-23/index.md) | Gaussian space-charge construction and reload discard placement | Collective effects | P2 | Wrong physics | 2026-11-07 (`3f06c87`, head) |  |  |
| [IO-01](IO-01/index.md) | Single-harmonic longitudinal generation uses proton mass for other species | Lattices, particles and I/O | P2 | Wrong physics | 2026-10-26 (`2633e52`, head) |  |  |
| [IO-14](IO-14/index.md) | MAD-X export loses embedded quadrupole dipole steering | Lattices, particles and I/O | P2 | Wrong physics | 2026-11-07 (`f09e0fb`, head) |  |  |
| [IO-20](IO-20/index.md) | Reverse MAD-X import keeps forward horizontal aperture offsets | Lattices, particles and I/O | P2 | Wrong physics | 2026-11-07 (`f09e0fb`, head) |  |  |
| [IO-21](IO-21/index.md) | Exported magnet body slices repeat the parent's integrated multipole strengths | Lattices, particles and I/O | P2 | Wrong physics | 2026-11-07 (`f09e0fb`, head) |  |  |
| [IO-23](IO-23/index.md) | Exported thin-mode slices occupy the parent's length a second time | Lattices, particles and I/O | P2 | Wrong physics | 2026-11-07 (`f09e0fb`, head) |  |  |
| [IO-24](IO-24/index.md) | Exported bend body slices each repeat the parent's edge faces | Lattices, particles and I/O | P2 | Wrong physics | 2026-11-07 (`f09e0fb`, head) |  |  |
| [IO-26](IO-26/index.md) | Exported Cavity slices each repeat the parent's full voltage | Lattices, particles and I/O | P2 | Wrong physics | 2026-11-07 (`f09e0fb`, head) |  |  |
| [IO-27](IO-27/index.md) | Bend export replaces the exit half-gap by the entrance half-gap <span class="bb-tag">partly-source-only</span> | Lattices, particles and I/O | P2 | Wrong physics | 2026-11-07 (`f09e0fb`, head) |  |  |
| [OPT-04](OPT-04/index.md) | Finite-difference transfer matrices reset the supplied RF clock | Optics and matching | P2 | Silent wrong result | 2026-10-26 (`3cd8bfc`, head) |  |  |
| [OPT-05](OPT-05/index.md) | 4D Twiss retains RF energy kicks from LineSegmentMap | Optics and matching | P2 | Silent wrong result | 2026-10-26 (`3cd8bfc`, head) |  |  |
| [OPT-06](OPT-06/index.md) | A subrange's second-order chromaticity depends on its absolute s position | Optics and matching | P2 | Silent wrong result | 2026-10-26 (`3cd8bfc`, head) |  |  |
| [OPT-07](OPT-07/index.md) | Periodic subrange slip factor uses the full parent line length | Optics and matching | P2 | Silent wrong result | 2026-10-26 (`3cd8bfc`, head) |  |  |
| [OPT-08](OPT-08/index.md) | Extracted Twiss init loses a downstream reference-momentum change | Optics and matching | P2 | Silent wrong result | 2026-10-26 (`3cd8bfc`, head) |  |  |
| [OPT-09](OPT-09/index.md) | Sliced decelerating cavities are counted as synchrotron-radiation loss | Optics and matching | P2 | Silent wrong result | 2026-10-26 (`3cd8bfc`, head) |  |  |
| [OPT-12](OPT-12/index.md) | Knob construction applies absolute limits to additive corrections | Optics and matching | P2 | Silent wrong result | 2026-10-29 (`f09e0fb`, head) |  |  |
| [OPT-13](OPT-13/index.md) | Reversed Twiss tables reconstruct incorrect transverse transfer maps | Optics and matching | P2 | Silent wrong result | 2026-10-29 (`f09e0fb`, head) |  |  |
| [OPT-18](OPT-18/index.md) | Twiss transfer matrices omit reference-change amplitudes | Optics and matching | P2 | Silent wrong result | 2026-11-07 (`f09e0fb`, head) |  |  |
| [OPT-19](OPT-19/index.md) | Normalized-emittance covariance uses the initialization rigidity | Optics and matching | P2 | Silent wrong result | 2026-11-07 (`f09e0fb`, head) |  |  |
| [OPT-20](OPT-20/index.md) | Open chromatic Twiss launches pζ dispersion as a δ derivative | Optics and matching | P2 | Silent wrong result | 2026-11-07 (`f09e0fb`, head) |  |  |
| [OPT-21](OPT-21/index.md) | Propagated longitudinal W uses the initialization reference speed | Optics and matching | P2 | Silent wrong result | 2026-11-07 (`f09e0fb`, head) |  |  |
| [OPT-25](OPT-25/index.md) | Periodic Twiss converts canonical time slip with reference-orbit factors | Optics and matching | P2 | Silent wrong result | 2026-11-07 (`f09e0fb`, head) |  |  |
| [TRK-03](TRK-03/index.md) | RF-derived linear segment map produces NaNs for electrons | Tracking | P3 | Wrong physics | 2026-10-26 (`3cd8bfc`, head) |  |  |
| [TRK-08](TRK-08/index.md) | SplineBoris omits species factors in physical charge and momentum | Tracking | P3 | Wrong physics | 2026-10-26 (`3cd8bfc`, head) |  |  |
| [TRK-10](TRK-10/index.md) | Linear dipole-edge finite-gap focusing has wrong species scaling | Tracking | P3 | Wrong physics | 2026-10-26 (`3cd8bfc`, head) |  |  |
| [TRK-11](TRK-11/index.md) | Elens has inverted species scaling and extra momentum dependence | Tracking | P3 | Wrong physics | 2026-10-29 (`f09e0fb`, head) |  |  |
| [TRK-12](TRK-12/index.md) | Wire kicks omit particle charge-to-mass scaling | Tracking | P3 | Wrong physics | 2026-10-29 (`f09e0fb`, head) |  |  |
| [TRK-13](TRK-13/index.md) | Exciter activates before its stated starting time | Tracking | P3 | Wrong physics | 2026-10-29 (`f09e0fb`, head) |  |  |
| [TRK-14](TRK-14/index.md) | Elens residual kicks fail forward/backward tracking | Tracking | P3 | Wrong physics | 2026-10-29 (`f09e0fb`, head) |  |  |
| [TRK-16](TRK-16/index.md) | Quantum photon rate omits reference charge squared | Tracking | P3 | Wrong physics | 2026-10-29 (`f09e0fb`, head) |  |  |
| [TRK-20](TRK-20/index.md) | Nearly cancelled bend focusing corrupts the expanded matrix map <span class="bb-tag">partly-source-only</span> | Tracking | P3 | Wrong physics | 2026-11-07 (`f09e0fb`, head) |  |  |
| [OPT-17](OPT-17/index.md) | Radiation integrals omit sextupole feed-down from the orbit gradient | Optics and matching | P3 | Wrong physics | 2026-11-07 (`f09e0fb`, head) |  |  |
| [COL-01](COL-01/index.md) | q-Gaussian normalization overflows near the Gaussian limit | Collective effects | P3 | Wrong physics | 2026-10-26 (`223b4fa`, head) |  |  |
| [COL-03](COL-03/index.md) | Particle-to-mesh deposition omits source charge ratios <span class="bb-tag">known-limitation</span> | Collective effects | P3 | Wrong physics | 2026-10-26 (`223b4fa`, head) |  |  |
| [COL-11](COL-11/index.md) | Interpolated electron lens omits the transit-speed factor | Collective effects | P3 | Wrong physics | 2026-10-29 (`3f06c87`, head) |  |  |
| [COL-12](COL-12/index.md) | ElectronCloud applies inconsistent reference-charge factors | Collective effects | P3 | Wrong physics | 2026-10-29 (`3f06c87`, head) |  |  |
| [COL-17](COL-17/index.md) | All-lost tracking calls freeze wake history | Collective effects | P3 | Wrong physics | 2026-10-29 (`3f06c87`, head) |  |  |
| [COL-18](COL-18/index.md) | Round-beam variance derivatives use an absolute near-axis cutoff <span class="bb-tag">partly-source-only</span> | Collective effects | P3 | Wrong physics | 2026-10-29 (`3f06c87`, head) |  |  |
| [COL-21](COL-21/index.md) | Wake kicks omit source charge and witness charge-to-mass scaling <span class="bb-tag">partly-source-only</span> | Collective effects | P3 | Wrong physics | 2026-11-07 (`3f06c87`, head) |  |  |
| [COL-22](COL-22/index.md) | Compound atom density counts formula units instead of atoms <span class="bb-tag">partly-source-only</span> | Collective effects | P3 | Wrong physics | 2026-11-06 (`64e13f5`, head) |  |  |
| [COL-24](COL-24/index.md) | 3D FFT Green-function corner subtraction corrupts sparse far-cell fields <span class="bb-tag">partly-source-only</span> | Collective effects | P3 | Wrong physics | 2026-11-07 (`3f06c87`, head) |  |  |
| [COL-25](COL-25/index.md) | Rounded CIC indices discard strictly interior upper-edge particles <span class="bb-tag">partly-source-only</span> | Collective effects | P3 | Wrong physics | 2026-11-07 (`3f06c87`, head) |  |  |
| [COL-26](COL-26/index.md) | Everest clock can precede the shortest flight between sampled endpoints <span class="bb-tag">partly-source-only</span> | Collective effects | P3 | Wrong physics | 2026-11-06 (`64e13f5`, head) |  |  |
| [IO-10](IO-10/index.md) | Linear longitudinal matching omits position-momentum correlation <span class="bb-tag">partly-source-only</span> | Lattices, particles and I/O | P3 | Wrong physics | 2026-10-29 (`2633e52`, head) |  |  |
| [TRK-25](TRK-25/index.md) | Loss refinement constructs false walls for apertures excluding the origin | Tracking | P3 | Silent wrong result | 2026-11-07 (`f09e0fb`, head) |  |  |
| [OPT-14](OPT-14/index.md) | Full radiation equilibrium covariance omits accumulated diffusion | Optics and matching | P3 | Silent wrong result | 2026-10-29 (`f09e0fb`, head) |  |  |
| [OPT-15](OPT-15/index.md) | Closest tune approach is inflated across an integer boundary | Optics and matching | P3 | Silent wrong result | 2026-10-29 (`f09e0fb`, head) |  |  |
| [OPT-26](OPT-26/index.md) | Full radiation covariance discards complex diffusion correlations <span class="bb-tag">partly-source-only</span> | Optics and matching | P3 | Silent wrong result | 2026-11-07 (`f09e0fb`, head) |  |  |
| [COL-07](COL-07/index.md) | Loss-map periodic aperture cells have negative lengths | Collective effects | P3 | Silent wrong result | 2026-10-29 (`64e13f5`, head) |  |  |
| [COL-10](COL-10/index.md) | Noninterpolated loss maps discard coordinate reversal | Collective effects | P3 | Silent wrong result | 2026-10-29 (`64e13f5`, head) |  |  |
| [COL-13](COL-13/index.md) | Everest deposited-energy accumulators omit macroparticle weights <span class="bb-tag">partly-source-only</span> | Collective effects | P3 | Silent wrong result | 2026-10-29 (`64e13f5`, head) |  |  |
| [COL-14](COL-14/index.md) | Loss-map primary energy uses secondary deposits | Collective effects | P3 | Silent wrong result | 2026-10-29 (`64e13f5`, head) |  |  |
| [COL-15](COL-15/index.md) | Loss-map export drops deposits without absorbed particles | Collective effects | P3 | Silent wrong result | 2026-10-29 (`64e13f5`, head) |  |  |
| [IO-04](IO-04/index.md) | Renaming a MAD-X deferred variable freezes its contribution to imported dependants | Lattices, particles and I/O | P3 | Silent wrong result | 2026-10-26 (`3cd8bfc`, head) |  |  |
| [IO-06](IO-06/index.md) | Negative-charge reference rigidity fails a constructor round trip | Lattices, particles and I/O | P3 | Silent wrong result | 2026-10-26 (`3cd8bfc`, head) |  |  |
| [TRK-21](TRK-21/index.md) | Dipole wedge returns NaNs for valid near-grazing momenta <span class="bb-tag">extreme-regime</span> <span class="bb-tag">partly-source-only</span> | Tracking | P3 | Crash or API hazard | 2026-11-07 (`f09e0fb`, head) |  |  |
| [OPT-24](OPT-24/index.md) | 4D polarization inverts the singular invariant-axis direction | Optics and matching | P3 | Crash or API hazard | 2026-11-06 (`ad5e808`; not on head 2026-11-06) |  | [[gh:https://github.com/xsuite/xtrack/pull/923]] |
| [COL-20](COL-20/index.md) | Zero horizontal dispersion invariant creates a NaN B&M rate | Collective effects | P3 | Crash or API hazard | 2026-10-29 (`3f06c87`, head) |  |  |
| [COL-27](COL-27/index.md) | Equal positive x–z collision eigenvalues make Nagaitsev IBS rates NaN <span class="bb-tag">partly-source-only</span> | Collective effects | P3 | Crash or API hazard | 2026-11-07 (`3f06c87`, head) |  |  |
| [IO-12](IO-12/index.md) | XYShift stores NaN for omitted zero offsets | Lattices, particles and I/O | P3 | Crash or API hazard | 2026-11-07 (`f09e0fb`, head) |  |  |
| [TRK-22](TRK-22/index.md) | Weak-field dipole wedge adds a spurious path at the rotation origin <span class="bb-tag">extreme-regime</span> <span class="bb-tag">partly-source-only</span> | Tracking | P4 | Wrong physics | 2026-11-07 (`f09e0fb`, head) |  | [[gh:https://github.com/xsuite/xtrack/pull/911]] |
| [OPT-01](OPT-01/index.md) | Native Taylor maps silently omit RFMultipole and CrabCavity kicks <span class="bb-tag">experimental</span> | Optics and matching | P4 | Wrong physics | 2026-10-26 (`3cd8bfc`, head) |  |  |
| [OPT-02](OPT-02/index.md) | Native Taylor maps ignore an absolute-time cavity's RF clock <span class="bb-tag">experimental</span> <span class="bb-tag">known-limitation</span> | Optics and matching | P4 | Silent wrong result | 2026-10-26 (`3cd8bfc`, head) |  |  |
| [OPT-03](OPT-03/index.md) | Solenoid Taylor map has zero strength sensitivity at zero field <span class="bb-tag">experimental</span> | Optics and matching | P4 | Silent wrong result | 2026-10-26 (`3cd8bfc`, head) |  |  |
| [OPT-10](OPT-10/index.md) | Sequential step clipping violates earlier variable bounds | Optics and matching | P4 | Silent wrong result | 2026-10-29 (`6484795`, head) |  |  |
| [OPT-11](OPT-11/index.md) | Variable weights distort physical maximum steps | Optics and matching | P4 | Silent wrong result | 2026-10-29 (`6484795`, head) |  |  |
| [OPT-22](OPT-22/index.md) | Native TPSA particle extraction restores the launch reference <span class="bb-tag">experimental</span> | Optics and matching | P4 | Silent wrong result | 2026-11-07 (`f09e0fb`, head) |  | [[gh:https://github.com/xsuite/xtrack/pull/879#issuecomment-6659447765]] |
| [COL-04](COL-04/index.md) | CollectiveMonitor slice emittance omits the covariance square | Collective effects | P4 | Silent wrong result | 2026-10-26 (`223b4fa`, head) |  |  |
| [COL-08](COL-08/index.md) | Loss-map initial energy double-counts preceding batches | Collective effects | P4 | Silent wrong result | 2026-10-29 (`64e13f5`, head) |  |  |
| [COL-09](COL-09/index.md) | Noninterpolated aperture losses are omitted from total energy | Collective effects | P4 | Silent wrong result | 2026-10-29 (`64e13f5`, head) |  |  |
| [COL-16](COL-16/index.md) | Touschek Monte Carlo diagnostic omits sampled angular width <span class="bb-tag">partly-source-only</span> | Collective effects | P4 | Silent wrong result | 2026-10-29 (`3f06c87`, head) |  |  |
| [IO-03](IO-03/index.md) | Checkpoint loading restores RNG words onto reordered particle coordinates | Lattices, particles and I/O | P4 | Silent wrong result | 2026-10-26 (`3cd8bfc`, head) |  |  |
| [IO-07](IO-07/index.md) | Manager.copy disconnects containers from dependency tasks | Lattices, particles and I/O | P4 | Silent wrong result | 2026-10-29 (`6484795`, head) |  |  |
| [IO-08](IO-08/index.md) | Accelerating RF contour emittance uses symmetric turning points | Lattices, particles and I/O | P4 | Silent wrong result | 2026-10-29 (`2633e52`, head) |  |  |
| [IO-09](IO-09/index.md) | RF phase setter leaves cached bucket geometry stale | Lattices, particles and I/O | P4 | Silent wrong result | 2026-10-29 (`2633e52`, head) |  |  |
| [IO-11](IO-11/index.md) | Survey assigns fictitious physical exits to thin elements | Lattices, particles and I/O | P4 | Silent wrong result | 2026-11-07 (`f09e0fb`, head) |  |  |
| [IO-13](IO-13/index.md) | Empty particle selections raise a zero-capacity exception <span class="bb-tag">partly-source-only</span> | Lattices, particles and I/O | P4 | Crash or API hazard | 2026-11-07 (`f09e0fb`, head) |  |  |
| [IO-15](IO-15/index.md) | Repeated element occurrences fail MAD-X sequence export | Lattices, particles and I/O | P4 | Crash or API hazard | 2026-11-07 (`f09e0fb`, head) |  |  |
| [IO-16](IO-16/index.md) | Distinct native names collide after MAD-X escaping | Lattices, particles and I/O | P4 | Crash or API hazard | 2026-11-07 (`f09e0fb`, head) |  |  |
| [IO-17](IO-17/index.md) | Reverse MAD-X loading crashes on string LINE components | Lattices, particles and I/O | P4 | Crash or API hazard | 2026-11-07 (`f09e0fb`, head) |  |  |
| [IO-18](IO-18/index.md) | MAD-X LINE export calls converters with obsolete arguments | Lattices, particles and I/O | P4 | Crash or API hazard | 2026-11-07 (`f09e0fb`, head) |  |  |
| [IO-19](IO-19/index.md) | Spin configuration rejects documented Boolean arguments | Lattices, particles and I/O | P4 | Crash or API hazard | 2026-11-06 (`ad5e808`; not on head 2026-11-07) |  | [[gh:https://github.com/xsuite/xtrack/pull/923]] |
| [IO-25](IO-25/index.md) | MAD-X rejects sliced-magnet exports containing kill_body | Lattices, particles and I/O | P4 | Crash or API hazard | 2026-11-07 (`f09e0fb`, head) |  |  |
| [IO-28](IO-28/index.md) | Sliced CrabCavity export raises AttributeError | Lattices, particles and I/O | P4 | Crash or API hazard | 2026-11-07 (`f09e0fb`, head) |  |  |
| [IO-22](IO-22/index.md) | Physics manual omits the squared reference speed in the exact speed equation | Lattices, particles and I/O | P4 | Documentation or example mismatch | 2026-11-07 (`fb7df34`, head) |  |  |
