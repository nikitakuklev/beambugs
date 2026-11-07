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

| ID | Priority | Defect | Area | Severity | Last reproduced | Issue | PR |
|---|---|---|---|---|---|---|---|
| [TRK-04](TRK-04/index.md) | P1 | Optimizing RF slices drops roll and phase taper | Tracking | Wrong physics | 2026-10-26 ([xtrack v0.118.4](https://github.com/xsuite/xtrack/tree/v0.118.4)) |  |  |
| [TRK-07](TRK-07/index.md) | P1 | Optimization disables radiation in curved-body RBend slices | Tracking | Wrong physics | 2026-10-26 ([xtrack v0.118.4](https://github.com/xsuite/xtrack/tree/v0.118.4)) |  |  |
| [IO-05](IO-05/index.md) | P1 | MAD-X bend roll errors are imported as design tilts | Lattices, particles and I/O | Wrong physics | 2026-10-26 ([xtrack v0.118.4](https://github.com/xsuite/xtrack/tree/v0.118.4)) |  |  |
| [CTX-01](CTX-01/index.md) | P1 | CPU linked-array views bypass dependent-state updates | Contexts and kernels | Wrong physics | 2026-10-29 ([xobjects v0.7.2](https://github.com/xsuite/xobjects/tree/v0.7.2)) |  |  |
| [CTX-02](CTX-02/index.md) | P1 | Copied nested fields disconnect from parent storage | Contexts and kernels | Wrong physics | 2026-10-29 ([xobjects v0.7.2](https://github.com/xsuite/xobjects/tree/v0.7.2)) |  |  |
| [TRK-01](TRK-01/index.md) | P2 | Solenoid body ignores per-particle charge-to-mass scaling | Tracking | Wrong physics | 2026-10-26 ([xtrack v0.118.4](https://github.com/xsuite/xtrack/tree/v0.118.4)) |  |  |
| [TRK-02](TRK-02/index.md) | P2 | Tracking optimization drops a full bend's multipole fringe | Tracking | Wrong physics | 2026-10-26 ([xtrack v0.118.4](https://github.com/xsuite/xtrack/tree/v0.118.4)) |  | [[gh:https://github.com/xsuite/xtrack/pull/911]] |
| [TRK-05](TRK-05/index.md) | P2 | Sliced thick Multipole repeats the full reference bend angle | Tracking | Wrong physics | 2026-10-26 ([xtrack v0.118.4](https://github.com/xsuite/xtrack/tree/v0.118.4)) |  |  |
| [TRK-06](TRK-06/index.md) | P2 | Zero linear chromaticity shifts the quadratic coefficient into the linear slot | Tracking | Wrong physics | 2026-10-26 ([xtrack v0.118.4](https://github.com/xsuite/xtrack/tree/v0.118.4)) |  |  |
| [TRK-09](TRK-09/index.md)<span class="bb-fixed"></span> | P2 | Solenoid spin tracking uses the wrong kinetic momentum | Tracking | Wrong physics | 2026-10-26 ([xtrack v0.118.2](https://github.com/xsuite/xtrack/tree/v0.118.2)) |  | [[gh:https://github.com/xsuite/xtrack/pull/923]] |
| [TRK-15](TRK-15/index.md) | P2 | Zero horizontal focusing loses accelerated-path terms | Tracking | Wrong physics | 2026-10-29 ([xtrack v0.118.6](https://github.com/xsuite/xtrack/tree/v0.118.6)) |  |  |
| [TRK-17](TRK-17/index.md) | P2 | Off-origin polygon apertures retain particles outside their boundary <span class="bb-tag">partly-source-only</span> | Tracking | Wrong physics | 2026-10-29 ([xtrack v0.118.6](https://github.com/xsuite/xtrack/tree/v0.118.6)) |  |  |
| [TRK-18](TRK-18/index.md) | P2 | Radiation tracking omits the field wedges of angled dipole faces <span class="bb-tag">partly-source-only</span> | Tracking | Wrong physics | 2026-11-07 ([xtrack v0.118.6](https://github.com/xsuite/xtrack/tree/v0.118.6)) |  |  |
| [TRK-19](TRK-19/index.md) | P2 | Spin precession omits tracked charge-to-mass scaling <span class="bb-tag">partly-source-only</span> | Tracking | Wrong physics | 2026-11-07 ([xtrack v0.118.6](https://github.com/xsuite/xtrack/tree/v0.118.6)) |  |  |
| [TRK-23](TRK-23/index.md) | P2 | Optimized thin bends scale geometric arrival shifts by charge-to-mass ratio | Tracking | Wrong physics | 2026-11-07 ([xtrack v0.118.6](https://github.com/xsuite/xtrack/tree/v0.118.6)) |  |  |
| [TRK-24](TRK-24/index.md) | P2 | Thin-element simplification discards quadrupole and dipole spin transport | Tracking | Wrong physics | 2026-11-07 ([xtrack v0.118.6](https://github.com/xsuite/xtrack/tree/v0.118.6)) |  |  |
| [TRK-26](TRK-26/index.md) | P2 | Backtracking silently skips spin transport and its rejection <span class="bb-tag">partly-source-only</span> | Tracking | Wrong physics | 2026-11-07 ([xtrack v0.118.6](https://github.com/xsuite/xtrack/tree/v0.118.6)) | [[gh:https://github.com/xsuite/xsuite/issues/647]] |  |
| [TRK-27](TRK-27/index.md) | P2 | Slicing helper names overwrite existing physical elements <span class="bb-tag">partly-source-only</span> | Tracking | Wrong physics | 2026-11-07 ([xtrack v0.118.6](https://github.com/xsuite/xtrack/tree/v0.118.6)) |  |  |
| [TRK-28](TRK-28/index.md) | P2 | Absolute-time RF phase loses particle arrival offsets at long clocks <span class="bb-tag">partly-source-only</span> | Tracking | Wrong physics | 2026-11-07 ([xtrack v0.118.6](https://github.com/xsuite/xtrack/tree/v0.118.6)) |  |  |
| [TRK-29](TRK-29/index.md) | P2 | Absolute-time RF clock accumulates rounding once per turn <span class="bb-tag">partly-source-only</span> | Tracking | Wrong physics | 2026-11-07 ([xtrack v0.118.6](https://github.com/xsuite/xtrack/tree/v0.118.6)) |  |  |
| [OPT-16](OPT-16/index.md) | P2 | Radiation integrals omit the dipole-face term of I₄ | Optics and matching | Wrong physics | 2026-11-07 ([xtrack v0.118.6](https://github.com/xsuite/xtrack/tree/v0.118.6)) |  |  |
| [OPT-23](OPT-23/index.md)<span class="bb-fixed"></span> | P2 | Periodic spin Twiss omits the longitudinal fixed-point constraint | Optics and matching | Wrong physics | 2026-11-06 ([xtrack v0.118.2](https://github.com/xsuite/xtrack/tree/v0.118.2)) |  | [[gh:https://github.com/xsuite/xtrack/pull/923]] |
| [COL-02](COL-02/index.md) | P2 | Displaced Gaussian space charge integrates the wrong longitudinal-kick ray | Collective effects | Wrong physics | 2026-10-26 ([xfields v0.27.5](https://github.com/xsuite/xfields/tree/v0.27.5)) |  |  |
| [COL-05](COL-05/index.md) | P2 | Strong-strong 2D beam-beam moments include lost particles | Collective effects | Wrong physics | 2026-10-26 ([xfields v0.27.5](https://github.com/xsuite/xfields/tree/v0.27.5)) |  |  |
| [COL-06](COL-06/index.md) | P2 | 2.5D FFT space charge: the longitudinal kick of a nonuniform bunch violates geometric similarity <span class="bb-tag">contract-unsettled</span> | Collective effects | Wrong physics | 2026-10-29 ([xfields v0.27.7](https://github.com/xsuite/xfields/tree/v0.27.7)) |  |  |
| [COL-19](COL-19/index.md) | P2 | B&M applies an extra beta factor to momentum dispersion | Collective effects | Wrong physics | 2026-10-29 ([xfields v0.27.7](https://github.com/xsuite/xfields/tree/v0.27.7)) |  |  |
| [COL-23](COL-23/index.md) | P2 | Gaussian space-charge construction and reload discard placement | Collective effects | Wrong physics | 2026-11-07 ([xfields v0.27.7](https://github.com/xsuite/xfields/tree/v0.27.7)) |  |  |
| [IO-01](IO-01/index.md) | P2 | Single-harmonic longitudinal generation uses proton mass for other species | Lattices, particles and I/O | Wrong physics | 2026-10-26 ([xpart v0.23.23](https://github.com/xsuite/xpart/tree/v0.23.23)) |  |  |
| [IO-14](IO-14/index.md) | P2 | MAD-X export loses embedded quadrupole dipole steering | Lattices, particles and I/O | Wrong physics | 2026-11-07 ([xtrack v0.118.6](https://github.com/xsuite/xtrack/tree/v0.118.6)) |  |  |
| [IO-20](IO-20/index.md) | P2 | Reverse MAD-X import keeps forward horizontal aperture offsets | Lattices, particles and I/O | Wrong physics | 2026-11-07 ([xtrack v0.118.6](https://github.com/xsuite/xtrack/tree/v0.118.6)) |  |  |
| [IO-21](IO-21/index.md) | P2 | Exported magnet body slices repeat the parent's integrated multipole strengths | Lattices, particles and I/O | Wrong physics | 2026-11-07 ([xtrack v0.118.6](https://github.com/xsuite/xtrack/tree/v0.118.6)) |  |  |
| [IO-23](IO-23/index.md) | P2 | Exported thin-mode slices occupy the parent's length a second time | Lattices, particles and I/O | Wrong physics | 2026-11-07 ([xtrack v0.118.6](https://github.com/xsuite/xtrack/tree/v0.118.6)) |  |  |
| [IO-24](IO-24/index.md) | P2 | Exported bend body slices each repeat the parent's edge faces | Lattices, particles and I/O | Wrong physics | 2026-11-07 ([xtrack v0.118.6](https://github.com/xsuite/xtrack/tree/v0.118.6)) |  |  |
| [IO-26](IO-26/index.md) | P2 | Exported Cavity slices each repeat the parent's full voltage | Lattices, particles and I/O | Wrong physics | 2026-11-07 ([xtrack v0.118.6](https://github.com/xsuite/xtrack/tree/v0.118.6)) |  |  |
| [IO-27](IO-27/index.md) | P2 | Bend export replaces the exit half-gap by the entrance half-gap <span class="bb-tag">partly-source-only</span> | Lattices, particles and I/O | Wrong physics | 2026-11-07 ([xtrack v0.118.6](https://github.com/xsuite/xtrack/tree/v0.118.6)) |  |  |
| [OPT-04](OPT-04/index.md) | P2 | Finite-difference transfer matrices reset the supplied RF clock | Optics and matching | Silent wrong result | 2026-10-26 ([xtrack v0.118.4](https://github.com/xsuite/xtrack/tree/v0.118.4)) |  |  |
| [OPT-05](OPT-05/index.md) | P2 | 4D Twiss retains RF energy kicks from LineSegmentMap | Optics and matching | Silent wrong result | 2026-10-26 ([xtrack v0.118.4](https://github.com/xsuite/xtrack/tree/v0.118.4)) |  |  |
| [OPT-06](OPT-06/index.md) | P2 | A subrange's second-order chromaticity depends on its absolute s position | Optics and matching | Silent wrong result | 2026-10-26 ([xtrack v0.118.4](https://github.com/xsuite/xtrack/tree/v0.118.4)) |  |  |
| [OPT-07](OPT-07/index.md) | P2 | Periodic subrange slip factor uses the full parent line length | Optics and matching | Silent wrong result | 2026-10-26 ([xtrack v0.118.4](https://github.com/xsuite/xtrack/tree/v0.118.4)) |  |  |
| [OPT-08](OPT-08/index.md) | P2 | Extracted Twiss init loses a downstream reference-momentum change | Optics and matching | Silent wrong result | 2026-10-26 ([xtrack v0.118.4](https://github.com/xsuite/xtrack/tree/v0.118.4)) |  |  |
| [OPT-09](OPT-09/index.md) | P2 | Sliced decelerating cavities are counted as synchrotron-radiation loss | Optics and matching | Silent wrong result | 2026-10-26 ([xtrack v0.118.4](https://github.com/xsuite/xtrack/tree/v0.118.4)) |  |  |
| [OPT-12](OPT-12/index.md) | P2 | Knob construction applies absolute limits to additive corrections | Optics and matching | Silent wrong result | 2026-10-29 ([xtrack v0.118.6](https://github.com/xsuite/xtrack/tree/v0.118.6)) |  |  |
| [OPT-13](OPT-13/index.md) | P2 | Reversed Twiss tables reconstruct incorrect transverse transfer maps | Optics and matching | Silent wrong result | 2026-10-29 ([xtrack v0.118.6](https://github.com/xsuite/xtrack/tree/v0.118.6)) |  |  |
| [OPT-18](OPT-18/index.md) | P2 | Twiss transfer matrices omit reference-change amplitudes | Optics and matching | Silent wrong result | 2026-11-07 ([xtrack v0.118.6](https://github.com/xsuite/xtrack/tree/v0.118.6)) |  |  |
| [OPT-19](OPT-19/index.md) | P2 | Normalized-emittance covariance uses the initialization rigidity | Optics and matching | Silent wrong result | 2026-11-07 ([xtrack v0.118.6](https://github.com/xsuite/xtrack/tree/v0.118.6)) |  |  |
| [OPT-20](OPT-20/index.md) | P2 | Open chromatic Twiss launches pζ dispersion as a δ derivative | Optics and matching | Silent wrong result | 2026-11-07 ([xtrack v0.118.6](https://github.com/xsuite/xtrack/tree/v0.118.6)) |  |  |
| [OPT-21](OPT-21/index.md) | P2 | Propagated longitudinal W uses the initialization reference speed | Optics and matching | Silent wrong result | 2026-11-07 ([xtrack v0.118.6](https://github.com/xsuite/xtrack/tree/v0.118.6)) |  |  |
| [OPT-25](OPT-25/index.md) | P2 | Periodic Twiss converts canonical time slip with reference-orbit factors | Optics and matching | Silent wrong result | 2026-11-07 ([xtrack v0.118.6](https://github.com/xsuite/xtrack/tree/v0.118.6)) |  |  |
| [TRK-03](TRK-03/index.md) | P3 | RF-derived linear segment map produces NaNs for electrons | Tracking | Wrong physics | 2026-10-26 ([xtrack v0.118.4](https://github.com/xsuite/xtrack/tree/v0.118.4)) |  |  |
| [TRK-08](TRK-08/index.md) | P3 | SplineBoris omits species factors in physical charge and momentum | Tracking | Wrong physics | 2026-10-26 ([xtrack v0.118.4](https://github.com/xsuite/xtrack/tree/v0.118.4)) |  |  |
| [TRK-10](TRK-10/index.md) | P3 | Linear dipole-edge finite-gap focusing has wrong species scaling | Tracking | Wrong physics | 2026-10-26 ([xtrack v0.118.4](https://github.com/xsuite/xtrack/tree/v0.118.4)) |  |  |
| [TRK-11](TRK-11/index.md) | P3 | Elens has inverted species scaling and extra momentum dependence | Tracking | Wrong physics | 2026-10-29 ([xtrack v0.118.6](https://github.com/xsuite/xtrack/tree/v0.118.6)) |  |  |
| [TRK-12](TRK-12/index.md) | P3 | Wire kicks omit particle charge-to-mass scaling | Tracking | Wrong physics | 2026-10-29 ([xtrack v0.118.6](https://github.com/xsuite/xtrack/tree/v0.118.6)) |  |  |
| [TRK-13](TRK-13/index.md) | P3 | Exciter activates before its stated starting time | Tracking | Wrong physics | 2026-10-29 ([xtrack v0.118.6](https://github.com/xsuite/xtrack/tree/v0.118.6)) |  |  |
| [TRK-14](TRK-14/index.md) | P3 | Elens residual kicks fail forward/backward tracking | Tracking | Wrong physics | 2026-10-29 ([xtrack v0.118.6](https://github.com/xsuite/xtrack/tree/v0.118.6)) |  |  |
| [TRK-16](TRK-16/index.md) | P3 | Quantum photon rate omits reference charge squared | Tracking | Wrong physics | 2026-10-29 ([xtrack v0.118.6](https://github.com/xsuite/xtrack/tree/v0.118.6)) |  |  |
| [TRK-20](TRK-20/index.md) | P3 | Nearly cancelled bend focusing corrupts the expanded matrix map <span class="bb-tag">partly-source-only</span> | Tracking | Wrong physics | 2026-11-07 ([xtrack v0.118.6](https://github.com/xsuite/xtrack/tree/v0.118.6)) |  |  |
| [OPT-17](OPT-17/index.md) | P3 | Radiation integrals omit sextupole feed-down from the orbit gradient | Optics and matching | Wrong physics | 2026-11-07 ([xtrack v0.118.6](https://github.com/xsuite/xtrack/tree/v0.118.6)) |  |  |
| [COL-01](COL-01/index.md) | P3 | q-Gaussian normalization overflows near the Gaussian limit | Collective effects | Wrong physics | 2026-10-26 ([xfields v0.27.5](https://github.com/xsuite/xfields/tree/v0.27.5)) |  |  |
| [COL-03](COL-03/index.md) | P3 | Particle-to-mesh deposition omits source charge ratios <span class="bb-tag">known-limitation</span> | Collective effects | Wrong physics | 2026-10-26 ([xfields v0.27.5](https://github.com/xsuite/xfields/tree/v0.27.5)) |  |  |
| [COL-11](COL-11/index.md) | P3 | Interpolated electron lens omits the transit-speed factor | Collective effects | Wrong physics | 2026-10-29 ([xfields v0.27.7](https://github.com/xsuite/xfields/tree/v0.27.7)) |  |  |
| [COL-12](COL-12/index.md) | P3 | ElectronCloud applies inconsistent reference-charge factors | Collective effects | Wrong physics | 2026-10-29 ([xfields v0.27.7](https://github.com/xsuite/xfields/tree/v0.27.7)) |  |  |
| [COL-17](COL-17/index.md) | P3 | All-lost tracking calls freeze wake history | Collective effects | Wrong physics | 2026-10-29 ([xfields v0.27.7](https://github.com/xsuite/xfields/tree/v0.27.7)) |  |  |
| [COL-18](COL-18/index.md) | P3 | Round-beam variance derivatives use an absolute near-axis cutoff <span class="bb-tag">partly-source-only</span> | Collective effects | Wrong physics | 2026-10-29 ([xfields v0.27.7](https://github.com/xsuite/xfields/tree/v0.27.7)) |  |  |
| [COL-21](COL-21/index.md) | P3 | Wake kicks omit source charge and witness charge-to-mass scaling <span class="bb-tag">partly-source-only</span> | Collective effects | Wrong physics | 2026-11-07 ([xfields v0.27.7](https://github.com/xsuite/xfields/tree/v0.27.7)) |  |  |
| [COL-22](COL-22/index.md) | P3 | Compound atom density counts formula units instead of atoms <span class="bb-tag">partly-source-only</span> | Collective effects | Wrong physics | 2026-11-06 ([xcoll v0.12.5](https://github.com/xsuite/xcoll/tree/v0.12.5)) |  |  |
| [COL-24](COL-24/index.md) | P3 | 3D FFT Green-function corner subtraction corrupts sparse far-cell fields <span class="bb-tag">partly-source-only</span> | Collective effects | Wrong physics | 2026-11-07 ([xfields v0.27.7](https://github.com/xsuite/xfields/tree/v0.27.7)) |  |  |
| [COL-25](COL-25/index.md) | P3 | Rounded CIC indices discard strictly interior upper-edge particles <span class="bb-tag">partly-source-only</span> | Collective effects | Wrong physics | 2026-11-07 ([xfields v0.27.7](https://github.com/xsuite/xfields/tree/v0.27.7)) |  |  |
| [COL-26](COL-26/index.md) | P3 | Everest clock can precede the shortest flight between sampled endpoints <span class="bb-tag">partly-source-only</span> | Collective effects | Wrong physics | 2026-11-06 ([xcoll v0.12.5](https://github.com/xsuite/xcoll/tree/v0.12.5)) |  |  |
| [IO-10](IO-10/index.md) | P3 | Linear longitudinal matching omits position-momentum correlation <span class="bb-tag">partly-source-only</span> | Lattices, particles and I/O | Wrong physics | 2026-10-29 ([xpart v0.23.23](https://github.com/xsuite/xpart/tree/v0.23.23)) |  |  |
| [TRK-25](TRK-25/index.md) | P3 | Loss refinement constructs false walls for apertures excluding the origin | Tracking | Silent wrong result | 2026-11-07 ([xtrack v0.118.6](https://github.com/xsuite/xtrack/tree/v0.118.6)) |  |  |
| [OPT-14](OPT-14/index.md) | P3 | Full radiation equilibrium covariance omits accumulated diffusion | Optics and matching | Silent wrong result | 2026-10-29 ([xtrack v0.118.6](https://github.com/xsuite/xtrack/tree/v0.118.6)) |  |  |
| [OPT-15](OPT-15/index.md) | P3 | Closest tune approach is inflated across an integer boundary | Optics and matching | Silent wrong result | 2026-10-29 ([xtrack v0.118.6](https://github.com/xsuite/xtrack/tree/v0.118.6)) |  |  |
| [OPT-26](OPT-26/index.md) | P3 | Full radiation covariance discards complex diffusion correlations <span class="bb-tag">partly-source-only</span> | Optics and matching | Silent wrong result | 2026-11-07 ([xtrack v0.118.6](https://github.com/xsuite/xtrack/tree/v0.118.6)) |  |  |
| [COL-07](COL-07/index.md) | P3 | Loss-map periodic aperture cells have negative lengths | Collective effects | Silent wrong result | 2026-10-29 ([xcoll v0.12.5](https://github.com/xsuite/xcoll/tree/v0.12.5)) |  |  |
| [COL-10](COL-10/index.md) | P3 | Noninterpolated loss maps discard coordinate reversal | Collective effects | Silent wrong result | 2026-10-29 ([xcoll v0.12.5](https://github.com/xsuite/xcoll/tree/v0.12.5)) |  |  |
| [COL-13](COL-13/index.md) | P3 | Everest deposited-energy accumulators omit macroparticle weights <span class="bb-tag">partly-source-only</span> | Collective effects | Silent wrong result | 2026-10-29 ([xcoll v0.12.5](https://github.com/xsuite/xcoll/tree/v0.12.5)) |  |  |
| [COL-14](COL-14/index.md) | P3 | Loss-map primary energy uses secondary deposits | Collective effects | Silent wrong result | 2026-10-29 ([xcoll v0.12.5](https://github.com/xsuite/xcoll/tree/v0.12.5)) |  |  |
| [COL-15](COL-15/index.md) | P3 | Loss-map export drops deposits without absorbed particles | Collective effects | Silent wrong result | 2026-10-29 ([xcoll v0.12.5](https://github.com/xsuite/xcoll/tree/v0.12.5)) |  |  |
| [IO-04](IO-04/index.md) | P3 | Renaming a MAD-X deferred variable freezes its contribution to imported dependants | Lattices, particles and I/O | Silent wrong result | 2026-10-26 ([xtrack v0.118.4](https://github.com/xsuite/xtrack/tree/v0.118.4)) |  |  |
| [IO-06](IO-06/index.md) | P3 | Negative-charge reference rigidity fails a constructor round trip | Lattices, particles and I/O | Silent wrong result | 2026-10-26 ([xtrack v0.118.4](https://github.com/xsuite/xtrack/tree/v0.118.4)) |  |  |
| [TRK-21](TRK-21/index.md) | P3 | Dipole wedge returns NaNs for valid near-grazing momenta <span class="bb-tag">extreme-regime</span> <span class="bb-tag">partly-source-only</span> | Tracking | Crash or API hazard | 2026-11-07 ([xtrack v0.118.6](https://github.com/xsuite/xtrack/tree/v0.118.6)) |  |  |
| [OPT-24](OPT-24/index.md)<span class="bb-fixed"></span> | P3 | 4D polarization inverts the singular invariant-axis direction | Optics and matching | Crash or API hazard | 2026-11-06 ([xtrack v0.118.2](https://github.com/xsuite/xtrack/tree/v0.118.2)) |  | [[gh:https://github.com/xsuite/xtrack/pull/923]] |
| [COL-20](COL-20/index.md) | P3 | Zero horizontal dispersion invariant creates a NaN B&M rate | Collective effects | Crash or API hazard | 2026-10-29 ([xfields v0.27.7](https://github.com/xsuite/xfields/tree/v0.27.7)) |  |  |
| [COL-27](COL-27/index.md) | P3 | Equal positive x–z collision eigenvalues make Nagaitsev IBS rates NaN <span class="bb-tag">partly-source-only</span> | Collective effects | Crash or API hazard | 2026-11-07 ([xfields v0.27.7](https://github.com/xsuite/xfields/tree/v0.27.7)) |  |  |
| [IO-12](IO-12/index.md) | P3 | XYShift stores NaN for omitted zero offsets | Lattices, particles and I/O | Crash or API hazard | 2026-11-07 ([xtrack v0.118.6](https://github.com/xsuite/xtrack/tree/v0.118.6)) |  |  |
| [TRK-22](TRK-22/index.md) | P4 | Weak-field dipole wedge adds a spurious path at the rotation origin <span class="bb-tag">extreme-regime</span> <span class="bb-tag">partly-source-only</span> | Tracking | Wrong physics | 2026-11-07 ([xtrack v0.118.6](https://github.com/xsuite/xtrack/tree/v0.118.6)) |  | [[gh:https://github.com/xsuite/xtrack/pull/911]] |
| [OPT-01](OPT-01/index.md) | P4 | Native Taylor maps silently omit RFMultipole and CrabCavity kicks <span class="bb-tag">experimental</span> | Optics and matching | Wrong physics | 2026-10-26 ([xtrack v0.118.4](https://github.com/xsuite/xtrack/tree/v0.118.4)) |  |  |
| [OPT-02](OPT-02/index.md) | P4 | Native Taylor maps ignore an absolute-time cavity's RF clock <span class="bb-tag">experimental</span> <span class="bb-tag">known-limitation</span> | Optics and matching | Silent wrong result | 2026-10-26 ([xtrack v0.118.4](https://github.com/xsuite/xtrack/tree/v0.118.4)) |  |  |
| [OPT-03](OPT-03/index.md) | P4 | Solenoid Taylor map has zero strength sensitivity at zero field <span class="bb-tag">experimental</span> | Optics and matching | Silent wrong result | 2026-10-26 ([xtrack v0.118.4](https://github.com/xsuite/xtrack/tree/v0.118.4)) |  |  |
| [OPT-10](OPT-10/index.md) | P4 | Sequential step clipping violates earlier variable bounds | Optics and matching | Silent wrong result | 2026-10-29 ([xdeps v0.11.1](https://github.com/xsuite/xdeps/tree/v0.11.1)) |  |  |
| [OPT-11](OPT-11/index.md) | P4 | Variable weights distort physical maximum steps | Optics and matching | Silent wrong result | 2026-10-29 ([xdeps v0.11.1](https://github.com/xsuite/xdeps/tree/v0.11.1)) |  |  |
| [OPT-22](OPT-22/index.md) | P4 | Native TPSA particle extraction restores the launch reference <span class="bb-tag">experimental</span> | Optics and matching | Silent wrong result | 2026-11-07 ([xtrack v0.118.6](https://github.com/xsuite/xtrack/tree/v0.118.6)) |  | [[gh:https://github.com/xsuite/xtrack/pull/879#issuecomment-6659447765]] |
| [COL-04](COL-04/index.md) | P4 | CollectiveMonitor slice emittance omits the covariance square | Collective effects | Silent wrong result | 2026-10-26 ([xfields v0.27.5](https://github.com/xsuite/xfields/tree/v0.27.5)) |  |  |
| [COL-08](COL-08/index.md) | P4 | Loss-map initial energy double-counts preceding batches | Collective effects | Silent wrong result | 2026-10-29 ([xcoll v0.12.5](https://github.com/xsuite/xcoll/tree/v0.12.5)) |  |  |
| [COL-09](COL-09/index.md) | P4 | Noninterpolated aperture losses are omitted from total energy | Collective effects | Silent wrong result | 2026-10-29 ([xcoll v0.12.5](https://github.com/xsuite/xcoll/tree/v0.12.5)) |  |  |
| [COL-16](COL-16/index.md) | P4 | Touschek Monte Carlo diagnostic omits sampled angular width <span class="bb-tag">partly-source-only</span> | Collective effects | Silent wrong result | 2026-10-29 ([xfields v0.27.7](https://github.com/xsuite/xfields/tree/v0.27.7)) |  |  |
| [IO-03](IO-03/index.md) | P4 | Checkpoint loading restores RNG words onto reordered particle coordinates | Lattices, particles and I/O | Silent wrong result | 2026-10-26 ([xtrack v0.118.4](https://github.com/xsuite/xtrack/tree/v0.118.4)) |  |  |
| [IO-07](IO-07/index.md) | P4 | Manager.copy disconnects containers from dependency tasks | Lattices, particles and I/O | Silent wrong result | 2026-10-29 ([xdeps v0.11.1](https://github.com/xsuite/xdeps/tree/v0.11.1)) |  |  |
| [IO-08](IO-08/index.md) | P4 | Accelerating RF contour emittance uses symmetric turning points | Lattices, particles and I/O | Silent wrong result | 2026-10-29 ([xpart v0.23.23](https://github.com/xsuite/xpart/tree/v0.23.23)) |  |  |
| [IO-09](IO-09/index.md) | P4 | RF phase setter leaves cached bucket geometry stale | Lattices, particles and I/O | Silent wrong result | 2026-10-29 ([xpart v0.23.23](https://github.com/xsuite/xpart/tree/v0.23.23)) |  |  |
| [IO-11](IO-11/index.md) | P4 | Survey assigns fictitious physical exits to thin elements | Lattices, particles and I/O | Silent wrong result | 2026-11-07 ([xtrack v0.118.6](https://github.com/xsuite/xtrack/tree/v0.118.6)) |  |  |
| [IO-13](IO-13/index.md) | P4 | Empty particle selections raise a zero-capacity exception <span class="bb-tag">partly-source-only</span> | Lattices, particles and I/O | Crash or API hazard | 2026-11-07 ([xtrack v0.118.6](https://github.com/xsuite/xtrack/tree/v0.118.6)) |  |  |
| [IO-15](IO-15/index.md) | P4 | Repeated element occurrences fail MAD-X sequence export | Lattices, particles and I/O | Crash or API hazard | 2026-11-07 ([xtrack v0.118.6](https://github.com/xsuite/xtrack/tree/v0.118.6)) |  |  |
| [IO-16](IO-16/index.md) | P4 | Distinct native names collide after MAD-X escaping | Lattices, particles and I/O | Crash or API hazard | 2026-11-07 ([xtrack v0.118.6](https://github.com/xsuite/xtrack/tree/v0.118.6)) |  |  |
| [IO-17](IO-17/index.md) | P4 | Reverse MAD-X loading crashes on string LINE components | Lattices, particles and I/O | Crash or API hazard | 2026-11-07 ([xtrack v0.118.6](https://github.com/xsuite/xtrack/tree/v0.118.6)) |  |  |
| [IO-18](IO-18/index.md) | P4 | MAD-X LINE export calls converters with obsolete arguments | Lattices, particles and I/O | Crash or API hazard | 2026-11-07 ([xtrack v0.118.6](https://github.com/xsuite/xtrack/tree/v0.118.6)) |  |  |
| [IO-19](IO-19/index.md)<span class="bb-fixed"></span> | P4 | Spin configuration rejects documented Boolean arguments | Lattices, particles and I/O | Crash or API hazard | 2026-11-06 ([xtrack v0.118.2](https://github.com/xsuite/xtrack/tree/v0.118.2)) |  | [[gh:https://github.com/xsuite/xtrack/pull/923]] |
| [IO-25](IO-25/index.md) | P4 | MAD-X rejects sliced-magnet exports containing kill_body | Lattices, particles and I/O | Crash or API hazard | 2026-11-07 ([xtrack v0.118.6](https://github.com/xsuite/xtrack/tree/v0.118.6)) |  |  |
| [IO-28](IO-28/index.md) | P4 | Sliced CrabCavity export raises AttributeError | Lattices, particles and I/O | Crash or API hazard | 2026-11-07 ([xtrack v0.118.6](https://github.com/xsuite/xtrack/tree/v0.118.6)) |  |  |
| [IO-22](IO-22/index.md) | P4 | Physics manual omits the squared reference speed in the exact speed equation | Lattices, particles and I/O | Documentation or example mismatch | 2026-11-07 ([`fb7df34`](https://github.com/xsuite/xsuite/commit/fb7df34faa78)) |  |  |
