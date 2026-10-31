---
title: ImpactX
hide:
  - toc
---

# ImpactX

Verified defects in [ImpactX](https://impactx.readthedocs.io) ([source](https://github.com/BLAST-ImpactX/impactx)).
Each entry links to its reproducer, the models that found and confirmed it, and any upstream issue or pull request.
Machine-readable: [defects.json](defects.json).
Tags are explained in [Methodology](../methodology.md#tags).

## Running the reproducers

Install the CPU build of ImpactX 26.10 (`pip install impactx-noacc==26.10`), on which
the reproducers ran. Download a reproducer and run it with `python <ID>.py`, one
script per process.

Some reproducers also need: cpymad, matplotlib, mpmath, pals_schema, quantiphy, scipy, sympy.

## Defects

<div class="bb-filters" data-column="Area"></div>

| ID | Defect | Area | Priority | Severity | Last reproduced | Issue | PR |
|---|---|---|---|---|---|---|---|
| [TRK-01](TRK-01/index.md) | `ExactCFbend` ignores the skew dipole `k_skew[0]` | Tracking | P2 | Wrong physics | 2026-10-29 (`fa0a43b`, development) | [[gh:https://github.com/BLAST-ImpactX/impactx/issues/1733]] | [[gh:https://github.com/BLAST-ImpactX/impactx/pull/1734]] |
| [TRK-02](TRK-02/index.md) | `RFCavity` spin transport rotates spin without a field and uses `sqrt(1+pt²)` for βγ <span class="bb-tag">experimental</span> | Tracking | P4 | Wrong physics | 2026-10-29 (`fa0a43b`, development) |  |  |
| [TRK-03](TRK-03/index.md) | `DipEdge(modify_ref_part=True)` deflects the reference by the beam-normalized kick | Tracking | P2 | Silent wrong result | 2026-10-29 (`fa0a43b`, development) | [[gh:https://github.com/BLAST-ImpactX/impactx/issues/1729]] | [[gh:https://github.com/BLAST-ImpactX/impactx/pull/1730]] |
| [TRK-04](TRK-04/index.md) | Soft-edge elements and their `reverse()` do not cancel in a lattice | Tracking | P2 | Silent wrong result | 2026-10-29 (`fa0a43b`, development) | [[gh:https://github.com/BLAST-ImpactX/impactx/issues/1731]] | [[gh:https://github.com/BLAST-ImpactX/impactx/pull/1732]] |
| [TRK-05](TRK-05/index.md) | Periodic `PolygonAperture` mirrors asymmetric polygons | Tracking | P3 | Silent wrong result | 2026-10-30 (`dfd49f2`, development) | [[gh:https://github.com/BLAST-ImpactX/impactx/issues/1740]] | [[gh:https://github.com/BLAST-ImpactX/impactx/pull/1741]] |
| [TRK-06](TRK-06/index.md) | Linear `CFbend` returns NaN at a zero focusing frequency | Tracking | P2 | Silent wrong result | 2026-10-29 (`fa0a43b`, development) |  |  |
| [TRK-07](TRK-07/index.md) | `ExactCFbend` with `k_normal[0]=0` returns NaN | Tracking | P3 | Crash or API hazard | 2026-10-29 (`fa0a43b`, development) |  |  |
| [TRK-08](TRK-08/index.md) | `Sol(ks=0)` returns NaN positions | Tracking | P3 | Crash or API hazard | 2026-10-29 (`fa0a43b`, development) |  |  |
| [TRK-09](TRK-09/index.md) | `ExactCFbend` octupole and decapole kicks lose all precision at small curvature | Tracking | P3 | Silent wrong result | 2026-10-29 (`fa0a43b`, development) | [[gh:https://github.com/BLAST-ImpactX/impactx/issues/1050]] |  |
| [TRK-10](TRK-10/index.md) | Defocusing `ChrPlasmaLens` has the wrong arrival-time sign term | Tracking | P3 | Wrong physics | 2026-10-30 (`dfd49f2`, development) | [[gh:https://github.com/BLAST-ImpactX/impactx/issues/1742]] | [[gh:https://github.com/BLAST-ImpactX/impactx/pull/1743]] |
| [TRK-11](TRK-11/index.md) | Integer factorial overflow corrupts multipole normalization | Tracking | P4 | Wrong physics | 2026-10-29 (`fa0a43b`, development) |  |  |
| [TRK-12](TRK-12/index.md) | `Buncher` uses the wrong transverse fields for spin transport <span class="bb-tag">experimental</span> | Tracking | P4 | Wrong physics | 2026-10-29 (`fa0a43b`, development) |  |  |
| [TRK-13](TRK-13/index.md) | Built-in H⁻ species has the wrong magnetic-anomaly sign <span class="bb-tag">experimental</span> | Tracking | P4 | Wrong physics | 2026-10-29 (`fa0a43b`, development) |  |  |
| [TRK-14](TRK-14/index.md) | Weak nonzero `ChrAcc` acceleration corrupts transport and reference time <span class="bb-tag">extreme-regime</span> | Tracking | P4 | Silent wrong result | 2026-10-29 (`fa0a43b`, development) | [[gh:https://github.com/BLAST-ImpactX/impactx/issues/1573]] |  |
| [TRK-15](TRK-15/index.md) | Weak `ExactSbend` bends lose precision through cancellation <span class="bb-tag">extreme-regime</span> | Tracking | P4 | Silent wrong result | 2026-10-29 (`fa0a43b`, development) |  |  |
| [TRK-16](TRK-16/index.md) | Rolled `ThinDipole`, `Sbend` and `ExactSbend` omit vertical steering <span class="bb-tag">contract-unsettled</span> | Tracking | P1 | Wrong physics | 2026-10-29 (`fa0a43b`, development) | [[gh:https://github.com/BLAST-ImpactX/impactx/issues/1725]] |  |
| [TRK-17](TRK-17/index.md) | `SoftSolenoid` omits transverse fringe-field spin precession <span class="bb-tag">experimental</span> | Tracking | P4 | Wrong physics | 2026-10-29 (`fa0a43b`, development) |  |  |
| [TRK-18](TRK-18/index.md) | Low-energy `ExactDrift` corrupts co-moving arrival times | Tracking | P3 | Wrong physics | 2026-10-29 (`fa0a43b`, development) | [[gh:https://github.com/BLAST-ImpactX/impactx/issues/1576]] |  |
| [TRK-19](TRK-19/index.md) | `ExactCFbend` reverses the skew-decapole horizontal kick <span class="bb-tag">experimental</span> | Tracking | P3 (spin P4) | Wrong physics | 2026-10-29 (`fa0a43b`, development) |  |  |
| [TRK-20](TRK-20/index.md) | `CFbend` spin coupling is singular at coincident frequencies <span class="bb-tag">experimental</span> | Tracking | P4 | Wrong physics | 2026-10-29 (`fa0a43b`, development) |  |  |
| [ENV-01](ENV-01/index.md) | Eigenemittance diagnostics corrupt small or repeated modes | Envelope | P2 | Silent wrong result | 2026-10-29 (`fa0a43b`, development) | [[gh:https://github.com/BLAST-ImpactX/impactx/issues/1481]] |  |
| [ENV-02](ENV-02/index.md) | Nonlinear `DipEdge` optics omit `K5` fringe focusing <span class="bb-tag">partly-source-only</span> | Envelope | P3 | Wrong physics | 2026-10-29 (`fa0a43b`, development) |  |  |
| [ENV-03](ENV-03/index.md) | Phase-space legends label geometric emittance as normalized | Envelope | P5 | Documentation or example mismatch | 2026-10-29 (`fa0a43b`, development) |  |  |
| [COL-01](COL-01/index.md) | PIC `2p5D` longitudinal kick uses a fixed 1 m reference length `r0`, which `Gauss2p5D` exposes | Collective effects | P1 | Wrong physics | 2026-10-28 (`b701a6e`, development) | [[gh:https://github.com/BLAST-ImpactX/impactx/issues/1718]] |  |
| [COL-02](COL-02/index.md) | ISR uses the electron classical radius for every species <span class="bb-tag">partly-source-only</span> | Collective effects | P2 | Wrong physics | 2026-10-29 (`fa0a43b`, development) | [[gh:https://github.com/BLAST-ImpactX/impactx/issues/1708]] |  |
| [COL-03](COL-03/index.md) | `wakeconvolution.w_t_rf` has the wrong functional form | Collective effects | P4 | Silent wrong result | 2026-10-29 (`fa0a43b`, development) |  |  |
| [COL-04](COL-04/index.md) | `Gauss2p5D` underweights the initial quadrature interval | Collective effects | P4 | Wrong physics | 2026-10-29 (`fa0a43b`, development) |  |  |
| [COL-05](COL-05/index.md) | The 3D PIC beta interface loses the supplied Lorentz factor <span class="bb-tag">extreme-regime</span> | Collective effects | P5 | Wrong physics | 2026-10-29 (`fa0a43b`, development) |  |  |
| [IO-01](IO-01/index.md) | A tracer can erase weighted beam size and emittance | Particles and I/O | P3 | Silent wrong result | 2026-10-29 (`fa0a43b`, development) |  |  |
| [IO-02](IO-02/index.md) | Phase-space charge histograms ignore particle weights | Particles and I/O | P4 | Silent wrong result | 2026-10-29 (`fa0a43b`, development) |  |  |
| [IO-03](IO-03/index.md) | Longitudinal plot scale and units disagree by 1000 | Particles and I/O | P5 | Documentation or example mismatch | 2026-10-29 (`fa0a43b`, development) |  |  |
| [IO-04](IO-04/index.md) | `Thermal` negative-charge initialization returns NaN positions | Particles and I/O | P3 | Crash or API hazard | 2026-10-29 (`fa0a43b`, development) |  |  |
| [API-01](API-01/index.md) | MAD-X `PC` leaves the reference energy at its default | Python API | P1 | Silent wrong result | 2026-10-29 (`fa0a43b`, development) | [[gh:https://github.com/BLAST-ImpactX/impactx/issues/1726]] | [[gh:https://github.com/BLAST-ImpactX/impactx/pull/1727]] |
| [API-02](API-02/index.md) | MAD-X ion import discards explicit mass and charge | Python API | P2 | Wrong physics | 2026-10-29 (`fa0a43b`, development) |  |  |
| [API-03](API-03/index.md) | Marker insertion multiplies an exact bend's angle | Python API | P1 | Wrong physics | 2026-10-29 (`fa0a43b`, development) | [[gh:https://github.com/BLAST-ImpactX/impactx/issues/1723]] | [[gh:https://github.com/BLAST-ImpactX/impactx/pull/1724]] |
| [API-04](API-04/index.md) | `before_slice` length changes retain stale collective-kick lengths <span class="bb-tag">partly-source-only</span> | Python API | P3 | Wrong physics | 2026-10-29 (`fa0a43b`, development) |  |  |
| [API-05](API-05/index.md) | Survey reverses normalized-quadrupole focusing for negative charge | Python API | P4 | Silent wrong result | 2026-10-29 (`fa0a43b`, development) |  |  |
| [API-06](API-06/index.md) | PALS import discards quadrupole tilt and skew | Python API | P2 | Wrong physics | 2026-10-29 (`fa0a43b`, development) |  |  |
| [API-07](API-07/index.md) | PALS import discards body offsets and active apertures <span class="bb-tag">partly-source-only</span> | Python API | P2 | Wrong physics | 2026-10-29 (`fa0a43b`, development) |  |  |
| [API-08](API-08/index.md) | Dashboard imports the bundled proton cyclotron with electron mass and charge <span class="bb-tag">partly-source-only</span> | Python API | P3 | Wrong physics | 2026-10-29 (`fa0a43b`, development) |  |  |
| [API-09](API-09/index.md) | Dashboard Twiss import corrupts scientific notation and drops negative alpha <span class="bb-tag">partly-source-only</span> | Python API | P3 | Wrong physics | 2026-10-29 (`fa0a43b`, development) |  |  |
| [API-10](API-10/index.md) | Dashboard envelope commands omit the selected beam intensity <span class="bb-tag">partly-source-only</span> | Python API | P3 | Wrong physics | 2026-10-29 (`fa0a43b`, development) |  |  |
| [API-11](API-11/index.md) | Dashboard silently accepts a partial lattice when repeated cells are unresolved <span class="bb-tag">partly-source-only</span> | Python API | P3 | Silent wrong result | 2026-10-29 (`fa0a43b`, development) |  |  |
| [API-12](API-12/index.md) | Dashboard reimports envelope and reference exports as particle tracking <span class="bb-tag">partly-source-only</span> | Python API | P4 | Crash or API hazard | 2026-10-29 (`fa0a43b`, development) |  |  |
| [API-13](API-13/index.md) | MAD-X `BV=-1` retains forward straight-multipole polarity | Python API | P2 | Wrong physics | 2026-10-29 (`fa0a43b`, development) |  |  |
| [API-14](API-14/index.md) | Harmonic-only MAD-X ring cavities import at zero frequency <span class="bb-tag">known-limitation</span> | Python API | P2 | Wrong physics | 2026-10-29 (`fa0a43b`, development) |  |  |
| [API-15](API-15/index.md) | Programmable reference callbacks lose mutations without a retained live wrapper | Python API | P3 | Wrong physics | 2026-10-29 (`fa0a43b`, development) |  |  |
| [API-16](API-16/index.md) | Dashboard variable refresh leaves equal elements at stale lengths <span class="bb-tag">partly-source-only</span> | Python API | P3 | Silent wrong result | 2026-10-29 (`fa0a43b`, development) |  |  |
| [API-17](API-17/index.md) | CSR-only dashboard exports omit required particle shape <span class="bb-tag">partly-source-only</span> | Python API | P4 | Crash or API hazard | 2026-10-29 (`fa0a43b`, development) |  |  |
| [API-18](API-18/index.md) | Marker insertion compresses field profiles and repeats linear maps in each piece | Python API | P1 | Wrong physics | 2026-10-29 (`fa0a43b`, development) | [[gh:https://github.com/BLAST-ImpactX/impactx/issues/1728]] |  |
