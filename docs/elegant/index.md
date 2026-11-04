---
title: Elegant
hide:
  - toc
---

# Elegant

Verified defects in [Elegant](https://github.com/rtsoliday/elegant) ([source](https://github.com/rtsoliday/elegant)).
Each entry links to its reproducer, the models that found and confirmed it, and any upstream issue or pull request.
Machine-readable: [defects.json](defects.json).
Tags are explained in [Methodology](../methodology.md#tags).

## Running the reproducers

Build serial Elegant, Pelegant and the SDDS tools from rtsoliday/elegant master
`0355dd6aca12` and rtsoliday/SDDS master `a3f730a14c49`, on which the reproducers ran
(CPU, Open MPI, no CUDA). Put the `elegant` and SDDS `bin` directories on `PATH` and set
`RPN_DEFNS` to the `defns.rpn` file of SDDS; Elegant does not start without it. Download a
reproducer with the `_*.py` helpers shown beside it and run `python <ID>.py`, which runs
serial Elegant; `--mpi` adds a 3-rank Pelegant run (needs `mpirun`), and entries that occur
only under MPI always run Pelegant with a serial control. SCT-01 also needs Valgrind,
DOC-01 an Elegant source checkout named by `ELEGANT_SOURCE`, and IO-02 to IO-04 `gcc`
with the SDDS libraries of the build.

Some reproducers also need: pysdds, scipy.

## Defects

<div class="bb-filters" data-column="Area"></div>

| ID | Defect | Area | Priority | Severity | Last reproduced | Issue | PR |
|---|---|---|---|---|---|---|---|
| [TRK-01](TRK-01/index.md) | Finite EHKICK bends sloped particles on the wrong radius and leaves yp unchanged <span class="bb-tag">partly-source-only</span> | Tracking | P3 | Wrong physics | 2026-11-04 (`0355dd6`, master) |  |  |
| [TRK-03](TRK-03/index.md) | KQUAD given by pole-tip field and bore tracks its fringes with zero strength until a matrix or Twiss request <span class="bb-tag">partly-source-only</span> | Tracking | P2 | Wrong physics | 2026-11-04 (`0355dd6`, master) |  |  |
| [TRK-04](TRK-04/index.md) | KQUAD fringes ignore the fractional strength error FSE that scales the body <span class="bb-tag">partly-source-only</span> | Tracking | P2 | Wrong physics | 2026-11-04 (`0355dd6`, master) |  |  |
| [TRK-05](TRK-05/index.md) | Zero-voltage MODRF and RAMPRF advance the arrival time by L instead of the slanted path length | Tracking | P3 | Wrong physics | 2026-11-04 (`0355dd6`, master) |  |  |
| [TRK-06](TRK-06/index.md) | RFDF drifts after a kick at the speed the particle had before it <span class="bb-tag">partly-source-only</span> | Tracking | P3 | Wrong physics | 2026-11-04 (`0355dd6`, master) |  |  |
| [TRK-07](TRK-07/index.md) | MODRF damped phase modulation keeps an undamped local frequency <span class="bb-tag">contract-unsettled</span> | Tracking | P4 | Wrong physics | 2026-11-04 (`0355dd6`, master) |  |  |
| [TRK-08](TRK-08/index.md) | Finite matrix RFCA assigns the whole transit to the entrance speed <span class="bb-tag">partly-source-only</span> | Tracking | P3 | Wrong physics | 2026-11-04 (`0355dd6`, master) |  |  |
| [TRK-09](TRK-09/index.md) | CCBEND plane changes add the axial projection of the flight, not its length, to the path | Tracking | P2 | Wrong physics | 2026-11-04 (`0355dd6`, master) |  |  |
| [TRK-10](TRK-10/index.md) | CSBEND edge model 3 is documented as symplectic but its finite-gap kick is not six-dimensionally symplectic <span class="bb-tag">contract-unsettled</span> <span class="bb-tag">partly-source-only</span> | Tracking | P4 | Documentation or example mismatch | 2026-11-04 (`0355dd6`, master) |  |  |
| [TRK-11](TRK-11/index.md) | SOLE given by its field B reverses the spin precession of positively charged particles | Tracking | P3 | Wrong physics | 2026-11-04 (`0355dd6`, master) |  |  |
| [TRK-13](TRK-13/index.md) | CSBEND spin tracking passes canonical momenta to the slope-based spin update <span class="bb-tag">partly-source-only</span> | Tracking | P2 | Wrong physics | 2026-11-04 (`0355dd6`, master) |  |  |
| [TRK-14](TRK-14/index.md) | POLAR random-direction modes give NaN spins along a longitudinal axis <span class="bb-tag">partly-source-only</span> | Tracking | P3 | Wrong physics | 2026-11-04 (`0355dd6`, master) |  |  |
| [TRK-15](TRK-15/index.md) | GKICKMAP applies classical radiation loss with SYNCH_RAD disabled | Tracking | P3 | Wrong physics | 2026-11-04 (`0355dd6`, master) |  |  |
| [TRK-16](TRK-16/index.md) | UKICKMAP loss compaction gives survivors extra radiation updates | Tracking | P3 | Wrong physics | 2026-11-04 (`0355dd6`, master) |  |  |
| [TRK-17](TRK-17/index.md) | Yawed kick maps sample half a slice behind the particle | Tracking | P2 | Wrong physics | 2026-11-04 (`0355dd6`, master) |  |  |
| [TRK-18](TRK-18/index.md) | LSRMDLTR radiation loss scales with the cube of the reference momentum | Tracking | P3 | Wrong physics | 2026-11-04 (`0355dd6`, master) |  |  |
| [TRK-19](TRK-19/index.md) | TMCF fiducial timing uses transverse positions as momentum components <span class="bb-tag">partly-source-only</span> | Tracking | P4 | Wrong physics | 2026-11-04 (`0355dd6`, master) |  |  |
| [TRK-20](TRK-20/index.md) | MAPSOLENOID combined misalignment rotations are not mutual inverses <span class="bb-tag">partly-source-only</span> | Tracking | P4 | Wrong physics | 2026-11-04 (`0355dd6`, master) |  |  |
| [TRK-21](TRK-21/index.md) | RFCA subtracts its CHANGE_T clock-wrap interval once per kick <span class="bb-tag">partly-source-only</span> | Tracking | P2 | Wrong physics | 2026-11-04 (`0355dd6`, master) |  |  |
| [TRK-22](TRK-22/index.md) | Downstream RFCA omits the accumulated clock phase for non-harmonic frequencies <span class="bb-tag">partly-source-only</span> | Tracking | P2 | Wrong physics | 2026-11-04 (`0355dd6`, master) |  |  |
| [TRK-23](TRK-23/index.md) | CSBEND dipole-only strength error rescales its octupole and higher multipoles <span class="bb-tag">partly-source-only</span> | Tracking | P2 | Wrong physics | 2026-11-04 (`0355dd6`, master) |  |  |
| [TRK-24](TRK-24/index.md) | KQUAD and KSEXT edge multipole cutoffs select rows by the body file's order labels <span class="bb-tag">partly-source-only</span> | Tracking | P2 | Wrong physics | 2026-11-04 (`0355dd6`, master) |  |  |
| [TRK-25](TRK-25/index.md) | Shared steering multipole tables carry one element's factor into another | Tracking | P2 | Wrong physics | 2026-11-04 (`0355dd6`, master) |  |  |
| [TRK-26](TRK-26/index.md) | Negative CCBEND edge multipole signs are chosen from the body table's order labels | Tracking | P2 | Wrong physics | 2026-11-04 (`0355dd6`, master) |  |  |
| [TRK-27](TRK-27/index.md) | CSBEND with EXPAND_HAMILTONIAN=1 does not transport the spin through the bend body <span class="bb-tag">partly-source-only</span> | Tracking | P2 | Wrong physics | 2026-11-04 (`0355dd6`, master) |  |  |
| [DOC-05](DOC-05/index.md) | Automatic CSBEND field expansion order is n+2, not the n+3 the manual states <span class="bb-tag">partly-source-only</span> | Tracking | P4 | Documentation or example mismatch | 2026-11-04 (`0355dd6`, master) |  |  |
| [OPT-01](OPT-01/index.md) | Unmatched alpha derivatives report phase derivatives | Optics | P3 | Wrong physics | 2026-11-04 (`0355dd6`, master) |  |  |
| [OPT-02](OPT-02/index.md) | Cubic compaction omits momentum–transverse quadratic terms | Optics | P3 | Wrong physics | 2026-11-04 (`0355dd6`, master) |  |  |
| [OPT-03](OPT-03/index.md) | CCBEND radiation totals omit repeated periods | Optics | P2 | Silent wrong result | 2026-11-04 (`0355dd6`, master) |  |  |
| [OPT-04](OPT-04/index.md) | Initial moments omit shared-dispersion covariance terms <span class="bb-tag">partly-source-only</span> | Optics | P2 | Wrong physics | 2026-11-04 (`0355dd6`, master) |  |  |
| [OPT-05](OPT-05/index.md) | Zero-length MULT quadrupole cubic matrix contains infinities | Optics | P3 | Wrong physics | 2026-11-04 (`0355dd6`, master) |  |  |
| [OPT-06](OPT-06/index.md) | Analytic QUAD cubic chromatic coefficients are wrong <span class="bb-tag">partly-source-only</span> | Optics | P2 | Wrong physics | 2026-11-04 (`0355dd6`, master) |  |  |
| [OPT-07](OPT-07/index.md) | Higher chromaticities mix cell and repeated-period phases <span class="bb-tag">partly-source-only</span> | Optics | P2 | Wrong physics | 2026-11-04 (`0355dd6`, master) |  |  |
| [OPT-09](OPT-09/index.md) | Quartic path term enters cubic momentum compaction | Optics | P3 | Wrong physics | 2026-11-04 (`0355dd6`, master) |  |  |
| [OPT-11](OPT-11/index.md) | MULT pole-tip optics use electron rigidity for other species <span class="bb-tag">partly-source-only</span> | Optics | P3 | Wrong physics | 2026-11-04 (`0355dd6`, master) |  |  |
| [OPT-12](OPT-12/index.md) | Unmatched chromatic derivatives use exit dispersion instead of entrance dispersion <span class="bb-tag">partly-source-only</span> | Optics | P3 | Wrong physics | 2026-11-04 (`0355dd6`, master) |  |  |
| [OPT-13](OPT-13/index.md) | Quick second chromaticity translates a supplied closed orbit twice <span class="bb-tag">partly-source-only</span> | Optics | P2 | Wrong physics | 2026-11-04 (`0355dd6`, master) |  |  |
| [OPT-14](OPT-14/index.md) | Slice normalized emittances omit the mean momentum offset | Optics | P2 | Silent wrong result | 2026-11-04 (`0355dd6`, master) |  |  |
| [OPT-15](OPT-15/index.md) | Individual matrix output keeps nonlinear coefficients of preceding elements | Optics | P4 | Silent wrong result | 2026-11-04 (`0355dd6`, master) |  |  |
| [DOC-01](DOC-01/index.md) | Manual gives the third-order momentum compaction term as δ² instead of δ³ | Optics | P4 | Documentation or example mismatch | 2026-11-04 (`0355dd6`, master) |  |  |
| [COL-02](COL-02/index.md) | Pelegant falsely warns that fast-orbit-feedback BPM readings are not reduced across ranks | Collective effects | P4 | Documentation or example mismatch | 2026-11-04 (`0355dd6`, master) |  |  |
| [COL-03](COL-03/index.md) | Round nonlinear space charge omits the integrating length and optics weighting <span class="bb-tag">partly-source-only</span> | Collective effects | P3 | Wrong physics | 2026-11-04 (`0355dd6`, master) |  |  |
| [COL-04](COL-04/index.md) | Nonlinear space charge ignores the horizontal and vertical plane switches | Collective effects | P2 | Wrong physics | 2026-11-04 (`0355dd6`, master) |  |  |
| [COL-05](COL-05/index.md) | Uniform nonlinear space charge is too weak by √12 | Collective effects | P2 | Wrong physics | 2026-11-04 (`0355dd6`, master) |  |  |
| [COL-06](COL-06/index.md) | Single-precision reciprocals in the complex error function reduce the Gaussian force accuracy <span class="bb-tag">partly-source-only</span> | Collective effects | P4 | Silent wrong result | 2026-11-04 (`a3f730a`, master) |  |  |
| [COL-07](COL-07/index.md) | WAKE cache substitutes the first requested column pair | Collective effects | P2 | Wrong physics | 2026-11-04 (`0355dd6`, master) |  |  |
| [COL-08](COL-08/index.md) | Broadband ZTRANSVERSE doubles its DC impedance | Collective effects | P2 | Wrong physics | 2026-11-04 (`0355dd6`, master) |  |  |
| [COL-09](COL-09/index.md) | Broadband ZTRANSVERSE uses the wrong Nyquist frequency and sign | Collective effects | P3 | Wrong physics | 2026-11-04 (`0355dd6`, master) |  |  |
| [COL-11](COL-11/index.md) | BIN_ONCE clears the grid required by attenuation-mode CSRDRIFT <span class="bb-tag">partly-source-only</span> | Collective effects | P2 | Wrong physics | 2026-11-04 (`0355dd6`, master) |  |  |
| [COL-12](COL-12/index.md) | BEAMBEAM cancels the tracked particle's charge sign, so positrons are kicked like electrons <span class="bb-tag">partly-source-only</span> | Collective effects | P2 | Wrong physics | 2026-11-04 (`0355dd6`, master) |  |  |
| [COL-13](COL-13/index.md) | Round Gaussian BEAMBEAM produces NaNs at its center <span class="bb-tag">partly-source-only</span> | Collective effects | P3 | Silent wrong result | 2026-11-04 (`0355dd6`, master) |  |  |
| [COL-14](COL-14/index.md) | BEAMBEAM kicks change arrival time because s is not rescaled | Collective effects | P3 | Wrong physics | 2026-11-04 (`0355dd6`, master) |  |  |
| [COL-15](COL-15/index.md) | FTRFMODE ignores RIGID_UNTIL_PASS | Collective effects | P2 | Wrong physics | 2026-11-04 (`0355dd6`, master) |  |  |
| [COL-16](COL-16/index.md) | Dipole resonators create momentum in the disabled plane | Collective effects | P3 | Wrong physics | 2026-11-04 (`0355dd6`, master) |  |  |
| [COL-17](COL-17/index.md) | Later SCMULT bunches use the exit beta function for the entrance weight | Collective effects | P2 | Wrong physics | 2026-11-04 (`0355dd6`, master) |  |  |
| [COL-18](COL-18/index.md) | SCMULT size-filter history is shared between bunches | Collective effects | P2 | Wrong physics | 2026-11-04 (`0355dd6`, master) |  |  |
| [COL-19](COL-19/index.md) | Empty SCMULT slice centroids break translation symmetry | Collective effects | P2 | Wrong physics | 2026-11-04 (`0355dd6`, master) |  |  |
| [COL-20](COL-20/index.md) | MPI SCMULT mixes bunch statistics when a rank holds no particles of a bunch <span class="bb-tag">partly-source-only</span> | Collective effects | P2 | Wrong physics | 2026-11-04 (`0355dd6`, master) |  |  |
| [COL-21](COL-21/index.md) | Sliced linear SCMULT normalizes every slice by the whole-bunch beam size <span class="bb-tag">partly-source-only</span> | Collective effects | P2 | Wrong physics | 2026-11-04 (`0355dd6`, master) |  |  |
| [COL-22](COL-22/index.md) | Constant RFMODE frequency waveforms repeatedly scale stored voltage <span class="bb-tag">partly-source-only</span> | Collective effects | P2 | Wrong physics | 2026-11-04 (`0355dd6`, master) |  |  |
| [COL-23](COL-23/index.md) | TRWAKE quadrupole probe coordinates omit element misalignment <span class="bb-tag">partly-source-only</span> | Collective effects | P3 | Wrong physics | 2026-11-04 (`0355dd6`, master) |  |  |
| [COL-24](COL-24/index.md) | Single-bunch TRWAKE fixed bins retain a zero time reference | Collective effects | P2 | Wrong physics | 2026-11-04 (`0355dd6`, master) |  |  |
| [COL-25](COL-25/index.md) | MPI TRWAKE adds fixed-bin histograms whose time origins differ between ranks | Collective effects | P2 | Wrong physics | 2026-11-04 (`0355dd6`, master) |  |  |
| [COL-26](COL-26/index.md) | MPI TRWAKE combines different bunches when ranks skip locally empty bunches <span class="bb-tag">partly-source-only</span> | Collective effects | P2 | Wrong physics | 2026-11-04 (`0355dd6`, master) |  |  |
| [DOC-04](DOC-04/index.md) | SCMULT manual prints the averaging filter with the current turn in its history term | Collective effects | P4 | Documentation or example mismatch | 2026-11-04 (`0355dd6`, master) |  |  |
| [SCT-04](SCT-04/index.md) | MATTER collision slices omit incoming transverse drift | Scattering and pressure | P2 | Wrong physics | 2026-11-04 (`0355dd6`, master) |  |  |
| [SCT-05](SCT-05/index.md) | Standalone electron recoil omits energy loss | Scattering and pressure | P2 | Wrong physics | 2026-11-04 (`0355dd6`, master) |  |  |
| [SCT-06](SCT-06/index.md) | Active recoil transfers use inconsistent energy normalization | Scattering and pressure | P2 | Wrong physics | 2026-11-04 (`0355dd6`, master) |  |  |
| [SCT-07](SCT-07/index.md) | Smooth IBS ignores the plane switches DO_X, DO_Y and DO_Z | Scattering and pressure | P2 | Wrong physics | 2026-11-04 (`0355dd6`, master) |  |  |
| [SCT-08](SCT-08/index.md) | Smooth IBS shifts the beam centroid when there is dispersion and a momentum offset | Scattering and pressure | P3 | Wrong physics | 2026-11-04 (`0355dd6`, master) |  |  |
| [SCT-01](SCT-01/index.md) | Pressure-region diagnostic reads an invalid sample index | Scattering and pressure | P4 | Crash or API hazard | 2026-11-04 (`0355dd6`, master) |  |  |
| [SCT-02](SCT-02/index.md) | Pelegant writes zero IBS rate columns and an unset vertical growth rate | Scattering and pressure | P4 | Silent wrong result | 2026-11-04 (`0355dd6`, master) |  |  |
| [SCT-03](SCT-03/index.md) | MPI Touschek generation crashes when tracking is disabled | Scattering and pressure | P3 | Crash or API hazard | 2026-11-04 (`0355dd6`, master) |  |  |
| [SCT-10](SCT-10/index.md) | CBSCAT computes a negative Klein–Nishina total cross section just above its Thomson cutoff | Scattering and pressure | P3 | Wrong physics | 2026-11-04 (`0355dd6`, master) |  |  |
| [SCT-11](SCT-11/index.md) | Approximate CBSCAT recoil depends on the reference momentum | Scattering and pressure | P3 | Wrong physics | 2026-11-04 (`0355dd6`, master) |  |  |
| [SCT-12](SCT-12/index.md) | Thin forced-MCS MATTER omits the flight distance from the transverse angles | Scattering and pressure | P4 | Wrong physics | 2026-11-04 (`0355dd6`, master) |  |  |
| [IO-01](IO-01/index.md) | Pelegant fails to load a multi-page beam file when a later page leaves a worker without rows | Lattices, particles and I/O | P3 | Crash or API hazard | 2026-11-04 (`0355dd6`, master) |  |  |
| [IO-02](IO-02/index.md) | SDDS Gill–Miller integral of a three-sample table is zero and leaves a result unset | Lattices, particles and I/O | P4 | Silent wrong result | 2026-11-04 (`a3f730a`, master) |  |  |
| [IO-03](IO-03/index.md) | SDDS complete elliptic integral E(k) returns π/4 at the endpoints k = ±1 | Lattices, particles and I/O | P4 | Silent wrong result | 2026-11-04 (`a3f730a`, master) |  |  |
| [IO-04](IO-04/index.md) | SDDS moment accumulation discards the earlier batches | Lattices, particles and I/O | P4 | Silent wrong result | 2026-11-04 (`a3f730a`, master) |  |  |
| [IO-05](IO-05/index.md) | The manual's bunch-input example assigns particle IDs 2001–3000 to bunch 1 instead of bunch 2 | Lattices, particles and I/O | P4 | Documentation or example mismatch | 2026-11-04 (`0355dd6`, master) |  |  |
