---
title: Elegant
hide:
  - toc
---

# Elegant

Verified defects in [Elegant](https://ops.aps.anl.gov/elegant.html) ([source](https://github.com/rtsoliday/elegant)).
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
only under MPI (COL-02, COL-20, COL-26, IO-01, SCT-02, SCT-03, SCT-24) always run Pelegant with a
serial control. SCT-24 uses five ranks to exercise distributed IBSCATTER integration;
the other MPI reproducers use three. The extra `COL-24-mpi.py` also always runs Pelegant
with serial controls;
`COL-24.py` runs the single-bunch case in serial by default. `TRK-03-fse.py` adds the
fractional-strength and zero-field controls for TRK-03. COL-29, COL-32, COL-34 to COL-46,
COL-50 to COL-52, DOC-06, DOC-07, DOC-10, DOC-11, OPT-16 to OPT-31, SCT-01, SCT-17 to SCT-23,
SCT-25 and TRK-47 run serial Elegant only. COL-32, COL-38, COL-40, COL-41, COL-50 and SCT-01
also need Valgrind (COL-50 can use `VALGRIND_OPTS=--vgdb=no` without disabling Memcheck),
and SCT-17 runs
the `ibsEmittance` program of the Elegant build. DOC-01, DOC-10 and DOC-11 need an Elegant
source checkout named by `ELEGANT_SOURCE`, and IO-02 to IO-04 `gcc` with the SDDS libraries
of the build. COL-51 accepts only the expected status-1 bunch-count abort; unrelated
failures raise, and its native run uses the shared 600 s timeout.

Some reproducers also need: pysdds, scipy.

## Defects

<div class="bb-filters" data-column="Area"></div>

| ID | Defect | Area | Priority | Severity | Last reproduced | Issue | PR |
|---|---|---|---|---|---|---|---|
| [OPT-07](OPT-07/index.md) | Higher chromaticities mix cell and repeated-period phases <span class="bb-tag">partly-source-only</span> | Optics | P1 | Wrong physics | 2026-11-04 (`0355dd6`, master) |  |  |
| [COL-44](COL-44/index.md) | An accelerating RFCW without a longitudinal wake evaluates its longitudinal space charge at the entrance momentum | Collective effects | P1 | Wrong physics | 2026-11-04 (`7514db3`, master) |  |  |
| [OPT-03](OPT-03/index.md) | CCBEND and LGBEND radiation totals omit repeated periods | Optics | P1 | Silent wrong result | 2026-11-04 (`0355dd6`, master) |  |  |
| [OPT-14](OPT-14/index.md) | Slice normalized emittances omit the mean momentum offset | Optics | P1 | Silent wrong result | 2026-11-04 (`0355dd6`, master) |  |  |
| [TRK-03](TRK-03/index.md) | KQUAD fringes use the stored gradient instead of the effective body gradient <span class="bb-tag">partly-source-only</span> | Tracking | P2 | Wrong physics | 2026-11-04 (`0355dd6`, master) |  |  |
| [TRK-08](TRK-08/index.md) | Finite matrix RFCA assigns the whole transit to the entrance speed <span class="bb-tag">partly-source-only</span> | Tracking | P2 | Wrong physics | 2026-11-04 (`0355dd6`, master) |  |  |
| [TRK-09](TRK-09/index.md) | CCBEND plane changes add the axial projection of the flight, not its length, to the path | Tracking | P2 | Wrong physics | 2026-11-04 (`0355dd6`, master) |  |  |
| [TRK-13](TRK-13/index.md) | CSBEND spin tracking passes canonical momenta to the slope-based spin update <span class="bb-tag">partly-source-only</span> | Tracking | P2 | Wrong physics | 2026-11-04 (`0355dd6`, master) |  |  |
| [TRK-17](TRK-17/index.md) | Yawed kick maps sample half a slice behind the particle | Tracking | P2 | Wrong physics | 2026-11-04 (`0355dd6`, master) |  |  |
| [TRK-21](TRK-21/index.md) | RFCA subtracts its CHANGE_T clock-wrap interval once per kick <span class="bb-tag">partly-source-only</span> | Tracking | P2 | Wrong physics | 2026-11-04 (`0355dd6`, master) |  |  |
| [TRK-22](TRK-22/index.md) | Downstream RFCA omits the accumulated clock phase for non-harmonic frequencies <span class="bb-tag">partly-source-only</span> | Tracking | P2 | Wrong physics | 2026-11-04 (`0355dd6`, master) |  |  |
| [TRK-23](TRK-23/index.md) | CSBEND dipole-only strength error rescales its octupole and higher multipoles <span class="bb-tag">partly-source-only</span> | Tracking | P2 | Wrong physics | 2026-11-04 (`0355dd6`, master) |  |  |
| [TRK-24](TRK-24/index.md) | KQUAD and KSEXT edge multipole cutoffs select rows by the body file's order labels <span class="bb-tag">partly-source-only</span> | Tracking | P2 | Wrong physics | 2026-11-04 (`0355dd6`, master) |  |  |
| [TRK-25](TRK-25/index.md) | Shared steering multipole tables carry one element's factor into another | Tracking | P2 | Wrong physics | 2026-11-04 (`0355dd6`, master) |  |  |
| [TRK-26](TRK-26/index.md) | Negative CCBEND edge multipole signs are chosen from the body table's order labels | Tracking | P2 | Wrong physics | 2026-11-04 (`0355dd6`, master) |  |  |
| [TRK-27](TRK-27/index.md) | CSBEND with EXPAND_HAMILTONIAN=1 does not transport the spin through the bend body <span class="bb-tag">partly-source-only</span> | Tracking | P2 | Wrong physics | 2026-11-04 (`0355dd6`, master) |  |  |
| [TRK-32](TRK-32/index.md) | CCBEND with ETILT as its only error skips the entrance misalignment transform | Tracking | P2 | Wrong physics | 2026-11-04 (`7514db3`, master) |  |  |
| [TRK-34](TRK-34/index.md) | CSBEND with EXPAND_HAMILTONIAN=1 scales its quantum excitation by the integration order <span class="bb-tag">partly-source-only</span> | Tracking | P2 | Wrong physics | 2026-11-04 (`7514db3`, master) |  |  |
| [TRK-35](TRK-35/index.md) | CSBEND with EXPAND_HAMILTONIAN=1 leaves the curvature term out of the path length | Tracking | P2 | Wrong physics | 2026-11-04 (`7514db3`, master) |  |  |
| [TRK-36](TRK-36/index.md) | Identical CCBENDs that share a tracking-based matrix lose the path-length reference correction | Tracking | P2 | Wrong physics | 2026-11-04 (`7514db3`, master) |  |  |
| [TRK-37](TRK-37/index.md) | CSBEND edge model 5 adds a constant path offset and drops the y² path term when K3 is small | Tracking | P2 | Wrong physics | 2026-11-04 (`7514db3`, master) |  |  |
| [TRK-38](TRK-38/index.md) | CSBEND FSE_CORRECTION cancels a dipole-only strength error and a horizontal steering kick | Tracking | P2 | Wrong physics | 2026-11-04 (`7514db3`, master) |  |  |
| [TRK-39](TRK-39/index.md) | CSBEND FSE_CORRECTION is applied only after a command computes the bend's matrix | Tracking | P2 | Wrong physics | 2026-11-04 (`7514db3`, master) |  |  |
| [TRK-41](TRK-41/index.md) | Standing-wave RFCA kick tracking fiducializes at the first kick, so PHASE=90 is not the crest | Tracking | P2 | Wrong physics | 2026-11-04 (`7514db3`, master) |  |  |
| [TRK-42](TRK-42/index.md) | RFCA END1_FOCUS and END2_FOCUS edge lenses omit the 1/β² factor <span class="bb-tag">partly-source-only</span> | Tracking | P2 | Wrong physics | 2026-11-04 (`7514db3`, master) |  |  |
| [TRK-45](TRK-45/index.md) | Element division repeats a one-sided RFCA end-focus lens at every piece <span class="bb-tag">partly-source-only</span> | Tracking | P2 | Wrong physics | 2026-11-05 (`0355dd6`) |  |  |
| [OPT-02](OPT-02/index.md) | Cubic compaction omits momentum–transverse quadratic terms | Optics | P2 | Wrong physics | 2026-11-04 (`0355dd6`, master) |  |  |
| [OPT-04](OPT-04/index.md) | Initial moments omit shared-dispersion covariance terms <span class="bb-tag">partly-source-only</span> | Optics | P2 | Wrong physics | 2026-11-04 (`0355dd6`, master) |  |  |
| [OPT-06](OPT-06/index.md) | Analytic QUAD cubic chromatic coefficients are wrong <span class="bb-tag">partly-source-only</span> | Optics | P2 | Wrong physics | 2026-11-04 (`0355dd6`, master) |  |  |
| [OPT-09](OPT-09/index.md) | Quartic path term enters cubic momentum compaction | Optics | P2 | Wrong physics | 2026-11-04 (`0355dd6`, master) |  |  |
| [OPT-12](OPT-12/index.md) | Unmatched chromatic derivatives use exit dispersion instead of entrance dispersion <span class="bb-tag">partly-source-only</span> | Optics | P2 | Wrong physics | 2026-11-04 (`0355dd6`, master) |  |  |
| [OPT-13](OPT-13/index.md) | Quick second chromaticity translates a supplied closed orbit twice <span class="bb-tag">partly-source-only</span> | Optics | P2 | Wrong physics | 2026-11-04 (`0355dd6`, master) |  |  |
| [OPT-16](OPT-16/index.md) | CCBEND radiation integral I4 omits the rectangular-edge terms | Optics | P2 | Wrong physics | 2026-11-04 (`7514db3`, master) |  |  |
| [OPT-17](OPT-17/index.md) | CCBEND radiation integral I4 subtracts the sextupole-centering offset twice | Optics | P2 | Wrong physics | 2026-11-04 (`7514db3`, master) |  |  |
| [OPT-18](OPT-18/index.md) | Third-order concatenation multiplies spurious path-origin terms of tracking-derived maps by the accumulated path length <span class="bb-tag">partly-source-only</span> | Optics | P2 | Wrong physics | 2026-11-04 (`7514db3`, master) |  |  |
| [OPT-23](OPT-23/index.md) | moments_output mode emittances normalize the (s, δ) pair with the transverse sign <span class="bb-tag">partly-source-only</span> | Optics | P2 | Wrong physics | 2026-11-04 (`7514db3`, master) |  |  |
| [OPT-29](OPT-29/index.md) | Resonance driving terms ignore the roll of sextupoles, octupoles, KQUSE and bends <span class="bb-tag">partly-source-only</span> | Optics | P2 | Wrong physics | 2026-11-05 (`0355dd6`) |  |  |
| [COL-04](COL-04/index.md) | Nonlinear space charge ignores the horizontal and vertical plane switches | Collective effects | P2 | Wrong physics | 2026-11-04 (`0355dd6`, master) |  |  |
| [COL-05](COL-05/index.md) | Uniform nonlinear space charge is too weak by √12 | Collective effects | P2 | Wrong physics | 2026-11-04 (`0355dd6`, master) |  |  |
| [COL-07](COL-07/index.md) | WAKE cache substitutes the first requested column pair | Collective effects | P2 | Wrong physics | 2026-11-04 (`0355dd6`, master) |  |  |
| [COL-08](COL-08/index.md) | Broadband ZTRANSVERSE doubles its DC impedance | Collective effects | P2 | Wrong physics | 2026-11-04 (`0355dd6`, master) |  |  |
| [COL-11](COL-11/index.md) | BIN_ONCE clears the grid required by attenuation-mode CSRDRIFT <span class="bb-tag">partly-source-only</span> | Collective effects | P2 | Wrong physics | 2026-11-04 (`0355dd6`, master) |  |  |
| [COL-15](COL-15/index.md) | FTRFMODE ignores RIGID_UNTIL_PASS | Collective effects | P2 | Wrong physics | 2026-11-04 (`0355dd6`, master) |  |  |
| [COL-17](COL-17/index.md) | Later SCMULT bunches use the exit beta function for the entrance weight | Collective effects | P2 | Wrong physics | 2026-11-04 (`0355dd6`, master) |  |  |
| [COL-18](COL-18/index.md) | SCMULT size-filter history is shared between bunches | Collective effects | P2 | Wrong physics | 2026-11-04 (`0355dd6`, master) |  |  |
| [COL-19](COL-19/index.md) | Empty SCMULT slice centroids break translation symmetry | Collective effects | P2 | Wrong physics | 2026-11-04 (`0355dd6`, master) |  |  |
| [COL-20](COL-20/index.md) | MPI SCMULT mixes bunch statistics when a rank holds no particles of a bunch <span class="bb-tag">partly-source-only</span> | Collective effects | P2 | Wrong physics | 2026-11-04 (`0355dd6`, master) |  |  |
| [COL-21](COL-21/index.md) | Sliced linear SCMULT normalizes every slice by the whole-bunch beam size <span class="bb-tag">partly-source-only</span> | Collective effects | P2 | Wrong physics | 2026-11-04 (`0355dd6`, master) |  |  |
| [COL-22](COL-22/index.md) | Constant RFMODE frequency waveforms repeatedly scale stored voltage <span class="bb-tag">partly-source-only</span> | Collective effects | P2 | Wrong physics | 2026-11-04 (`0355dd6`, master) |  |  |
| [COL-24](COL-24/index.md) | TRWAKE fixed bins use an absent or rank-local time origin | Collective effects | P2 | Wrong physics | 2026-11-04 (`0355dd6`, master) |  |  |
| [COL-26](COL-26/index.md) | MPI TRWAKE combines different bunches when ranks skip locally empty bunches <span class="bb-tag">partly-source-only</span> | Collective effects | P2 | Wrong physics | 2026-11-04 (`0355dd6`, master) |  |  |
| [COL-27](COL-27/index.md) | Averaged nonlinear SCMULT normalizes its kick with the unfiltered beam sizes <span class="bb-tag">contract-unsettled</span> | Collective effects | P2 | Wrong physics | 2026-11-04 (`0355dd6`, master) |  |  |
| [COL-33](COL-33/index.md) | SCMULT discards the strength accumulated after a ring's last kick on every turn <span class="bb-tag">partly-source-only</span> | Collective effects | P2 | Wrong physics | 2026-11-04 (`7514db3`, master) |  |  |
| [COL-34](COL-34/index.md) | Transient CSRCSBEND trapezoid integration gives its first regular sample weight 3/4 instead of 1/2 | Collective effects | P2 | Wrong physics | 2026-11-04 (`7514db3`, master) |  |  |
| [COL-37](COL-37/index.md) | CSRCSBEND with LINEARIZE=1 mixes arrival time with path length, applies ETILT twice and drops SYNCH_RAD | Collective effects | P2 | Wrong physics | 2026-11-04 (`7514db3`, master) |  |  |
| [COL-39](COL-39/index.md) | Stupakov-model CSRDRIFT trapezoid integration omits the first interval behind each particle | Collective effects | P2 | Wrong physics | 2026-11-04 (`7514db3`, master) |  |  |
| [COL-43](COL-43/index.md) | CSRCSBEND and CSRDRIFT apply the CSR wake half a bin toward the bunch tail <span class="bb-tag">partly-source-only</span> | Collective effects | P2 | Wrong physics | 2026-11-04 (`7514db3`, master) |  |  |
| [COL-46](COL-46/index.md) | WAKE gives each particle the full W(0) of its own bin, so full-origin wakes such as CORGPIPE's overstate the loss <span class="bb-tag">contract-unsettled</span> <span class="bb-tag">partly-source-only</span> | Collective effects | P2 | Wrong physics | 2026-11-04 (`7514db3`, master) |  |  |
| [COL-47](COL-47/index.md) | LRWAKE keeps more turns of wake history than TURNS_TO_KEEP after an end bunch is lost <span class="bb-tag">partly-source-only</span> | Collective effects | P2 | Wrong physics | 2026-11-05 (`0355dd6`) |  |  |
| [SCT-04](SCT-04/index.md) | MATTER collision slices omit incoming transverse drift | Scattering and pressure | P2 | Wrong physics | 2026-11-04 (`0355dd6`, master) |  |  |
| [SCT-05](SCT-05/index.md) | Standalone electron recoil omits energy loss | Scattering and pressure | P2 | Wrong physics | 2026-11-04 (`0355dd6`, master) |  |  |
| [SCT-06](SCT-06/index.md) | Active recoil transfers use inconsistent energy normalization | Scattering and pressure | P2 | Wrong physics | 2026-11-04 (`0355dd6`, master) |  |  |
| [SCT-07](SCT-07/index.md) | Smooth IBS ignores the plane switches DO_X, DO_Y and DO_Z | Scattering and pressure | P2 | Wrong physics | 2026-11-04 (`0355dd6`, master) |  |  |
| [SCT-08](SCT-08/index.md) | Smooth IBS shifts the beam centroid when there is dispersion and a momentum offset | Scattering and pressure | P2 | Wrong physics | 2026-11-04 (`0355dd6`, master) |  |  |
| [SCT-14](SCT-14/index.md) | FORCE_MCS removes MATTER bremsstrahlung and electron recoil | Scattering and pressure | P2 | Wrong physics | 2026-11-04 (`0355dd6`, master) |  |  |
| [SCT-17](SCT-17/index.md) | ibsEmittance stops the equilibrium emittance at ε₀/(1+κ) when τx > τy and reports convergence | Scattering and pressure | P2 | Wrong physics | 2026-11-04 (`7514db3`, master) |  |  |
| [SCT-24](SCT-24/index.md) | Pelegant's distributed IBSCATTER integration uses the wrong reference momenta in accelerating lines <span class="bb-tag">partly-source-only</span> | Scattering and pressure | P2 | Wrong physics | 2026-11-05 (`0355dd6`) |  |  |
| [OPT-24](OPT-24/index.md) | closed_orbit ignores start_from_centroid and start_from_dp_centroid | Optics | P2 | Silent wrong result | 2026-11-04 (`7514db3`, master) |  |  |
| [OPT-30](OPT-30/index.md) | slice_analysis with final_values_only=1 writes no data and leaves its optimization symbols at zero <span class="bb-tag">partly-source-only</span> | Optics | P2 | Silent wrong result | 2026-11-05 (`0355dd6`) |  |  |
| [COL-29](COL-29/index.md) | insert_sceffects clear leaves a negative specification count, crashing or silently dropping every SCMULT | Collective effects | P2 | Silent wrong result | 2026-11-04 (`0355dd6`, master) |  |  |
| [COL-30](COL-30/index.md) | insert_sceffects with skip = 0 inserts no SCMULT when name or type matches nothing | Collective effects | P2 | Silent wrong result | 2026-11-04 (`0355dd6`, master) |  |  |
| [TRK-01](TRK-01/index.md) | Finite EHKICK bends sloped particles on the wrong radius and leaves yp unchanged <span class="bb-tag">partly-source-only</span> | Tracking | P3 | Wrong physics | 2026-11-04 (`0355dd6`, master) |  |  |
| [TRK-05](TRK-05/index.md) | Zero-voltage MODRF and RAMPRF advance the arrival time by L instead of the slanted path length | Tracking | P3 | Wrong physics | 2026-11-04 (`0355dd6`, master) |  |  |
| [TRK-06](TRK-06/index.md) | RFDF drifts after a kick at the speed the particle had before it <span class="bb-tag">partly-source-only</span> | Tracking | P3 | Wrong physics | 2026-11-04 (`0355dd6`, master) |  |  |
| [TRK-11](TRK-11/index.md) | SOLE given by its field B reverses the spin precession of positively charged particles | Tracking | P3 | Wrong physics | 2026-11-04 (`0355dd6`, master) |  |  |
| [TRK-14](TRK-14/index.md) | POLAR random-direction modes give NaN spins along a longitudinal axis <span class="bb-tag">partly-source-only</span> | Tracking | P3 | Wrong physics | 2026-11-04 (`0355dd6`, master) |  |  |
| [TRK-15](TRK-15/index.md) | GKICKMAP applies classical radiation loss with SYNCH_RAD disabled | Tracking | P3 | Wrong physics | 2026-11-04 (`0355dd6`, master) |  |  |
| [TRK-16](TRK-16/index.md) | UKICKMAP loss compaction gives survivors extra radiation updates | Tracking | P3 | Wrong physics | 2026-11-04 (`0355dd6`, master) |  |  |
| [TRK-18](TRK-18/index.md) | LSRMDLTR radiation loss scales with the cube of the reference momentum | Tracking | P3 | Wrong physics | 2026-11-04 (`0355dd6`, master) |  |  |
| [TRK-28](TRK-28/index.md) | SOLE spin transport misses the precession along off-axis trajectories and the edge-field rotations | Tracking | P3 | Wrong physics | 2026-11-04 (`0355dd6`, master) |  |  |
| [TRK-30](TRK-30/index.md) | Matrix concatenation drops the spin rotation of SOLE and ROTATE | Tracking | P3 | Wrong physics | 2026-11-04 (`0355dd6`, master) |  |  |
| [TRK-33](TRK-33/index.md) | KQUAD and KSEXT steering kicks radiate nothing horizontally and four times too much vertically <span class="bb-tag">partly-source-only</span> | Tracking | P3 | Wrong physics | 2026-11-04 (`7514db3`, master) |  |  |
| [TRK-40](TRK-40/index.md) | KQUAD with RADIAL=1 radiates from the unscaled field shape instead of the applied kick | Tracking | P3 | Wrong physics | 2026-11-04 (`7514db3`, master) |  |  |
| [TRK-43](TRK-43/index.md) | Matrix-method RFCA tracking uses a different R12 from its own transfer matrix | Tracking | P3 | Wrong physics | 2026-11-04 (`7514db3`, master) |  |  |
| [TRK-44](TRK-44/index.md) | RFCA with Q ramps its voltage as the square root of a stored-energy fill, not as a resonant fill <span class="bb-tag">contract-unsettled</span> | Tracking | P3 | Wrong physics | 2026-11-04 (`7514db3`, master) |  |  |
| [TRK-46](TRK-46/index.md) | p_central_mev multiplies a custom species' reference momentum by its charge magnitude | Tracking | P3 | Wrong physics | 2026-11-05 (`0355dd6`) |  |  |
| [TRK-47](TRK-47/index.md) | CSBEND Gaussian quantum diffusion scales as charge to the fourth power <span class="bb-tag">partly-source-only</span> | Tracking | P3 | Wrong physics | 2026-11-05 (`0355dd6`) |  |  |
| [OPT-01](OPT-01/index.md) | Unmatched alpha derivatives report phase derivatives | Optics | P3 | Wrong physics | 2026-11-04 (`0355dd6`, master) |  |  |
| [OPT-05](OPT-05/index.md) | Zero-length MULT quadrupole cubic matrix contains infinities | Optics | P3 | Wrong physics | 2026-11-04 (`0355dd6`, master) |  |  |
| [OPT-11](OPT-11/index.md) | MULT pole-tip optics use electron rigidity for other species <span class="bb-tag">partly-source-only</span> | Optics | P3 | Wrong physics | 2026-11-04 (`0355dd6`, master) |  |  |
| [OPT-20](OPT-20/index.md) | Matrix-method tune shift with amplitude uses the wrong betatron slope phase | Optics | P3 | Wrong physics | 2026-11-04 (`7514db3`, master) |  |  |
| [OPT-26](OPT-26/index.md) | moments_output omits the radiation of KSEXT and SEXT sextupoles | Optics | P3 | Wrong physics | 2026-11-04 (`7514db3`, master) |  |  |
| [OPT-28](OPT-28/index.md) | rf_setup sets the electron's synchronous phase for positrons and protons <span class="bb-tag">partly-source-only</span> | Optics | P3 | Wrong physics | 2026-11-04 (`7514db3`, master) |  |  |
| [OPT-31](OPT-31/index.md) | Resonance driving terms omit fractional strength errors applied by tracking <span class="bb-tag">partly-source-only</span> | Optics | P3 | Wrong physics | 2026-11-05 (`0355dd6`) |  |  |
| [COL-03](COL-03/index.md) | Round nonlinear space charge omits the integrating length and optics weighting <span class="bb-tag">partly-source-only</span> | Collective effects | P3 | Wrong physics | 2026-11-04 (`0355dd6`, master) |  |  |
| [COL-09](COL-09/index.md) | Broadband ZTRANSVERSE uses the wrong Nyquist frequency and sign | Collective effects | P3 | Wrong physics | 2026-11-04 (`0355dd6`, master) |  |  |
| [COL-12](COL-12/index.md) | BEAMBEAM cancels the tracked particle's charge sign, so positrons are kicked like electrons <span class="bb-tag">partly-source-only</span> | Collective effects | P3 | Wrong physics | 2026-11-04 (`0355dd6`, master) |  |  |
| [COL-14](COL-14/index.md) | BEAMBEAM kicks change arrival time because s is not rescaled | Collective effects | P3 | Wrong physics | 2026-11-04 (`0355dd6`, master) |  |  |
| [COL-16](COL-16/index.md) | Dipole resonators create momentum in the disabled plane | Collective effects | P3 | Wrong physics | 2026-11-04 (`0355dd6`, master) |  |  |
| [COL-23](COL-23/index.md) | TRWAKE quadrupole probe coordinates omit element misalignment <span class="bb-tag">partly-source-only</span> | Collective effects | P3 | Wrong physics | 2026-11-04 (`0355dd6`, master) |  |  |
| [COL-28](COL-28/index.md) | SCMULT keeps stale beam sizes across zero-length elements that change them <span class="bb-tag">known-limitation</span> <span class="bb-tag">partly-source-only</span> | Collective effects | P3 | Wrong physics | 2026-11-04 (`0355dd6`, master) |  |  |
| [COL-35](COL-35/index.md) | CSRCSBEND applies quantum excitation to a single particle only when ISR1PART=0 | Collective effects | P3 | Wrong physics | 2026-11-04 (`7514db3`, master) |  |  |
| [COL-36](COL-36/index.md) | CSRCSBEND computes CSR with the nominal bend radius, ignoring the field error FSE <span class="bb-tag">partly-source-only</span> | Collective effects | P3 | Wrong physics | 2026-11-04 (`7514db3`, master) |  |  |
| [COL-42](COL-42/index.md) | CSRCSBEND with zero or negligible ANGLE tracks a finite K1 as a drift | Collective effects | P3 | Wrong physics | 2026-11-04 (`7514db3`, master) |  |  |
| [COL-45](COL-45/index.md) | Longitudinal space charge uses an unpadded FFT, so an isolated bunch gets the field of a train spaced by its own length <span class="bb-tag">partly-source-only</span> | Collective effects | P3 | Wrong physics | 2026-11-04 (`7514db3`, master) |  |  |
| [COL-48](COL-48/index.md) | CSRCSBEND and CSRDRIFT reverse self-CSR kicks for positrons | Collective effects | P3 | Wrong physics | 2026-11-05 (`0355dd6`) |  |  |
| [COL-49](COL-49/index.md) | LSC voltage generators omit the source-charge sign and reverse positron kicks | Collective effects | P3 | Wrong physics | 2026-11-05 (`0355dd6`) |  |  |
| [SCT-10](SCT-10/index.md) | CBSCAT computes a negative Klein–Nishina total cross section just above its Thomson cutoff | Scattering and pressure | P3 | Wrong physics | 2026-11-04 (`0355dd6`, master) |  |  |
| [SCT-11](SCT-11/index.md) | Approximate CBSCAT recoil depends on the reference momentum | Scattering and pressure | P3 | Wrong physics | 2026-11-04 (`0355dd6`, master) |  |  |
| [SCT-13](SCT-13/index.md) | MATTER recoil probability per slice exceeds one and saturates | Scattering and pressure | P3 | Wrong physics | 2026-11-04 (`0355dd6`, master) |  |  |
| [SCT-15](SCT-15/index.md) | MATTER converts between mass and atom density with the proton mass <span class="bb-tag">partly-source-only</span> | Scattering and pressure | P3 | Wrong physics | 2026-11-04 (`0355dd6`, master) |  |  |
| [SCT-18](SCT-18/index.md) | IBSCATTER uses the electron's classical radius and mass for every species <span class="bb-tag">partly-source-only</span> | Scattering and pressure | P3 | Wrong physics | 2026-11-04 (`7514db3`, master) |  |  |
| [SCT-19](SCT-19/index.md) | moments_output IBS Coulomb logarithm mixes electron-unit momenta with the species mass <span class="bb-tag">partly-source-only</span> | Scattering and pressure | P3 | Wrong physics | 2026-11-04 (`7514db3`, master) |  |  |
| [SCT-20](SCT-20/index.md) | Touschek scattering rates use the electron's classical radius for every species | Scattering and pressure | P3 | Wrong physics | 2026-11-04 (`7514db3`, master) |  |  |
| [SCT-21](SCT-21/index.md) | moments_output IBS uses a negative particle count for positrons and protons | Scattering and pressure | P3 | Wrong physics | 2026-11-04 (`7514db3`, master) |  |  |
| [SCT-23](SCT-23/index.md) | moments_output IBS rotates the coupled diffusion tensor with transposed eigenvectors | Scattering and pressure | P3 | Wrong physics | 2026-11-05 (`0355dd6`) |  |  |
| [SCT-25](SCT-25/index.md) | touschek_scatter converts normalized emittance with γ instead of βγ | Scattering and pressure | P3 | Wrong physics | 2026-11-05 (`0355dd6`) |  |  |
| [TRK-29](TRK-29/index.md) | SOLE tracks with B when KS is also set, while its matrix and Twiss output use KS | Tracking | P3 | Silent wrong result | 2026-11-04 (`0355dd6`, master) |  |  |
| [OPT-22](OPT-22/index.md) | Coupling integral reads a KSEXT's YAW as its vertical offset DY | Optics | P3 | Silent wrong result | 2026-11-04 (`7514db3`, master) |  |  |
| [COL-13](COL-13/index.md) | Round Gaussian BEAMBEAM produces NaNs at its center <span class="bb-tag">partly-source-only</span> | Collective effects | P3 | Silent wrong result | 2026-11-04 (`0355dd6`, master) |  |  |
| [COL-38](COL-38/index.md) | CSRDRIFT spread mode with WAVELENGTH_MODE="peak-to-peak" uses an uninitialized wavelength | Collective effects | P3 | Silent wrong result | 2026-11-04 (`7514db3`, master) |  |  |
| [COL-50](COL-50/index.md) | CSRCSBEND BIN_ONCE wake output corrupts retained histograms and accesses past their ends | Collective effects | P3 | Silent wrong result | 2026-11-05 (`0355dd6`) |  |  |
| [COL-40](COL-40/index.md) | CSRCSBEND with IGF=1 and BIN_SIZE writes past its Green-function arrays when the bunch lengthens | Collective effects | P3 | Crash or API hazard | 2026-11-04 (`7514db3`, master) |  |  |
| [COL-41](COL-41/index.md) | CSRCSBEND with IGF=1 leaves dangling Green-function pointers, so a second step, pass or CSRCSBEND aborts | Collective effects | P3 | Crash or API hazard | 2026-11-04 (`7514db3`, master) |  |  |
| [SCT-03](SCT-03/index.md) | MPI Touschek generation crashes when tracking is disabled | Scattering and pressure | P3 | Crash or API hazard | 2026-11-04 (`0355dd6`, master) |  |  |
| [IO-01](IO-01/index.md) | Pelegant fails to load a multi-page beam file when a later page leaves a worker without rows | Lattices, particles and I/O | P3 | Crash or API hazard | 2026-11-04 (`0355dd6`, master) |  |  |
| [TRK-07](TRK-07/index.md) | MODRF damped phase modulation keeps an undamped local frequency <span class="bb-tag">contract-unsettled</span> | Tracking | P4 | Wrong physics | 2026-11-04 (`0355dd6`, master) |  |  |
| [TRK-19](TRK-19/index.md) | TMCF fiducial timing uses transverse positions as momentum components <span class="bb-tag">partly-source-only</span> | Tracking | P4 | Wrong physics | 2026-11-04 (`0355dd6`, master) |  |  |
| [TRK-20](TRK-20/index.md) | MAPSOLENOID combined misalignment rotations are not mutual inverses <span class="bb-tag">partly-source-only</span> | Tracking | P4 | Wrong physics | 2026-11-04 (`0355dd6`, master) |  |  |
| [SCT-12](SCT-12/index.md) | Thin forced-MCS MATTER omits the flight distance from the transverse angles | Scattering and pressure | P4 | Wrong physics | 2026-11-04 (`0355dd6`, master) |  |  |
| [SCT-16](SCT-16/index.md) | ENERGY_STRAGGLE shifts the mean energy loss and narrows its stated spread | Scattering and pressure | P4 | Wrong physics | 2026-11-04 (`0355dd6`, master) |  |  |
| [SCT-22](SCT-22/index.md) | touschek_scatter Monte Carlo rate does not converge when the momentum aperture lies inside the sampled energy spread <span class="bb-tag">extreme-regime</span> <span class="bb-tag">contract-unsettled</span> <span class="bb-tag">partly-source-only</span> | Scattering and pressure | P4 | Wrong physics | 2026-11-04 (`7514db3`, master) |  |  |
| [TRK-31](TRK-31/index.md) | A custom particle keeps the anomalous magnetic moment of the previously selected species | Tracking | P4 | Silent wrong result | 2026-11-04 (`0355dd6`, master) |  |  |
| [OPT-15](OPT-15/index.md) | Individual matrix output keeps nonlinear coefficients of preceding elements | Optics | P4 | Silent wrong result | 2026-11-04 (`0355dd6`, master) |  |  |
| [COL-06](COL-06/index.md) | Single-precision reciprocals in the complex error function reduce the Gaussian force accuracy <span class="bb-tag">partly-source-only</span> | Collective effects | P4 | Silent wrong result | 2026-11-04 (`a3f730a`, master) |  |  |
| [COL-52](COL-52/index.md) | CSRCSBEND BIN_ONCE wake output overwrites retained kick components | Collective effects | P4 | Silent wrong result | 2026-11-05 (`0355dd6`) |  |  |
| [SCT-02](SCT-02/index.md) | Pelegant writes zero IBS rate columns and an unset vertical growth rate | Scattering and pressure | P4 | Silent wrong result | 2026-11-04 (`0355dd6`, master) |  |  |
| [IO-02](IO-02/index.md) | SDDS Gill–Miller integral of a three-sample table is zero and leaves a result unset | Lattices, particles and I/O | P4 | Silent wrong result | 2026-11-04 (`a3f730a`, master) |  |  |
| [IO-03](IO-03/index.md) | SDDS complete elliptic integral E(k) returns π/4 at the endpoints k = ±1 | Lattices, particles and I/O | P4 | Silent wrong result | 2026-11-04 (`a3f730a`, master) |  |  |
| [IO-04](IO-04/index.md) | SDDS moment accumulation discards the earlier batches | Lattices, particles and I/O | P4 | Silent wrong result | 2026-11-04 (`a3f730a`, master) |  |  |
| [OPT-19](OPT-19/index.md) | tracking_matrix_cleanup=1 crashes at second order | Optics | P4 | Crash or API hazard | 2026-11-04 (`7514db3`, master) |  |  |
| [OPT-21](OPT-21/index.md) | Tracking tune shift with amplitude ignores nuy_roi_width and the amplitude scaling factors | Optics | P4 | Crash or API hazard | 2026-11-04 (`7514db3`, master) |  |  |
| [OPT-25](OPT-25/index.md) | moments_output crashes on a one-element beamline | Optics | P4 | Crash or API hazard | 2026-11-04 (`7514db3`, master) |  |  |
| [OPT-27](OPT-27/index.md) | Tracking tune shift with amplitude retries without end when NAFF finds no tune in its window <span class="bb-tag">extreme-regime</span> | Optics | P4 | Crash or API hazard | 2026-11-04 (`7514db3`, master) |  |  |
| [COL-31](COL-31/index.md) | insert_sceffects aborts without a name although the manual's default insertion needs none | Collective effects | P4 | Crash or API hazard | 2026-11-04 (`0355dd6`, master) |  |  |
| [COL-32](COL-32/index.md) | insert_sceffects never stores verbosity, so the insertion notice reads uninitialized memory | Collective effects | P4 | Crash or API hazard | 2026-11-04 (`0355dd6`, master) |  |  |
| [COL-51](COL-51/index.md) | LRWAKE inherits the preceding bunch count across independent new lattices | Collective effects | P4 | Crash or API hazard | 2026-11-05 (`0355dd6`) |  |  |
| [SCT-01](SCT-01/index.md) | Pressure-region diagnostic reads an invalid sample index | Scattering and pressure | P4 | Crash or API hazard | 2026-11-04 (`0355dd6`, master) |  |  |
| [TRK-10](TRK-10/index.md) | CSBEND edge model 3 is documented as symplectic but its finite-gap kick is not six-dimensionally symplectic <span class="bb-tag">contract-unsettled</span> <span class="bb-tag">partly-source-only</span> | Tracking | P4 | Documentation or example mismatch | 2026-11-04 (`0355dd6`, master) |  |  |
| [DOC-05](DOC-05/index.md) | Automatic CSBEND field expansion order is n+2, not the n+3 the manual states <span class="bb-tag">partly-source-only</span> | Tracking | P4 | Documentation or example mismatch | 2026-11-04 (`0355dd6`, master) |  |  |
| [DOC-01](DOC-01/index.md) | Manual gives the third-order momentum compaction term as δ² instead of δ³ | Optics | P4 | Documentation or example mismatch | 2026-11-04 (`0355dd6`, master) |  |  |
| [DOC-10](DOC-10/index.md) | Manual gives the tune-shift amplitude as (1 + α)q²/β and the lines_only default as 0 | Optics | P4 | Documentation or example mismatch | 2026-11-04 (`7514db3`, master) |  |  |
| [DOC-11](DOC-11/index.md) | Manual orders tune_shift_with_amplitude after twiss_output, where Elegant rejects or ignores it | Optics | P4 | Documentation or example mismatch | 2026-11-04 (`7514db3`, master) |  |  |
| [COL-02](COL-02/index.md) | Pelegant falsely warns that fast-orbit-feedback BPM readings are not reduced across ranks | Collective effects | P4 | Documentation or example mismatch | 2026-11-04 (`7514db3`, master) |  |  |
| [DOC-04](DOC-04/index.md) | SCMULT manual prints the averaging filter with the current turn in its history term | Collective effects | P4 | Documentation or example mismatch | 2026-11-04 (`0355dd6`, master) |  |  |
| [DOC-06](DOC-06/index.md) | WAKE and TRWAKE manuals say ACAUSAL_ALLOWED=0 enables acausal wakes; Elegant requires 1 | Collective effects | P4 | Documentation or example mismatch | 2026-11-04 (`7514db3`, master) |  |  |
| [DOC-07](DOC-07/index.md) | WAKE manual says ALLOW_LONG_BEAM ignores the beam's far part, but Elegant keeps the whole beam | Collective effects | P4 | Documentation or example mismatch | 2026-11-04 (`7514db3`, master) |  |  |
| [IO-05](IO-05/index.md) | The manual's bunch-input example assigns particle IDs 2001–3000 to bunch 1 instead of bunch 2 | Lattices, particles and I/O | P4 | Documentation or example mismatch | 2026-11-04 (`0355dd6`, master) |  |  |
