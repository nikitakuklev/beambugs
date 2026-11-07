---
title: Bmad
hide:
  - toc
---

# Bmad

Verified defects in [Bmad](https://www.classe.cornell.edu/bmad/) ([source](https://github.com/bmad-sim/bmad-ecosystem)).
Each entry links to its reproducer, the models that found and confirmed it, and any upstream issue or pull request.
Machine-readable: [defects.json](defects.json).
Tags are explained in [Methodology](../methodology.md#tags).

## Running the reproducers

1. Build Bmad ecosystem release `20261031-0` (`c4c629fc9071`) from source, with GNU Fortran,
   double precision, OpenMP enabled and serial Bmad libraries (the reproducers ran with GNU
   Fortran 15.3.0): configure `util/dist_prefs` for GNU Fortran and the installed dependencies,
   source `util/dist_source_me`, run `util/dist_build_production`, then run `mk` inside
   `regression_tests`. This builds Tao, `bmad_to_mad_sad_elegant` and the regression executables.
2. Keep the complete build tree, including its CMake compile/link recipes, libraries, Fortran
   module files and the compiler that built them; the reproducers compile inline Fortran with
   those recipes.
3. Set `BMAD_DIST` to that tree's absolute path (the directory containing `production` and
   `regression_tests`) and `BMAD_SOURCE` to the matching ecosystem source tree; a distribution
   build containing its archived sources can serve both variables.
4. Install NumPy and the additional Python packages listed below. MAD-X checks use cpymad 1.19.0
   with MAD-X 5.09.03.
5. Download a reproducer and the `_*.py` helpers linked on its entry page from the shared
   [helpers directory](helpers/index.md), and save the helpers beside the script.
6. Run `python <ID>.py` with `OMP_NUM_THREADS=1`, one script per process.

A script that needs more than this states it in a `Setup:` paragraph of its description.
Scripts that run Elegant used serial Elegant from rtsoliday/elegant `0355dd6aca12` and SDDS
from rtsoliday/SDDS `a3f730a14c49`.

Some reproducers also need: cpymad, h5py, mpmath, pysdds, scipy.

## Defects

<div class="bb-filters" data-column="Area"></div>

| ID | Defect | Area | Priority | Severity | Last reproduced | Issue | PR |
|---|---|---|---|---|---|---|---|
| [TRK-14](TRK-14/index.md) | An lcavity with is_on = False still accelerates and focuses the beam | Tracking, RF, fields and spin | P1 | Wrong physics | 2026-10-21 (`19b8afb`; not on head 2026-11-05) | [[gh:https://github.com/bmad-sim/bmad-ecosystem/issues/2222 filed]] | [[gh:https://github.com/bmad-sim/bmad-ecosystem/pull/2223 filed]] |
| [COL-09](COL-09/index.md) | S-based FFT space charge gives full kicks at both element endpoints | Collective effects | P1 | Wrong physics | 2026-11-05 (`c4c629f`, head) |  |  |
| [XLT-05](XLT-05/index.md) | Embedded steering kicks delete the host magnet on export <span class="bb-tag">partly-source-only</span> | Translators | P1 | Wrong physics | 2026-11-05 (`c4c629f`, head) |  |  |
| [XLT-06](XLT-06/index.md) | MAD-X bend export omits the design field and doubles its error <span class="bb-tag">partly-source-only</span> | Translators | P1 | Wrong physics | 2026-11-05 (`c4c629f`, head) |  |  |
| [XLT-07](XLT-07/index.md) | MAD-X bend import treats total k0 as a field error | Translators | P1 | Wrong physics | 2026-11-05 (`c4c629f`, head) |  |  |
| [XLT-13](XLT-13/index.md) | MAD-X skew-multipole export reverses the kick <span class="bb-tag">partly-source-only</span> | Translators | P1 | Wrong physics | 2026-10-20 (`19b8afb`; not on head 2026-11-05) | [[gh:https://github.com/bmad-sim/bmad-ecosystem/issues/2228 filed]] | [[gh:https://github.com/bmad-sim/bmad-ecosystem/pull/2229 filed]] |
| [XLT-16](XLT-16/index.md) | MAD-X import drops pure-skew normal-magnet strengths | Translators | P1 | Wrong physics | 2026-11-05 (`c4c629f`, head) |  |  |
| [XLT-17](XLT-17/index.md) | The manual and MAD-X RF exporter use the wrong phase sign <span class="bb-tag">partly-source-only</span> | Translators | P1 | Wrong physics | 2026-11-05 (`c4c629f`, head) |  |  |
| [XLT-21](XLT-21/index.md) | MAD-X and MAD8 export ignores is_on <span class="bb-tag">partly-source-only</span> | Translators | P1 | Wrong physics | 2026-11-05 (`c4c629f`, head) |  |  |
| [TRK-01](TRK-01/index.md) | Grid-map superposition duplicates transverse fields on axis | Tracking, RF, fields and spin | P2 | Wrong physics | 2026-11-05 (`c4c629f`, head) |  |  |
| [TRK-02](TRK-02/index.md) | Laboratory-frame electric multipole evaluation cancels element tilt | Tracking, RF, fields and spin | P2 | Wrong physics | 2026-11-05 (`c4c629f`, head) |  |  |
| [TRK-03](TRK-03/index.md) | Laboratory-frame field evaluation rotates fixed quadrupole steering <span class="bb-tag">partly-source-only</span> | Tracking, RF, fields and spin | P2 | Wrong physics | 2026-11-05 (`c4c629f`, head) | [[gh:https://github.com/bmad-sim/bmad-ecosystem/issues/2239]] |  |
| [TRK-04](TRK-04/index.md) | Planar-wiggler longitudinal field derivatives disagree with the field | Tracking, RF, fields and spin | P2 | Wrong physics | 2026-11-05 (`c4c629f`, head) |  |  |
| [TRK-13](TRK-13/index.md) | PTC electric multipoles remain active after being set to zero | Tracking, RF, fields and spin | P2 | Wrong physics | 2026-11-05 (`c4c629f`, head) |  |  |
| [TRK-15](TRK-15/index.md) | PTC lcavity updates convert the RF phase to radians twice <span class="bb-tag">partly-source-only</span> | Tracking, RF, fields and spin | P2 | Wrong physics | 2026-11-05 (`c4c629f`, head) |  |  |
| [TRK-16](TRK-16/index.md) | DC gun autoscaling restores the old amplitude | Tracking, RF, fields and spin | P2 | Wrong physics | 2026-11-05 (`c4c629f`, head) |  |  |
| [TRK-22](TRK-22/index.md) | PTC fibre update does not refresh cavity geometry | Tracking, RF, fields and spin | P2 | Wrong physics | 2026-11-05 (`c4c629f`, head) |  |  |
| [TRK-25](TRK-25/index.md) | Spin Taylor tracking omits spin-frame rotations <span class="bb-tag">partly-source-only</span> | Tracking, RF, fields and spin | P2 | Wrong physics | 2026-11-05 (`c4c629f`, head) |  |  |
| [TRK-26](TRK-26/index.md) | Spin Taylor evaluation uses the orbital map’s expansion origin <span class="bb-tag">partly-source-only</span> | Tracking, RF, fields and spin | P2 | Wrong physics | 2026-11-05 (`c4c629f`, head) |  |  |
| [TRK-27](TRK-27/index.md) | Magnus solenoid spin fails backward-time inversion | Tracking, RF, fields and spin | P2 | Wrong physics | 2026-11-05 (`c4c629f`, head) |  |  |
| [TRK-31](TRK-31/index.md) | Negative-time combined-function bends select invalid small-argument series | Tracking, RF, fields and spin | P2 | Wrong physics | 2026-11-05 (`c4c629f`, head) |  |  |
| [TRK-32](TRK-32/index.md) | Reverse quadrupole timing correction repeats the full length at every step | Tracking, RF, fields and spin | P2 | Wrong physics | 2026-11-05 (`c4c629f`, head) |  |  |
| [TRK-33](TRK-33/index.md) | Reverse sol_quad and bend maps omit backward reference-clock motion | Tracking, RF, fields and spin | P2 | Wrong physics | 2026-11-05 (`c4c629f`, head) |  |  |
| [TRK-51](TRK-51/index.md) | A straight_reference multipole gives no on-momentum native dipole kick | Tracking, RF, fields and spin | P2 | Wrong physics | 2026-11-05 (`c4c629f`, head) |  |  |
| [TRK-52](TRK-52/index.md) | A skew multipole loses its whole-element tilt in native consumers <span class="bb-tag">partly-source-only</span> | Tracking, RF, fields and spin | P2 | Wrong physics | 2026-11-05 (`c4c629f`, head) |  |  |
| [TRK-53](TRK-53/index.md) | PTC drops the thin reference-bend terms of a multipole | Tracking, RF, fields and spin | P2 | Wrong physics | 2026-11-05 (`c4c629f`, head) |  |  |
| [TRK-56](TRK-56/index.md) | Native Lie radiation ignores radiation_zero_average <span class="bb-tag">partly-source-only</span> | Tracking, RF, fields and spin | P2 | Wrong physics | 2026-11-05 (`c4c629f`, head) |  |  |
| [TRK-59](TRK-59/index.md) | Slicing suppresses Runge–Kutta hard-edge solenoid spin rotations <span class="bb-tag">partly-source-only</span> | Tracking, RF, fields and spin | P2 | Wrong physics | 2026-11-06 (`c4c629f`) |  |  |
| [TRK-60](TRK-60/index.md) | Native spin tracking rejects interior solenoid boundary conversions <span class="bb-tag">partly-source-only</span> | Tracking, RF, fields and spin | P2 | Wrong physics | 2026-11-06 (`c4c629f`) |  |  |
| [TRK-61](TRK-61/index.md) | Slice boundary distances count the parent displacement twice <span class="bb-tag">partly-source-only</span> | Tracking, RF, fields and spin | P2 | Wrong physics | 2026-11-06 (`c4c629f`) |  |  |
| [TRK-65](TRK-65/index.md) | PTC retains the first initialized species across lattice parses <span class="bb-tag">partly-source-only</span> | Tracking, RF, fields and spin | P2 | Wrong physics | 2026-11-07 (`c4c629f`) |  |  |
| [TRK-68](TRK-68/index.md) | Stepped native lcavities apply each RF energy-kick spin rotation twice <span class="bb-tag">partly-source-only</span> | Tracking, RF, fields and spin | P2 | Wrong physics | 2026-11-06 (`c4c629f`) |  |  |
| [TRK-69](TRK-69/index.md) | Native tracking ignores the selected spin method in stepped lcavities, patches, crab cavities and beambeam elements | Tracking, RF, fields and spin | P2 | Wrong physics | 2026-11-06 (`c4c629f`) |  |  |
| [TRK-71](TRK-71/index.md) | Hard-edge RF fringe spin rotations apply the particle charge twice <span class="bb-tag">partly-source-only</span> | Tracking, RF, fields and spin | P2 | Wrong physics | 2026-11-06 (`c4c629f`) |  |  |
| [TRK-72](TRK-72/index.md) | PTC applies RF cavity fringe kicks without their spin rotation <span class="bb-tag">partly-source-only</span> | Tracking, RF, fields and spin | P2 | Wrong physics | 2026-11-06 (`c4c629f`) |  |  |
| [OPT-01](OPT-01/index.md) | Ohmi damping integrand omits forward transport | Optics | P2 | Wrong physics | 2026-11-05 (`c4c629f`, head) |  |  |
| [OPT-02](OPT-02/index.md) | Radiation-map setup changes the original bend field model | Optics | P2 | Wrong physics | 2026-11-05 (`c4c629f`, head) |  |  |
| [OPT-03](OPT-03/index.md) | Radiation quadrature aliases complete wiggler periods <span class="bb-tag">partly-source-only</span> | Optics | P2 | Wrong physics | 2026-11-05 (`c4c629f`, head) |  |  |
| [OPT-04](OPT-04/index.md) | Chromaticity optional orbit outputs select the wrong branch | Optics | P2 | Wrong physics | 2026-11-05 (`c4c629f`, head) |  |  |
| [OPT-06](OPT-06/index.md) | Radiation convergence tests multiply errors by tolerances <span class="bb-tag">partly-source-only</span> | Optics | P2 | Wrong physics | 2026-11-05 (`c4c629f`, head) |  |  |
| [OPT-07](OPT-07/index.md) | Curvature-vector rotation overwrites its second input | Optics | P2 | Wrong physics | 2026-11-05 (`c4c629f`, head) |  |  |
| [OPT-08](OPT-08/index.md) | Matrix errors leave quantum fluctuations disabled globally <span class="bb-tag">partly-source-only</span> | Optics | P2 | Wrong physics | 2026-11-05 (`c4c629f`, head) |  |  |
| [OPT-09](OPT-09/index.md) | Linear-map construction retains off-momentum beta | Optics | P2 | Wrong physics | 2026-11-05 (`c4c629f`, head) |  |  |
| [OPT-13](OPT-13/index.md) | Omitting the transfer endpoint returns identity instead of one turn | Optics | P2 | Wrong physics | 2026-11-05 (`c4c629f`, head) |  |  |
| [OPT-14](OPT-14/index.md) | Periodic vertical and b-mode dispersion slopes have a denominator typo | Optics | P2 | Wrong physics | 2026-11-05 (`c4c629f`, head) |  |  |
| [OPT-15](OPT-15/index.md) | Periodic dispersion mixes normalized and raw matrix coordinates | Optics | P2 | Wrong physics | 2026-11-05 (`c4c629f`, head) |  |  |
| [OPT-16](OPT-16/index.md) | Normalized Twiss propagation corrupts ordinary dispersion | Optics | P2 | Wrong physics | 2026-11-05 (`c4c629f`, head) |  |  |
| [OPT-17](OPT-17/index.md) | Projected dispersion slopes use a momentum ratio as momentum | Optics | P2 | Wrong physics | 2026-11-05 (`c4c629f`, head) |  |  |
| [OPT-24](OPT-24/index.md) | symp_lie_bmad applies each step's radiation twice <span class="bb-tag">partly-source-only</span> | Optics | P2 | Wrong physics | 2026-11-05 (`c4c629f`, head) |  |  |
| [OPT-25](OPT-25/index.md) | srdt counts skew and rotated multipoles as normal ones <span class="bb-tag">partly-source-only</span> | Optics | P2 | Wrong physics | 2026-11-05 (`c4c629f`, head) |  |  |
| [OPT-34](OPT-34/index.md) | Transverse normalized beam emittance depends on reference momentum | Optics | P2 | Wrong physics | 2026-11-05 (`c4c629f`, head) |  |  |
| [OPT-35](OPT-35/index.md) | Transverse mode dispersion omits the longitudinal-mode divisor <span class="bb-tag">partly-source-only</span> | Optics | P2 | Wrong physics | 2026-11-05 (`c4c629f`, head) |  |  |
| [OPT-36](OPT-36/index.md) | Radiation integrals omit displaced sextupoles and octupoles | Optics | P2 | Wrong physics | 2026-11-05 (`c4c629f`, head) |  |  |
| [OPT-37](OPT-37/index.md) | Averaged-wiggler I0 omits length and misnormalizes the planar average <span class="bb-tag">partly-source-only</span> | Optics | P2 | Wrong physics | 2026-11-05 (`c4c629f`, head) |  |  |
| [OPT-38](OPT-38/index.md) | Inclined bend faces introduce a forbidden vertical radiation-damping column <span class="bb-tag">partly-source-only</span> | Optics | P2 | Wrong physics | 2026-11-07 (`c4c629f`) |  |  |
| [OPT-39](OPT-39/index.md) | Displaced-reference radiation maps omit the constant bend-face wedge loss <span class="bb-tag">partly-source-only</span> | Optics | P2 | Wrong physics | 2026-11-07 (`c4c629f`) |  |  |
| [OPT-40](OPT-40/index.md) | Higher-order PTC compaction and slip series omit particle-speed conversion <span class="bb-tag">partly-source-only</span> | Optics | P2 | Wrong physics | 2026-11-07 (`c4c629f`) |  |  |
| [OPT-42](OPT-42/index.md) | Bend exit-face radiation damping uses the entrance orbit and transport <span class="bb-tag">partly-source-only</span> | Optics | P2 | Wrong physics | 2026-11-07 (`c4c629f`) |  |  |
| [OPT-43](OPT-43/index.md) | Downstream tracking radiation kick omits the inverse half-element transport <span class="bb-tag">partly-source-only</span> | Optics | P2 | Wrong physics | 2026-11-06 (`c4c629f`) |  |  |
| [COL-03](COL-03/index.md) | Off-axis slice LSC coefficients omit the kick normalization | Collective effects | P2 | Wrong physics | 2026-11-05 (`c4c629f`, head) |  |  |
| [COL-06](COL-06/index.md) | The 1D CSR fast kick leaves beta and z inconsistent <span class="bb-tag">partly-source-only</span> | Collective effects | P2 | Wrong physics | 2026-11-05 (`c4c629f`, head) |  |  |
| [COL-08](COL-08/index.md) | CSR and both slice-SC coefficients divide by mass twice | Collective effects | P2 | Wrong physics | 2026-11-05 (`c4c629f`, head) | [[gh:https://github.com/bmad-sim/bmad-ecosystem/issues/2214]] | [[gh:https://github.com/bmad-sim/bmad-ecosystem/pull/2215]] |
| [COL-11](COL-11/index.md) | Slice-width flooring discards all but the last numerator | Collective effects | P2 | Wrong physics | 2026-11-05 (`c4c629f`, head) |  |  |
| [COL-13](COL-13/index.md) | The high-energy SC matrix rotates the kick with the wrong sign | Collective effects | P2 | Wrong physics | 2026-11-05 (`c4c629f`, head) |  |  |
| [COL-15](COL-15/index.md) | Cathode image cutoff compares against the total field instead of the bunch field | Collective effects | P2 | Wrong physics | 2026-11-05 (`c4c629f`, head) |  |  |
| [COL-21](COL-21/index.md) | Transverse slice space charge inherits the last shielding-image factor <span class="bb-tag">partly-source-only</span> | Collective effects | P2 | Wrong physics | 2026-11-05 (`c4c629f`, head) |  |  |
| [COL-23](COL-23/index.md) | Coupled IBS diffusion is rotated with transposed eigenvectors | Collective effects | P2 | Wrong physics | 2026-11-05 (`c4c629f`, head) |  |  |
| [COL-24](COL-24/index.md) | IBS formulas use charge magnitude instead of charge squared in the classical radius <span class="bb-tag">partly-source-only</span> | Collective effects | P2 | Wrong physics | 2026-10-21 (`19b8afb`; not on head 2026-11-05) | [[gh:https://github.com/bmad-sim/bmad-ecosystem/issues/2230 filed]] | [[gh:https://github.com/bmad-sim/bmad-ecosystem/pull/2231 filed]] |
| [COL-26](COL-26/index.md) | The default disabled envelope excitation injects negative diffusion | Collective effects | P2 | Wrong physics | 2026-11-05 (`c4c629f`, head) |  |  |
| [COL-27](COL-27/index.md) | Longitudinal IBS equilibrium balances RMS spread instead of variance | Collective effects | P2 | Wrong physics | 2026-11-05 (`c4c629f`, head) |  |  |
| [COL-28](COL-28/index.md) | Normal quadrupole gradient creates a fictitious vertical-energy damping term | Collective effects | P2 | Wrong physics | 2026-11-05 (`c4c629f`, head) |  |  |
| [COL-29](COL-29/index.md) | CIMP interpolation violates its stated one-percent accuracy | Collective effects | P2 | Wrong physics | 2026-11-05 (`c4c629f`, head) |  |  |
| [COL-37](COL-37/index.md) | Basic Touschek lifetime averaging reads beyond the tracking elements <span class="bb-tag">partly-source-only</span> | Collective effects | P2 | Wrong physics | 2026-11-05 (`c4c629f`, head) |  |  |
| [COL-48](COL-48/index.md) | Classic and Raubenheimer Coulomb logs omit the documented upper impact-parameter cap <span class="bb-tag">partly-source-only</span> | Collective effects | P2 | Wrong physics | 2026-11-05 (`c4c629f`, head) |  |  |
| [COL-49](COL-49/index.md) | Polarized long-range wake modes project onto the wrong geometric axis | Collective effects | P2 | Wrong physics | 2026-11-05 (`c4c629f`, head) |  |  |
| [COL-50](COL-50/index.md) | Fundamental wakes lose elapsed phase during absolute-time tracking | Collective effects | P2 | Wrong physics | 2026-11-05 (`c4c629f`, head) |  |  |
| [COL-58](COL-58/index.md) | Gaussian table smoothing omits the squared index | Collective effects | P2 | Wrong physics | 2026-10-20 (`19b8afb`; not on head 2026-11-05) |  | [[gh:https://github.com/bmad-sim/bmad-ecosystem/pull/2225]] |
| [COL-59](COL-59/index.md) | Tabulated-wake origin depends on particle storage order <span class="bb-tag">partly-source-only</span> | Collective effects | P2 | Wrong physics | 2026-11-05 (`c4c629f`, head) |  |  |
| [COL-60](COL-60/index.md) | A table-support error marks only the final array particle lost <span class="bb-tag">partly-source-only</span> | Collective effects | P2 | Wrong physics | 2026-11-05 (`c4c629f`, head) |  |  |
| [COL-66](COL-66/index.md) | Wake kicks omit the witness charge state and source species sign <span class="bb-tag">partly-source-only</span> | Collective effects | P2 | Wrong physics | 2026-11-05 (`c4c629f`, head) |  |  |
| [COL-70](COL-70/index.md) | Entrance-aperture rejects contribute to the first slice space-charge kick <span class="bb-tag">partly-source-only</span> | Collective effects | P2 | Wrong physics | 2026-11-06 (`c4c629f`) |  |  |
| [COL-71](COL-71/index.md) | S-based FFT space charge uses the exit energy throughout energy-changing elements <span class="bb-tag">partly-source-only</span> | Collective effects | P2 | Wrong physics | 2026-11-06 (`c4c629f`) |  |  |
| [COL-72](COL-72/index.md) | Wake evaluation uses the reference frame instead of the placed structure frame <span class="bb-tag">partly-source-only</span> | Collective effects | P2 | Wrong physics | 2026-11-06 (`c4c629f`) |  |  |
| [LAT-04](LAT-04/index.md) | HDF5 momentum-centroid shifts leave beta inconsistent <span class="bb-tag">partly-source-only</span> | Lattices, particles and I/O | P2 | Wrong physics | 2026-11-05 (`c4c629f`, head) |  |  |
| [LAT-11](LAT-11/index.md) | An omitted HDF5 momentum policy selects inconsistent defaults <span class="bb-tag">partly-source-only</span> | Lattices, particles and I/O | P2 | Wrong physics | 2026-11-05 (`c4c629f`, head) |  |  |
| [LAT-12](LAT-12/index.md) | Explicit false HDF5 reference changes retain the old beta | Lattices, particles and I/O | P2 | Wrong physics | 2026-11-05 (`c4c629f`, head) |  |  |
| [LAT-13](LAT-13/index.md) | HDF5 reading discards the axial-momentum direction sign <span class="bb-tag">partly-source-only</span> | Lattices, particles and I/O | P2 | Wrong physics | 2026-11-05 (`c4c629f`, head) |  |  |
| [LAT-23](LAT-23/index.md) | Mixed Gaussian/ellipse renormalization ignores charge weights | Lattices, particles and I/O | P2 | Wrong physics | 2026-11-05 (`c4c629f`, head) |  |  |
| [LAT-24](LAT-24/index.md) | Multipass bookkeeping deletes multipole arrays, losing kicks or crashing <span class="bb-tag">partly-source-only</span> | Lattices, particles and I/O | P2 | Wrong physics | 2026-11-05 (`c4c629f`, head) |  |  |
| [LAT-26](LAT-26/index.md) | A full-length one-lord slice loses its entrance aperture | Lattices, particles and I/O | P2 | Wrong physics | 2026-11-05 (`c4c629f`, head) |  |  |
| [LAT-27](LAT-27/index.md) | Taylor slices each apply the complete map | Lattices, particles and I/O | P2 | Wrong physics | 2026-11-05 (`c4c629f`, head) |  |  |
| [LAT-28](LAT-28/index.md) | RF slicing invents couplers and doubles both-end kicks | Lattices, particles and I/O | P2 | Wrong physics | 2026-11-05 (`c4c629f`, head) |  |  |
| [LAT-29](LAT-29/index.md) | A girder's off state is lost at creation and ignored during alignment | Lattices, particles and I/O | P2 | Wrong physics | 2026-11-05 (`c4c629f`, head) |  |  |
| [LAT-30](LAT-30/index.md) | Center-fiducial bend angle changes corrupt symmetric geometry | Lattices, particles and I/O | P2 | Wrong physics | 2026-11-05 (`c4c629f`, head) |  |  |
| [LAT-31](LAT-31/index.md) | Super-lord spin-method edits do not reach single-lord slaves | Lattices, particles and I/O | P2 | Wrong physics | 2026-11-05 (`c4c629f`, head) |  |  |
| [XLT-08](XLT-08/index.md) | SAD MULT fallback invents multipoles through order 21 <span class="bb-tag">partly-source-only</span> | Translators | P2 | Wrong physics | 2026-11-05 (`c4c629f`, head) |  |  |
| [XLT-14](XLT-14/index.md) | MAD export uses incorrect finite-beta starting T and PT <span class="bb-tag">partly-source-only</span> | Translators | P2 | Wrong physics | 2026-11-05 (`c4c629f`, head) |  |  |
| [XLT-15](XLT-15/index.md) | MAD-X initial dispersion omits the PT derivative conversion | Translators | P2 | Wrong physics | 2026-11-05 (`c4c629f`, head) |  |  |
| [XLT-20](XLT-20/index.md) | MAD-X bend k0s import applies the wrong strength normalization | Translators | P2 | Wrong physics | 2026-11-05 (`c4c629f`, head) |  |  |
| [XLT-22](XLT-22/index.md) | MAD-X export of a bend vkick is lost outside PTC <span class="bb-tag">partly-source-only</span> | Translators | P2 | Wrong physics | 2026-11-05 (`c4c629f`, head) |  |  |
| [XLT-23](XLT-23/index.md) | MAD-X sbend k1 export ignores the changed reference momentum | Translators | P2 | Wrong physics | 2026-11-05 (`c4c629f`, head) |  |  |
| [XLT-27](XLT-27/index.md) | Generated element names can replace lattice elements of the same name <span class="bb-tag">partly-source-only</span> | Translators | P2 | Wrong physics | 2026-11-05 (`c4c629f`, head) |  |  |
| [XLT-29](XLT-29/index.md) | MAD-X export loses order-zero multipole kicks at the reference momentum <span class="bb-tag">partly-source-only</span> | Translators | P2 | Wrong physics | 2026-11-05 (`c4c629f`, head) |  |  |
| [XLT-30](XLT-30/index.md) | MAD-X export drops the whole-element tilt of a multipole <span class="bb-tag">partly-source-only</span> | Translators | P2 | Wrong physics | 2026-11-05 (`c4c629f`, head) |  |  |
| [XLT-38](XLT-38/index.md) | SciBmad export reuses one definition for distinct elements whose generated names collide <span class="bb-tag">partly-source-only</span> | Translators | P2 | Wrong physics | 2026-11-06 (`c4c629f`) |  |  |
| [TRK-18](TRK-18/index.md) | Reverse-time rfcavity matrices use the forward energy change | Tracking, RF, fields and spin | P2 | Silent wrong result | 2026-11-05 (`c4c629f`, head) |  |  |
| [TRK-35](TRK-35/index.md) | Exact curved-multipole bends overwrite the kick-factor derivative | Tracking, RF, fields and spin | P2 | Silent wrong result | 2026-11-05 (`c4c629f`, head) |  |  |
| [TRK-36](TRK-36/index.md) | Energy-changing patch Jacobians mix momentum normalizations | Tracking, RF, fields and spin | P2 | Silent wrong result | 2026-11-05 (`c4c629f`, head) |  |  |
| [TRK-37](TRK-37/index.md) | Thin-multipole dispatch ignores and fails to compose the supplied matrix <span class="bb-tag">partly-source-only</span> | Tracking, RF, fields and spin | P2 | Silent wrong result | 2026-11-05 (`c4c629f`, head) |  |  |
| [OPT-11](OPT-11/index.md) | Tracking-derived quaternion derivatives have reversed signs | Optics | P2 | Silent wrong result | 2026-11-05 (`c4c629f`, head) |  |  |
| [OPT-30](OPT-30/index.md) | Tilted Taylor-map Jacobians lose the coordinate-frame transformations | Optics | P2 | Silent wrong result | 2026-11-05 (`c4c629f`, head) |  |  |
| [OPT-31](OPT-31/index.md) | AC-kicker matrices omit waveform derivatives <span class="bb-tag">partly-source-only</span> | Optics | P2 | Silent wrong result | 2026-11-05 (`c4c629f`, head) |  | [[gh:https://github.com/bmad-sim/bmad-ecosystem/pull/2108]] |
| [OPT-32](OPT-32/index.md) | Helical-wiggler horizontal octupole derivatives use vertical coordinates <span class="bb-tag">partly-source-only</span> | Optics | P2 | Silent wrong result | 2026-11-05 (`c4c629f`, head) |  |  |
| [OPT-33](OPT-33/index.md) | Periodic-wiggler matrices omit the leading momentum response of focusing <span class="bb-tag">partly-source-only</span> | Optics | P2 | Silent wrong result | 2026-11-05 (`c4c629f`, head) |  | [[gh:https://github.com/bmad-sim/bmad-ecosystem/pull/2108]] |
| [OPT-41](OPT-41/index.md) | Runge–Kutta integrated matrices reverse physical-time perturbations <span class="bb-tag">partly-source-only</span> | Optics | P2 | Silent wrong result | 2026-11-07 (`c4c629f`) |  |  |
| [COL-51](COL-51/index.md) | Whole-lattice wake reset leaves secondary-branch history intact | Collective effects | P2 | Silent wrong result | 2026-11-05 (`c4c629f`, head) |  |  |
| [LAT-01](LAT-01/index.md) | Live-only ASCII output corrupts retained particles | Lattices, particles and I/O | P2 | Silent wrong result | 2026-11-05 (`c4c629f`, head) |  |  |
| [LAT-02](LAT-02/index.md) | ASCII reading consumes the next bunch’s first header line <span class="bb-tag">partly-source-only</span> | Lattices, particles and I/O | P2 | Silent wrong result | 2026-11-05 (`c4c629f`, head) |  |  |
| [LAT-03](LAT-03/index.md) | Final initialization counts lost particles and leaves live charge stale | Lattices, particles and I/O | P2 | Silent wrong result | 2026-11-05 (`c4c629f`, head) |  |  |
| [LAT-07](LAT-07/index.md) | Modern particle-file readers ignore the requested bunch count <span class="bb-tag">partly-source-only</span> | Lattices, particles and I/O | P2 | Silent wrong result | 2026-11-05 (`c4c629f`, head) |  |  |
| [LAT-08](LAT-08/index.md) | One short HDF5 bunch reduces later bunches’ particle limits | Lattices, particles and I/O | P2 | Silent wrong result | 2026-11-05 (`c4c629f`, head) |  |  |
| [LAT-10](LAT-10/index.md) | openPMD beam reading converts units with the inverse unitSI ratio <span class="bb-tag">partly-source-only</span> | Lattices, particles and I/O | P2 | Silent wrong result | 2026-11-05 (`c4c629f`, head) |  |  |
| [LAT-14](LAT-14/index.md) | Ordinary HDF5 reads leave the live-particle count unset | Lattices, particles and I/O | P2 | Silent wrong result | 2026-11-05 (`c4c629f`, head) |  |  |
| [TAO-01](TAO-01/index.md) | Filtered variable penalties differentiate the wrong variable <span class="bb-tag">partly-source-only</span> | Tao | P2 | Silent wrong result | 2026-11-05 (`c4c629f`, head) |  |  |
| [TAO-02](TAO-02/index.md) | Geodesic LM omits variable-penalty weights <span class="bb-tag">partly-source-only</span> | Tao | P2 | Silent wrong result | 2026-11-05 (`c4c629f`, head) |  |  |
| [TAO-03](TAO-03/index.md) | Data namelist export omits every datum in its row-format path <span class="bb-tag">partly-source-only</span> | Tao | P2 | Silent wrong result | 2026-11-05 (`c4c629f`, head) |  |  |
| [TAO-05](TAO-05/index.md) | Single-element absolute constraints discard the reference value | Tao | P2 | Silent wrong result | 2026-11-05 (`c4c629f`, head) |  |  |
| [TAO-06](TAO-06/index.md) | RMS reduction integrates squared endpoints instead of the linear interpolant | Tao | P2 | Silent wrong result | 2026-11-05 (`c4c629f`, head) |  |  |
| [TAO-07](TAO-07/index.md) | Four a-mode chromatic pipe selectors read b-mode fields <span class="bb-tag">partly-source-only</span> | Tao | P2 | Silent wrong result | 2026-11-05 (`c4c629f`, head) |  |  |
| [TAO-09](TAO-09/index.md) | Track-only lattice listing uses branch zero’s tracking count | Tao | P2 | Silent wrong result | 2026-11-05 (`c4c629f`, head) |  |  |
| [TAO-10](TAO-10/index.md) | Wrapped average, integral, and RMS pass the unreordered value buffer <span class="bb-tag">partly-source-only</span> | Tao | P2 | Silent wrong result | 2026-11-05 (`c4c629f`, head) |  |  |
| [TAO-11](TAO-11/index.md) | Wrapped extrema compare the second candidate with itself <span class="bb-tag">partly-source-only</span> | Tao | P2 | Silent wrong result | 2026-11-05 (`c4c629f`, head) |  |  |
| [TAO-17](TAO-17/index.md) | Phase residual recentering depends on array names rather than data types <span class="bb-tag">contract-unsettled</span> | Tao | P2 | Silent wrong result | 2026-11-05 (`c4c629f`, head) |  |  |
| [TAO-24](TAO-24/index.md) | normal.h driving terms are the complex conjugates of Bengtsson’s | Tao | P2 | Silent wrong result | 2026-11-05 (`c4c629f`, head) |  |  |
| [TAO-25](TAO-25/index.md) | New normal.h monomials return zero after the first cached calculation | Tao | P2 | Silent wrong result | 2026-11-05 (`c4c629f`, head) |  |  |
| [TAO-26](TAO-26/index.md) | Ad hoc pipe srdt expressions return uncomputed values as valid | Tao | P2 | Silent wrong result | 2026-11-05 (`c4c629f`, head) |  |  |
| [TRK-05](TRK-05/index.md) | AC cylindrical magnetic modes contain an extra radial azimuthal factor <span class="bb-tag">partly-source-only</span> | Tracking, RF, fields and spin | P3 | Wrong physics | 2026-11-05 (`c4c629f`, head) |  |  |
| [TRK-06](TRK-06/index.md) | Multilayer mirrors use the wrong π vacuum-interface coefficient | Tracking, RF, fields and spin | P3 | Wrong physics | 2026-11-05 (`c4c629f`, head) |  |  |
| [TRK-07](TRK-07/index.md) | Recognized crystal materials falsely lose photons in transmitting masks | Tracking, RF, fields and spin | P3 | Wrong physics | 2026-11-05 (`c4c629f`, head) |  |  |
| [TRK-08](TRK-08/index.md) | Segmented surfaces reverse their height slopes | Tracking, RF, fields and spin | P3 | Wrong physics | 2026-11-05 (`c4c629f`, head) |  |  |
| [TRK-09](TRK-09/index.md) | Polynomial surface derivatives corrupt curved-detector coordinates | Tracking, RF, fields and spin | P3 | Wrong physics | 2026-11-05 (`c4c629f`, head) |  |  |
| [TRK-10](TRK-10/index.md) | Coherent forward-Laue transport gives the β branch an extra π phase | Tracking, RF, fields and spin | P3 | Wrong physics | 2026-11-05 (`c4c629f`, head) |  |  |
| [TRK-11](TRK-11/index.md) | Oblique mask transmission omits the longer material path | Tracking, RF, fields and spin | P3 | Wrong physics | 2026-11-05 (`c4c629f`, head) |  |  |
| [TRK-12](TRK-12/index.md) | Laue displacement includes an absent polarization component | Tracking, RF, fields and spin | P3 | Wrong physics | 2026-11-05 (`c4c629f`, head) |  |  |
| [TRK-17](TRK-17/index.md) | RF and crab kicks use charge-to-mass instead of charge | Tracking, RF, fields and spin | P3 | Wrong physics | 2026-11-05 (`c4c629f`, head) |  |  |
| [TRK-19](TRK-19/index.md) | Native lcavity body kicks omit the tracked/reference charge ratio | Tracking, RF, fields and spin | P3 | Wrong physics | 2026-11-05 (`c4c629f`, head) |  |  |
| [TRK-23](TRK-23/index.md) | Legacy lcavity tracking skips reference-momentum changes at zero length and in backward time <span class="bb-tag">partly-source-only</span> | Tracking, RF, fields and spin | P3 | Wrong physics | 2026-11-05 (`c4c629f`, head) |  |  |
| [TRK-28](TRK-28/index.md) | Magnus solenoid spin omits the tracked/reference charge ratio | Tracking, RF, fields and spin | P3 | Wrong physics | 2026-11-05 (`c4c629f`, head) |  |  |
| [TRK-29](TRK-29/index.md) | Near-axis bends suppress opposite-charge deflection | Tracking, RF, fields and spin | P3 | Wrong physics | 2026-11-05 (`c4c629f`, head) |  |  |
| [TRK-30](TRK-30/index.md) | Solenoid strength counts tracking charge twice | Tracking, RF, fields and spin | P3 | Wrong physics | 2026-11-05 (`c4c629f`, head) |  |  |
| [TRK-34](TRK-34/index.md) | Moved wall-transition apertures compare different coordinate frames <span class="bb-tag">partly-source-only</span> | Tracking, RF, fields and spin | P3 | Wrong physics | 2026-11-05 (`c4c629f`, head) |  |  |
| [TRK-38](TRK-38/index.md) | Radiation kicks leave the stored particle beta stale | Tracking, RF, fields and spin | P3 | Wrong physics | 2026-11-05 (`c4c629f`, head) |  |  |
| [TRK-43](TRK-43/index.md) | FOREST and Bmad rest masses give different drift timing <span class="bb-tag">partly-source-only</span> | Tracking, RF, fields and spin | P3 | Wrong physics | 2026-11-05 (`c4c629f`, head) |  |  |
| [TRK-54](TRK-54/index.md) | Foils apply full-thickness scattering at every integration step <span class="bb-tag">partly-source-only</span> | Tracking, RF, fields and spin | P3 | Wrong physics | 2026-11-05 (`c4c629f`, head) |  |  |
| [TRK-55](TRK-55/index.md) | Foil energy loss leaves longitudinal position inconsistent with arrival time | Tracking, RF, fields and spin | P3 | Wrong physics | 2026-11-05 (`c4c629f`, head) |  |  |
| [TRK-62](TRK-62/index.md) | Placed wall3d apertures ignore element placement at laboratory-coordinate checks | Tracking, RF, fields and spin | P3 | Wrong physics | 2026-11-06 (`c4c629f`) |  |  |
| [TRK-63](TRK-63/index.md) | Combined-function bends lose the zero-focusing and small-phase limits | Tracking, RF, fields and spin | P3 | Wrong physics | 2026-11-06 (`c4c629f`) |  |  |
| [TRK-64](TRK-64/index.md) | Legacy standing-wave lcavities lose finite transverse focusing near RF zero crossing <span class="bb-tag">partly-source-only</span> | Tracking, RF, fields and spin | P3 | Wrong physics | 2026-11-06 (`c4c629f`) |  |  |
| [TRK-66](TRK-66/index.md) | Full bend fringes omit the tracked particle charge factor <span class="bb-tag">partly-source-only</span> | Tracking, RF, fields and spin | P3 | Wrong physics | 2026-11-06 (`c4c629f`) |  |  |
| [OPT-23](OPT-23/index.md) | Radiation-integral and symp_lie_bmad radiation omit the charge squared <span class="bb-tag">partly-source-only</span> | Optics | P3 | Wrong physics | 2026-11-05 (`c4c629f`, head) |  |  |
| [OPT-29](OPT-29/index.md) | Radiation uses the reference species for a tracked particle of another charge state <span class="bb-tag">partly-source-only</span> | Optics | P3 | Wrong physics | 2026-11-05 (`c4c629f`, head) |  |  |
| [COL-04](COL-04/index.md) | S-based FFT kicks omit the test particle's charge state | Collective effects | P3 | Wrong physics | 2026-11-05 (`c4c629f`, head) |  |  |
| [COL-05](COL-05/index.md) | A low live-particle count leaves stale space-charge fields <span class="bb-tag">partly-source-only</span> | Collective effects | P3 | Wrong physics | 2026-11-05 (`c4c629f`, head) |  |  |
| [COL-10](COL-10/index.md) | The FFT Coulomb prefactor loses default-real precision <span class="bb-tag">partly-source-only</span> | Collective effects | P3 | Wrong physics | 2026-11-05 (`c4c629f`, head) |  |  |
| [COL-14](COL-14/index.md) | A one-element high-energy SC branch receives only half its length | Collective effects | P3 | Wrong physics | 2026-11-05 (`c4c629f`, head) |  |  |
| [COL-25](COL-25/index.md) | Touschek integration discards the tail above τ = 1 <span class="bb-tag">extreme-regime</span> | Collective effects | P3 | Wrong physics | 2026-11-05 (`c4c629f`, head) |  |  |
| [COL-30](COL-30/index.md) | Touschek-background spread propagation discards its compression correction <span class="bb-tag">partly-source-only</span> | Collective effects | P3 | Wrong physics | 2026-11-05 (`c4c629f`, head) |  |  |
| [COL-35](COL-35/index.md) | Envelope radiation diffusion retains electron recoil for every species <span class="bb-tag">partly-source-only</span> | Collective effects | P3 | Wrong physics | 2026-11-05 (`c4c629f`, head) |  |  |
| [COL-42](COL-42/index.md) | Beam–beam longitudinal kick hardcodes equal beam speeds | Collective effects | P3 | Wrong physics | 2026-11-05 (`c4c629f`, head) |  |  |
| [COL-43](COL-43/index.md) | ibs_linac converts normalized emittance with gamma instead of beta gamma | Collective effects | P3 | Wrong physics | 2026-11-05 (`c4c629f`, head) |  |  |
| [COL-44](COL-44/index.md) | ibs_linac omits charge-squared scaling from ISR energy-spread growth | Collective effects | P3 | Wrong physics | 2026-11-05 (`c4c629f`, head) |  |  |
| [COL-47](COL-47/index.md) | Tail-cut Coulomb logarithms exceed the classical scattering bound <span class="bb-tag">partly-source-only</span> | Collective effects | P3 | Wrong physics | 2026-11-05 (`c4c629f`, head) |  |  |
| [COL-52](COL-52/index.md) | Replacing a table wake with modes leaves an active empty table | Collective effects | P3 | Wrong physics | 2026-11-05 (`c4c629f`, head) |  |  |
| [COL-55](COL-55/index.md) | BBU charge ramp patterns skip their final entry | Collective effects | P3 | Wrong physics | 2026-11-05 (`c4c629f`, head) |  |  |
| [COL-56](COL-56/index.md) | Sparse legacy wake rows take polarization from the wrong row | Collective effects | P3 | Wrong physics | 2026-11-05 (`c4c629f`, head) |  |  |
| [COL-61](COL-61/index.md) | Tolerated table outliers receive wrapped or extrapolated kicks <span class="bb-tag">partly-source-only</span> | Collective effects | P3 | Wrong physics | 2026-11-05 (`c4c629f`, head) |  |  |
| [COL-73](COL-73/index.md) | Distant cathode images lose integrated-cell field accuracy and produce NaNs <span class="bb-tag">partly-source-only</span> | Collective effects | P3 | Wrong physics | 2026-11-06 (`c4c629f`) |  |  |
| [COL-74](COL-74/index.md) | Time-based FFT space charge reconstructs the boost from accumulated rounded speeds <span class="bb-tag">partly-source-only</span> | Collective effects | P3 | Wrong physics | 2026-11-06 (`c4c629f`) |  |  |
| [LAT-05](LAT-05/index.md) | Photon initialization leaves even particles dark | Lattices, particles and I/O | P3 | Wrong physics | 2026-11-05 (`c4c629f`, head) |  |  |
| [XLT-01](XLT-01/index.md) | Converter angular cuts increase yield through an inverted acceptance factor | Translators | P3 | Wrong physics | 2026-11-05 (`c4c629f`, head) |  |  |
| [XLT-02](XLT-02/index.md) | Converter momentum cuts fail to restrict generated momenta | Translators | P3 | Wrong physics | 2026-11-05 (`c4c629f`, head) |  |  |
| [XLT-03](XLT-03/index.md) | Converter beta changes rescale longitudinal position backwards | Translators | P3 | Wrong physics | 2026-11-05 (`c4c629f`, head) |  |  |
| [XLT-04](XLT-04/index.md) | Converter polarization interpolation leaves its output uninitialized | Translators | P3 | Wrong physics | 2026-11-05 (`c4c629f`, head) |  |  |
| [XLT-28](XLT-28/index.md) | MAD8 export of a zero-length lcavity drops its energy gain | Translators | P3 | Wrong physics | 2026-11-05 (`c4c629f`, head) |  |  |
| [XLT-31](XLT-31/index.md) | SAD export turns a K0L reference bend into a dipole kick <span class="bb-tag">partly-source-only</span> | Translators | P3 | Wrong physics | 2026-11-05 (`c4c629f`, head) |  |  |
| [XLT-34](XLT-34/index.md) | SAD import reverses ROTATE on zero-length QUAD, SEXT and OCT | Translators | P3 | Wrong physics | 2026-11-05 (`c4c629f`, head) |  |  |
| [XLT-35](XLT-35/index.md) | SAD import of a zero-length BEND drops its kick or writes invalid attributes <span class="bb-tag">partly-source-only</span> | Translators | P3 | Wrong physics | 2026-11-05 (`c4c629f`, head) |  |  |
| [OPT-22](OPT-22/index.md) | RF-off equilibrium leaves the no-opening-angle longitudinal emittance positive <span class="bb-tag">known-limitation</span> | Optics | P3 | Silent wrong result | 2026-11-05 (`c4c629f`, head) |  |  |
| [LAT-37](LAT-37/index.md) | After expand_lattice, element references longer than 40 characters are rejected or silently truncated <span class="bb-tag">partly-source-only</span> | Lattices, particles and I/O | P3 | Silent wrong result | 2026-11-07 (`c4c629f`) |  |  |
| [TRK-24](TRK-24/index.md) | A zero-frequency lcavity gives NaN with absolute-time tracking | Tracking, RF, fields and spin | P3 | Crash or API hazard | 2026-11-05 (`c4c629f`, head) |  |  |
| [TRK-57](TRK-57/index.md) | PTC branch tracker reverses its optional error flag | Tracking, RF, fields and spin | P3 | Crash or API hazard | 2026-11-05 (`c4c629f`, head) |  |  |
| [OPT-05](OPT-05/index.md) | Backward transfer matrix dereferences an absent affine vector | Optics | P3 | Crash or API hazard | 2026-11-05 (`c4c629f`, head) |  |  |
| [OPT-26](OPT-26/index.md) | srdt returns NaN for zero-length non-multipole elements with multipoles <span class="bb-tag">partly-source-only</span> | Optics | P3 | Crash or API hazard | 2026-11-05 (`c4c629f`, head) |  |  |
| [COL-07](COL-07/index.md) | Valid Green-function offsets expose removable singularities | Collective effects | P3 | Crash or API hazard | 2026-11-05 (`c4c629f`, head) |  |  |
| [COL-17](COL-17/index.md) | High-energy space-charge setup reads beyond a tightly allocated branch | Collective effects | P3 | Crash or API hazard | 2026-11-05 (`c4c629f`, head) |  |  |
| [COL-53](COL-53/index.md) | BBU ramping overwrites the circular bunch-queue index | Collective effects | P3 | Crash or API hazard | 2026-11-05 (`c4c629f`, head) |  |  |
| [COL-75](COL-75/index.md) | Automatic FFT mesh bounds permit deposition beyond the density array <span class="bb-tag">partly-source-only</span> | Collective effects | P3 | Crash or API hazard | 2026-11-06 (`c4c629f`) |  |  |
| [LAT-25](LAT-25/index.md) | Overlay bookkeeping overruns its 20-attribute buffers | Lattices, particles and I/O | P3 | Crash or API hazard | 2026-11-05 (`c4c629f`, head) |  |  |
| [TAO-20](TAO-20/index.md) | Documented normal.dhdj datum dereferences an unpopulated Taylor series | Tao | P3 | Crash or API hazard | 2026-11-05 (`c4c629f`, head) |  |  |
| [COL-63](COL-63/index.md) | Modal momentum kicks omit the manual’s finite-velocity factor <span class="bb-tag">contract-unsettled</span> | Collective effects | P3 | Documentation or example mismatch | 2026-11-05 (`c4c629f`, head) |  |  |
| [TRK-21](TRK-21/index.md) | PTC-to-Bmad cavity update inverts voltage and lcavity phase wrongly <span class="bb-tag">partly-source-only</span> | Tracking, RF, fields and spin | P4 | Wrong physics | 2026-11-05 (`c4c629f`, head) |  |  |
| [COL-01](COL-01/index.md) | Uniform-charge deposition uses an uninitialized charge at 50,000 particles or more | Collective effects | P4 | Wrong physics | 2026-11-05 (`c4c629f`, head) |  |  |
| [COL-02](COL-02/index.md) | The FFT wrapper ignores mixed transform directions <span class="bb-tag">partly-source-only</span> | Collective effects | P4 | Wrong physics | 2026-11-05 (`c4c629f`, head) |  |  |
| [COL-12](COL-12/index.md) | CSR3D boundary derivatives use integer division <span class="bb-tag">experimental</span> | Collective effects | P4 | Wrong physics | 2026-11-05 (`c4c629f`, head) |  |  |
| [COL-16](COL-16/index.md) | CSR3D momentum kicks leave the arrival-time coordinate inconsistent <span class="bb-tag">experimental</span> <span class="bb-tag">known-limitation</span> | Collective effects | P4 | Wrong physics | 2026-11-05 (`c4c629f`, head) |  |  |
| [COL-45](COL-45/index.md) | Potential-well routines omit the ion charge state <span class="bb-tag">partly-source-only</span> | Collective effects | P4 | Wrong physics | 2026-11-05 (`c4c629f`, head) |  |  |
| [COL-46](COL-46/index.md) | multi_coulomb_log uses the root branch species for secondary branches | Collective effects | P4 | Wrong physics | 2026-11-05 (`c4c629f`, head) |  |  |
| [XLT-09](XLT-09/index.md) | Elegant export writes quadratic Taylor terms as linear entries <span class="bb-tag">experimental</span> | Translators | P4 | Wrong physics | 2026-11-05 (`c4c629f`, head) |  |  |
| [XLT-10](XLT-10/index.md) | Elegant collimator export reads another element's length <span class="bb-tag">experimental</span> | Translators | P4 | Wrong physics | 2026-11-05 (`c4c629f`, head) |  |  |
| [XLT-32](XLT-32/index.md) | Elegant export drops order-zero multipole components <span class="bb-tag">experimental</span> <span class="bb-tag">partly-source-only</span> | Translators | P4 | Wrong physics | 2026-11-05 (`c4c629f`, head) |  |  |
| [TRK-49](TRK-49/index.md) | Upstream pytest comparator accepts missing trailing records | Tracking, RF, fields and spin | P4 | Silent wrong result | 2026-11-05 (`c4c629f`, head) |  |  |
| [TRK-58](TRK-58/index.md) | Kinematic conversion helpers omit charge from magnetic rigidity <span class="bb-tag">partly-source-only</span> | Tracking, RF, fields and spin | P4 | Silent wrong result | 2026-11-06 (`c4c629f`) |  |  |
| [TRK-67](TRK-67/index.md) | Intrinsic-spin lookup returns zero for helion and antihelion <span class="bb-tag">partly-source-only</span> | Tracking, RF, fields and spin | P4 | Silent wrong result | 2026-11-06 (`c4c629f`) |  |  |
| [TRK-70](TRK-70/index.md) | Stepped native lcavities ignore spin_fringe_on | Tracking, RF, fields and spin | P4 | Silent wrong result | 2026-11-06 (`c4c629f`) |  |  |
| [OPT-10](OPT-10/index.md) | Linear tracking stores incompatible map-reference records | Optics | P4 | Silent wrong result | 2026-11-05 (`c4c629f`, head) |  |  |
| [OPT-27](OPT-27/index.md) | srdt_lsq_solution gives every variable the same strength <span class="bb-tag">partly-source-only</span> | Optics | P4 | Silent wrong result | 2026-11-05 (`c4c629f`, head) |  |  |
| [OPT-28](OPT-28/index.md) | srdt per-element terms have wrong factors and unset fields <span class="bb-tag">partly-source-only</span> | Optics | P4 | Silent wrong result | 2026-11-05 (`c4c629f`, head) |  |  |
| [COL-22](COL-22/index.md) | S-based cathode_fft_3d tracking silently applies no space charge | Collective effects | P4 | Silent wrong result | 2026-11-05 (`c4c629f`, head) |  |  |
| [COL-33](COL-33/index.md) | ibs_rates0.out uses one thousand times the requested population | Collective effects | P4 | Silent wrong result | 2026-11-05 (`c4c629f`, head) |  |  |
| [COL-67](COL-67/index.md) | Initial IBS rates use equilibrium beam sizes instead of natural sizes | Collective effects | P4 | Silent wrong result | 2026-11-05 (`c4c629f`, head) |  |  |
| [COL-68](COL-68/index.md) | Saved strong-beam centers are overwritten with the weak particle position | Collective effects | P4 | Silent wrong result | 2026-11-05 (`c4c629f`, head) |  |  |
| [LAT-06](LAT-06/index.md) | Lazy seeding initializes the wrong random state <span class="bb-tag">partly-source-only</span> | Lattices, particles and I/O | P4 | Silent wrong result | 2026-11-05 (`c4c629f`, head) |  |  |
| [LAT-09](LAT-09/index.md) | ASCII output omits persistent particle identifiers | Lattices, particles and I/O | P4 | Silent wrong result | 2026-11-05 (`c4c629f`, head) |  |  |
| [LAT-34](LAT-34/index.md) | An explicit END marker does not suppress the automatic marker | Lattices, particles and I/O | P4 | Silent wrong result | 2026-11-05 (`c4c629f`, head) |  |  |
| [LAT-35](LAT-35/index.md) | Repeated HDF5 suffixes make the beam writer emit ASCII | Lattices, particles and I/O | P4 | Silent wrong result | 2026-11-05 (`c4c629f`, head) |  |  |
| [TAO-14](TAO-14/index.md) | call -no_calc saves flags without disabling lattice calculation | Tao | P4 | Silent wrong result | 2026-11-05 (`c4c629f`, head) |  |  |
| [TAO-15](TAO-15/index.md) | A later lattice-file option cancels explicit -noinit | Tao | P4 | Silent wrong result | 2026-11-05 (`c4c629f`, head) |  |  |
| [TAO-18](TAO-18/index.md) | Disabled lattice calculation makes pipe merit report exactly zero <span class="bb-tag">partly-source-only</span> | Tao | P4 | Silent wrong result | 2026-11-05 (`c4c629f`, head) | [[gh:https://github.com/bmad-sim/bmad-ecosystem/issues/1996]] |  |
| [TRK-39](TRK-39/index.md) | track_all accepts an orbit array one record too short | Tracking, RF, fields and spin | P4 | Crash or API hazard | 2026-11-05 (`c4c629f`, head) |  |  |
| [TRK-40](TRK-40/index.md) | The interior-start guard compares particle state with a location enum | Tracking, RF, fields and spin | P4 | Crash or API hazard | 2026-11-05 (`c4c629f`, head) |  |  |
| [TRK-48](TRK-48/index.md) | Legacy regression runner passes a list object to open | Tracking, RF, fields and spin | P4 | Crash or API hazard | 2026-11-05 (`c4c629f`, head) |  |  |
| [OPT-12](OPT-12/index.md) | Single-element matrix cache hits report an error | Optics | P4 | Crash or API hazard | 2026-11-05 (`c4c629f`, head) |  |  |
| [COL-54](COL-54/index.md) | BBU HOM-order filtering breaks mode-array allocation | Collective effects | P4 | Crash or API hazard | 2026-11-05 (`c4c629f`, head) |  |  |
| [COL-57](COL-57/index.md) | Randomizing frequencies dereferences an absent wake | Collective effects | P4 | Crash or API hazard | 2026-11-05 (`c4c629f`, head) |  |  |
| [LAT-22](LAT-22/index.md) | openPMD readers reject a numeric or absent openPMDextension <span class="bb-tag">partly-source-only</span> | Lattices, particles and I/O | P4 | Crash or API hazard | 2026-11-05 (`c4c629f`, head) |  |  |
| [LAT-32](LAT-32/index.md) | Indexed attribute access omits the spin method and cannot reach electric multipoles | Lattices, particles and I/O | P4 | Crash or API hazard | 2026-11-05 (`c4c629f`, head) |  |  |
| [LAT-36](LAT-36/index.md) | Initial PTC lcavity parsing reaches tracking before step bookkeeping | Lattices, particles and I/O | P4 | Crash or API hazard | 2026-11-06 (`c4c629f`) |  |  |
| [XLT-11](XLT-11/index.md) | MAD-X multipole-array import is shadowed by an earlier branch | Translators | P4 | Crash or API hazard | 2026-11-05 (`c4c629f`, head) |  |  |
| [XLT-12](XLT-12/index.md) | Ordinary MAD-X aperture import unpacks the wrong array slice | Translators | P4 | Crash or API hazard | 2026-11-05 (`c4c629f`, head) |  |  |
| [XLT-19](XLT-19/index.md) | Five Elegant importer mappings emit invalid Bmad element classes <span class="bb-tag">experimental</span> | Translators | P4 | Crash or API hazard | 2026-11-05 (`c4c629f`, head) |  |  |
| [XLT-24](XLT-24/index.md) | MAD export without a reference orbit aborts for a full-fringe quadrupole <span class="bb-tag">partly-source-only</span> | Translators | P4 | Crash or API hazard | 2026-11-05 (`c4c629f`, head) |  |  |
| [XLT-25](XLT-25/index.md) | Elegant export of a zero-length magnet writes NaN <span class="bb-tag">experimental</span> <span class="bb-tag">partly-source-only</span> | Translators | P4 | Crash or API hazard | 2026-11-05 (`c4c629f`, head) |  |  |
| [XLT-33](XLT-33/index.md) | Elegant export leaves later copies of a split multipole undefined <span class="bb-tag">experimental</span> | Translators | P4 | Crash or API hazard | 2026-11-05 (`c4c629f`, head) |  |  |
| [TAO-04](TAO-04/index.md) | Data namelist export writes an unquoted character universe selector | Tao | P4 | Crash or API hazard | 2026-11-05 (`c4c629f`, head) |  |  |
| [TAO-08](TAO-08/index.md) | Pipe global duplicates two optimizer parameter records | Tao | P4 | Crash or API hazard | 2026-11-05 (`c4c629f`, head) |  |  |
| [TAO-16](TAO-16/index.md) | Startup and batch-command failures exit with process status zero | Tao | P4 | Crash or API hazard | 2026-11-05 (`c4c629f`, head) |  |  |
| [TAO-19](TAO-19/index.md) | ele:param tags matrix and vector arrays as scalar REAL | Tao | P4 | Crash or API hazard | 2026-11-05 (`c4c629f`, head) |  |  |
| [TRK-20](TRK-20/index.md) | The LCavity tracking manual doubles the implemented fringe coefficient <span class="bb-tag">partly-source-only</span> | Tracking, RF, fields and spin | P4 | Documentation or example mismatch | 2026-11-05 (`c4c629f`, head) |  |  |
| [TRK-42](TRK-42/index.md) | Two compiled constants disagree with the claimed CODATA 2022 update | Tracking, RF, fields and spin | P4 | Documentation or example mismatch | 2026-11-05 (`c4c629f`, head) |  |  |
| [TRK-44](TRK-44/index.md) | Manual constants tables disagree with executable constants | Tracking, RF, fields and spin | P4 | Documentation or example mismatch | 2026-11-05 (`c4c629f`, head) |  |  |
| [TRK-45](TRK-45/index.md) | offset_particle promises multipole kicks that it does not apply | Tracking, RF, fields and spin | P4 | Documentation or example mismatch | 2026-11-05 (`c4c629f`, head) |  |  |
| [TRK-46](TRK-46/index.md) | Aperture header states an obsolete transverse-momentum limit | Tracking, RF, fields and spin | P4 | Documentation or example mismatch | 2026-11-05 (`c4c629f`, head) |  |  |
| [TRK-47](TRK-47/index.md) | Custom-tracker example registers an undeclared procedure pointer | Tracking, RF, fields and spin | P4 | Documentation or example mismatch | 2026-11-05 (`c4c629f`, head) |  |  |
| [TRK-50](TRK-50/index.md) | Legacy regression documentation names the wrong script and list file | Tracking, RF, fields and spin | P4 | Documentation or example mismatch | 2026-11-05 (`c4c629f`, head) |  |  |
| [OPT-18](OPT-18/index.md) | The mode-flip manual example reverses forward conjugation | Optics | P4 | Documentation or example mismatch | 2026-11-05 (`c4c629f`, head) |  |  |
| [OPT-21](OPT-21/index.md) | Integral bunch length uses summed cavity M65 instead of one-turn M65 <span class="bb-tag">known-limitation</span> | Optics | P4 | Documentation or example mismatch | 2026-11-05 (`c4c629f`, head) |  |  |
| [COL-18](COL-18/index.md) | CSR step-control documentation names two nonexistent attributes <span class="bb-tag">partly-source-only</span> | Collective effects | P4 | Documentation or example mismatch | 2026-11-05 (`c4c629f`, head) |  |  |
| [COL-19](COL-19/index.md) | The CSR manual describes a second staggered deposition absent from the calculation | Collective effects | P4 | Documentation or example mismatch | 2026-11-05 (`c4c629f`, head) |  |  |
| [COL-20](COL-20/index.md) | The CSR method list omits the experimental three-dimensional selector <span class="bb-tag">experimental</span> <span class="bb-tag">known-limitation</span> <span class="bb-tag">partly-source-only</span> | Collective effects | P4 | Documentation or example mismatch | 2026-11-05 (`c4c629f`, head) |  |  |
| [COL-31](COL-31/index.md) | IBS selector documentation names different algorithms from those executed | Collective effects | P4 | Documentation or example mismatch | 2026-11-05 (`c4c629f`, head) |  |  |
| [COL-32](COL-32/index.md) | The documented ibs_ring Kubo selector and cache option are unavailable | Collective effects | P4 | Documentation or example mismatch | 2026-11-05 (`c4c629f`, head) |  |  |
| [COL-38](COL-38/index.md) | Beam–beam slices use mid-quantiles instead of documented charge centroids | Collective effects | P4 | Documentation or example mismatch | 2026-11-05 (`c4c629f`, head) |  |  |
| [COL-39](COL-39/index.md) | Beam–beam crab coefficients define an angle instead of the documented displacement <span class="bb-tag">partly-source-only</span> | Collective effects | P4 | Documentation or example mismatch | 2026-11-05 (`c4c629f`, head) |  |  |
| [COL-40](COL-40/index.md) | The beam–beam manual names a nonexistent Twiss-reference attribute | Collective effects | P4 | Documentation or example mismatch | 2026-11-05 (`c4c629f`, head) |  |  |
| [COL-62](COL-62/index.md) | Short-range wake equations interchange source and witness offsets | Collective effects | P4 | Documentation or example mismatch | 2026-11-05 (`c4c629f`, head) |  |  |
| [COL-64](COL-64/index.md) | Manual long-range accumulator signs disagree with stored coefficients | Collective effects | P4 | Documentation or example mismatch | 2026-11-05 (`c4c629f`, head) |  |  |
| [COL-65](COL-65/index.md) | The long-range damping example omits a factor of two pi | Collective effects | P4 | Documentation or example mismatch | 2026-11-05 (`c4c629f`, head) |  |  |
| [COL-69](COL-69/index.md) | Potential-well matrix header reverses the implemented kick order | Collective effects | P4 | Documentation or example mismatch | 2026-11-05 (`c4c629f`, head) |  |  |
| [LAT-16](LAT-16/index.md) | The beam-writer header documents the wrong default format | Lattices, particles and I/O | P4 | Documentation or example mismatch | 2026-11-05 (`c4c629f`, head) |  |  |
| [LAT-18](LAT-18/index.md) | The documented Gaussian default differs from both source defaults | Lattices, particles and I/O | P4 | Documentation or example mismatch | 2026-11-05 (`c4c629f`, head) |  |  |
| [LAT-19](LAT-19/index.md) | The manual contradicts the additive initialization time offset <span class="bb-tag">partly-source-only</span> | Lattices, particles and I/O | P4 | Documentation or example mismatch | 2026-11-05 (`c4c629f`, head) |  |  |
| [LAT-20](LAT-20/index.md) | The HDF5 manual places species metadata one group too high | Lattices, particles and I/O | P4 | Documentation or example mismatch | 2026-11-05 (`c4c629f`, head) |  |  |
| [LAT-21](LAT-21/index.md) | Bunch-index documentation disagrees with initialization | Lattices, particles and I/O | P4 | Documentation or example mismatch | 2026-11-05 (`c4c629f`, head) |  |  |
| [LAT-33](LAT-33/index.md) | The field-map assignment manual overstates shared storage | Lattices, particles and I/O | P4 | Documentation or example mismatch | 2026-11-05 (`c4c629f`, head) |  |  |
| [XLT-18](XLT-18/index.md) | noaperture documentation contradicts the MAD-X output guard | Translators | P4 | Documentation or example mismatch | 2026-11-05 (`c4c629f`, head) |  |  |
| [TAO-12](TAO-12/index.md) | Aliases with placeholders reject documented surplus arguments | Tao | P4 | Documentation or example mismatch | 2026-11-05 (`c4c629f`, head) |  |  |
| [TAO-13](TAO-13/index.md) | Command-file nesting contradicts the documented unlimited depth | Tao | P4 | Documentation or example mismatch | 2026-11-05 (`c4c629f`, head) |  |  |
| [TAO-23](TAO-23/index.md) | Single-element show matrix syntax disagrees with its manual | Tao | P4 | Documentation or example mismatch | 2026-11-05 (`c4c629f`, head) |  |  |
| [LAT-15](LAT-15/index.md) | HDF5 macro-charge weights advertise dimensionless units | Lattices, particles and I/O | P5 | Silent wrong result | 2026-11-05 (`c4c629f`, head) |  |  |
| [OPT-19](OPT-19/index.md) | Chromaticity documentation labels a momentum derivative as energy <span class="bb-tag">partly-source-only</span> | Optics | P5 | Documentation or example mismatch | 2026-11-05 (`c4c629f`, head) |  |  |
| [COL-34](COL-34/index.md) | Envelope IBS argument comments call a damping time a rate <span class="bb-tag">partly-source-only</span> | Collective effects | P5 | Documentation or example mismatch | 2026-11-05 (`c4c629f`, head) |  |  |
| [COL-36](COL-36/index.md) | Touschek emittance comments reverse the open and closed conventions | Collective effects | P5 | Documentation or example mismatch | 2026-11-05 (`c4c629f`, head) |  |  |
| [LAT-17](LAT-17/index.md) | The manual calls a momentum spread an energy spread | Lattices, particles and I/O | P5 | Documentation or example mismatch | 2026-11-05 (`c4c629f`, head) |  |  |
| [XLT-36](XLT-36/index.md) | Energy-time coordinate discussion contains three spelling errors | Translators | P5 | Documentation or example mismatch | 2026-11-05 (`c4c629f`) |  |  |
| [XLT-37](XLT-37/index.md) | Sad_Mult documentation interchanges normal and skew coefficients | Translators | P5 | Documentation or example mismatch | 2026-11-05 (`c4c629f`, head) |  |  |
| [TAO-21](TAO-21/index.md) | show universe labels momentum chromaticity as an energy derivative | Tao | P5 | Documentation or example mismatch | 2026-11-05 (`c4c629f`, head) |  |  |
| [TAO-22](TAO-22/index.md) | Default radiation output labels a damped transport matrix as undamped | Tao | P5 | Documentation or example mismatch | 2026-11-05 (`c4c629f`, head) |  |  |
