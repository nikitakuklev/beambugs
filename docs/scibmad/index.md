---
title: SciBmad
hide:
  - toc
---

# SciBmad

Verified defects in [SciBmad](https://github.com/bmad-sim/SciBmad.jl) ([source](https://github.com/bmad-sim/SciBmad.jl)).
Each entry links to its reproducer, the models that found and confirmed it, and any upstream issue or pull request.
Machine-readable: [defects.json](defects.json).
Tags are explained in [Methodology](../methodology.md#tags).

## Running the reproducers

Install Julia 1.13 and the packages `SciBmad` (0.6.0), `SpaceCharge` and `ForwardDiff`.
Download a reproducer and run it with `julia <ID>.jl`, one script per process and one
thread (`JULIA_NUM_THREADS=1`).


## Defects

<div class="bb-filters" data-column="Area"></div>

| ID | Defect | Area | Priority | Severity | Last reproduced | Issue | PR |
|---|---|---|---|---|---|---|---|
| [TRK-01](TRK-01/index.md) | Ordinary RF decelerates at the declared accelerating crest | Tracking | P0 | Wrong physics | 2026-10-24 (`d8e45fd`, main) |  |  |
| [OPT-17](OPT-17/index.md) | Orbital caps of 1 drop the start-point dispersion feed-down (default `twiss(chrom=2)`, documented knob descriptors) | Optics and maps | P0 | Silent wrong result | 2026-10-31 (`ae9423d`, main) |  |  |
| [TRK-20](TRK-20/index.md) | Zero alignment applies a bend's reference roll twice | Tracking | P1 | Wrong physics | 2026-10-29 (`0bf3585`, main) |  |  |
| [OPT-06](OPT-06/index.md) | Summary `alphac` (momentum compaction) equals the slip factor `etac` | Optics and maps | P1 | Wrong physics | 2026-10-31 (`ae9423d`, main) | [[gh:https://github.com/bmad-sim/SciBmad.jl/issues/152]] | [[gh:https://github.com/bmad-sim/SciBmad.jl/pull/153]] |
| [COL-01](COL-01/index.md) | IBS source statistics include lost particles | Collective effects and spin | P1 | Wrong physics | 2026-10-31 (`0bf3585`, main) | [[gh:https://github.com/bmad-sim/BeamTracking.jl/issues/380]] | [[gh:https://github.com/bmad-sim/BeamTracking.jl/pull/381]] |
| [COL-10](COL-10/index.md) | IBS statistics are taken before the element's entrance alignment and aperture <span class="bb-tag">partly-source-only</span> | Collective effects and spin | P1 | Wrong physics | 2026-10-23 (`d8e45fd`, main) |  |  |
| [LAT-18](LAT-18/index.md) | Bmad SciBmad writer copies `phi0` (rad/2π) into a radians field with the wrong sign convention <span class="bb-tag">partly-source-only</span> | Lattice and API | P1 | Wrong physics | 2026-10-22 (`19b8afb`) |  |  |
| [TRK-18](TRK-18/index.md) | Builders skip the frame rotation at a nominally zero angle and drop its derivative <span class="bb-tag">partly-source-only</span> | Tracking | P1 | Silent wrong result | 2026-10-24 (`d8e45fd`, main) |  |  |
| [OPT-02](OPT-02/index.md) | Scalar `d<col>_<n>` returns f⁽ⁿ⁾/n, not the nth derivative | Optics and maps | P1 | Silent wrong result | 2026-10-22 (`62399f9`; not on main) | [[gh:https://github.com/bmad-sim/SciBmad.jl/issues/154]] | [[gh:https://github.com/bmad-sim/SciBmad.jl/pull/155]] |
| [OPT-03](OPT-03/index.md) | `h<ijkl>` columns are degree-weighted, and the scalar and Taylor paths disagree | Optics and maps | P1 | Silent wrong result | 2026-10-22 (`62399f9`; not on main) | [[gh:https://github.com/bmad-sim/SciBmad.jl/issues/156]] | [[gh:https://github.com/bmad-sim/SciBmad.jl/pull/157]] |
| [OPT-20](OPT-20/index.md) | `getterm` returns 0.0 for summary coefficients the normal form never computes | Optics and maps | P1 | Silent wrong result | 2026-10-23 (`5f0fb83`, main) |  |  |
| [LAT-21](LAT-21/index.md) | Bmad SciBmad writer drops the last exponent digit of aperture limits | Lattice and API | P1 | Silent wrong result | 2026-10-30 (`19b8afb`) |  |  |
| [TRK-02](TRK-02/index.md) | `SaganCavity` swaps the below/above-transition zero phases | Tracking | P2 | Wrong physics | 2026-10-31 (`0bf3585`, main) |  |  |
| [TRK-07](TRK-07/index.md) | `R_to_v` and `R_to_gamma` are wrong for q ≠ ±1 | Tracking | P2 | Wrong physics | 2026-10-22 (`65e343f`; not on main) | [[gh:https://github.com/bmad-sim/BeamTracking.jl/issues/366]] | [[gh:https://github.com/bmad-sim/BeamTracking.jl/pull/368]] |
| [TRK-17](TRK-17/index.md) | `BendKick` applies a straight dipole's other multipoles unrotated in the dipole's frame | Tracking | P2 | Wrong physics | 2026-10-24 (`d8e45fd`, main) | [[gh:https://github.com/bmad-sim/BeamTracking.jl/issues/376]] | [[gh:https://github.com/bmad-sim/BeamTracking.jl/pull/377]] |
| [TRK-19](TRK-19/index.md) | Callback out/in transforms change momentum in a time-dependent potential | Tracking | P2 | Wrong physics | 2026-10-29 (`0bf3585`, main) |  |  |
| [COL-02](COL-02/index.md) | Stochastic radiation in a tilted bend or separator is computed in the wrong frame | Collective effects and spin | P2 | Wrong physics | 2026-10-24 (`d8e45fd`, main) | [[gh:https://github.com/bmad-sim/BeamTracking.jl/issues/378]] | [[gh:https://github.com/bmad-sim/BeamTracking.jl/pull/379]] |
| [COL-11](COL-11/index.md) | Sagan RF energy kicks double the leading off-axis spin rotation | Collective effects and spin | P2 | Wrong physics | 2026-10-29 (`0bf3585`, main) |  |  |
| [COL-15](COL-15/index.md) | IBS uses canonical momenta for flight time inside a solenoid <span class="bb-tag">partly-source-only</span> | Collective effects and spin | P2 | Wrong physics | 2026-10-29 (`0bf3585`, main) |  |  |
| [LAT-03](LAT-03/index.md) | `R_to_v` and `R_to_gamma` omit the charge state; ion harmonic RF frequency is superluminal | Lattice and API | P2 | Wrong physics | 2026-10-29 (`35fe016`, main) | [[gh:https://github.com/bmad-sim/BeamTracking.jl/issues/366]] | [[gh:https://github.com/bmad-sim/BeamTracking.jl/pull/368]] |
| [LAT-04](LAT-04/index.md) | Harmonic RF frequency uses branch-global `s`, not the containing Beamline length <span class="bb-tag">contract-unsettled</span> | Lattice and API | P2 | Wrong physics | 2026-10-31 (`35fe016`, main) |  |  |
| [LAT-25](LAT-25/index.md) | Simultaneous magnetic-strength convention changes rescale the partner twice | Lattice and API | P2 | Wrong physics | 2026-10-29 (`35fe016`, main) |  |  |
| [TRK-03](TRK-03/index.md) | `Exact()` bend ignores `e1`, `e2` and `fringe_at` | Tracking | P2 | Silent wrong result | 2026-10-31 (`0bf3585`, main) |  |  |
| [TRK-05](TRK-05/index.md) | `Patch(dt)` shifts z but not `Bunch.t_ref` | Tracking | P2 | Silent wrong result | 2026-10-31 (`0bf3585`, main) |  |  |
| [TRK-06](TRK-06/index.md) | `SaganCavity` leaves `Bunch.p_over_q_ref` stale; the next element erases the energy gain | Tracking | P2 | Silent wrong result | 2026-10-31 (`0bf3585`, main) | [[gh:https://github.com/bmad-sim/BeamTracking.jl/issues/367]] | [[gh:https://github.com/bmad-sim/BeamTracking.jl/pull/369]] |
| [TRK-08](TRK-08/index.md) | Interior callbacks get the entrance `t_ref` in RF cavities and separators | Tracking | P2 | Silent wrong result | 2026-10-31 (`0bf3585`, main) |  |  |
| [TRK-10](TRK-10/index.md) | Methods other than `RungeKutta` silently ignore `EMFieldParams` <span class="bb-tag">partly-source-only</span> | Tracking | P2 | Silent wrong result | 2026-10-31 (`0bf3585`, main) |  |  |
| [OPT-01](OPT-01/index.md) | Coasting `twiss` ignores `delta0`; `a_initial` runs ignore `delta0` and `v0` | Optics and maps | P2 | Silent wrong result | 2026-10-31 (`ae9423d`, main) |  |  |
| [OPT-04](OPT-04/index.md) | Linear De Moivre `B`/`E` use columns of A⁻¹ instead of rows <span class="bb-tag">partly-source-only</span> | Optics and maps | P2 | Silent wrong result | 2026-10-31 (`ae9423d`, main) |  |  |
| [OPT-07](OPT-07/index.md) | With RF, the parametric `slip`/`etac`/`alphac` drop sin(2πφ₃) | Optics and maps | P2 | Silent wrong result | 2026-10-31 (`ae9423d`, main) |  |  |
| [OPT-16](OPT-16/index.md) | NAFF's Gram–Schmidt step is conjugated and unnormalized | Optics and maps | P2 | Silent wrong result | 2026-10-31 (`f208dc0`, main) |  |  |
| [OPT-18](OPT-18/index.md) | Summary tunes and `phi` columns alias when a sampled segment exceeds half a turn | Optics and maps | P2 | Silent wrong result | 2026-10-31 (`ae9423d`, main) |  |  |
| [OPT-21](OPT-21/index.md) | `getterm(…, as_taylor_series=true)` drops the constant coefficient | Optics and maps | P2 | Silent wrong result | 2026-10-23 (`5f0fb83`, main) |  |  |
| [OPT-30](OPT-30/index.md) | Automatic damping detection gives nonperiodic periodic Twiss functions | Optics and maps | P2 | Silent wrong result | 2026-10-29 (`ae9423d`, main) |  |  |
| [OPT-32](OPT-32/index.md) | Montague functions omit momentum normalization | Optics and maps | P2 | Silent wrong result | 2026-10-29 (`ae9423d`, main) |  |  |
| [COL-07](COL-07/index.md) | Degenerate automatic z bounds misplace the cathode image | Collective effects and spin | P2 | Silent wrong result | 2026-10-31 (`ae30463`, main) |  |  |
| [COL-08](COL-08/index.md) | Cathode-image field loses Float64 accuracy at large distance/cell ratio | Collective effects and spin | P2 | Silent wrong result | 2026-10-31 (`ae30463`, main) |  |  |
| [LAT-05](LAT-05/index.md) | `0 - BatchParam(array)` returns the array unnegated | Lattice and API | P2 | Silent wrong result | 2026-10-22 (`65e343f`; not on main) | [[gh:https://github.com/bmad-sim/BeamTracking.jl/issues/374]] | [[gh:https://github.com/bmad-sim/BeamTracking.jl/pull/375]] |
| [LAT-23](LAT-23/index.md) | `BatchParam` scalar shortcuts test `≈ 0` and `≈ 1`, dropping factors, derivatives and knobs | Lattice and API | P2 | Silent wrong result | 2026-10-29 (`0bf3585`, main) |  |  |
| [TRK-21](TRK-21/index.md) | Stored zero dipole strength corrupts coordinates at bend faces | Tracking | P3 | Wrong physics | 2026-10-29 (`0bf3585`, main) |  |  |
| [COL-12](COL-12/index.md) | Sextupole spin fringes use twice the longitudinal field | Collective effects and spin | P3 | Wrong physics | 2026-10-29 (`0bf3585`, main) |  |  |
| [COL-14](COL-14/index.md) | Sagan RF spin impulses omit the superimposed solenoid vector potential | Collective effects and spin | P3 | Wrong physics | 2026-10-29 (`0bf3585`, main) |  |  |
| [TRK-16](TRK-16/index.md) | `sincu`/`sinhcu` use the two-term series for arguments up to 0.004 in magnitude in Float64 | Tracking | P3 | Silent wrong result | 2026-10-31 (`0bf3585`, main) |  |  |
| [OPT-08](OPT-08/index.md) | Row 2 of the first table column reuses row 1's cache | Optics and maps | P3 | Silent wrong result | 2026-10-31 (`ae9423d`, main) |  |  |
| [OPT-22](OPT-22/index.md) | `copy!` into a `VectorField` drops the spin part, so `log` never corrects its spin generator | Optics and maps | P3 | Silent wrong result | 2026-10-31 (`7e40539`, main) |  |  |
| [OPT-26](OPT-26/index.md) | `TPSAMap` composition transports covariance at the wrong expansion point | Optics and maps | P3 | Silent wrong result | 2026-10-29 (`7e40539`, main) |  |  |
| [OPT-27](OPT-27/index.md) | `q_map` initialization is skipped unless `q` is supplied | Optics and maps | P3 | Silent wrong result | 2026-10-29 (`7e40539`, main) |  |  |
| [OPT-28](OPT-28/index.md) | `I - map` retains off-diagonal, nonlinear and spin coefficient signs | Optics and maps | P3 | Silent wrong result | 2026-10-29 (`7e40539`, main) |  |  |
| [OPT-31](OPT-31/index.md) | Degenerate damped modes corrupt eigenvectors and equilibrium covariance | Optics and maps | P3 | Silent wrong result | 2026-10-29 (`7e40539`, main) | [[gh:https://github.com/bmad-sim/NonlinearNormalForm.jl/issues/65]] |  |
| [OPT-33](OPT-33/index.md) | Scaled identity addition and subtraction ignore the scale <span class="bb-tag">partly-source-only</span> | Optics and maps | P3 | Silent wrong result | 2026-10-29 (`7e40539`, main) |  |  |
| [OPT-35](OPT-35/index.md) | GTPSA inversion loses parameter-dependent orbital coefficients | Optics and maps | P3 | Silent wrong result | 2026-10-29 (`a5790a7`, main) |  |  |
| [OPT-36](OPT-36/index.md) | Cross-descriptor TPS replacement retains destination coefficients | Optics and maps | P3 | Silent wrong result | 2026-10-29 (`a5790a7`, main) |  |  |
| [LAT-01](LAT-01/index.md) | Rigidity sign "correction" keeps a negative rigidity for positive species | Lattice and API | P3 | Silent wrong result | 2026-10-31 (`35fe016`, main) |  |  |
| [LAT-02](LAT-02/index.md) | Relative reference getters return wrong units or repeat the upstream increment | Lattice and API | P3 | Silent wrong result | 2026-10-31 (`35fe016`, main) |  |  |
| [LAT-07](LAT-07/index.md) | GTPSA special functions of a `DefExpr` discard the evaluation Context <span class="bb-tag">partly-source-only</span> | Lattice and API | P3 | Silent wrong result | 2026-10-31 (`35fe016`, main) |  | [[gh:https://github.com/bmad-sim/Beamlines.jl/pull/171]] |
| [LAT-11](LAT-11/index.md) | Aperture kernels overwrite the state of particles that are not alive <span class="bb-tag">partly-source-only</span> | Lattice and API | P3 | Silent wrong result | 2026-10-31 (`0bf3585`, main) |  |  |
| [OPT-05](OPT-05/index.md) | Map-valued `E1..E3` throw `UndefVarError: nj_mat` <span class="bb-tag">partly-source-only</span> | Optics and maps | P3 | Crash or API hazard | 2026-10-31 (`ae9423d`, main) |  |  |
| [OPT-12](OPT-12/index.md) | Spin normal form gives NaN for an exactly vertical (or absent) spin rotation | Optics and maps | P3 | Crash or API hazard | 2026-10-24 (`7e40539`, main) |  |  |
| [OPT-23](OPT-23/index.md) | `twiss` with `h` columns and `spin=true` ends the process (division by a zero quaternion) | Optics and maps | P3 | Crash or API hazard | 2026-10-31 (`ae9423d`, main) |  |  |
| [OPT-29](OPT-29/index.md) | ForwardDiff of a four-potential strength fails in implicit tracking | Optics and maps | P3 | Crash or API hazard | 2026-10-29 (`0bf3585`, main) |  |  |
| [OPT-34](OPT-34/index.md) | Full-circle dynamic-aperture scans abort during analysis | Optics and maps | P3 | Crash or API hazard | 2026-10-29 (`ae9423d`, main) |  |  |
| [COL-03](COL-03/index.md) | SaganCavity fluctuations with multipoles throw `MethodError` <span class="bb-tag">partly-source-only</span> | Collective effects and spin | P3 | Crash or API hazard | 2026-10-30 (`0bf3585`, main) |  |  |
| [COL-04](COL-04/index.md) | `twiss(spin=true)` fails when the spin axis is exactly vertical | Collective effects and spin | P3 | Crash or API hazard | 2026-10-24 (`7e40539`, main) |  |  |
| [COL-13](COL-13/index.md) | Automatic Float32 mesh bounds invalidate CIC cells for input particles | Collective effects and spin | P3 | Crash or API hazard | 2026-10-29 (`ae30463`, main) |  |  |
| [TRK-09](TRK-09/index.md) | RF with voltage but no frequency loses every live particle with NaN coordinates | Tracking | P4 | Silent wrong result | 2026-10-31 (`0bf3585`, main) |  |  |
| [OPT-11](OPT-11/index.md) | The spin Lie bracket uses the result field instead of the second input | Optics and maps | P4 | Silent wrong result | 2026-10-31 (`7e40539`, main) |  |  |
| [OPT-14](OPT-14/index.md) | Batched Newton returns the previous iterate's `f` and `jac` <span class="bb-tag">partly-source-only</span> | Optics and maps | P4 | Silent wrong result | 2026-10-31 (`ac41c09`, main) |  |  |
| [COL-06](COL-06/index.md) | `track_spin` does not validate a matrix initial spin <span class="bb-tag">partly-source-only</span> | Collective effects and spin | P4 | Silent wrong result | 2026-10-31 (`ae9423d`, main) |  |  |
| [TRK-12](TRK-12/index.md) | Sagan tracking guide uses nonexistent names | Tracking | P4 | Crash or API hazard | 2026-10-31 (`ae9423d`, main) |  |  |
| [OPT-10](OPT-10/index.md) | Experimental torus and aperture code calls APIs that do not exist <span class="bb-tag">experimental</span> <span class="bb-tag">partly-source-only</span> | Optics and maps | P4 | Crash or API hazard | 2026-10-24 (`5f0fb83`, main) |  |  |
| [OPT-13](OPT-13/index.md) | Out-of-place `F * m`, `a * m`, `lb`, mixed-type `exp(F, m)` and `VectorField(F; init)` are unusable | Optics and maps | P4 | Crash or API hazard | 2026-10-31 (`7e40539`, main) |  |  |
| [OPT-19](OPT-19/index.md) | Reading a monomial the descriptor does not hold terminates the process <span class="bb-tag">partly-source-only</span> | Optics and maps | P4 | Crash or API hazard | 2026-10-31 (`a5790a7`, main) |  |  |
| [OPT-24](OPT-24/index.md) | `twiss` overwrites the caller's `v0` with the closed orbit <span class="bb-tag">contract-unsettled</span> | Optics and maps | P4 | Crash or API hazard | 2026-10-24 (`5f0fb83`, main) |  |  |
| [OPT-25](OPT-25/index.md) | `VectorField` keyword constructor fails unless `v` is given | Optics and maps | P4 | Crash or API hazard | 2026-10-31 (`7e40539`, main) |  |  |
| [COL-05](COL-05/index.md) | `DAMap` keyword constructor cannot infer `nv` without `v` | Collective effects and spin | P4 | Crash or API hazard | 2026-10-31 (`7e40539`, main) |  |  |
| [LAT-06](LAT-06/index.md) | Multipole tilt assignment needing type promotion errors | Lattice and API | P4 | Crash or API hazard | 2026-10-31 (`35fe016`, main) |  |  |
| [LAT-08](LAT-08/index.md) | `track` rejects the documented vector `weight` | Lattice and API | P4 | Crash or API hazard | 2026-10-31 (`ae9423d`, main) |  |  |
| [LAT-09](LAT-09/index.md) | `track(...; use_KA=true)` on CPU errors; SIMD default ignores the override | Lattice and API | P4 | Crash or API hazard | 2026-10-31 (`ae9423d`, main) |  |  |
| [LAT-10](LAT-10/index.md) | Exported `ParticleView` cannot be constructed <span class="bb-tag">partly-source-only</span> | Lattice and API | P4 | Crash or API hazard | 2026-10-31 (`0bf3585`, main) |  |  |
| [LAT-12](LAT-12/index.md) | `writebl` fails for any electric multipole | Lattice and API | P4 | Crash or API hazard | 2026-10-31 (`35fe016`, main) |  |  |
| [LAT-19](LAT-19/index.md) | Bmad SciBmad writer emits property names Beamlines does not define (`bsol_field`, `roll`, `is_on`, controlled `z_rot` and `t_offset`) <span class="bb-tag">partly-source-only</span> | Lattice and API | P4 | Crash or API hazard | 2026-10-22 (`19b8afb`) |  |  |
| [LAT-20](LAT-20/index.md) | Bmad SciBmad writer emits `reverse(ele)` without a warning; no such method exists <span class="bb-tag">known-limitation</span> | Lattice and API | P4 | Crash or API hazard | 2026-10-22 (`19b8afb`) |  |  |
| [LAT-22](LAT-22/index.md) | Bmad SciBmad writer's Context `c1` overwrites a Bmad element named `c1` | Lattice and API | P4 | Crash or API hazard | 2026-10-30 (`19b8afb`) |  |  |
| [LAT-24](LAT-24/index.md) | A ForwardDiff or GTPSA scalar combined with a Float32 `BatchParam` throws | Lattice and API | P4 | Crash or API hazard | 2026-10-29 (`0bf3585`, main) |  |  |
| [TRK-11](TRK-11/index.md) | Sagan physics guide: edge kicks ×2, contradictory signs, ponderomotive H sign | Tracking | P4 | Documentation or example mismatch | 2026-10-31 (`ae9423d`, main) |  |  |
| [TRK-15](TRK-15/index.md) | Stale docstrings: `Bunch` callback signature, `SciBmadStandard` "Yoshida" | Tracking | P4 | Documentation or example mismatch | 2026-10-29 (`0bf3585`, main) |  | [[gh:https://github.com/bmad-sim/BeamTracking.jl/pull/370]] |
| [TRK-22](TRK-22/index.md) | Neutron g_spin conflicts with its own-mass normalization contract | Tracking | P4 | Documentation or example mismatch | 2026-10-29 (`750a9f2`, main) |  |  |
| [OPT-15](OPT-15/index.md) | Allocating `newton`/`brent` document caller mutation they do not do | Optics and maps | P4 | Documentation or example mismatch | 2026-10-31 (`ac41c09`, main) |  |  |
| [COL-09](COL-09/index.md) | AGS spin-ramp example: plotted time span and crossing-rate factor | Collective effects and spin | P4 | Documentation or example mismatch | 2026-10-31 (`ae9423d`, main) |  |  |
| [LAT-13](LAT-13/index.md) | `Bunch` docstring gives a three-argument callback; tracking calls nine | Lattice and API | P4 | Documentation or example mismatch | 2026-10-22 (`65e343f`; not on main) |  | [[gh:https://github.com/bmad-sim/BeamTracking.jl/pull/370]] |
| [LAT-15](LAT-15/index.md) | GPU batch documentation assigns a raw array instead of a `BatchParam` <span class="bb-tag">partly-source-only</span> | Lattice and API | P4 | Documentation or example mismatch | 2026-10-31 (`ae9423d`, main) |  |  |
| [LAT-16](LAT-16/index.md) | `nonlinear-twiss.ipynb` Python example has a syntax error | Lattice and API | P4 | Documentation or example mismatch | 2026-10-31 (`ae9423d`, main) |  |  |
| [LAT-17](LAT-17/index.md) | `ags-spin-ramp.ipynb` Python example calls the removed `rotate_spins` | Lattice and API | P4 | Documentation or example mismatch | 2026-10-31 (`ae9423d`, main) |  |  |
| [OPT-09](OPT-09/index.md) | An invalid `getterm` throws `UndefVarError: s` | Optics and maps | P5 | Crash or API hazard | 2026-10-22 (`62399f9`; not on main) | [[gh:https://github.com/bmad-sim/SciBmad.jl/issues/150]] | [[gh:https://github.com/bmad-sim/SciBmad.jl/pull/151]]<br>[[gh:https://github.com/bmad-sim/SciBmad.jl/pull/158]] |
| [LAT-14](LAT-14/index.md) | `TrackingResult.elapsed_time` documented in minutes, stores seconds | Lattice and API | P5 | Documentation or example mismatch | 2026-10-31 (`ae9423d`, main) |  |  |
