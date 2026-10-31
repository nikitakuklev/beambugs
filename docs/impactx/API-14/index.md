---
title: "API-14: Harmonic-only MAD-X ring cavities import at zero frequency"
hide:
  - toc
---

# API-14 — Harmonic-only MAD-X ring cavities import at zero frequency

| | |
|---|---|
| Code | [ImpactX](../index.md) · Python API |
| Priority | P2 |
| Severity | Wrong physics |
| Evidence | Reproduced, independently rechecked |
| Last reproduced | 2026-10-29 on ImpactX development (`fa0a43b`) |
| Tags | <span class="bb-tag">known-limitation</span> [?](../../methodology.md#tags) |
| Source | [`MADXParser.py#L746-L763`](https://github.com/BLAST-ImpactX/impactx/blob/b701a6e9d1f9e30cefb9a3e18e82ae7e02b9d943/src/python/impactx/MADXParser.py#L746-L763)<br>[`_parse_beam_command()`](https://github.com/BLAST-ImpactX/impactx/blob/b701a6e9d1f9e30cefb9a3e18e82ae7e02b9d943/src/python/impactx/MADXParser.py#L2243-L2254)<br>[`getFreq0()`](https://github.com/BLAST-ImpactX/impactx/blob/b701a6e9d1f9e30cefb9a3e18e82ae7e02b9d943/src/python/impactx/MADXParser.py#L2668-L2670)<br>[RF branch](https://github.com/BLAST-ImpactX/impactx/blob/b701a6e9d1f9e30cefb9a3e18e82ae7e02b9d943/src/python/impactx/madx_to_impactx.py#L1480-L1553)<br>[public import contract](https://github.com/BLAST-ImpactX/impactx/blob/615921c6867c895cf3a8fcf1d12991079d0f7d75/docs/source/usage/python.rst#L1139-L1246) |

Importing into ImpactX a MAD-X ring whose `RFCAVITY` gives only the harmonic number h (`HARMON`), without `FREQ` or a beam `FREQ0`, creates a `ShortRF` with zero frequency and prints a warning. MAD-X derives the frequency from the ring circumference C as f = hβc/C, and native MAD-X does so for the same decks: a 1.4 GeV (total energy) proton ring with C = 73 m and h = 17 needs 51.82 MHz. For a 0.73 MV cavity at zero crossing, the slope of the energy kick along the bunch is 0 instead of −7.630×10⁻⁴ m⁻¹, so the cavity gives no longitudinal focusing. Cavities with an explicit `FREQ` or `FREQ0` import correctly; a source comment requires `HARMON` and `FREQ0` together, but the importer's documentation does not.

??? example "Reproducer: API-14.py"

    [Download API-14.py](API-14.py)

    ```python
    """API-14: Harmonic-only MAD-X ring cavities import at zero frequency

    Importing into ImpactX a MAD-X ring whose `RFCAVITY` gives only the harmonic number h (`HARMON`),
    without `FREQ` or a beam `FREQ0`, creates a `ShortRF` with zero frequency and prints a warning.
    MAD-X derives the frequency from the ring circumference C as f = hβc/C, and native MAD-X does so for
    the same decks: a 1.4 GeV (total energy) proton ring with C = 73 m and h = 17 needs 51.82 MHz. For a
    0.73 MV cavity at zero crossing, the slope of the energy kick along the bunch is 0 instead of
    −7.630×10⁻⁴ m⁻¹, so the cavity gives no longitudinal focusing. Cavities with an explicit `FREQ` or
    `FREQ0` import correctly; a source comment requires `HARMON` and `FREQ0` together, but the
    importer's documentation does not.

    The script writes four closed MAD-X rings of four 90° bends and four straight sections, two of them
    as `SEQUENCE` layouts with gaps, for protons at 1.4 and 4.7 GeV and electrons at 0.05 GeV, with C =
    73 or 146 m, h = 17 to 51 and three RF phases. It imports each complete ring with a harmonic-only,
    an explicit-`FREQ` and an explicit-`FREQ0` cavity, injects 16 particles just before the RF kick and
    compares their six coordinates and the kick slope with the energy-kick map at f = hβc/C; native
    MAD-X, run through cpymad, must resolve the same frequency. The explicit-frequency cavities are the
    controls.
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
    os.sched_setaffinity(0, {min(os.sched_getaffinity(0))})

    import importlib.metadata
    import json
    import math
    import sys
    import warnings
    from pathlib import Path
    from tempfile import TemporaryDirectory

    import numpy as np
    from impactx import Config, ImpactX, RefPart, elements, read_beam

    ROOT = Path(__file__).resolve().parent
    C = 299792458.0
    N = 16


    def emit(**row):
        print(json.dumps(row), flush=True)


    def deck(name, text):
        path = ROOT / (name + ".madx")
        path.write_text(text)
        return str(path)


    def load(sim, path, floor="linear", slices=1):
        sim.lattice.clear()
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            ref = RefPart()
            read_beam(ref, path)
            sim.lattice.load_file(path, nslice=slices, min_model=floor)
        return ref, [str(w.message) for w in caught]


    def track(sim, ref, q):
        """Coordinates are (x, px, y, py, t, pt), with pt = -delta(E)/(P0*c)."""
        b = sim.beam
        b.clear_particles()
        b.set_ref_particle(ref)
        b.add_n_particles(*q[:, [0, 2, 4, 1, 3, 5]].T.copy(), qm=ref.qm_ratio_SI, w=np.ones(N))
        sim.track_particles()
        frame = b.to_df(local=True).reset_index(drop=True).sort_values("idcpu")
        assert len(frame) == N
        return frame[
            ["position_x", "momentum_x", "position_y", "momentum_y", "position_t", "momentum_t"]
        ].to_numpy(), b.ref.copy()


    def ring_text(
        energy, circumference, harmonic, voltage, lag, mode, frequency, species="proton", sequence=False
    ):
        beam_extra = f",freq0={frequency / harmonic / 1e6:.17g}" if mode == "freq0" else ""
        rf_extra = f",freq={frequency / 1e6:.17g}" if mode == "freq" else ""
        begin = f"beam,particle={species},energy={energy}{beam_extra};\n"
        bend = f"b:sbend,l={circumference / 8},angle=pi/2;\n"
        rf = f"rf:rfcavity,volt={voltage},harmon={harmonic},lag={lag}{rf_extra};\n"
        if sequence:
            # Four bends plus true SEQUENCE gaps: the declared circumference must count.
            layout = f"""ring:sequence,l={circumference},refer=entry;
    b,at=0; b,at={circumference / 4}; b,at={circumference / 2};
    b,at={3 * circumference / 4}; rf,at={circumference}; endsequence;
    """
        else:
            layout = f"d:drift,l={circumference / 8};\nring:line=(b,d,b,d,b,d,b,d,rf);\n"
        return begin + bend + rf + layout + "use,period=ring;\n"


    def check_rf(sim):
        # Each closed ring has four 90-degree bends and four straight sections.
        specs = [
            (1.4, 73.0, 17, 0.0, "proton", False),
            (1.4, 73.0, 34, 0.0, "proton", True),
            (4.7, 146.0, 51, 0.125, "proton", False),
            (0.05, 73.0, 17, -0.19, "electron", True),
        ]
        for energy, circumference, harmonic, lag, species, sequence in specs:
            # Obtain mass through a BEAM-only import, avoiding duplicated mass constants.
            beamfile = deck("beam_for_rf", f"beam,particle={species},energy={energy};\n")
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                ref = RefPart()
                read_beam(ref, beamfile)
            frequency = harmonic * ref.beta * C / circumference
            q = np.zeros((N, 6))
            q[:, 4] = np.linspace(-0.047, 0.053, N)
            # Nonzero incoming energy and momenta test the accelerating normalization too.
            q[:, 5] = np.linspace(-2e-4, 3e-4, N)
            q[:, 1] = 2e-5
            q[:, 3] = -3e-5
            for mode in ("implicit", "freq", "freq0"):
                text = ring_text(
                    energy, circumference, harmonic, 0.73, lag, mode, frequency, species, sequence
                )
                tag = f"ring_{species}_E{energy}_C{circumference}_h{harmonic}_{mode}"
                path = deck(tag, text)
                ref, messages = load(sim, path)
                imported_length = sum(float(e.ds) for e in sim.lattice)
                assert abs(imported_length - circumference) < 1e-12
                cavities = [e.copy() for e in sim.lattice if isinstance(e, elements.ShortRF)]
                assert len(cavities) == 1
                actual_freq = cavities[0].freq
                # Inject immediately before RF: transport in the ring cannot obscure its kick.
                sim.lattice.clear()
                sim.lattice.extend(cavities)
                out, final_ref = track(sim, ref, q)
                mass = ref.mass_MeV
                initial_pc = mass * ref.beta_gamma
                gain_ref = 0.73 * math.sin(2 * math.pi * lag)
                final_energy = 1000 * energy + gain_ref
                final_pc = math.sqrt(final_energy**2 - mass**2)
                gain = 0.73 * np.sin(2 * math.pi * (lag + frequency * q[:, 4] / C))
                oracle = q.copy()
                oracle[:, [1, 3]] *= initial_pc / final_pc
                oracle[:, 5] = (initial_pc * q[:, 5] - gain + gain_ref) / final_pc
                err = float(np.max(np.abs(out - oracle)))
                # Cancellation in the off-crest energy difference sets an absolute
                # roundoff scale even when the final energy deviation is near zero.
                roundoff = float(
                    8
                    * np.finfo(float).eps
                    * np.max(np.abs(initial_pc * q[:, 5]) + np.abs(gain) + abs(gain_ref))
                    / final_pc
                )
                assert abs(final_ref.kin_energy_MeV + mass - final_energy) < 2e-12
                assert err > 1e-5 if mode == "implicit" else err < roundoff
                assert (
                    actual_freq == 0 if mode == "implicit" else abs(actual_freq / frequency - 1) < 3e-16
                )
                assert any("resulting RF frequency is 0" in w for w in messages) == (mode == "implicit")
                derivative_q = np.zeros((N, 6))
                derivative_q[:, 4] = np.tile([-1e-4, 1e-4], N // 2)
                derivative_out, _ = track(sim, ref, derivative_q)
                measured_slope = float((derivative_out[1, 5] - derivative_out[0, 5]) / 2e-4)
                expected_slope = (
                    -0.73
                    * math.cos(2 * math.pi * lag)
                    * math.sin(2 * math.pi * frequency / C * 1e-4)
                    / 1e-4
                    / final_pc
                )
                assert (
                    abs(measured_slope) < 1e-15
                    if mode == "implicit"
                    else abs(measured_slope / expected_slope - 1) < 2e-10
                )
                emit(
                    case="rf",
                    mode=mode,
                    energy=energy,
                    species=species,
                    circumference=circumference,
                    harmonic=harmonic,
                    lag=lag,
                    sequence=sequence,
                    imported_length=imported_length,
                    expected_frequency=frequency,
                    actual_frequency=actual_freq,
                    max_error=err,
                    pt_observed=float(out[0, 5]),
                    pt_oracle=float(oracle[0, 5]),
                    gain_ref_MeV=gain_ref,
                    pt_per_m_observed=measured_slope,
                    pt_per_m_oracle=expected_slope,
                    warnings=messages,
                )


    def native():
        from cpymad.madx import Madx

        emit(case="native_version", cpymad=importlib.metadata.version("cpymad"))
        for path in sorted(ROOT.glob("ring_*_implicit.madx")):
            with Madx(stdout=False) as madx:
                madx.call(str(path))
                madx.twiss(betx=1, bety=1)
                h = float(madx.elements.rf.harmon)
                circ, beta = float(madx.beam.circ), float(madx.beam.beta)
                actual = float(madx.elements.rf.freq) * 1e6
                expected = h * beta * C / circ
                assert abs(actual / expected - 1) < 4e-16
                emit(
                    case="native_rf",
                    deck=path.name,
                    circumference=circ,
                    harmonic=h,
                    beta=beta,
                    expected_frequency=expected,
                    actual_frequency=actual,
                )


    def run():
        global ROOT
        emit(
            case="environment",
            python=sys.version.split()[0],
            impactx=importlib.metadata.version("impactx-noacc"),
            numpy=np.__version__,
            scipy=importlib.metadata.version("scipy"),
            build=Config.to_dict(),
        )
        print(
            "Oracle: f=h*beta*c/C; G(t)=0.73 MeV*sin(2*pi*(lag+f*t/c)); "
            "pt_f=(P_i*c*pt_i-G(t)+G(0))/(P_f*c).",
            flush=True,
        )
        config = Config.to_dict()
        assert not any(config[k] for k in ("have_gpu", "have_mpi", "have_omp"))
        assert config["precision"] == config["precision_particles"] == "DOUBLE"
        with TemporaryDirectory(prefix="api-14-", dir=ROOT) as directory:
            ROOT = Path(directory)
            sim = ImpactX()
            try:
                sim.tiny_profiler = sim.memory_profiler = False
                sim.particle_shape = 1
                sim.space_charge = "off"
                sim.csr = sim.isr = sim.spin = sim.diagnostics = sim.slice_step_diagnostics = False
                sim.verbose = 0
                sim.init_grids()
                check_rf(sim)
            finally:
                sim.finalize()
            native()
        print("DEFECT REPRODUCED", flush=True)


    if __name__ == "__main__":
        run()

    ```

## Provenance

| Role | Model | Effort | Date | Code examined |
|---|---|---|---|---|
| Found | `gpt-6-astra` | xhigh | 2026-10-29 | 26.10 (`615921c6867c`), impactx-noacc wheel, CPU, double, serial |
| Confirmed | `gpt-6.1-sol` | xhigh | 2026-10-29 | 26.10 (`615921c6867c`), impactx-noacc wheel, CPU, double, serial |

