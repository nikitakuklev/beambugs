---
title: "COL-44: An accelerating RFCW without a longitudinal wake evaluates its longitudinal space charge at the entrance momentum"
hide:
  - toc
---

# COL-44 — An accelerating RFCW without a longitudinal wake evaluates its longitudinal space charge at the entrance momentum

| | |
|---|---|
| Code | [Elegant](../index.md) · Collective effects |
| Priority | P1 |
| Severity | Wrong physics |
| Evidence | Reproduced, independently rechecked |
| Last reproduced | 2026-11-04 on Elegant master (`7514db3`) |
| Source | [kick split, simple_rfca.c:525–532](https://github.com/rtsoliday/elegant/blob/0355dd6aca12c861971fbf8f3ae78a0ded991734/src/simple_rfca.c#L525-L532)<br>[LSC call, simple_rfca.c:792–793](https://github.com/rtsoliday/elegant/blob/0355dd6aca12c861971fbf8f3ae78a0ded991734/src/simple_rfca.c#L792-L793)<br>[impedance, lsc.c:625–628](https://github.com/rtsoliday/elegant/blob/0355dd6aca12c861971fbf8f3ae78a0ded991734/src/lsc.c#L625-L628)<br>[lsc.c:502](https://github.com/rtsoliday/elegant/blob/0355dd6aca12c861971fbf8f3ae78a0ded991734/src/lsc.c#L502)<br>[reference update, simple_rfca.c:1116–1117](https://github.com/rtsoliday/elegant/blob/0355dd6aca12c861971fbf8f3ae78a0ded991734/src/simple_rfca.c#L1116-L1117)<br>[simple_rfca.c:1207](https://github.com/rtsoliday/elegant/blob/0355dd6aca12c861971fbf8f3ae78a0ded991734/src/simple_rfca.c#L1207)<br>[1278](https://github.com/rtsoliday/elegant/blob/0355dd6aca12c861971fbf8f3ae78a0ded991734/src/simple_rfca.c#L1278)<br>[wake.c:318–319](https://github.com/rtsoliday/elegant/blob/0355dd6aca12c861971fbf8f3ae78a0ded991734/src/wake.c#L318-L319)<br>[RFCW manual, rfcw.tex:14–19](https://github.com/rtsoliday/elegant/blob/0355dd6aca12c861971fbf8f3ae78a0ded991734/doc/rfcw.tex#L14-L19) |

With LSC=1 and no longitudinal wake updating the reference between kicks, an accelerating RFCW evaluates its longitudinal space-charge impedance at the entrance momentum throughout the cavity; increasing N_KICKS does not update that momentum. For a 0.8 m cavity accelerating an electron bunch of 160 μm rms length and 240 μm transverse rms size from γ = 6 to 30, the +1σ energy kick is 3.590 times the local-momentum result. Splitting it into 480 one-kick RFCWs with CHANGE_P0=1 gives a result within 0.3468%. An enabled longitudinal wake with CHANGE_P0=1 also updates the reference between kicks, avoiding the frozen entrance momentum.

??? example "Reproducer: COL-44.py"

    [Download COL-44.py](COL-44.py)

    ```python
    """COL-44: An accelerating RFCW without a longitudinal wake evaluates its longitudinal space charge at the entrance momentum

    With LSC=1 and no longitudinal wake updating the reference between kicks, an accelerating RFCW
    evaluates its longitudinal space-charge impedance at the entrance momentum throughout the
    cavity; increasing N_KICKS does not update that momentum. For a 0.8 m cavity accelerating an
    electron bunch of 160 μm rms length and 240 μm transverse rms size from γ = 6 to 30, the +1σ
    energy kick is 3.590 times the local-momentum result. Splitting it into 480 one-kick RFCWs with
    CHANGE_P0=1 gives a result within 0.3468%. An enabled longitudinal wake with CHANGE_P0=1 also
    updates the reference between kicks, avoiding the frozen entrance momentum.

    The script tracks 65536 electrons (Gaussian quantiles, sigma_ct = 160 um, rms sizes 240 um,
    1 fC) through a 0.8 m RFCW with FREQ=0 and PHASE=90, which accelerates every particle uniformly
    from gamma = 6 to 30, with LSC=1, 1024 LSC bins and 480 kicks, and through the same cavity split
    into 480 one-kick RFCWs with CHANGE_P0=1. It rebuilds both results in NumPy from Elegant's
    periodic LSC kick, summed over the 480 kicks with the reference momentum each one uses, and
    compares the single cavity with the same sum at the local momentum of each kick. The control runs
    the cavity with LSC=0.
    """

    import numpy as np
    from _elegant import verdict
    from _lsc import ME_EV, RADIUS_FACTOR, C, at, energy, periodic_kick, round_beam, track

    COUNT, SIGMA, SIGMA_R, CHARGE, BINS = 65536, 160e-6, 240e-6, 1e-15, 1024
    LENGTH, KICKS, GAMMA_IN, GAMMA_OUT = 0.8, 480, 6.0, 30.0
    RB = RADIUS_FACTOR * SIGMA_R
    VOLT = (GAMMA_OUT - GAMMA_IN) * ME_EV  # FREQ=0, PHASE=90: every kick adds VOLT/KICKS
    P_IN = float(np.sqrt(GAMMA_IN**2 - 1))
    RTOL = 1e-3  # rebuilt sums against Elegant; 1 fC makes the bunch's own redistribution negligible
    MIN_RATIO = 1.1  # defect if the single cavity exceeds the local-momentum sum by more than 10%

    BEAM = round_beam(COUNT, SIGMA, SIGMA_R, P_IN)
    Z = C * BEAM["t"]


    def cavity(segments: int, lsc: int) -> str:
        """`segments` equal RFCWs of KICKS/segments kicks each, together the 0.8 m cavity."""
        return (
            f"R: RFCW,L={LENGTH / segments!r},CELL_LENGTH={LENGTH / segments!r},"
            f"VOLT={VOLT / segments!r},FREQ=0,PHASE=90,CHANGE_P0=1,N_KICKS={KICKS // segments},"
            f"LSC={lsc},LSC_BINS={BINS},ZWAKE=0,TRWAKE=0"
        )


    def lsc_kick(segments: int, lsc: int = 1) -> np.ndarray:
        """Energy change (eV) of each particle beyond the common gain VOLT.

        The bunch is symmetric about z = 0 and the LSC kick antisymmetric, so pairing particle i
        with its mirror image removes the rounding of the 12 MeV gain.
        """
        out = track(
            f"Q: CHARGE,TOTAL={CHARGE!r}\n{cavity(segments, lsc)}",
            f"Q,{segments}*R",
            BEAM,
            P_IN,
        )
        gain = energy(out["p"]) - energy(P_IN) - VOLT
        return (gain - gain[::-1]) / 2


    def rebuilt(momenta: np.ndarray) -> np.ndarray:
        """Sum of Elegant's periodic LSC kick over the kicks, each of length LENGTH/KICKS, with the
        momentum in the impedance taken from `momenta` (one per kick)."""
        total = np.zeros(COUNT)
        for p in np.unique(momenta):
            count = int(np.sum(momenta == p))
            total += count * periodic_kick(Z, float(p), RB, CHARGE, LENGTH / KICKS, BINS)[0]
        return total


    # Momentum at each kick. FREQ=0 makes gamma rise by (GAMMA_OUT - GAMMA_IN)/KICKS per kick, each
    # RF kick preceding its LSC kick. One cavity: the reference stays at the entrance momentum until
    # CHANGE_P0 acts at the exit. 480 one-kick cavities: each LSC kick uses the momentum at the
    # entrance of its cavity (the previous cavity's CHANGE_P0). Local physics: the impedance
    # integrated along the cavity with the beam's momentum at each point, here by the midpoint rule,
    # gamma = GAMMA_IN + (k + 1/2) dgamma for kick k, whose error is second order in dgamma.
    step = (GAMMA_OUT - GAMMA_IN) / KICKS
    frozen = np.full(KICKS, P_IN)
    split = np.sqrt((GAMMA_IN + step * np.arange(KICKS)) ** 2 - 1)
    local = np.sqrt((GAMMA_IN + step * (np.arange(KICKS) + 0.5)) ** 2 - 1)
    expected = at(Z, rebuilt(local), SIGMA)

    control = np.max(np.abs(lsc_kick(1, lsc=0)))
    print(f"serial one RFCW, LSC=0 (control): max |antisymmetric energy change| = {control:.3g} eV")
    if control > 1e-4:
        raise SystemExit("the cavity without LSC does not give every particle the same gain")

    defect = False
    for label, segments, momenta in (
        ("one RFCW, 480 kicks", 1, frozen),
        ("480 one-kick RFCWs with CHANGE_P0=1", KICKS, split),
    ):
        observed = at(Z, lsc_kick(segments), SIGMA)
        model = at(Z, rebuilt(momenta), SIGMA)
        print(
            f"serial {label}: LSC kick at +1 sigma observed={observed:.9g} eV "
            f"expected (local momentum)={expected:.9g} eV ratio={observed / expected:.6f}; "
            f"rebuilt with the momenta Elegant uses {model:.9g} eV"
        )
        if abs(observed / model - 1) > RTOL:
            raise SystemExit(f"{label}: the rebuild does not reproduce Elegant")
        if segments == 1:
            defect = observed / expected > MIN_RATIO
    print(f"rebuild tolerance {RTOL:g} relative; defect if the one-cavity ratio exceeds {MIN_RATIO}")
    verdict(defect)

    ```

??? example "Helper: _elegant.py"

    [Download _elegant.py](_elegant.py)

    ```python
    """Shared runner for the Elegant reproducers: write inputs, run elegant or Pelegant, read SDDS output.

    `elegant` must be on PATH, and `Pelegant` with `mpirun` for MPI runs. RPN_DEFNS must name the
    `defns.rpn` file shipped with SDDS; elegant refuses to start without it. MPI runs default to three ranks
    (rank 0 coordinates, two workers track), oversubscribed so that they also run on fewer cores.
    Each run uses a fresh temporary directory and one numerical thread.
    """

    from __future__ import annotations

    import argparse
    import contextlib
    import io
    import os
    import shutil
    import subprocess
    import sys
    import tempfile
    from pathlib import Path
    from typing import Any

    import numpy as np
    import pysdds

    MPI_RANKS = 3
    TIMEOUT = 600  # seconds per native run


    class RunFailure(RuntimeError):
        """A native run that exited with a nonzero status; `log` is its full stdout and stderr."""

        def __init__(self, program: str, status: int, log: str):
            super().__init__(f"{program} exited with status {status}:\n{log[-3000:]}")
            self.program, self.status, self.log = program, status, log


    def modes(mpi_only: bool = False) -> tuple[str, ...]:
        """Execution modes from the command line: serial, plus 3-rank Pelegant with --mpi.

        An MPI-only defect always runs Pelegant; --mpi is then implied.
        """
        parser = argparse.ArgumentParser()
        parser.add_argument("--mpi", action="store_true", help="also run 3-rank Pelegant")
        args = parser.parse_args()
        if mpi_only:
            return ("mpi",)
        return ("serial", "mpi") if args.mpi else ("serial",)


    def write_beam(path: Path, beam: Any) -> None:
        """Binary SDDS particle file; Pelegant reads only binary input.

        `beam` is one page or a list of pages. In a page, array values are columns (integer arrays
        such as particleID become SDDS long) and scalar values are parameters (for example
        IDSlotsPerBunch); all pages need the same keys. Columns x, xp, y, yp, t, p default to zero;
        p is in units of m c of the species.
        """
        pages: list[dict[str, Any]] = [beam] if isinstance(beam, dict) else list(beam)
        arrays = [{k: np.asarray(v) for k, v in page.items() if np.ndim(v) == 1} for page in pages]
        counts = [len(next(iter(a.values()))) for a in arrays]
        units = {"x": "m", "xp": "", "y": "m", "yp": "", "t": "s", "p": "m$be$nc"}
        sdds = pysdds.SDDSFile(add_data_nm=True)
        sdds.mode = "binary"
        sdds.n_pages = len(pages)
        for name in dict.fromkeys([*units, *arrays[0]]):
            data = [a.get(name, np.zeros(n)) for a, n in zip(arrays, counts, strict=True)]
            if np.issubdtype(data[0].dtype, np.integer):
                sdds.add_column(name, "long", data=[d.astype(np.int32) for d in data])
            else:
                data = [d.astype(np.float64) for d in data]
                sdds.add_column(name, "double", units=units.get(name, ""), data=data)
        for name, value in pages[0].items():
            if np.ndim(value) != 0:
                continue
            values = [page[name] for page in pages]
            # pysdds checks the exact NumPy scalar type, despite its Python-scalar annotation.
            if isinstance(value, (int, np.integer)):
                data = [np.int32(v) for v in values]
                sdds.add_parameter(name, "long", data=data)  # ty: ignore[invalid-argument-type]
            else:
                sdds.add_parameter(name, "double", data=[np.float64(v) for v in values])
        # pysdds 0.6.0 prints every parameter value while validating; keep the output clean.
        with contextlib.redirect_stdout(io.StringIO()):
            pysdds.write(sdds, path, overwrite=True)


    def run(
        files: dict[str, str],
        mode: str = "serial",
        beams: dict[str, Any] | None = None,
        command: str = "run.ele",
        outputs: tuple[str, ...] = ("out.sdds",),
        *,
        ranks: int = MPI_RANKS,
    ) -> dict:
        """Run one case and return {output name: pysdds file} plus "log" (stdout and stderr).

        files: text inputs by name (lattice, command file, tables); beams: particle pages by file
        name (see write_beam), written as binary SDDS. ranks: Pelegant rank count, including rank 0.
        A nonzero exit status raises RunFailure; a timeout raises RuntimeError.
        """
        if "RPN_DEFNS" not in os.environ:
            sys.exit("Set RPN_DEFNS to the defns.rpn file of SDDS (see the elegant installation).")
        program = "elegant" if mode == "serial" else "Pelegant"
        if shutil.which(program) is None or (mode == "mpi" and shutil.which("mpirun") is None):
            sys.exit(f"{program}{' and mpirun' if mode == 'mpi' else ''} must be on PATH.")
        argv = [program, command]
        if mode == "mpi":
            argv = [
                "mpirun",
                "--oversubscribe",
                "--bind-to",
                "none",
                "-x",
                "RPN_DEFNS",
                "-x",
                "OMP_NUM_THREADS",
                "-n",
                str(ranks),
                *argv,
            ]
        env = dict(os.environ, OMP_NUM_THREADS="1", OPENBLAS_NUM_THREADS="1", MKL_NUM_THREADS="1")
        with tempfile.TemporaryDirectory(prefix="elegant-repro-") as tmp:
            work = Path(tmp)
            for name, text in files.items():
                (work / name).write_text(text)
            for name, columns in (beams or {}).items():
                write_beam(work / name, columns)
            try:
                done = subprocess.run(
                    argv,
                    cwd=work,
                    env=env,
                    capture_output=True,
                    text=True,
                    timeout=TIMEOUT,
                    check=False,
                )
            except subprocess.TimeoutExpired:
                raise RuntimeError(f"{program} exceeded {TIMEOUT} s") from None
            log = done.stdout + done.stderr
            if done.returncode:
                raise RunFailure(program, done.returncode, log)
            result: dict = {"log": log}
            for name in outputs:
                result[name] = pysdds.read(work / name)
        return result


    def particles(sdds, page: int = 0) -> dict[str, np.ndarray]:
        """Columns of one page, rows ordered by particleID when present (MPI output order varies).

        Raises RuntimeError when a coordinate column (x, xp, y, yp, t, p) holds a nonfinite value: such
        output is an invalid run, not evidence for a numerical discrepancy.
        """
        names = [c.name for c in sdds.columns]
        data = {n: np.asarray(sdds[page, n]) for n in names}
        for n in ("x", "xp", "y", "yp", "t", "p"):
            if n in data and not np.all(np.isfinite(data[n])):
                raise RuntimeError(f"Elegant output column {n} holds nonfinite values")
        if "particleID" in data:
            order = np.argsort(data["particleID"], kind="stable")
            data = {n: v[order] for n, v in data.items()}
        return data


    def verdict(defect: bool) -> None:
        print("DEFECT REPRODUCED" if defect else "NOT REPRODUCED")

    ```

??? example "Helper: _lsc.py"

    [Download _lsc.py](_lsc.py)

    ```python
    """Shared setup for the longitudinal space-charge reproducers: Elegant's physical constants,
    deterministic round electron beams, one-run tracking, Elegant's periodic FFT kick rebuilt in NumPy,
    and the open-boundary (isolated-bunch) kick of the same uniform-disk model.

    Lengths are in m, momenta p = beta*gamma in units of m_e c, energies in eV, and z = c t grows
    toward the tail of the bunch. Every model here uses p, as Elegant's impedance does (its Bessel
    argument is k r_b / p), so the comparisons isolate the boundary condition and the reference
    momentum rather than the p-versus-gamma choice.
    """

    from __future__ import annotations

    import numpy as np
    from _elegant import particles, run
    from scipy.integrate import quad
    from scipy.special import erf, k1, ndtri

    # SDDS constants.h, the values Elegant computes with.
    C = 2.99792458e8
    EPSILON_0 = 1e7 / (4 * np.pi * C**2)
    Z0 = 4e-7 * np.pi * C
    ME_EV = 0.51099906e6
    RADIUS_FACTOR = 1.7  # Elegant's default: r_b = 1.7 (sigma_x + sigma_y)/2


    def round_beam(count: int, sigma_ct: float, sigma_r: float, p: float) -> dict:
        """Gaussian quantiles in z = c t with rms sigma_ct, all at momentum p, zero slopes.

        Transverse positions lie on a golden-angle circle rescaled to rms sigma_r exactly in x and y,
        so Elegant's effective radius is RADIUS_FACTOR * sigma_r. Quantiles (i + 1/2)/count make the
        bunch symmetric about z = 0.
        """
        z = sigma_ct * ndtri((np.arange(count) + 0.5) / count)
        phi = np.arange(count) * np.pi * (np.sqrt(5) - 1)
        x, y = np.cos(phi), np.sin(phi)
        x = (x - x.mean()) / x.std() * sigma_r
        y = (y - y.mean()) / y.std() * sigma_r
        return {
            "x": x,
            "y": y,
            "t": z / C,
            "p": np.full(count, p),
            "particleID": np.arange(1, count + 1),
        }


    def track(elements: str, line: str, beam: dict, p0: float) -> dict[str, np.ndarray]:
        """Serial run of `beam` through BL: LINE=(line); final coordinates ordered by particle ID."""
        commands = (
            "&global_settings echo_namelists=0 &end\n"
            f"&run_setup lattice=lattice.lte,use_beamline=BL,p_central={p0!r},output=out.sdds &end\n"
            "&run_control n_steps=1,n_passes=1 &end\n"
            "&sdds_beam input=beam.sdds &end\n"
            "&track &end\n"
        )
        lattice = elements.rstrip("\n") + f"\nBL: LINE=({line})\n"
        result = run({"lattice.lte": lattice, "run.ele": commands}, beams={"beam.sdds": beam})
        out = particles(result["out.sdds"])
        if len(out["p"]) != len(beam["p"]):
            raise SystemExit("particles were lost")
        return out


    def energy(p: np.ndarray | float) -> np.ndarray:
        """Total energy in eV of electrons with momentum p (m_e c)."""
        return ME_EV * np.sqrt(1 + np.asarray(p) ** 2)


    def periodic_kick(z: np.ndarray, p: float, rb: float, charge: float, length: float, bins: int):
        """Elegant's LSC energy kick (eV) at each particle, rebuilt in NumPy, and the grid period (m).

        As in lsc.c: bin width h = (z_max - z_min)/(bins - 3), centers at z_min + i h, nearest-center
        deposition, voltage V = IFFT[-i Z(k) FFT(counts)] q_m c/h over `bins` points with
        Z(k) = Z0 length/(pi k rb^2) [1 - a K1(a)], a = k rb/p, k = 2 pi f/c, zero at DC and Nyquist,
        and linear interpolation between bin centers; the kick is -V. A discrete transform of length
        `bins` is periodic, so this is the field of a train of copies spaced bins*h apart.
        """
        lower = z.min()
        h = (z.max() - lower) / (bins - 3)
        counts = np.bincount(np.floor((z - lower) / h + 0.5).astype(int), minlength=bins)
        k = 2 * np.pi * np.fft.rfftfreq(bins, h)
        impedance = np.zeros(len(k))
        a = k[1:] * rb / p
        impedance[1:] = Z0 * length / (np.pi * rb**2 * k[1:]) * (1 - a * k1(a))
        impedance[-1] = 0
        voltage = np.fft.irfft(-1j * impedance * np.fft.rfft(counts), bins) * charge / len(z) * C / h
        return -np.interp(z, lower + h * np.arange(bins), voltage), bins * h


    def open_shape(q: float, big_r: float) -> float:
        """Dimensionless on-axis field h(q, R) of an isolated Gaussian bunch of uniform disks.

        A disk of radius r_b carrying charge dQ, seen in its rest frame from a distance gamma*u on
        the axis, has the longitudinal field dQ/(2 pi eps0 r_b^2) [sign(u) - gamma u /
        sqrt(gamma^2 u^2 + r_b^2)]; E_z is unchanged by the boost. Summing over a Gaussian line
        density of rms sigma, with q = z/sigma and R = r_b/(gamma sigma):
            h(q, R) = erf(q/sqrt 2) - int phi(y) (q - y)/sqrt((q - y)^2 + R^2) dy,
        phi the unit normal density. Its Fourier transform is the impedance Z(k) above, with no
        periodic images.
        """
        integrand = lambda y: (
            np.exp(-y * y / 2) / np.sqrt(2 * np.pi) * (q - y) / np.sqrt((q - y) ** 2 + big_r**2)
        )
        smooth = quad(integrand, -14, 14, points=[q], epsabs=1e-14, epsrel=1e-13, limit=400)[0]
        return float(erf(q / np.sqrt(2)) - smooth)


    def open_kick(q: float, sigma: float, p: float, rb: float, charge: float, length: float) -> float:
        """Energy kick (eV) of an electron at z = q sigma from an isolated Gaussian bunch of total
        charge `charge` over `length`: -charge length/(2 pi eps0 rb^2) h(q, rb/(p sigma))."""
        return -charge * length / (2 * np.pi * EPSILON_0 * rb**2) * open_shape(q, rb / (p * sigma))


    def at(z: np.ndarray, kick: np.ndarray, z0: float) -> float:
        """Kick at z0, interpolated between the sorted particle positions (quantile beams)."""
        order = np.argsort(z)
        return float(np.interp(z0, z[order], kick[order]))

    ```

## Provenance

| Role | Model | Effort | Date | Code examined |
|---|---|---|---|---|
| Found | `claude-opus-5-5` | high | 2026-11-04 | master (`7514db36f764`), source with SDDS a3f730a14c49, CPU, serial |
| Confirmed (accelerating RFCW; the standalone off-reference LSCDRIFT sub-claim is the reference-momentum convention (Rejected candidates)) | `gpt-6.1-sol` | xhigh | 2026-11-04 | master (`7514db36f764`), source with SDDS a3f730a14c49, CPU, serial |

