---
title: "COL-45: Longitudinal space charge uses an unpadded FFT, so an isolated bunch gets the field of a train spaced by its own length"
hide:
  - toc
---

# COL-45 — Longitudinal space charge uses an unpadded FFT, so an isolated bunch gets the field of a train spaced by its own length

| | |
|---|---|
| Code | [Elegant](../index.md) · Collective effects |
| Priority | P3 |
| Severity | Wrong physics |
| Evidence | Reproduced, independently rechecked |
| Last reproduced | 2026-11-04 on Elegant master (`7514db3`) |
| Tags | <span class="bb-tag">partly-source-only</span> [?](../../methodology.md#tags) |
| Source | [LSCDRIFT window, lsc.c:145](https://github.com/rtsoliday/elegant/blob/0355dd6aca12c861971fbf8f3ae78a0ded991734/src/lsc.c#L145)<br>[forward FFT, lsc.c:246](https://github.com/rtsoliday/elegant/blob/0355dd6aca12c861971fbf8f3ae78a0ded991734/src/lsc.c#L246)<br>[impedance and inverse FFT, lsc.c:337–358](https://github.com/rtsoliday/elegant/blob/0355dd6aca12c861971fbf8f3ae78a0ded991734/src/lsc.c#L337-L358)<br>[CSRDRIFT calls, csbend.c:5212](https://github.com/rtsoliday/elegant/blob/0355dd6aca12c861971fbf8f3ae78a0ded991734/src/csbend.c#L5212)<br>[5229](https://github.com/rtsoliday/elegant/blob/0355dd6aca12c861971fbf8f3ae78a0ded991734/src/csbend.c#L5229)<br>[window, lsc.c:446](https://github.com/rtsoliday/elegant/blob/0355dd6aca12c861971fbf8f3ae78a0ded991734/src/lsc.c#L446)<br>[FFT, lsc.c:543](https://github.com/rtsoliday/elegant/blob/0355dd6aca12c861971fbf8f3ae78a0ded991734/src/lsc.c#L543)<br>[impedance and inverse FFT, lsc.c:625–642](https://github.com/rtsoliday/elegant/blob/0355dd6aca12c861971fbf8f3ae78a0ded991734/src/lsc.c#L625-L642)<br>[allocation, lsc.c:96–101](https://github.com/rtsoliday/elegant/blob/0355dd6aca12c861971fbf8f3ae78a0ded991734/src/lsc.c#L96-L101)<br>[LSCDRIFT manual, lscdrift.tex:1–6](https://github.com/rtsoliday/elegant/blob/0355dd6aca12c861971fbf8f3ae78a0ded991734/doc/lscdrift.tex#L1-L6)<br>[rfcw.tex:14–19](https://github.com/rtsoliday/elegant/blob/0355dd6aca12c861971fbf8f3ae78a0ded991734/doc/rfcw.tex#L14-L19) |

LSCDRIFT, and the longitudinal space charge of RFCW and CSRDRIFT, apply the impedance with a discrete Fourier transform over a window three bins longer than the bunch's sampled extent, without zero padding, so the field is that of a periodic train of bunch copies. For short, wide, low-energy bunches the images cancel much of the field. At Lorentz factor γ = 4, a 1 pC Gaussian bunch of 80 μm rms length and 2 mm rms size receives at +1σ a kick 21.43% weaker than the isolated-bunch field from a real-space Coulomb integral, and 6.004% weaker at γ = 15. More bins do not help; stretching the window to 80σ brings the kick within 0.09324% of the isolated value.

??? example "Reproducer: COL-45.py"

    [Download COL-45.py](COL-45.py)

    ```python
    """COL-45: Longitudinal space charge uses an unpadded FFT, so an isolated bunch gets the field of a train spaced by its own length

    LSCDRIFT, and the longitudinal space charge of RFCW and CSRDRIFT, apply the impedance with a
    discrete Fourier transform over a window three bins longer than the bunch's sampled extent, without
    zero padding, so the field is that of a periodic train of bunch copies. For short, wide, low-energy
    bunches the images cancel much of the field. At Lorentz factor γ = 4, a 1 pC Gaussian bunch of 80 μm
    rms length and 2 mm rms size receives at +1σ a kick 21.43% weaker than the isolated-bunch field from
    a real-space Coulomb integral, and 6.004% weaker at γ = 15. More bins do not help; stretching the
    window to 80σ brings the kick within 0.09324% of the isolated value.

    The script tracks 262144 electrons (Gaussian quantiles, sigma_ct = 80 um, rms sizes 2 mm,
    1 pC) through one thin LSCDRIFT (L=0, LEFFECTIVE=0.2 m, 2048 bins) at gamma = 4 and 15, and at
    gamma = 4 through a 0.2 m RFCW with VOLT=0 and LSC=1, and compares the energy kick at +1 sigma
    with the open-boundary field of the same uniform-disk model, an independent real-space integral
    with no FFT. A NumPy rebuild of Elegant's periodic transform reproduces every kick. The control
    adds two spectator particles at +-40 sigma, which barely change the open field but stretch the
    FFT period, and shows the kick converging to the open value.
    """

    import numpy as np
    from _elegant import verdict
    from _lsc import (
        EPSILON_0,
        RADIUS_FACTOR,
        C,
        at,
        energy,
        open_kick,
        periodic_kick,
        round_beam,
        track,
    )

    COUNT, SIGMA, SIGMA_R, CHARGE, LEFF = 262144, 80e-6, 2e-3, 1e-12, 0.2
    RB = RADIUS_FACTOR * SIGMA_R
    REBUILD_ATOL = 1e-6  # eV; the NumPy rebuild of the periodic FFT matches Elegant to ~1e-9 eV
    CONTROL_RTOL = 2e-3  # the long-period control must approach the open-boundary kick this closely
    MIN_DEFICIT = 0.01  # defect if the gamma = 4 kick misses the open-boundary kick by more than 1%


    def lsc_kick(beam: dict, p: float, element: str, charge: float) -> np.ndarray:
        """Energy change (eV) of each particle through the CHARGE + LSC element."""
        out = track(f"Q: CHARGE,TOTAL={charge!r}\nL1: {element}", "Q,L1", beam, p)
        return energy(out["p"]) - energy(p)


    def case(
        label: str,
        beam: dict,
        p: float,
        element: str,
        bins: int,
        open_value: float,
        charge: float = CHARGE,
    ) -> float:
        """Print the kick at +1 sigma against the open-boundary value; return observed/open - 1."""
        z = C * beam["t"]
        kick = lsc_kick(beam, p, element, charge)
        rebuilt, period = periodic_kick(z, p, RB, charge, LEFF, bins)
        rebuild_error = float(np.max(np.abs(kick - rebuilt)))
        if rebuild_error > REBUILD_ATOL:
            raise SystemExit(f"{label}: the periodic rebuild misses Elegant by {rebuild_error:g} eV")
        observed = at(z, kick, SIGMA)
        print(
            f"serial {label}: kick at +1 sigma observed={observed:.9g} eV "
            f"expected (open boundary)={open_value:.9g} eV ratio={observed / open_value:.6f}; "
            f"FFT period {period / SIGMA:.4f} sigma, periodic rebuild within {rebuild_error:.2g} eV"
        )
        return observed / open_value - 1


    defect = False
    for gamma in (4.0, 15.0):
        p = float(np.sqrt(gamma**2 - 1))
        beam = round_beam(COUNT, SIGMA, SIGMA_R, p)
        open_value = open_kick(1.0, SIGMA, p, RB, CHARGE, LEFF)
        deficit = case(
            f"LSCDRIFT gamma={gamma:g}",
            beam,
            p,
            f"LSCDRIFT,L=0,LEFFECTIVE={LEFF},BINS=2048,INTERPOLATE=1,SMOOTHING=0",
            2048,
            open_value,
        )
        if gamma == 4.0:
            defect = deficit < -MIN_DEFICIT
            # The RF cavity's LSC (addLSCKick) uses the same window. VOLT=0 and zero slopes leave the
            # bunch unchanged over the 0.2 m, so its one kick of length 0.2 m is the same thin kick;
            # 1 fC keeps the plasma length long enough for one kick (the cavity refuses a kick
            # spacing above a tenth of it), and the kicks scale linearly with charge.
            case(
                "RFCW gamma=4 (VOLT=0, one kick)",
                beam,
                p,
                f"RFCW,L={LEFF},CELL_LENGTH={LEFF},VOLT=0,FREQ=0,N_KICKS=1,LSC=1,LSC_BINS=2048,"
                "ZWAKE=0,TRWAKE=0",
                2048,
                open_value * 1e-3,
                charge=CHARGE * 1e-3,
            )
            # Control: two spectators at -+40 sigma, total charge and rms sizes unchanged, 16384
            # bins to keep the bin width. Their open-space field at +1 sigma is the single-disk
            # kernel sign(v) - v/sqrt(v^2 + R^2), v = 1 -+ 40 in units of sigma, R = r_b/(p sigma);
            # the Gaussian part keeps COUNT of the COUNT + 2 macroparticles.
            spect = {
                "x": np.r_[0.0, beam["x"], 0.0],
                "y": np.r_[0.0, beam["y"], 0.0],
                "t": np.r_[-40 * SIGMA / C, beam["t"], 40 * SIGMA / C],
                "p": np.full(COUNT + 2, p),
                "particleID": np.arange(1, COUNT + 3),
            }
            for axis in ("x", "y"):
                spect[axis] = spect[axis] / spect[axis].std() * SIGMA_R
            big_r = RB / (p * SIGMA)
            kernel = sum(np.sign(v) - v / np.hypot(v, big_r) for v in (41.0, -39.0))
            scale = -CHARGE * LEFF / (2 * np.pi * EPSILON_0 * RB**2)
            open_spect = (COUNT * open_value + scale * kernel) / (COUNT + 2)
            control = case(
                "LSCDRIFT gamma=4, spectators at -+40 sigma (control)",
                spect,
                p,
                f"LSCDRIFT,L=0,LEFFECTIVE={LEFF},BINS=16384,INTERPOLATE=1,SMOOTHING=0",
                16384,
                open_spect,
            )
            if abs(control) > CONTROL_RTOL:
                raise SystemExit("the long-period control does not approach the open-boundary kick")
    print(
        f"defect if the gamma=4 LSCDRIFT kick is more than {MIN_DEFICIT:.0%} below the open-boundary "
        f"kick; control tolerance {CONTROL_RTOL:g} relative"
    )
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
| Confirmed (isolated bunches whose periodic images have an appreciable field; LSCDRIFT executed, the window depends on extrema, macroparticle count and bins) | `gpt-6.1-sol` | xhigh | 2026-11-04 | master (`7514db36f764`), source with SDDS a3f730a14c49, CPU, serial |

