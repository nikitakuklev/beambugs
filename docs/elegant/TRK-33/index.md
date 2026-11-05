---
title: "TRK-33: KQUAD and KSEXT steering kicks radiate nothing horizontally and four times too much vertically"
hide:
  - toc
---

# TRK-33 — KQUAD and KSEXT steering kicks radiate nothing horizontally and four times too much vertically

| | |
|---|---|
| Code | [Elegant](../index.md) · Tracking |
| Priority | P3 |
| Severity | Wrong physics |
| Evidence | Reproduced, independently rechecked |
| Last reproduced | 2026-11-04 on Elegant master (`7514db3`) |
| Tags | <span class="bb-tag">partly-source-only</span> [?](../../methodology.md#tags) |
| Source | [`integrate_kick_multipole_ordn`, multipole.c:1430–1444](https://github.com/rtsoliday/elegant/blob/0355dd6aca12c861971fbf8f3ae78a0ded991734/src/multipole.c#L1430-L1444)<br>[multipole.h:186–209](https://github.com/rtsoliday/elegant/blob/0355dd6aca12c861971fbf8f3ae78a0ded991734/src/multipole.h#L186-L209)<br>[multipole.c:1488–1510](https://github.com/rtsoliday/elegant/blob/0355dd6aca12c861971fbf8f3ae78a0ded991734/src/multipole.c#L1488-L1510)<br>[manual, kquad.tex:64–67](https://github.com/rtsoliday/elegant/blob/0355dd6aca12c861971fbf8f3ae78a0ded991734/doc/kquad.tex#L64-L67)<br>[kquad.tex:119–123](https://github.com/rtsoliday/elegant/blob/0355dd6aca12c861971fbf8f3ae78a0ded991734/doc/kquad.tex#L119-L123)<br>[SYNCH_RAD, track_data.c:1171](https://github.com/rtsoliday/elegant/blob/0355dd6aca12c861971fbf8f3ae78a0ded991734/src/track_data.c#L1171) |

A KQUAD or KSEXT with SYNCH_RAD=1 and a steering kick HKICK or VKICK computes the radiating field after subtracting the horizontal kick and doubling the vertical one. For a 0.8 m magnet with zero main strength and a 1 mrad kick at 3.066 GeV, the classical radiation integral along the uniform steering field gives a fractional energy loss of 5.072×10⁻¹⁰ in either plane; HKICK gives zero and VKICK 2.029×10⁻⁹, four times too much. The moments_output quantum excitation of the KQUAD is zero for HKICK and eight times the Sands diffusion integral for VKICK. This steering-field error is absent when HKICK and VKICK are both zero.

??? example "Reproducer: TRK-33.py"

    [Download TRK-33.py](TRK-33.py)

    ```python
    """TRK-33: KQUAD and KSEXT steering kicks radiate nothing horizontally and four times too much vertically

    A KQUAD or KSEXT with SYNCH_RAD=1 and a steering kick HKICK or VKICK computes the radiating
    field after subtracting the horizontal kick and doubling the vertical one. For a 0.8 m magnet
    with zero main strength and a 1 mrad kick at 3.066 GeV, the classical radiation integral along
    the uniform steering field gives a fractional energy loss of 5.072×10⁻¹⁰ in either plane; HKICK
    gives zero and VKICK 2.029×10⁻⁹, four times too much. The moments_output quantum excitation of
    the KQUAD is zero for HKICK and eight times the Sands diffusion integral for VKICK. This
    steering-field error is absent when HKICK and VKICK are both zero.

    The script tracks an on-axis electron through a KQUAD and a KSEXT of zero main strength with
    SYNCH_RAD=1 and a horizontal (HKICK) or vertical (VKICK) steering kick, and compares the momentum
    loss with the classical radiation integral along the uniform steering field. It also compares the
    moments_output energy-spread growth of the KQUAD with ISR=1 against the quantum-diffusion integral.
    The control is a KQUAD with K₁ ≠ 0 and no steering, tracked off axis, whose loss is checked against an
    independent integration of the orbit and its radiation.
    """

    import sys

    import numpy as np
    from _elegant import modes, particles, run, verdict
    from scipy.integrate import solve_ivp

    P0 = 6000.0  # reference momentum in m_e c (3.066 GeV electrons)
    LENGTH, KICK, SLICES = 0.8, 0.001, 256
    RE = 2.8179403205e-15  # classical electron radius (m), CODATA 2022
    LAMBDA_C = 3.8615926744e-13  # reduced Compton wavelength of the electron (m), CODATA 2022
    # The slice quadrature of the radiation integral and the energy loss fed back into the orbit
    # (about 1e-9 relative) agree with the continuous integral to about 1e-6 relative at 256 slices.
    RTOL = 1e-4
    N = 12  # identical particles, so that every MPI worker tracks some


    def fmt(values):
        return "[" + ", ".join(f"{v:.10g}" for v in np.atleast_1d(values)) + "]"


    def lattice(element, params):
        extra = "".join(f",{k}={v!r}" for k, v in params.items())
        # The marker keeps moments_output off its one-element-beamline crash.
        return (
            f"M: {element},L={LENGTH!r},N_SLICES={SLICES},INTEGRATION_ORDER=4{extra}\n"
            "END: MARK\nBL: LINE=(M,END)\n"
        )


    def loss(mode, element, params, x0=0.0):
        """Classical momentum loss -delta of electrons entering at x = x0 with SYNCH_RAD=1, ISR=0."""
        commands = (
            f"&run_setup lattice=lattice.lte, use_beamline=BL, p_central={P0!r}, output=out.sdds &end\n"
            "&run_control n_steps=1, n_passes=1 &end\n"
            "&sdds_beam input=beam.sdds &end\n"
            "&track &end\n"
        )
        files = {
            "lattice.lte": lattice(element, {**params, "SYNCH_RAD": 1, "ISR": 0}),
            "run.ele": commands,
        }
        beam = {"x": np.full(N, x0), "p": np.full(N, P0)}
        out = particles(run(files, mode, {"beam.sdds": beam})["out.sdds"])
        if len(out["p"]) != N or np.ptp(out["p"]) > 0:
            sys.exit(f"{mode} {element} {params}: particles lost or not identical: invalid")
        return 1 - out["p"][0] / P0


    def diffusion(mode, element, params):
        """Growth of the rms energy spread squared through the element, from moments_output."""
        commands = (
            f"&run_setup lattice=lattice.lte, use_beamline=BL, p_central={P0!r} &end\n"
            "&moments_output filename=out.mom, matched=0, equilibrium=0, radiation=1,"
            " beta_x=2, beta_y=3, beta_z=1 &end\n"
            "&run_control &end\n&bunched_beam &end\n&track &end\n"
        )
        files = {
            "lattice.lte": lattice(element, {**params, "SYNCH_RAD": 1, "ISR": 1}),
            "run.ele": commands,
        }
        moments = run(files, mode, outputs=("out.mom",))["out.mom"]
        s6 = np.asarray(moments[0, "s6"])
        return float(s6[-1] ** 2 - s6[0] ** 2)


    def compare(label, observed, expected):
        bad = abs(observed - expected) > RTOL * abs(expected)
        ratio = observed / expected
        print(
            f"{label}: observed={observed:.10g} expected={expected:.10g} observed/expected={ratio:.7g}"
        )
        return bad


    # A steering kick k over length L is a uniform field of curvature h = k/L in the Cartesian frame; on
    # the unperturbed orbit the normalized transverse momentum grows linearly to k, so the path is L*asin(k)/k. The classical loss
    # is (2/3) r_e gamma^3 h^2 * path, and the quantum diffusion of delta is
    # (55/(24 sqrt 3)) r_e lambda_c gamma^5 h^3 * path (Sands); gamma = P0 to 1.4e-8 here.
    path = LENGTH * np.arcsin(KICK) / KICK
    gamma = np.hypot(1.0, P0)
    expected_loss = 2 / 3 * RE * gamma**3 * (KICK / LENGTH) ** 2 * path
    expected_diffusion = 55 / (24 * np.sqrt(3)) * RE * LAMBDA_C * gamma**5 * (KICK / LENGTH) ** 3 * path


    def quad_loss(k1, x0):
        """Classical loss of an electron entering a quadrupole at x = x0, by direct integration.

        Cartesian equations with canonical slope q = P_x/P0: x' = q/sqrt(1 - q^2), q' = -K1*x; the
        field magnitude in curvature units is |K1*x|, and the loss integrand is
        (2/3) r_e gamma^3 (K1 x)^2 sqrt(1 + x'^2).
        """

        def rhs(_, z):
            x, q, _ = z
            xp = q / np.sqrt(1 - q * q)
            return [xp, -k1 * x, (k1 * x) ** 2 * np.sqrt(1 + xp * xp)]

        sol = solve_ivp(rhs, (0, LENGTH), [x0, 0.0, 0.0], method="DOP853", rtol=1e-12, atol=1e-18)
        return 2 / 3 * RE * gamma**3 * sol.y[2, -1]


    defect = []
    print(f"p0={P0:g} m_e c, L={LENGTH} m, kick={KICK} rad, {SLICES} slices, INTEGRATION_ORDER=4")
    for mode in modes():
        # Control: radiation from the main quadrupole field, no steering.
        control = loss(mode, "KQUAD", {"K1": 0.5}, x0=0.001)
        if compare(
            f"{mode} KQUAD K1=0.5 x0=1 mm, no steering (control)", control, quad_loss(0.5, 0.001)
        ):
            sys.exit("The quadrupole radiation control disagrees with the integration: invalid.")
        for element in ("KQUAD", "KSEXT"):
            for plane in ("HKICK", "VKICK"):
                observed = loss(mode, element, {plane: KICK})
                defect.append(compare(f"{mode} {element} {plane}={KICK} loss", observed, expected_loss))
        for plane in ("HKICK", "VKICK"):
            observed = diffusion(mode, "KQUAD", {plane: KICK})
            defect.append(
                compare(
                    f"{mode} KQUAD {plane}={KICK} moments sigma_delta^2 growth",
                    observed,
                    expected_diffusion,
                )
            )
    print(f"comparison: relative tolerance {RTOL:g}")
    verdict(all(defect))

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

## Provenance

| Role | Model | Effort | Date | Code examined |
|---|---|---|---|---|
| Found | `gemini-4-argon-medium` |  | 2026-11-04 | master (`7514db36f764`), source with SDDS a3f730a14c49, CPU, serial |
| Found | `gemini-4-argon-medium` |  | 2026-11-04 | master (`7514db36f764`), source reading, not executed for this claim |
| Found (comparison with the analytic loss and a CSBEND reference) | `claude-opus-5-5` | high | 2026-11-04 | master (`7514db36f764`), source with SDDS a3f730a14c49, CPU, serial |
| Confirmed | `gpt-6.1-sol` | xhigh | 2026-11-04 | master (`7514db36f764`), source with SDDS a3f730a14c49, CPU, serial |

