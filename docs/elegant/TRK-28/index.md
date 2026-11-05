---
title: "TRK-28: SOLE spin transport misses the precession along off-axis trajectories and the edge-field rotations"
hide:
  - toc
---

# TRK-28 — SOLE spin transport misses the precession along off-axis trajectories and the edge-field rotations

| | |
|---|---|
| Code | [Elegant](../index.md) · Tracking |
| Priority | P3 |
| Severity | Wrong physics |
| Evidence | Reproduced, independently rechecked |
| Last reproduced | 2026-11-04 on Elegant master (`0355dd6`) |
| Source | [SOLE tracking, do_tracking.c:2380–2382](https://github.com/rtsoliday/elegant/blob/0355dd6aca12c861971fbf8f3ae78a0ded991734/src/do_tracking.c#L2380-L2382)<br>[`updateSpinForSolenoid`, spinUpdate.c:190–206](https://github.com/rtsoliday/elegant/blob/0355dd6aca12c861971fbf8f3ae78a0ded991734/src/spinUpdate.c#L190-L206)<br>[`updateSpinQuaternionLocalFS`, spinUpdate.c:47–125](https://github.com/rtsoliday/elegant/blob/0355dd6aca12c861971fbf8f3ae78a0ded991734/src/spinUpdate.c#L47-L125)<br>[kernel contract, spinUpdate.c:4–29](https://github.com/rtsoliday/elegant/blob/0355dd6aca12c861971fbf8f3ae78a0ded991734/src/spinUpdate.c#L4-L29)<br>[`solenoid_matrix`, compute_matrices.c:1144–1183](https://github.com/rtsoliday/elegant/blob/0355dd6aca12c861971fbf8f3ae78a0ded991734/src/compute_matrices.c#L1144-L1183)<br>[CSBEND edge spin rotation, csbend.c:1464–1472](https://github.com/rtsoliday/elegant/blob/0355dd6aca12c861971fbf8f3ae78a0ded991734/src/csbend.c#L1464-L1472)<br>[SOLE parameters, track_data.c:553–561](https://github.com/rtsoliday/elegant/blob/0355dd6aca12c861971fbf8f3ae78a0ded991734/src/track_data.c#L553-L561)<br>[spin_tracking, elegant.tex:7291–7292](https://github.com/rtsoliday/elegant/blob/0355dd6aca12c861971fbf8f3ae78a0ded991734/doc/elegant.tex#L7291-L7292)<br>[spin-aware elements, elegant.tex:11191–11192](https://github.com/rtsoliday/elegant/blob/0355dd6aca12c861971fbf8f3ae78a0ded991734/doc/elegant.tex#L11191-L11192) |

With spin tracking enabled, a SOLE solenoid rotates each spin once about its longitudinal field at the exit slopes, so off-axis or sloped particles miss both the precession along their rotating trajectory in the body and the spin rotations by the radial edge fields that the orbital matrix includes. For an electron of momentum 1000 mₑc, 1 mm off axis in a 1 m, 0.8523 T solenoid with longitudinal spin, Thomas–BMT along the same hard-edge trajectory gives transverse spin (sₓ, sᵧ) = (−2.568×10⁻⁴, 1.027×10⁻⁴); Elegant gives (1.701×10⁻⁵, 3.111×10⁻⁵), an error of 2.830×10⁻⁴ rad. The error is first order in offset and slope, and dividing the element does not remove it. The on-axis rotation of a KS-specified solenoid is exact.

??? example "Reproducer: TRK-28.py"

    [Download TRK-28.py](TRK-28.py)

    ```python
    """TRK-28: SOLE spin transport misses the precession along off-axis trajectories and the edge-field rotations

    With spin tracking enabled, a SOLE solenoid rotates each spin once about its longitudinal field at
    the exit slopes, so off-axis or sloped particles miss both the precession along their rotating
    trajectory in the body and the spin rotations by the radial edge fields that the orbital matrix
    includes. For an electron of momentum 1000 mₑc, 1 mm off axis in a 1 m, 0.8523 T solenoid with
    longitudinal spin, Thomas–BMT along the same hard-edge trajectory gives transverse spin (sₓ, sᵧ) =
    (−2.568×10⁻⁴, 1.027×10⁻⁴); Elegant gives (1.701×10⁻⁵, 3.111×10⁻⁵), an error of 2.830×10⁻⁴ rad. The
    error is first order in offset and slope, and dividing the element does not remove it. The on-axis
    rotation of a KS-specified solenoid is exact.

    The script tracks electrons with longitudinal spin through KS-specified SOLE elements and compares
    the exit spin with the first-order solution of the Lorentz and Thomas–BMT equations for the
    hard-edge solenoid: the body generator exponentiated over the length between the entrance and exit
    edge kicks implied by div B = 0. A full cyclotron turn isolates the body precession (also with 100
    element divisions), a low-momentum offset case isolates the edge rotations, and a 1 mm offset at
    511.0 MeV/c gives a typical size. Every off-axis case first checks that the tracked orbit equals
    the oracle's orbit, and an on-axis control must match the exact precession.
    """

    import sys

    import numpy as np
    from _elegant import modes, verdict
    from _sole import A_ELECTRON, axial_rotation, compare, on_axis_angle, track
    from scipy.linalg import expm

    ORBIT_RTOL = 1e-12
    # The first-order map neglects transverse spin terms of third order in the amplitudes; an
    # integration of the full spin rotation along the same orbit, with finite edge rotations, agrees
    # with it within 1.1e-7 of |expected| at x = 1 mm and 6e-9 in the other cases.
    SPIN_RTOL = 1e-6


    def first_order_map(ks, length, p, anomaly=A_ELECTRON):
        """Linear map of (x, y, u, v, Sx, Sy) through a hard-edge solenoid, for spin near +z.

        u, v are the mechanical slopes and Sz = 1 to first order in the transverse amplitudes. Per unit
        length, with k = KS = q Bz/(m c p) and slopes small:
        - Lorentz force, dv/dt = -(q Bz/(m gamma)) z-hat x v:  u' = k v, v' = -k u.
        - Thomas-BMT, dS/dt = Omega x S with
          Omega = -(q/m)[(a + 1/gamma) B - (a gamma/(gamma + 1))(beta.B) beta]; with
          beta = beta (u, v, 1) and B = Bz z-hat, Omega/(beta c) = (c u, c v, -k (1 + a)) where
          c = k a (gamma - 1), so Sx' = k (1 + a) Sy + c v and Sy' = -k (1 + a) Sx - c u.
        At a field step dk, div B = 0 gives B_perp = -(x, y) dBz/dz / 2, which integrates to a slope
        kick (du, dv) = dk (y, -x)/2 and, the field being transverse to the velocity, a spin rotation
        vector (1 + a gamma) dk (x, y)/2, that is (dSx, dSy) = (1 + a gamma) dk (y, -x)/2. The map is
        edge(-k) body(L) edge(+k); the transverse spin terms neglected are third order.
        """
        gamma = np.sqrt(1 + p * p)
        c = ks * anomaly * (gamma - 1)
        body = np.zeros((6, 6))
        body[0, 2] = body[1, 3] = 1
        body[2, 3], body[3, 2] = ks, -ks
        body[4, 5], body[4, 3] = ks * (1 + anomaly), c
        body[5, 4], body[5, 2] = -ks * (1 + anomaly), -c

        def edge(dk):
            m = np.eye(6)
            g = (1 + anomaly * gamma) * dk / 2
            m[2, 1], m[3, 0] = dk / 2, -dk / 2
            m[4, 1], m[5, 0] = g, -g
            return m

        return edge(-ks) @ expm(body * length) @ edge(ks)


    def offaxis_case(mode, label, ks, length, p, x=0.0, xp=0.0, divisions=1):
        """Track a particle with longitudinal spin; return True when the spin disagrees.

        The orbit must agree with the first-order hard-edge map, which shows that SOLE's matrix and the
        oracle describe the same field and trajectory; only then is the spin comparison a verdict.
        """
        setup = f", element_divisions={divisions}" if divisions > 1 else ""
        out, _ = track(
            mode, f"SOLE,L={length!r},KS={ks!r}", p, {"x": x, "xp": xp, "spz": 1.0}, setup=setup
        )
        expected = first_order_map(ks, length, p) @ np.array([x, 0.0, xp, 0.0, 0.0, 0.0])
        orbit = [out["x"], out["xp"], out["y"], out["yp"]]
        scale = max(abs(x), abs(xp))
        if not compare(
            f"{label}: orbit (x, xp, y, yp)", orbit, expected[[0, 2, 1, 3]], atol=ORBIT_RTOL * scale
        ):
            sys.exit(f"{label}: the orbit disagrees with the hard-edge map; the spin test is invalid")
        return not compare(
            f"{label}: spin (sx, sy)",
            [out["spx"], out["spy"]],
            expected[4:],
            atol=1e-14,
            rtol=SPIN_RTOL,
        )


    defect = False
    for mode in modes():
        # Control: on the axis the field is purely longitudinal and the exact rotation applies.
        ks, length, p = 0.9, 0.8, 600.0
        out, _ = track(mode, f"SOLE,L={length!r},KS={ks!r}", p, {"spx": 1.0})
        if not compare(
            f"{mode} on-axis control: spin",
            [out["spx"], out["spy"], out["spz"]],
            axial_rotation(on_axis_angle(ks, length, A_ELECTRON)),
            atol=1e-13,
        ):
            sys.exit("The on-axis control disagrees with the exact precession: it is not valid.")
        # Body: one full cyclotron turn (KS L = 2 pi) returns the trajectory to the axis at the exit,
        # so to first order the edge fields act at x = y = 0 and only the body precession remains.
        for n in (1, 100):
            defect |= offaxis_case(
                mode,
                f"{mode} body, xp=2e-6, element_divisions={n}",
                2 * np.pi / 0.8,
                0.8,
                600.0,
                xp=2e-6,
                divisions=n,
            )
        # Edges: at p = 0.02 the velocity-dependent body term c = k a (gamma - 1) is 2.3e-7 k, while
        # the edge rotation coefficient (1 + a gamma) is 1, so the edge rotations dominate.
        defect |= offaxis_case(mode, f"{mode} edges, x=3e-7", 1.1, 0.6, 0.02, x=3e-7)
        # Typical case: 511.0 MeV/c electron 1 mm off axis in a 1 m, 0.8523 T solenoid (KS = 0.5 /m).
        defect |= offaxis_case(mode, f"{mode} p=1000, KS=0.5, L=1, x=1 mm", 0.5, 1.0, 1000.0, x=1e-3)
    print("tolerances: spin 1e-14 + 1e-6 |expected|; orbit 1e-12 of the initial amplitude")
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

??? example "Helper: _sole.py"

    [Download _sole.py](_sole.py)

    ```python
    """Shared runner for the SOLE spin-tracking reproducers: one solenoid, one pass, spin tracking on.

    `track` sends identical copies of one particle (so that each MPI worker tracks some) through a
    lattice holding a single element and returns that particle's final coordinates and spin. Spin
    components (spx, spy, spz) are in the fixed straight element frame; p is βγ. Constants are those of
    SDDS `constants.h` and of the species table of Elegant's `change_particle` command, so that a
    different choice of constants cannot pass for an algorithm defect.
    """

    from __future__ import annotations

    import sys

    import numpy as np
    from _elegant import particles, run

    C, QE, ME = 299792458.0, 1.60217733e-19, 9.1093897e-31  # SDDS constants.h
    A_ELECTRON = 0.00115965218128  # electron and positron anomalous magnetic moment in Elegant
    A_PROTON = 1.792847
    A_MUON = 1.165921e-3
    COPIES = 6
    COORDINATES = ("x", "xp", "y", "yp", "t", "p")
    SPIN = ("spx", "spy", "spz")


    def track(
        mode: str,
        element: str,
        p0: float,
        start: dict[str, float],
        before: str = "",
        setup: str = "",
        after_setup: str = "",
        outputs: tuple[str, ...] = ("out.sdds",),
    ) -> tuple[dict[str, float], dict]:
        """Track COPIES copies of one particle through `S: <element>` and return its exit state.

        start: initial x, xp, y, yp, spx, spy, spz (missing values are zero; p is p0). before: commands
        ahead of run_setup (change_particle); setup: extra run_setup fields; after_setup: commands
        between run_control and the beam (matrix_output). Returns ({name: value}, run result).
        """
        lattice = f"S: {element}\nBL: LINE=(S)\n"
        commands = (
            before
            + f"&run_setup lattice=lattice.lte, use_beamline=BL, p_central={p0!r}, output=out.sdds, "
            f"spin_tracking=1, default_order=2{setup} &end\n"
            "&run_control n_steps=1, n_passes=1 &end\n"
            + after_setup
            + "&sdds_beam input=beam.sdds &end\n&track &end\n"
        )
        beam = {n: np.full(COPIES, float(start.get(n, 0.0))) for n in ("x", "xp", "y", "yp", *SPIN)}
        beam["p"] = np.full(COPIES, p0)
        result = run(
            {"lattice.lte": lattice, "run.ele": commands}, mode, {"beam.sdds": beam}, outputs=outputs
        )
        out = particles(result["out.sdds"])
        if len(out["p"]) != COPIES:
            sys.exit(f"{mode}: {COPIES - len(out['p'])} particles lost")
        state = np.column_stack([out[n] for n in (*COORDINATES, *SPIN)])
        if not np.all(np.isfinite(state)):
            sys.exit(f"{mode}: nonfinite orbit or spin")
        if np.ptp(state, axis=0).max() != 0:
            sys.exit(f"{mode}: identical particles ended in different states")
        if abs(np.linalg.norm(state[0, 6:]) - 1) > 1e-14:
            sys.exit(f"{mode}: the spin norm is not 1")
        return {n: float(v) for n, v in zip((*COORDINATES, *SPIN), state[0], strict=True)}, result


    def axial_rotation(theta: float, spin=(1.0, 0.0, 0.0)) -> np.ndarray:
        """Spin rotated by the angle theta about +z (counterclockwise seen from +z)."""
        c, s = np.cos(theta), np.sin(theta)
        return np.array([c * spin[0] - s * spin[1], s * spin[0] + c * spin[1], spin[2]])


    def on_axis_angle(ks: float, length: float, anomaly: float) -> float:
        """Spin rotation angle about +z on the axis of a solenoid of strength KS, on momentum.

        KS = q B/(m c p0) for the signed physical charge q (Elegant's "-Bs/(B*Rho)" with its legacy
        charge sign). With the velocity along B, magnetic Thomas-BMT reduces to
        dS/dt = -(q/m)(1 + a) B/gamma z-hat x S; dt = gamma dz/(beta gamma c) turns it into
        dtheta/dz = -q (1 + a) B/(m c p) = -KS (1 + a).
        """
        return -ks * length * (1 + anomaly)


    def fmt(values) -> str:
        return "[" + ", ".join(f"{v:.16g}" for v in np.atleast_1d(values)) + "]"


    def compare(label: str, observed, expected, atol: float, rtol: float = 0.0) -> bool:
        """Print one case; return True when observed and expected agree within atol + rtol*|expected|."""
        print(f"{label}: observed={fmt(observed)} expected={fmt(expected)}")
        return bool(np.allclose(observed, expected, rtol=rtol, atol=atol))

    ```

## Provenance

| Role | Model | Effort | Date | Code examined |
|---|---|---|---|---|
| Found | `gpt-6-astra` | xhigh | 2026-11-04 | master (`0355dd6aca12`), source with SDDS a3f730a14c49, CPU, serial |
| Confirmed | `gpt-6.1-sol` | xhigh | 2026-11-04 | master (`0355dd6aca12`), source with SDDS a3f730a14c49, CPU, serial |

