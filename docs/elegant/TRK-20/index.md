---
title: "TRK-20: MAPSOLENOID combined misalignment rotations are not mutual inverses"
hide:
  - toc
---

# TRK-20 — MAPSOLENOID combined misalignment rotations are not mutual inverses

| | |
|---|---|
| Code | [Elegant](../index.md) · Tracking |
| Priority | P4 |
| Severity | Wrong physics |
| Evidence | Reproduced, independently rechecked |
| Last reproduced | 2026-11-04 on Elegant master (`0355dd6`) |
| Tags | <span class="bb-tag">partly-source-only</span> [?](../../methodology.md#tags) |
| Source | [`set_up_derivatives`, MAPSOLENOID rotations, motion.cc:481–482](https://github.com/rtsoliday/elegant/blob/0355dd6aca12c861971fbf8f3ae78a0ded991734/src/motion.cc#L481-L482)<br>[`setupRotate3Matrix`, tilt_matrices.c:218–255](https://github.com/rtsoliday/elegant/blob/0355dd6aca12c861971fbf8f3ae78a0ded991734/src/tilt_matrices.c#L218-L255)<br>[`derivatives_mapSolenoid` position rotation, motion.cc:1055](https://github.com/rtsoliday/elegant/blob/0355dd6aca12c861971fbf8f3ae78a0ded991734/src/motion.cc#L1055)<br>[`derivatives_mapSolenoid` field rotation, motion.cc:1101](https://github.com/rtsoliday/elegant/blob/0355dd6aca12c861971fbf8f3ae78a0ded991734/src/motion.cc#L1101)<br>[MAPSOLENOID dictionary, track_data.c:2781–2783](https://github.com/rtsoliday/elegant/blob/0355dd6aca12c861971fbf8f3ae78a0ded991734/src/track_data.c#L2781-L2783) |

In Elegant's MAPSOLENOID, the misalignment angles ETILT, EYAW and EPITCH rotate the particle into the element frame with one matrix and rotate the field back with another that is not its inverse when more than one angle is nonzero. A particle travelling along the map's own axis, parallel to a uniform axial field, must feel no force, but the field points in a different direction. With a 0.7 m map of 1.200 mT, p = 13 mₑc, ETILT = 0.17, EYAW = −0.11 and EPITCH = 0.13 rad, the slopes change by −8.214×10⁻⁴ (x′) and −8.546×10⁻⁴ (y′) instead of zero; each angle alone is exact. The error is second order in the angles.

??? example "Reproducer: TRK-20.py"

    [Download TRK-20.py](TRK-20.py)

    ```python
    """TRK-20: MAPSOLENOID combined misalignment rotations are not mutual inverses

    In Elegant's MAPSOLENOID, the misalignment angles ETILT, EYAW and EPITCH rotate the particle into
    the element frame with one matrix and rotate the field back with another that is not its inverse
    when more than one angle is nonzero. A particle travelling along the map's own axis, parallel to a
    uniform axial field, must feel no force, but the field points in a different direction. With a 0.7 m
    map of 1.200 mT, p = 13 mₑc, ETILT = 0.17, EYAW = −0.11 and EPITCH = 0.13 rad, the slopes change by
    −8.214×10⁻⁴ (x′) and −8.546×10⁻⁴ (y′) instead of zero; each angle alone is exact. The error is
    second order in the angles.

    The script tracks electrons along the axis of a MAPSOLENOID with a constant axial field map, for
    each single misalignment angle and for the three combined, and compares the exit slopes with the
    entrance slopes, which must be equal in the absence of force. For the combined case it also compares
    them with an analytic helix around the field direction the code applies, R(angles)ez, which explains
    the observed slope change. The single-angle cases are the controls.
    """

    import sys

    import numpy as np
    from _elegant import modes, particles, run, verdict
    from scipy.optimize import brentq

    # SDDS constants.h, as Elegant uses them.
    C, QE, ME = 299792458.0, 1.60217733e-19, 9.1093897e-31
    LENGTH, P, FIELD, STEPS = 0.7, 13.0, 0.0012, 1000
    COPIES = 12
    CONTROL_ATOL = 1e-14  # exit slope against entrance slope for one rotation angle
    HELIX_ATOL = 1e-14  # exit slope against the helix around the field the code applies
    # (roll, yaw, pitch) = (ETILT, EYAW, EPITCH), rad
    CASES = ((0.17, 0.0, 0.0), (0.0, -0.11, 0.0), (0.0, 0.0, 0.13), (0.17, -0.11, 0.13))


    def field_map():
        """Constant axial field Bz = FIELD, Br = 0, on three radii: one SDDS page per radius."""
        z = np.linspace(0, LENGTH, 8).tolist()
        pages = "".join(
            f"{len(z)}\n" + "".join(f"{zz!r} {r!r} 0 {FIELD!r}\n" for zz in z) for r in (0.0, 0.2, 0.4)
        )
        return (
            "SDDS1\n"
            '&column name=z, type=double, units="m" &end\n'
            '&column name=r, type=double, units="m" &end\n'
            '&column name=Br, type=double, units="T" &end\n'
            '&column name=Bz, type=double, units="T" &end\n'
            "&data mode=ascii &end\n" + pages
        )


    def axis_rotation(axis, angle):
        """Rotation matrix about a coordinate axis (0: x, 1: y, 2: z) by angle, Rodrigues' formula."""
        unit = np.eye(3)[axis]
        cross = np.array([[0, -unit[2], unit[1]], [unit[2], 0, -unit[0]], [-unit[1], unit[0], 0]])
        return (
            np.eye(3) * np.cos(angle)
            + (1 - np.cos(angle)) * np.outer(unit, unit)
            + np.sin(angle) * cross
        )


    def rotation(roll, yaw, pitch):
        # setupRotate3Matrix applies yaw, then pitch, then roll (3-2-1 sequence of Goldstein, with
        # accelerator (x, y, z) = Goldstein (Y, Z, X)): in accelerator coordinates this is
        # Rz(-roll) Rx(-pitch) Ry(-yaw), each factor a rotation by the negative angle about its axis.
        return axis_rotation(2, -roll) @ axis_rotation(0, -pitch) @ axis_rotation(1, -yaw)


    def helix_slopes(axis, field_axis):
        """Exit slopes (xp, yp) of a particle starting along `axis` in a uniform field along field_axis.

        The motion is a helix: the momentum component along the field is constant and the
        perpendicular part rotates at omega = eB/(m c gamma) per unit c*t. The particle enters at
        the entrance plane through the point that puts the element centre on its line and is followed
        until z = LENGTH.
        """
        momentum = P * axis
        start = np.array([-LENGTH / 2 * axis[0] / axis[2], -LENGTH / 2 * axis[1] / axis[2], 0])
        gamma = np.sqrt(1 + P * P)
        omega = QE * FIELD / (ME * C * gamma)
        parallel = field_axis * np.dot(momentum, field_axis)
        perpendicular = momentum - parallel
        cross = np.cross(field_axis, momentum)

        def position(ct):
            angle = omega * ct
            return (
                start
                + (
                    parallel * ct
                    + perpendicular * np.sin(angle) / omega
                    + cross * (2 * np.sin(angle / 2) ** 2) / omega
                )
                / gamma
            )

        ct = brentq(
            lambda ct: position(ct)[2] - LENGTH, 0, 1.5 * LENGTH * gamma / (P * axis[2]), xtol=5e-15
        )
        final = parallel + perpendicular * np.cos(omega * ct) + cross * np.sin(omega * ct)
        return final[:2] / final[2]


    def track(mode, angles, axis_slopes):
        roll, yaw, pitch = angles
        lattice = (
            f'K: MAPSOLENOID,L={LENGTH},INPUTFILE="map.sdds",RCOLUMN="r",ZCOLUMN="z",BRCOLUMN="Br",'
            f'BZCOLUMN="Bz",FACTOR=1,N_STEPS={STEPS},ACCURACY=1e-10,'
            f'METHOD="non-adaptive runge-kutta",ETILT={roll!r},EYAW={yaw!r},EPITCH={pitch!r}\n'
            "BL: LINE=(K)\n"
        )
        commands = (
            "&change_particle name=electron &end\n"
            f"&run_setup lattice=lattice.lte, use_beamline=BL, p_central={P!r}, "
            "output=out.sdds, random_number_seed=752913 &end\n"
            "&run_control n_steps=1, n_passes=1 &end\n"
            "&sdds_beam input=beam.sdds &end\n"
            "&track soft_failure=0 &end\n"
        )
        # Start on the line along the map axis through the element centre (the map centre is at
        # z = L/2 on the axis), with the slopes of that axis.
        beam = {
            "x": np.full(COPIES, -LENGTH / 2 * axis_slopes[0]),
            "y": np.full(COPIES, -LENGTH / 2 * axis_slopes[1]),
            "xp": np.full(COPIES, axis_slopes[0]),
            "yp": np.full(COPIES, axis_slopes[1]),
            "p": np.full(COPIES, P),
        }
        files = {"lattice.lte": lattice, "run.ele": commands, "map.sdds": field_map()}
        out = particles(run(files, mode, {"beam.sdds": beam})["out.sdds"])
        if len(out["p"]) != COPIES or not all(np.all(np.isfinite(v)) for v in out.values()):
            sys.exit(f"{mode} angles={angles}: particles were lost or not finite")
        return np.array([out["xp"][0], out["yp"][0]])


    def compare(label, observed, expected):
        print(f"{label}: observed={observed.tolist()!r} expected={expected.tolist()!r}")


    def main():
        defect = False
        for mode in modes():
            for angles in CASES:
                roll, yaw, pitch = angles
                # The code rotates positions into the element frame with R(-angles) and the field back
                # with R(angles); the element axis in the laboratory is the image of ez under the
                # inverse of the position rotation, R(-angles)^T ez. A field along ez in the element
                # frame must point along that same axis, which requires R(angles) = R(-angles)^T,
                # and then a particle on the axis feels no force. They agree for a single angle.
                to_element = rotation(-roll, -yaw, -pitch)
                to_lab = rotation(roll, yaw, pitch)
                axis = to_element.T @ np.array([0, 0, 1.0])
                slopes = axis[:2] / axis[2]
                inverse_error = np.max(np.abs(to_lab @ to_element - np.eye(3)))
                applied_field_axis = to_lab @ np.array([0, 0, 1.0])
                observed = track(mode, angles, slopes)
                label = (
                    f"{mode} (ETILT, EYAW, EPITCH)={angles}, rotation inverse error={inverse_error:.3g}"
                )
                compare(label, observed, slopes)
                bad = bool(np.max(np.abs(observed - slopes)) > CONTROL_ATOL)
                if sum(a != 0 for a in angles) == 1:
                    if bad:
                        sys.exit(
                            "A single-axis rotation changes the slopes: the comparison is not valid."
                        )
                    continue
                # The code's applied field is along R(angles) ez, not along the axis: a uniform-field
                # helix around it gives the observed slopes.
                helix = helix_slopes(axis, applied_field_axis)
                print(f"{label}: false kick={(observed - slopes).tolist()!r}")
                compare(label + " vs helix around the applied field", observed, helix)
                if np.max(np.abs(observed - helix)) > HELIX_ATOL:
                    sys.exit("The false kick is not explained by the rotated field direction.")
                defect |= bad
        print(f"comparison: atol controls={CONTROL_ATOL:g}, helix={HELIX_ATOL:g}")
        verdict(defect)


    main()

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
| Found | `gpt-6-astra` | xhigh | 2026-10-28 | master (`75a2e1a1a58b`), source with SDDS a60b9bbed7cc, CPU, serial |
| Confirmed | `gpt-6.1-sol` | xhigh | 2026-10-28 | master (`75a2e1a1a58b`), source with SDDS a60b9bbed7cc, CPU, serial |

