---
title: "Bmad shared helpers"
---

# Bmad shared helpers

These modules support the [Bmad reproducers](../index.md).
Each entry links all the helpers it needs, including those imported by another helper.
Save the linked helpers in the same directory as the downloaded reproducer scripts.

## `_bmad.py` { #_bmad }

[Download _bmad.py](_bmad.py)

??? example "Source: _bmad.py"

    ```python
    """Shared Bmad runner: isolated inputs, build-derived Fortran recipes, numerical verdicts.

    BMAD_DIST names the build root containing production/{bin,lib,modules}; BMAD_SOURCE
    names a source tree for source-reading reproducers. Native processes run serially with
    one numerical thread. Scratch directories live under the system temp directory's
    bmad-scratch/ and are removed on success, failure and timeout.
    """

    from __future__ import annotations

    import os
    import re
    import shlex
    import signal
    import subprocess
    import tempfile
    from collections.abc import Mapping, Sequence
    from dataclasses import dataclass
    from pathlib import Path

    import numpy as np
    from numpy.typing import ArrayLike

    TIMEOUT = 290  # seconds per compilation, link or native run
    _THREADS = ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "NUMEXPR_NUM_THREADS")


    class RunFailure(RuntimeError):
        """A failed process; status and the complete combined output remain available."""

        def __init__(self, program: str, status: int, log: str):
            super().__init__(f"{program} exited with status {status}:\n{log[-3000:]}")
            self.program, self.status, self.log = program, status, log


    @dataclass
    class Result:
        """Combined stdout/stderr and requested files as bytes (decode text with .decode())."""

        log: str
        outputs: dict[str, bytes]


    def _root(variable: str) -> Path:
        value = os.environ.get(variable)
        if not value:
            raise RuntimeError(
                f"Set {variable} to the Bmad {'build' if variable == 'BMAD_DIST' else 'source'} root."
            )
        root = Path(value).expanduser().resolve()
        if not root.is_dir():
            raise RuntimeError(f"{variable} is not a directory: {root}")
        return root


    def build() -> Path:
        """Resolve BMAD_DIST and require its production binaries, libraries and modules."""
        root = _root("BMAD_DIST")
        for name in ("bin", "lib", "modules"):
            if not (root / "production" / name).is_dir():
                raise RuntimeError(
                    f"BMAD_DIST must name the build root; missing {root / 'production' / name}"
                )
        return root


    def source() -> Path:
        """Resolve BMAD_SOURCE independently of the executable build."""
        root = _root("BMAD_SOURCE")
        if not (root / "bmad/doc/bmad.tex").is_file():
            raise RuntimeError(f"BMAD_SOURCE lacks bmad/doc/bmad.tex: {root}")
        return root


    def _scratch() -> tempfile.TemporaryDirectory[str]:
        parent = Path(tempfile.gettempdir()) / "bmad-scratch"
        parent.mkdir(exist_ok=True)
        return tempfile.TemporaryDirectory(prefix="repro-", dir=parent)


    def _path(work: Path, name: str) -> Path:
        if not name or Path(name).is_absolute() or ".." in Path(name).parts:
            raise ValueError(f"Expected a relative case file name: {name!r}")
        return work / name


    def _execute(argv: Sequence[str], work: Path, stdin: str, timeout: float) -> str:
        env = dict(os.environ, **dict.fromkeys(_THREADS, "1"), OMP_DYNAMIC="FALSE")
        with subprocess.Popen(
            argv,
            cwd=work,
            env=env,
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            start_new_session=True,
        ) as process:
            try:
                log, _ = process.communicate(stdin, timeout=timeout)
            except subprocess.TimeoutExpired:
                # Compilers spawn a linker; terminate the whole group before removing scratch.
                os.killpg(process.pid, signal.SIGKILL)
                log, _ = process.communicate()
                raise TimeoutError(f"{argv[0]} exceeded {timeout} s:\n{log[-3000:]}") from None
            if process.returncode:
                raise RunFailure(argv[0], process.returncode, log)
        return log


    def _case(
        argv: Sequence[str],
        work: Path,
        files: Mapping[str, str | bytes] | None,
        outputs: Sequence[str],
        stdin: str,
        timeout: float,
    ) -> Result:
        work.mkdir()
        for name, content in (files or {}).items():
            path = _path(work, name)
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_bytes(content.encode() if isinstance(content, str) else content)
        log = _execute(argv, work, stdin, timeout)
        return Result(log, {name: _path(work, name).read_bytes() for name in outputs})


    def run(
        program: str,
        args: Sequence[str] = (),
        *,
        files: Mapping[str, str | bytes] | None = None,
        outputs: Sequence[str] = (),
        stdin: str = "",
        timeout: float = TIMEOUT,
    ) -> Result:
        """Run a named production/bin executable in fresh scratch; collect outputs before cleanup.

        Pass lattice, Tao command and namelist files inline. Nonzero exits raise RunFailure;
        timeouts raise TimeoutError; a missing requested output raises FileNotFoundError.
        Check completion markers and case-specific diagnostics even when the exit status is zero.
        """
        if Path(program).name != program:
            raise ValueError("program must be a name in BMAD_DIST/production/bin")
        executable = build() / "production/bin" / program
        if not executable.is_file() or not os.access(executable, os.X_OK):
            raise RuntimeError(f"Required executable missing or not executable: {executable}")
        with _scratch() as tmp:
            return _case([str(executable), *args], Path(tmp) / "run", files, outputs, stdin, timeout)


    def _recipe(tao: bool) -> tuple[list[str], list[str]]:
        relative = (
            "tao/production/CMakeFiles/tao-exe.dir"
            if tao
            else "regression_tests/production/CMakeFiles/tracking_method_test-exe.dir"
        )
        recipe = build() / relative
        flags_text = (recipe / "flags.make").read_text()
        flags = []
        for key in ("Fortran_DEFINES", "Fortran_INCLUDES", "Fortran_FLAGS"):
            match = re.search(rf"^{key} = (.*)$", flags_text, re.MULTILINE)
            if match is None:
                raise RuntimeError(f"Missing {key} in {recipe / 'flags.make'}")
            flags.extend(shlex.split(match[1]))
        custom = re.findall(r"^# Custom flags: .*?_FLAGS = (.*)$", flags_text, re.MULTILINE)
        if len(custom) != 1:
            raise RuntimeError(f"Expected one source compile recipe in {recipe}")
        flags.extend(shlex.split(custom[0]))
        # The supplied GNU recipes use joined -J paths. Keep generated modules in scratch.
        if "-J" in flags:
            raise RuntimeError(f"Expected joined -J module path in {recipe}")
        flags = [flag for flag in flags if not flag.startswith("-J")]
        link = shlex.split((recipe / "link.txt").read_text())
        objects = [i for i, token in enumerate(link) if token.endswith(".o")]
        if not link or len(objects) != 1 or link.count("-o") != 1 or "@" in " ".join(link):
            raise RuntimeError(f"Expected a single-object executable link recipe in {recipe}")
        if not Path(link[0]).is_file():
            raise RuntimeError(f"Build compiler missing: {link[0]}")
        link[objects[0]] = "probe.o"
        link[link.index("-o") + 1] = "probe"
        # These diagnostics point into the original build; retain all actual link options in order.
        link = [
            token for token in link if not token.startswith(("-Wl,--dependency-file=", "-Wl,-Map="))
        ]
        return [link[0], *flags, "-J.", "-c", "probe.f90", "-o", "probe.o"], link


    def fortran(
        program: str,
        args: Sequence[str] = (),
        *,
        tao: bool = False,
        files: Mapping[str, str | bytes] | None = None,
        outputs: Sequence[str] = (),
        stdin: str = "",
        timeout: float = TIMEOUT,
    ) -> Result:
        """Compile/link inline Fortran with the build's recipe, then run it in fresh scratch.

        The tracking_method_test recipe supplies Bmad/PTC libraries; tao=True selects Tao's
        own recipe for Tao modules. The compiler and every library come from flags.make/link.txt.
        The build must retain these CMake files and the compiler that produced its modules.
        """
        compile_argv, link_argv = _recipe(tao)
        with _scratch() as tmp:
            work = Path(tmp)
            (work / "probe.f90").write_text(program)
            _execute(compile_argv, work, "", timeout)
            _execute(link_argv, work, "", timeout)
            return _case([str(work / "probe"), *args], work / "run", files, outputs, stdin, timeout)


    def compare(
        label: str,
        observed: ArrayLike,
        expected: ArrayLike,
        *,
        atol: float,
        rtol: float = 0.0,
    ) -> bool:
        """Print observed/expected values and return True for a discrepancy beyond the tolerances.

        Use a label that names the case, units and expectation. Shapes must match; empty arrays,
        nonfinite numbers and invalid tolerances raise ValueError. Crash/NaN claims need their
        own explicit checks. A passing control requires compare(...) to return False.
        """
        actual, target = np.asarray(observed, dtype=float), np.asarray(expected, dtype=float)
        if actual.shape != target.shape or not actual.size:
            raise ValueError(
                f"{label}: expected equal nonempty shapes, got {actual.shape}, {target.shape}"
            )
        if not (np.isfinite(actual).all() and np.isfinite(target).all()):
            raise ValueError(f"{label}: comparison requires finite values")
        if not (np.isfinite([atol, rtol]).all() and atol >= 0 and rtol >= 0):
            raise ValueError("Comparison tolerances must be finite and nonnegative")
        defect = not np.allclose(actual, target, atol=atol, rtol=rtol)
        print(
            f"{label}: observed={actual.tolist()} expected={target.tolist()} atol={atol:g} rtol={rtol:g}"
        )
        return defect


    def verdict(defect: bool) -> None:
        """Print the final verdict after all required cases and controls have completed."""
        print("DEFECT REPRODUCED" if defect else "NOT REPRODUCED")

    ```

## `_collective_contracts.py` { #_collective_contracts }

[Download _collective_contracts.py](_collective_contracts.py)

??? example "Source: _collective_contracts.py"

    ```python
    """Small parsers and source readers for the collective-contract reproductions."""

    import numpy as np
    from _bmad import fortran, source


    def source_text(path: str) -> str:
        """Read the explicitly supplied upstream source tree."""
        return (source() / path).read_text()


    def rows(log: str) -> dict[str, np.ndarray]:
        """Require unique labelled numeric results; preserve native failures."""
        result = {}
        for line in log.splitlines():
            if not line.startswith("RESULT "):
                continue
            _, key, *values = line.split()
            if key in result or not values:
                raise RuntimeError(f"Invalid result record: {line}")
            result[key] = np.array([float(v.replace("D", "E")) for v in values])
        if not result:
            raise RuntimeError(f"No RESULT records:\n{log}")
        return result


    def parse_lattice(element: str, extra: str = "") -> tuple[bool, str]:
        """Exercise the production parser, retaining its explicit error return."""
        result = fortran(
            """program parse
    use bmad
    implicit none
    type(lat_struct) lat
    logical err
    global_com%exit_on_error = .false.
    call bmad_parser('lat.bmad', lat, err_flag=err)
    write(*,'(a,i0)') 'RESULT error ', merge(1,0,err)
    if (.not. err) then
      write(*,'(a,i0)') 'RESULT csr_method ', lat%ele(1)%csr_method
    endif
    end program
    """,
            files={
                "lat.bmad": "parameter[particle]=electron\nparameter[p0c]=1e9\n"
                + extra
                + "e: "
                + element
                + "\nl: line=(e)\nuse, l\n"
            },
        )
        data = rows(result.log)
        if data["error"].shape != (1,):
            raise RuntimeError(result.log)
        return bool(data["error"][0] == 0), result.log


    def section(text: str, start: str, end: str) -> str:
        """Select one explicit source passage; refuse missing or ambiguous anchors."""
        if text.count(start) != 1:
            raise RuntimeError(f"Expected one source anchor: {start}")
        tail = text.split(start, 1)[1]
        if end not in tail:
            raise RuntimeError(f"Missing source endpoint: {end}")
        return start + tail.split(end, 1)[0] + end


    def bounds_checked(program: str, args: list[str]) -> str:
        """Recompile exact upstream source with bounds checks, using the build's recipe."""
        from pathlib import Path

        from _bmad import TIMEOUT, _execute, _recipe, _scratch

        compile_command, link_command = _recipe(False)
        # Last compiler options take precedence over production optimization settings.
        compile_command.extend(["-O0", "-fcheck=bounds"])
        with _scratch() as tmp:
            work = Path(tmp)
            (work / "probe.f90").write_text(program)
            _execute(compile_command, work, "", TIMEOUT)
            _execute(link_command, work, "", TIMEOUT)
            return _execute([str(work / "probe"), *args], work, "", TIMEOUT)

    ```

## `_collective_fields.py` { #_collective_fields }

[Download _collective_fields.py](_collective_fields.py)

??? example "Source: _collective_fields.py"

    ```python
    """Parsing and independent cell quadrature for the collective-fields reproducers."""

    import re

    import numpy as np
    from _bmad import run


    def rows(log: str) -> dict[str, np.ndarray]:
        """Require unique labelled numerical records and reject native diagnostics."""
        if re.search(r"\[(?:ERROR|FATAL|ABORT|SEVERE)\b|Fortran runtime error|ERROR STOP", log):
            raise RuntimeError(log)
        result = {}
        for line in log.splitlines():
            if not line.startswith("RESULT "):
                continue
            _, label, *values = line.split()
            if label in result or not values:
                raise ValueError(f"Invalid result record: {line}")
            result[label] = np.array([float(v.replace("D", "E")) for v in values])
        if not result:
            raise ValueError(f"No result records: {log}")
        return result


    def cell_field(target, center, charge, spacing, order=40):
        """Coulomb field of a uniform cell, by tensor-product Gauss–Legendre integration."""
        nodes, weights = np.polynomial.legendre.leggauss(order)
        xyz = np.stack(np.meshgrid(nodes, nodes, nodes, indexing="ij"), axis=-1)
        weight = np.einsum("i,j,k->ijk", weights, weights, weights) / 8
        displacement = np.asarray(target) - np.asarray(center) - xyz * np.asarray(spacing) / 2
        radius2 = np.sum(displacement**2, axis=-1)
        # The weights average over the cell: charge is the total cell charge, in C.
        return (
            299792458.0**2
            * 1.00000000055e-7
            * charge
            * np.sum(displacement / radius2[..., None] ** 1.5 * weight[..., None], axis=(0, 1, 2))
        )


    def tao_covariance(method: str, images: int = 0) -> tuple[np.ndarray, bool]:
        """Fresh deterministic Tao bunch; return covariances and configuration rejection."""
        lattice = f"""no_digested
    parameter[geometry] = open
    parameter[particle] = electron
    parameter[p0c] = 10e6
    beginning[beta_a] = 100
    beginning[beta_b] = 100
    bmad_com[csr_and_space_charge_on] = T
    space_charge_com[n_bin] = 20
    space_charge_com[n_shield_images] = {images}
    space_charge_com[beam_chamber_height] = 0.01
    d: drift, l = 0.01, space_charge_method = {method}, csr_ds_step = 0.01
    lat: line = (d)
    use, lat
    """
        commands = """set beam_init n_particle = 2000
    set beam_init random_engine = quasi
    set beam_init a_emit = 1e-9
    set beam_init b_emit = 1e-9
    set beam_init sig_z = 1e-3
    set beam_init bunch_charge = 1e-11
    set global track_type = beam
    pipe bunch_params end
    quit
    """
        log = run(
            "tao",
            ["-noplot", "-noinit", "-lat", "lat.bmad", "-startup", "run.tao"],
            files={"lat.bmad": lattice, "run.tao": commands},
        ).log
        if "Tao: quit" not in log:
            raise RuntimeError(log)
        diagnostic = re.search(r"\[(?:ERROR|FATAL|ABORT|SEVERE|WARNING)\b", log)
        # A future rejection must identify this unsupported combination, not an unrelated failure.
        rejected = bool(
            diagnostic
            and method == "cathode_fft_3d"
            and "CATHODE_FFT_3D" in log
            and "TRACKING" in log.upper()
        )
        if diagnostic and not rejected:
            raise RuntimeError(log)
        if rejected:
            return np.array([]), True
        values = {}
        for key in ("sigma_12", "sigma_56", "n_particle_live"):
            matches = re.findall(rf"^{key};[^;]+;[^;]+;\s*([^;\n]+)", log, re.MULTILINE)
            if len(matches) != 1:
                raise ValueError(f"Missing or duplicate {key}: {log}")
            values[key] = float(matches[0])
        if values["n_particle_live"] != 2000:
            raise ValueError(f"Particle-count control failed: {values}")
        return np.array([values["sigma_12"], values["sigma_56"]]), False

    ```

## `_dispersion_and_moments.py` { #_dispersion_and_moments }

[Download _dispersion_and_moments.py](_dispersion_and_moments.py)

??? example "Source: _dispersion_and_moments.py"

    ```python
    """Deterministic dispersion and beam-moment probes, using the selected Bmad build."""

    import re

    import numpy as np
    from _bmad import fortran


    def records(log):
        """Read unique RESULT records and reject native diagnostics."""
        if re.search(r"\[(?:ERROR|FATAL|ABORT|SEVERE)\b|ERROR STOP|Fortran runtime error", log):
            raise RuntimeError(log)
        result = {}
        for line in log.splitlines():
            if not line.startswith("RESULT "):
                continue
            _, key, *values = line.split()
            if key in result:
                raise RuntimeError(f"Duplicate record: {key}")
            result[key] = np.array([float(x) for x in values])
            if not result[key].size or not np.isfinite(result[key]).all():
                raise RuntimeError(f"Invalid numeric record: {line}")
        if not result:
            raise RuntimeError(f"No result records:\n{log}")
        return result


    def execute(program, files=None, args=()):
        return records(fortran(program, files=files, args=args).log)


    def require(actual, expected, *, atol=1e-15, rtol=3e-14):
        """Controls fail the run instead of supplying a defect verdict."""
        if not np.isfinite(np.asarray(actual)).all() or not np.isfinite(np.asarray(expected)).all():
            raise ValueError("Controls require finite observed and expected values")
        np.testing.assert_allclose(actual, expected, atol=atol, rtol=rtol, equal_nan=False)


    def lattice(elements, *, p0c=1e6, species="electron", start=None, extra="", line="e", closed=False):
        start = np.zeros(6) if start is None else start
        return (
            f"parameter[geometry] = {'closed' if closed else 'open'}\n"
            f"parameter[particle] = {species}\nparameter[p0c] = {p0c:.17g}\n"
            "beginning[beta_a] = 3\nbeginning[alpha_a] = 0.4\nbeginning[beta_b] = 4\n"
            + "\n".join(
                f"particle_start[{k}] = {v:.17g}"
                for k, v in zip(("x", "px", "y", "py", "z", "pz"), start, strict=True)
            )
            + f"\n{elements}\n{extra}\nbl: line = ({line})\nuse, bl\n"
        )


    def quad(k, length, momentum=1.0):
        root = np.sqrt(abs(k) / momentum)
        phase = root * length
        c, s = (np.cos(phase), np.sin(phase)) if k > 0 else (np.cosh(phase), np.sinh(phase))
        return np.array([[c, s / (momentum * root)], [-np.sign(k) * momentum * root * s, c]])


    def fodo_matrix(delta=0.0, *, skew=False):
        momentum = 1 + delta
        drift = np.array([[1, 0.8 / momentum], [0, 1]])
        m = np.zeros((4, 4))
        for block, sign in [(slice(0, 2), 1), (slice(2, 4), -1)]:
            m[block, block] = (
                drift @ quad(-0.96 * sign, 0.2, momentum) @ drift @ quad(sign, 0.2, momentum)
            )
        if skew:
            kick = np.eye(4)
            kick[1, 2] = kick[3, 0] = -0.003
            m = kick @ m
        return m


    PERIODIC = r"""
    program dispersion_oracle
    use bmad
    implicit none
    type(ele_struct) ele
    real(rp) m(6,6), orb(6), growth
    integer i, stat
    logical stable, normalized
    open(10, file='input.dat')
    read(10,*) normalized
    read(10,*) orb
    do i=1,6
      read(10,*) m(i,:)
    enddo
    close(10)
    bmad_com%normalize_twiss = normalized
    call twiss_from_mat6(m, orb, ele, stable, growth, stat, .true.)
    if (.not. stable .or. stat /= ok$) error stop 'unstable test map'
    write(*,'(a,8es25.17)') 'RESULT dispersion ', ele%x%eta, ele%x%etap, ele%y%eta, ele%y%etap, &
      ele%x%deta_ds, ele%y%deta_ds, ele%a%deta_ds, ele%b%deta_ds
    end program
    """


    def periodic(normalized):
        orbit = np.array([0, 0.03, 0, -0.02, 0, 0.2])
        eta = np.array([0.8, 0.13, 0.4, -0.05])
        mat = np.eye(6)
        mat[:4, :4] = fodo_matrix(orbit[5])
        mat[:4, 5] = (np.eye(4) - mat[:4, :4]) @ eta
        # Symplectic completion of the longitudinal row for the imposed dispersion.
        j4 = np.kron(np.eye(2), [[0, 1], [-1, 0]])
        mat[4, :4] = mat[:4, 5] @ j4 @ mat[:4, :4]
        text = ("T" if normalized else "F") + "\n"
        text += " ".join(map(str, orbit)) + "\n"
        text += "\n".join(" ".join(map(str, row)) for row in mat) + "\n"
        values = execute(PERIODIC, {"input.dat": text})["dispersion"]
        return values, orbit, eta, mat


    OPEN_LINE = r"""
    program open_line
    use bmad
    implicit none
    type(lat_struct) lat
    type(coord_struct), allocatable :: orb(:)
    integer stat
    call bmad_parser('lat.bmad',lat)
    call twiss_and_track(lat,orb,stat,use_particle_start=.true.)
    if (stat /= ok$) error stop 'twiss_and_track failed'
    write(*,'(a,4es25.17)') 'RESULT entrance ',lat%ele(0)%x%eta,lat%ele(0)%x%etap, &
     lat%ele(0)%y%eta,lat%ele(0)%y%etap
    write(*,'(a,4es25.17)') 'RESULT exit ',lat%ele(1)%x%eta,lat%ele(1)%x%etap, &
     lat%ele(1)%y%eta,lat%ele(1)%y%etap
    write(*,'(a,4es25.17)') 'RESULT slopes ',lat%ele(1)%x%deta_ds,lat%ele(1)%y%deta_ds, &
     lat%ele(1)%a%deta_ds,lat%ele(1)%b%deta_ds
    write(*,'(a,6es25.17)') 'RESULT orbit ',orb(1)%vec
    if (orb(1)%state /= alive$) error stop 'lost particle'
    end program
    """


    MOMENTS = r"""
    program moments
    use bmad
    use beam_utils
    implicit none
    type(lat_struct) lat
    type(ele_struct) ele
    type(bunch_struct) bunch
    type(bunch_params_struct) bp
    type(coord_struct) before
    real(rp) v(6), q, pbase, pnew, nm(6,6), invariant_error(4)
    integer iu,nc,ic,n,i,j,species
    logical err
    character(32) species_label,tag
    call bmad_parser('lat.bmad',lat)
    ele=lat%ele(0)
    open(newunit=iu,file='input.dat',status='old')
    read(iu,*) nc
    do ic=1,nc
      read(iu,*) n,species_label,pbase,pnew
      species=species_id(species_label)
      if (species == invalid$) error stop 'invalid species'
      ele%value(p0c$)=pbase
      ele%value(e_tot$)=sqrt(pbase**2+mass_of(species)**2)
      call reallocate_bunch(bunch,n)
      invariant_error=0
      do i=1,n
        read(iu,*) q,v
        call init_coord(bunch%particle(i),v,ele,downstream_end$,species,shift_vec6=.false.)
        bunch%particle(i)%charge=q
        before=bunch%particle(i)
        call orbit_reference_energy_correction(bunch%particle(i),pnew)
        invariant_error(1)=max(invariant_error(1),maxval(abs( &
          before%p0c*before%vec([2,4])-bunch%particle(i)%p0c*bunch%particle(i)%vec([2,4])))/pbase)
        invariant_error(2)=max(invariant_error(2),abs( &
          before%p0c*(1+before%vec(6))-bunch%particle(i)%p0c*(1+bunch%particle(i)%vec(6)))/pbase)
        invariant_error(3)=max(invariant_error(3),abs(before%t-bunch%particle(i)%t))
        invariant_error(4)=max(invariant_error(4),abs(before%beta-bunch%particle(i)%beta))
      enddo
      if (any(bunch%particle%state/=alive$)) error stop 'lost test particle'
      call calc_bunch_params(bunch,bp,err,n_mat=nm)
      if (err .or. .not.bp%twiss_valid) error stop 'moment calculation failed'
      write(tag,'(a,i0)') 'meta',ic
      call row(tag,[mass_of(species),invariant_error])
      write(tag,'(a,i0)') 'emit',ic
      call row(tag,[bp%x%emit,bp%x%norm_emit,bp%y%emit,bp%y%norm_emit,bp%a%norm_emit,bp%b%norm_emit])
      write(tag,'(a,i0)') 'disp',ic
      call row(tag,[bp%x%eta,bp%x%etap,bp%y%eta,bp%y%etap,bp%a%eta,bp%a%etap,bp%b%eta,bp%b%etap,bp%c%gamma])
      do j=1,6
        write(tag,'(a,i0,a,i0)') 'sigma',ic,'_',j
        call row(tag,bp%sigma(j,:))
        if (bp%sigma(5,5)>0 .and. bp%sigma(6,6)>0) then
          write(tag,'(a,i0,a,i0)') 'nmat',ic,'_',j
          call row(tag,nm(j,:))
        endif
      enddo
    enddo
    close(iu)
    contains
    subroutine row(name,values)
    character(*) name
    real(rp) values(:)
    write(*,'(a,a,100(1x,es25.17))') 'RESULT ',trim(name),values
    end subroutine
    end program
    """


    def twiss(beta, alpha, emit):
        return np.sqrt(emit) * np.array(
            [[np.sqrt(beta), 0], [-alpha / np.sqrt(beta), 1 / np.sqrt(beta)]]
        )


    def moments(points, weights):
        weights = weights / weights.sum()
        mean = weights @ points
        offset = points - mean
        return mean, (offset.T * weights) @ offset


    def moment_cases(cases, weights):
        lines = [str(len(cases))]
        for case in cases:
            lines.append(f"64 {case['species']} {case['pbase']:.17e} {case['pnew']:.17e}")
            lines.extend(
                " ".join(f"{x:.17e}" for x in [w, *p])
                for w, p in zip(weights, case["points"], strict=True)
            )
        return execute(
            MOMENTS, {"input.dat": "\n".join(lines) + "\n", "lat.bmad": lattice("e: drift, l=0.1")}
        )


    RING = r"""
    ! 16-cell FODO ring of 32 identical sector bends (rho = 32/pi m), gamma = 1000 for every species.
    ! Strengths are geometric (angle, k1), so the optics do not depend on the species.
    parameter[particle] = SPECIES
    parameter[geometry] = GEOMETRY
    beginning[e_tot] = 1000 * mass_of(SPECIES)
    beginning[beta_a] = 10
    beginning[beta_b] = 10
    b: sbend, l = 2, angle = pi/16
    qf: quadrupole, l = 0.5, k1 = 0.6
    qd: quadrupole, l = 0.5, k1 = -0.6
    d: drift, l = 1
    cell: line = (qf, d, b, d, qd, d, b, d)
    ring: line = (16*cell)
    use, ring
    """


    def radiation_ring(species="electron", closed=True, rf=False):
        text = RING.replace("SPECIES", species).replace("GEOMETRY", "closed" if closed else "open")
        if rf:
            text = text.replace(
                "ring: line = (16*cell)",
                "rf: rfcavity, l=0, voltage=1e4, harmon=100, phi0=0\nring: line = (8*cell,rf,8*cell,rf)",
            )
        return text


    RADIATION = r"""
    program radiation
    use bmad
    use rad_6d_mod
    implicit none
    type(lat_struct),target :: lat
    type(coord_struct),allocatable :: orbit(:)
    type(normal_modes_struct) mode,mode6
    type(rad_map_struct) rmap
    real(rp) sigma(6,6),mc2,energy,gamma,rc,cq,m(6,6),m65,nd(6,6)
    integer stat,i,opening
    logical err
    character(32) tag
    call bmad_parser('lat.bmad',lat)
    if (lat%param%geometry == closed$) then
      call twiss_and_track(lat,orbit,stat)
      if (stat /= ok$) error stop 'twiss_and_track failed'
    else
      call reallocate_coord(orbit,lat%n_ele_track)
      call init_coord(orbit(0),lat%ele(0),downstream_end$)
      call track_all(lat,orbit,err_flag=err)
      if (err) error stop 'open-line tracking failed'
      call twiss_propagate_all(lat,err_flag=err)
      if (err) error stop 'open-line Twiss failed'
    endif
    if (any(orbit%state/=alive$)) error stop 'orbit lost'
    call radiation_integrals(lat,orbit,mode)
    mc2=mass_of(lat%param%particle)
    energy=lat%ele(0)%value(e_tot$)
    gamma=energy/mc2
    rc=classical_radius(lat%param%particle)
    cq=55*h_bar_planck*c_light/(32*sqrt(3.0_rp)*mc2)
    call row('constants',[mc2,energy,gamma,rc,cq,real(charge_of(lat%param%particle),rp)])
    call row('integrals',mode%synch_int)
    call row('loss',[mode%e_loss,mode%a%alpha_damp,mode%b%alpha_damp,mode%z%alpha_damp])
    call row('partition',[mode%a%j_damp,mode%b%j_damp,mode%z%j_damp])
    call row('linac',[mode%lin%sig_E1,mode%lin%i3_E7,mode%lin%a_emittance_end, &
     mode%lin%b_emittance_end,mode%lin%i5a_E6,mode%lin%i5b_E6])
    call row('spread',[mode%sig_z,mode%sigE_E,mode%z%emittance])
    if (lat%param%geometry == closed$) then
      call transfer_matrix_calc(lat,m)
      m65=0
      do i=1,lat%n_ele_track
        if (lat%ele(i)%key == rfcavity$) m65=m65+lat%ele(i)%mat6(6,5)
      enddo
      call row('m65',[m65,m(6,5)])
      do opening=0,1
        call emit_6d(lat%ele(0),opening==1,mode6,sigma,closed_orbit=orbit)
        write(tag,'(a,i0)') 'native',opening
        call row(tag,[mode6%z%emittance,mode6%z%emittance_no_vert,mode6%sig_z,mode6%sigE_E, &
                      mode6%e_loss,mode6%dpz_damp,sigma(5,5),sigma(6,6),mode6%a%alpha_damp])
      enddo
      call rad_damp_and_stoc_mats(lat%ele(0),lat%ele(0),.false.,rmap,mode6,nd,err,closed_orbit=orbit)
      if (err) error stop 'radiation map failed'
      call row('damped65',[rmap%xfer_damp_mat(6,5)])
    endif
    contains
    subroutine row(name,values)
    character(*) name
    real(rp) values(:)
    write(*,'(a,a,100(1x,es25.17))') 'RESULT ',trim(name),values
    end subroutine
    end program
    """

    ```

## `_ibs_charge_and_modes.py` { #_ibs_charge_and_modes }

[Download _ibs_charge_and_modes.py](_ibs_charge_and_modes.py)

??? example "Source: _ibs_charge_and_modes.py"

    ```python
    """Build the BBU probe with the pin's libbsim and the shared Bmad recipe."""

    from pathlib import Path

    from _bmad import Result, _case, _execute, _recipe, _scratch, build


    def bbu_fortran(program: str, files: dict[str, str]) -> Result:
        compile_argv, link_argv = _recipe(False)
        library = build() / "production/lib/libbsim.a"
        if not library.is_file():
            raise RuntimeError(f"Missing BBU library: {library}")
        # Static libraries must precede the Bmad libraries that satisfy their references.
        link_argv.insert(link_argv.index("probe.o") + 1, str(library))
        with _scratch() as tmp:
            work = Path(tmp)
            (work / "probe.f90").write_text(program)
            _execute(compile_argv, work, "", 290)
            _execute(link_argv, work, "", 290)
            return _case([str(work / "probe")], work / "run", files, (), "", 290)

    ```

## `_lattice_import_export.py` { #_lattice_import_export }

[Download _lattice_import_export.py](_lattice_import_export.py)

??? example "Source: _lattice_import_export.py"

    ```python
    """Inline lattice and translator runners for the lattice import/export cases."""

    import os
    import re
    import subprocess
    import sys
    import tempfile
    from contextlib import contextmanager
    from pathlib import Path

    from _bmad import Result, fortran, run, source

    HEADER = """no_digested
    parameter[geometry] = open
    parameter[particle] = electron
    parameter[p0c] = 3e9
    beginning[beta_a] = 10
    beginning[beta_b] = 10
    """
    PROBE = """program probe
    use bmad
    implicit none
    type(lat_struct) lat
    type(coord_struct), allocatable :: orbit(:)
    character(20) fmt, mode
    integer i, status, ix_pole
    real(rp) aa(0:n_pole_maxx), bb(0:n_pole_maxx)
    logical err
    call get_command_argument(1, fmt)
    call get_command_argument(2, mode)
    global_com%exit_on_error = .false.
    call bmad_parser('lat.bmad', lat, err_flag=err)
    print *, 'PARSE_ERROR ', err
    if (err) stop
    global_com%exit_on_error = .true.
    ! Imported LINE definitions default to closed geometry; track each fixture once as a transfer line.
    lat%branch(0)%param%geometry = open$
    ! Importers omit initial Twiss parameters; give those fixtures explicit finite optical initial conditions.
    if (lat%ele(0)%a%beta == 0) lat%ele(0)%a%beta = 10
    if (lat%ele(0)%b%beta == 0) lat%ele(0)%b%beta = 10
    if (mode == 'noorbit') then
      call write_lattice_foreign_format(trim(fmt), 'out', lat, err=err)
    else
      call twiss_and_track(lat, orbit, status, orb_start=lat%particle_start)
      if (status /= ok$) stop 2
      do i=0,lat%n_ele_track
        print '(a,i0,1x,a,1x,a)', 'ELEMENT ', i, trim(lat%ele(i)%name), trim(key_name(lat%ele(i)%key))
        print '(a,6es25.16)', 'ORBIT ', orbit(i)%vec
        print '(a,4es25.16)', 'DISP ', lat%ele(i)%a%eta,lat%ele(i)%a%etap,lat%ele(i)%b%eta,lat%ele(i)%b%etap
        print '(a,es25.16)', 'P0C ',lat%ele(i)%value(p0c$)
        if (lat%ele(i)%key == sbend$) then
          call multipole_ele_to_ab(lat%ele(i), .false., ix_pole, aa, bb, magnetic$, include_kicks$)
          print '(a,es25.16)', 'FIELD ',aa(0)
        endif
        if (associated(lat%ele(i)%a_pole)) print '(a,es25.16)', 'A0 ',lat%ele(i)%a_pole(0)
      enddo
      if (len_trim(fmt) > 0) call write_lattice_foreign_format(trim(fmt), 'out', lat, orbit, err=err)
    endif
    if (len_trim(fmt) > 0 .and. err) stop 3
    print *, 'COMPLETE'
    end program
    """


    def inspect(lattice: str, fmt: str = "", *, noorbit: bool = False) -> Result:
        return fortran(
            PROBE,
            [fmt, "noorbit" if noorbit else "orbit"],
            files={"lat.bmad": lattice},
            outputs=["out"] if fmt else [],
        )


    def valid(result: Result) -> None:
        if "PARSE_ERROR  F" not in result.log or "COMPLETE" not in result.log:
            raise RuntimeError(result.log)
        if re.search(r"\[(?:ERROR|FATAL|ABORT)", result.log):
            raise RuntimeError(result.log)


    def export(lattice: str, fmt: str = "madx", *, noaperture: bool = False) -> str:
        suffix = {"madx": "madx", "elegant": "lte", "sad": "sad"}[fmt]
        args = [f"-{fmt}"] + (["-noaperture"] if noaperture else []) + ["lat.bmad"]
        result = run(
            "bmad_to_mad_sad_elegant", args, files={"lat.bmad": lattice}, outputs=[f"lat.{suffix}"]
        )
        if "Written" not in result.log or re.search(r"\[(?:ERROR|FATAL|ABORT)", result.log):
            raise RuntimeError(result.log)
        return result.outputs[f"lat.{suffix}"].decode()


    def translate(lattice: str, kind: str = "madx") -> tuple[int, str, str]:
        relative, suffix = {
            "madx": ("mad_to_bmad/madx_to_bmad.py", "madx"),
            "elegant": ("elegant_to_bmad/elegant_to_bmad.py", "lte"),
        }[kind]
        script = source() / "util_programs" / relative
        if not script.is_file():
            raise RuntimeError(f"Missing translator: {script}")
        parent = Path(tempfile.gettempdir()) / "bmad-scratch"
        parent.mkdir(exist_ok=True)
        with tempfile.TemporaryDirectory(prefix="lattice-import-export-", dir=parent) as tmp:
            work = Path(tmp)
            (work / f"lat.{suffix}").write_text(lattice)
            result = subprocess.run(
                [sys.executable, str(script), f"lat.{suffix}"],
                cwd=work,
                env=dict(
                    os.environ, OMP_NUM_THREADS="1", OPENBLAS_NUM_THREADS="1", MKL_NUM_THREADS="1"
                ),
                text=True,
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                timeout=290,
                check=False,
            )
            output = (work / "lat.bmad").read_text() if result.returncode == 0 else ""
            return result.returncode, result.stdout, output


    def definition(text: str, name: str) -> str:
        flat = re.sub(r"&\s*\n\s*", " ", text)
        flat = re.sub(r",\s*\n\s*", ", ", flat)
        match = re.search(
            rf"^\s*{re.escape(name)}\s*:\s*(.*?)(?:;|$)", flat, re.MULTILINE | re.IGNORECASE
        )
        if match is None:
            raise RuntimeError(f"Missing definition {name}:\n{text}")
        return match[1]


    def number(text: str, key: str) -> float:
        match = re.search(rf"\b{re.escape(key)}\s*=\s*([^,;\s}})]+)", text, re.IGNORECASE)
        if match is None:
            raise RuntimeError(f"Missing explicit {key}: {text}")
        return float(match[1])


    def records(result: Result, label: str) -> list[list[float]]:
        rows = [
            [float(x) for x in line.split()[1:]]
            for line in result.log.splitlines()
            if line.startswith(label + " ")
        ]
        if not rows:
            raise RuntimeError(f"Missing {label}: {result.log}")
        return rows


    @contextmanager
    def madx():
        """Keep MAD-X/PTC diagnostic files inside an automatically removed scratch directory."""
        from cpymad.madx import Madx

        parent = Path(tempfile.gettempdir()) / "bmad-scratch"
        parent.mkdir(exist_ok=True)
        with (
            tempfile.TemporaryDirectory(prefix="lattice-import-export-madx-", dir=parent) as tmp,
            Madx(stdout=False, cwd=tmp) as instance,
        ):
            yield instance

    ```

## `_multipass_slicing.py` { #_multipass_slicing }

[Download _multipass_slicing.py](_multipass_slicing.py)

??? example "Source: _multipass_slicing.py"

    ```python
    """Input and output helpers for the multipass and slicing reproducers."""

    import re

    from _bmad import compare

    HEADER = """no_digested
    parameter[particle]=electron
    parameter[p0c]=1e9
    parameter[geometry]=open
    beginning[beta_a]=10
    beginning[beta_b]=10
    """


    def rows(log: str) -> dict[str, list[float]]:
        """Read unique RESULT records, rejecting Bmad diagnostics and malformed output."""
        if re.search(r"\[(?:ERROR|FATAL|ABORT|SEVERE)\b|ERROR STOP|Fortran runtime error:", log):
            raise RuntimeError(log)
        result = {}
        for line in log.splitlines():
            if not line.startswith("RESULT "):
                continue
            _, name, *values = line.split()
            if name in result or not values:
                raise ValueError(f"Duplicate or empty RESULT {name}: {log}")
            result[name] = [float(v.replace("D", "E")) for v in values]
        if not result:
            raise ValueError(f"Missing RESULT records: {log}")
        return result


    def control(label, observed, expected, *, atol=0.0, rtol=0.0):
        """A failed prerequisite is an execution failure, never a defect verdict."""
        if compare(label, observed, expected, atol=atol, rtol=rtol):
            raise RuntimeError(f"Control failed: {label}")

    ```

## `_orbital_matrices_radiation.py` { #_orbital_matrices_radiation }

[Download _orbital_matrices_radiation.py](_orbital_matrices_radiation.py)

??? example "Source: _orbital_matrices_radiation.py"

    ```python
    """Record parsing and checked-source execution for orbital-matrix/radiation cases."""

    from pathlib import Path

    import numpy as np
    from _bmad import TIMEOUT, Result, _case, _execute, _recipe, _scratch


    def records(log: str) -> dict[str, np.ndarray]:
        """Require unique, finite labelled records from a completed native probe."""
        rows = {}
        for line in log.splitlines():
            if not line.startswith("RESULT "):
                continue
            _, key, *values = line.split()
            if key in rows or not values:
                raise RuntimeError(f"Malformed or duplicate record: {line}")
            row = np.array([float(value) for value in values])
            if not np.isfinite(row).all():
                raise RuntimeError(f"Nonfinite record: {line}")
            rows[key] = row
        if not rows:
            raise RuntimeError(f"Missing numerical records:\n{log}")
        return rows


    def checked_fortran(program: str, lattice: str, args: tuple[str, ...] = ()) -> Result:
        """Compile an inline source/driver with bounds checking and the pin's recipe."""
        compile_argv, link_argv = _recipe(False)
        with _scratch() as tmp:
            work = Path(tmp)
            (work / "probe.f90").write_text(program)
            _execute([*compile_argv, "-O0", "-g", "-fcheck=all"], work, "", TIMEOUT)
            _execute(link_argv, work, "", TIMEOUT)
            return _case(
                [str(work / "probe"), *args], work / "run", {"lat.bmad": lattice}, (), "", TIMEOUT
            )

    ```

## `_rf_and_spin.py` { #_rf_and_spin }

[Download _rf_and_spin.py](_rf_and_spin.py)

??? example "Source: _rf_and_spin.py"

    ```python
    """Parsing and independent spin algebra for the RF and spin conversion chunk."""

    import math

    import numpy as np
    from _bmad import compare


    def rows(log, prefix="RESULT "):
        records = {}
        for line in log.splitlines():
            if line.startswith(prefix):
                key, *values = line[len(prefix) :].split()
                if key in records or not values:
                    raise ValueError(f"Duplicate or empty result: {line}")
                records[key] = np.array([float(v.replace("D", "E")) for v in values])
        if not records:
            raise ValueError(f"Missing numerical records:\n{log}")
        return records


    def control(label, observed, expected, *, atol, rtol=0.0):
        if compare(label, observed, expected, atol=atol, rtol=rtol):
            raise AssertionError(f"Control failed: {label}")


    def rotate(v, angle):
        c, s = math.cos(angle), math.sin(angle)
        return np.array([c * v[0] - s * v[1], s * v[0] + c * v[1], v[2]])


    def spin(q, initial):
        # Hamilton quaternion action; normalize the polynomial as Bmad's spin map requires.
        q = q / np.linalg.norm(q)
        return (
            (q[0] ** 2 - np.dot(q[1:], q[1:])) * initial
            + 2 * np.dot(q[1:], initial) * q[1:]
            + 2 * q[0] * np.cross(q[1:], initial)
        )


    def evaluate(terms, x, reference):
        output = np.zeros(4)
        for component, coefficient, exponents in terms:
            output[component] += coefficient * math.prod(
                float(v) ** power for v, power in zip(x - reference, exponents, strict=True)
            )
        return output


    def spin_data(log):
        records, terms = {}, {}
        for line in log.splitlines():
            fields = line.split()
            if not fields:
                continue
            if fields[0] == "TERM":
                tag, component, coefficient = fields[1], int(fields[2]), float(fields[3])
                terms.setdefault(tag, []).append((component, coefficient, tuple(map(int, fields[4:]))))
            elif fields[0].startswith(("P_", "R_")):
                if fields[0] in records:
                    raise ValueError(f"Duplicate result: {line}")
                records[fields[0]] = np.array(list(map(float, fields[1:])))
        return records, terms


    TAYLOR_PROGRAM = """program spin_check
    use bmad
    use ptc_interface_mod
    implicit none
    type(lat_struct), target :: lat
    type(ele_struct), pointer :: e
    type(coord_struct) o, ending
    integer n,i,j,k
    real(rp) x
    logical err
    character(16) arg
    character(80) prefix
    call get_command_argument(1,arg)
    read(arg,*) n
    call bmad_parser('verify.bmad',lat,err_flag=err)
    if(err) error stop 'parse'
    bmad_com%spin_tracking_on=.true.
    if(n==2) then
      call set_ptc(taylor_order=2)
      do i=1,2
        e=>lat%ele(i)
        call init_coord(o,[0.0012_rp,0.00008_rp,-0.0007_rp,-0.00003_rp,0.0_rp,0.0_rp],e,upstream_end$)
        o%spin=[0.3_rp,0.4_rp,sqrt(0.75_rp)]
        do j=0,1
          e%taylor_map_includes_offsets=(j==1)
          call kill_taylor(e%taylor)
          call kill_taylor(e%spin_taylor)
          e%tracking_method=bmad_standard$
          call ele_to_taylor(e)
          e%tracking_method=taylor$
          call track1(o,e,lat%param,ending,err_flag=err)
          if(err) error stop 'track'
          if(ending%state/=alive$) error stop 'spin Taylor particle lost'
          write(prefix,'(a,i0,a,i0)') 'P_',i-1,'_',j
          call emit(trim(prefix)//'_ANGLE',[e%value(tilt_tot$)])
          call emit(trim(prefix)//'_IN',o%vec)
          call emit(trim(prefix)//'_S0',o%spin)
          call emit(trim(prefix)//'_ACTUAL',ending%spin)
          call emit(trim(prefix)//'_ORBIT',ending%vec)
          call map_dump(prefix,e)
        enddo
      enddo
    else if(n==3) then
      e=>lat%ele(3)
      do i=0,2
        x=0
        if(i==1) x=0.001_rp
        if(i==2) x=-0.001_rp
        call init_coord(o,[x,0.0_rp,0.0_rp,0.0_rp,0.0_rp,0.0_rp],e,upstream_end$)
        o%spin=[0.0_rp,0.0_rp,1.0_rp]
        do j=0,1
          call kill_taylor(e%spin_taylor)
          if(j==0) call sprint_spin_taylor_map(e)
          ! j=1 deliberately leaves the map absent: normal track1 creates it.
          call track1(o,e,lat%param,ending,err_flag=err)
          if(err) error stop 'track'
          if(ending%state/=alive$) error stop 'sprint particle lost'
          write(prefix,'(a,i0,a,i0)') 'R_',i,'_',j
          call emit(trim(prefix)//'_IN',o%vec)
          call emit(trim(prefix)//'_ACTUAL',ending%spin)
          call emit(trim(prefix)//'_PARAM',[e%value(k1$),e%value(l$),e%value(e_tot$)/mass_of(e%ref_species),anomalous_moment_of(e%ref_species)])
          call map_dump(prefix,e)
        enddo
      enddo
    else
      error stop 'unknown spin case'
    endif
    contains
    subroutine emit(label,values)
    character(*) label
    real(rp) values(:)
    write(*,'(a,1x,*(es25.16e3,1x))') trim(label),values
    end subroutine
    subroutine map_dump(label,ele)
    character(*) label
    type(ele_struct) ele
    integer a,b
    call emit(trim(label)//'_ORBITREF',ele%taylor%ref)
    call emit(trim(label)//'_SPINREF',ele%spin_taylor_ref_orb_in)
    do a=0,3
      do b=1,size(ele%spin_taylor(a)%term)
        write(*,'(a,1x,a,1x,i0,1x,es25.16e3,1x,6(i0,1x))') 'TERM',trim(label),a,ele%spin_taylor(a)%term(b)%coef,ele%spin_taylor(a)%term(b)%expn
      enddo
    enddo
    end subroutine
    end program
    """

    TAYLOR_LATTICE = """no_digested
    parameter[geometry]=open
    parameter[particle]=electron
    parameter[p0c]=1e9
    beginning[beta_a]=10
    beginning[beta_b]=10
    p0: quadrupole,l=.6,k1=.65,fringe_type=none,spin_tracking_method=tracking
    p1: quadrupole,l=.6,k1=.65,tilt=.43,fringe_type=none,spin_tracking_method=tracking
    r: quadrupole,l=1,k1=.5,fringe_type=none,spin_tracking_method=sprint
    lat: line=(p0,p1,r)
    use,lat
    """


    MAGNUS_PROGRAM = """program spin_check
    use bmad
    implicit none
    type(lat_struct), target :: lat
    type(ele_struct), pointer :: ele
    type(coord_struct) :: a, fwd, returned, back, back_out
    type(em_field_struct) :: field
    real(rp) :: v(6), seed(3), deltas(3)
    integer :: config, ie, isp, id, ispin, method, species
    logical :: err
    character(40) :: filename

    bmad_com%spin_tracking_on = .true.
    bmad_com%electric_dipole_moment = 0
    bmad_com%radiation_damping_on = .false.
    bmad_com%radiation_fluctuations_on = .false.
    bmad_com%rel_tol_adaptive_tracking = 1e-12_rp
    bmad_com%abs_tol_adaptive_tracking = 1e-14_rp
    deltas = [-0.25_rp, 0.0_rp, 0.4_rp]
    do config = 1, 4
      write(filename, '(a,i0,a)') 'spin_', config, '.bmad'
      call bmad_parser(trim(filename), lat, err_flag=err)
      if(err) error stop 'parse'
      do ie = 1, 3
        ele => lat%ele(ie)
        do isp = 1, 2
          species = electron$
          if (isp == 2) species = positron$
          do id = 1, 3
            v = 0
            v(6) = deltas(id)
            do ispin = 1, 2
              seed = [0.36_rp, 0.48_rp, 0.8_rp]
              if (ispin == 2) seed = [0.0_rp, 0.0_rp, 1.0_rp]
              do method = 1, 3
                ele%tracking_method = bmad_standard$
                ele%spin_tracking_method = tracking$
                if (method == 1) ele%spin_tracking_method = magnus$
                if (method == 3) ele%tracking_method = runge_kutta$
                call init_coord(a, v, ele, upstream_end$, particle=species)
                a%spin = seed
                call em_field_calc(ele, lat%param, ele%value(l$)/2, a, .true., field, err_flag=err)
                if (err) error stop 'field error'
                call track1(a, ele, lat%param, fwd, err_flag=err)
                if (err .or. fwd%state /= alive$) error stop 'forward error'
                fwd%time_dir = -1
                call track1(fwd, ele, lat%param, returned, err_flag=err)
                if (err .or. returned%state /= alive$) error stop 'return error'
                call init_coord(back, v, ele, downstream_end$, particle=species)
                back%time_dir = -1
                back%spin = seed
                call track1(back, ele, lat%param, back_out, err_flag=err)
                if (err .or. back_out%state /= alive$) error stop 'backward error'
                write(*,'(a,6i4,*(es25.16e3,1x))') 'SPIN', config,ie,isp,id,ispin,method, &
                  real(charge_of(species),rp),real(charge_of(ele%ref_species),rp),anomalous_moment_of(species), &
                  a%p0c*(1+v(6)),field%B(3),ele%value(l$),ele%value(ks$),seed, &
                  fwd%spin,returned%spin,back_out%spin,maxval(abs(returned%vec-a%vec)),returned%t-a%t
              enddo
            enddo
          enddo
        enddo
      enddo
    enddo
    end program
    """


    def magnus_records():
        """432 axial-solenoid records; independent physical-field and integration controls."""
        from _bmad import fortran

        files = {}
        for config, (species, p0c) in enumerate(
            ((species, p0c) for species in ("electron", "positron") for p0c in (3e6, 8e8)), 1
        ):
            lines = [
                "no_digested",
                f"parameter[particle]={species}",
                f"parameter[p0c]={p0c}",
                "parameter[geometry]=open",
                "beginning[beta_a]=10",
                "beginning[beta_b]=10",
            ]
            for index, (length, ks) in enumerate(((0.17, 0.24), (1.3, -1.1), (0.7, 0.0)), 1):
                lines.append(
                    f"s{index}: solenoid, l={length}, ks={ks}, fringe_type=none, spin_tracking_method=magnus"
                )
            lines += ["lat: line=(s1,s2,s3)", "use,lat"]
            files[f"spin_{config}.bmad"] = "\n".join(lines) + "\n"
        log = fortran(MAGNUS_PROGRAM, files=files).log
        records = []
        maximum_orbit = maximum_clock = 0.0
        integration_error = {2: 0.0, 3: 0.0}
        endpoint_error = {2: 0.0, 3: 0.0}
        for line in log.splitlines():
            if not line.startswith("SPIN "):
                continue
            columns = line.split()
            config, ie, isp, idelta, ispin, method = map(int, columns[1:7])
            q, qref, anomaly, momentum, bz, length, ks, *values = map(float, columns[7:])
            seed, forward, returned, backward = np.array(values[:12]).reshape(4, 3)
            orbit_error, clock_error = values[12:]
            if not np.isfinite([q, qref, anomaly, momentum, bz, length, ks, *values]).all():
                raise ValueError("Nonfinite spin sweep output")
            p0c = 3e6 if config % 2 else 8e8
            if abs(bz - ks * p0c / (qref * 299792458.0)) > 1e-14:
                raise AssertionError("Solenoid field/rigidity control failed")
            # Axial T-BMT uses the physical B in tesla, charge in e units, and pc in eV.
            theta = -q * 299792458.0 * bz * (1 + anomaly) * length / momentum
            maximum_orbit = max(maximum_orbit, orbit_error)
            maximum_clock = max(maximum_clock, abs(clock_error))
            if method != 1:
                integration_error[method] = max(
                    integration_error[method], float(np.max(np.abs(returned - seed)))
                )
                # The spline integrator omits 1e-11 m at each edge. RK starts 1e-12 m inside.
                excluded = 2e-11 if method == 2 else 1e-12
                integrated = theta * (1 - excluded / length)
                endpoint_error[method] = max(
                    endpoint_error[method],
                    float(np.max(np.abs(forward - rotate(seed, integrated)))),
                    float(np.max(np.abs(backward - rotate(seed, -integrated)))),
                )
            if ie == 3 or ispin == 2:
                if (
                    max(
                        np.max(np.abs(forward - seed)),
                        np.max(np.abs(returned - seed)),
                        np.max(np.abs(backward - seed)),
                    )
                    > 3e-15
                ):
                    raise AssertionError("Zero-field or longitudinal-spin control failed")
            records.append(
                (
                    (config, ie, isp, idelta, ispin, method),
                    q == qref,
                    theta,
                    seed,
                    forward,
                    returned,
                    backward,
                )
            )
        if len(records) != 432 or len({r[0] for r in records}) != 432:
            raise ValueError(f"Expected 432 unique spin records, got {len(records)}")
        control("orbital round-trip maximum residual", maximum_orbit, 0, atol=1e-15)
        control("clock round-trip maximum residual (s)", maximum_clock, 0, atol=2e-23)
        for method in (2, 3):
            control(
                f"integration method {method} spin round-trip maximum",
                integration_error[method],
                0,
                atol=3e-15,
            )
            control(
                f"integration method {method} endpoint-adjusted T-BMT maximum",
                endpoint_error[method],
                0,
                atol=3e-15,
            )
        return records

    ```

## `_s3_11_particle_io.py` { #_s3_11_particle_io }

[Download _s3_11_particle_io.py](_s3_11_particle_io.py)

??? example "Source: _s3_11_particle_io.py"

    ```python
    """Small native particle-file cases and strict result parsing for this family of reproducers."""

    from collections.abc import Mapping

    import numpy as np
    from _bmad import fortran

    HEADER_FILE = """# species = electron
    # charge_tot = 1
    #! x charge state
    0 1 alive
    # species = proton
    # charge_tot = 12
    #! x charge state
    0 1 alive
    """
    COUNT_FILE = """# species = electron
    # p0c = 1e9
    # charge_tot = 6
    #! x charge state
    0 1 alive
    0 2 lost
    0 3 alive
    """


    def program(body: str) -> str:
        return (
            """
    program particle_io
    use beam_utils
    use random_mod
    implicit none
    type(ele_struct) ele
    type(lat_param_struct) param
    type(beam_init_struct) bi
    type(beam_struct), target :: beam, back
    type(coord_struct), pointer :: p
    type(random_state_struct) rs, initial
    real(rp) x, v(6), scalar(6), beta0
    integer i, j, k
    logical err
    character(40) label
    call ran_seed_put(12345)
    global_com%exit_on_error = .false.
    call init_ele(ele, drift$)
    ele%value(p0c$) = 1e9_rp
    ele%value(p0c_start$) = 1e9_rp
    ele%value(e_tot$) = hypot(1e9_rp, mass_of(electron$))
    ele%value(e_tot_start$) = ele%value(e_tot$)
    ele%a%beta = 1
    ele%b%beta = 1
    ele%orientation = 1
    param%particle = electron$
    bi%n_particle = 0
    """
            + body
            + """
    print *, 'COMPLETED'
    contains
    subroutine emit(name, values)
    character(*) name
    real(rp) values(:)
    write(*, '(3a, *(es26.17e3))') 'RESULT ', name, ' ', values
    end subroutine
    end program
    """
        )


    def rows(log: str) -> dict[str, np.ndarray]:
        result = {}
        for line in log.splitlines():
            if line.startswith("RESULT "):
                _, name, *values = line.split()
                if name in result or not values:
                    raise RuntimeError(f"Duplicate or empty result: {line}")
                values_array = np.array([float(v) for v in values])
                if not np.isfinite(values_array).all():
                    raise RuntimeError(f"Nonfinite result: {line}")
                result[name] = values_array
        if not result:
            raise RuntimeError(f"No results:\n{log}")
        return result


    def execute(body: str, *, files: Mapping[str, str | bytes] | None = None) -> dict[str, np.ndarray]:
        log = fortran(program(body), files=files).log
        if "COMPLETED" not in log or any(s in log for s in ("[ERROR", "[FATAL", "[ABORT")):
            raise RuntimeError(log)
        return rows(log)


    # Each read uses a fresh process, so reader global state cannot leak between policies.
    POLICY_SETUP = """
    ele%value(p0c$) = 1e6_rp
    ele%value(p0c_start$) = 1e6_rp
    ele%value(e_tot$) = hypot(1e6_rp, m_electron)
    call reallocate_beam(beam, 1, 1)
    p => beam%bunch(1)%particle(1)
    call init_coord(p, [0.001_rp, 0.04_rp, -0.002_rp, -0.02_rp, 0.01_rp, 0.1_rp], &
                    ele, upstream_end$, electron$)
    p%charge = 1e-12_rp
    p%t = 4.1e-9_rp
    beam%bunch(1)%charge_tot = 1e-12_rp
    beam%bunch(1)%charge_live = 1e-12_rp
    beam%bunch(1)%n_live = 1
    call emit('input', [p%vec, p%p0c, p%beta, real(p%t,rp), m_electron])
    call hdf5_write_beam('beam.h5', beam%bunch, .false., err)
    if (err) error stop 'HDF5 write failed'
    ele%value(p0c$) = 1.7e6_rp
    """


    def policy(mode: str) -> dict[str, np.ndarray]:
        options = {
            "omitted": "",
            "true": ", conserve_momentum=.true.",
            "false": ", conserve_momentum=.false.",
        }
        return execute(
            POLICY_SETUP
            + f"""
    call read_beam_file('beam.h5', back, bi, err, ele{options[mode]})
    if (err) error stop 'HDF5 read failed'
    if (size(back%bunch) /= 1) error stop 'bunch count changed'
    if (size(back%bunch(1)%particle) /= 1) error stop 'particle count changed'
    p => back%bunch(1)%particle(1)
    call emit('output', [p%vec, p%p0c, p%beta, real(p%t,rp)])
    """
        )

    ```

## `_s3_12_particle_contracts.py` { #_s3_12_particle_contracts }

[Download _s3_12_particle_contracts.py](_s3_12_particle_contracts.py)

??? example "Source: _s3_12_particle_contracts.py"

    ```python
    """Inline particle-contract fixtures for the LAT-13 through LAT-23 reproducers."""

    import io
    import re

    import h5py
    import numpy as np
    from _bmad import fortran, source

    BEAM = """
    program beam_io
      use bmad
      use beam_file_io
      implicit none
      type(ele_struct) ele
      type(beam_struct), target :: beam, back
      type(beam_init_struct) init
      type(coord_struct), pointer :: p
      integer i, ib, mode
      logical err
      character(30) arg, fname, label
      call get_command_argument(1, arg)
      read(arg, *) mode
      call init_ele(ele, drift$)
      ele%value(p0c$) = 1e6_rp
      ele%value(p0c_start$) = 1e6_rp
      ele%value(e_tot$) = hypot(1e6_rp, m_electron)
      if (mode == 0) then
        call reallocate_beam(beam, 2, 3)
        do ib = 1, 2
          do i = 1, 3
            p => beam%bunch(ib)%particle(i)
            call init_coord(p, [0.001_rp*i, 0.04_rp*i, -0.002_rp*i, -0.02_rp*i, &
                            0.01_rp*i, 0.1_rp*i], ele, upstream_end$, electron$)
            p%charge = i*1e-12_rp
            p%t = ib*4e-9_rp + i*1e-10_rp
            p%s = 0.7_rp*ib
            p%ix_ele = 10+ib
            p%ix_branch = ib
            p%ix_user = 100*ib+i
            p%spin = [0.3_rp, -0.4_rp, 0.5_rp]
            if (i == 2) p%state = lost$
            if (i == 3) p%direction = -1
          enddo
          beam%bunch(ib)%charge_tot = 6e-12_rp
          beam%bunch(ib)%charge_live = 4e-12_rp
          beam%bunch(ib)%n_live = 2
        enddo
        call write_beam_file('beam.txt', beam, file_format=ascii$)
        call hdf5_write_beam('beam.h5', beam%bunch, .false., err)
        if (err) error stop 'hdf5 write failed'
        call hdf5_write_beam('patches.h5', beam%bunch, .false., err, as_patches=.true.)
        if (err) error stop 'patch write failed'
        call write_beam_file('default_beam', beam)
        call write_beam_file('default.h5', beam)
        call write_beam_file('default.hdf5', beam)
        print *, 'WRITE COMPLETE'
        stop
      endif
      call get_command_argument(2, fname)
      call read_beam_file(trim(fname), back, init, err)
      if (err) error stop 'beam read failed'
      write(*, '(a, 2i8)') 'RESULT constants ', electron$, lost$
      do ib = 1, size(back%bunch)
        write(label, '(a,i0)') 'b', ib
        write(*, '(3a, 3es26.17e3)') 'RESULT ', trim(label), ' ', back%bunch(ib)%charge_tot, &
           back%bunch(ib)%charge_live, real(back%bunch(ib)%n_live,rp)
        do i = 1, size(back%bunch(ib)%particle)
          p => back%bunch(ib)%particle(i)
          write(label, '(a,i0,a,i0)') 'b', ib, 'p', i
          write(*, '(3a, 30es26.17e3)') 'RESULT ', trim(label), ' ', p%vec, p%p0c, &
            p%beta, real(p%t,rp), p%charge, p%s, real(p%direction,rp), real(p%state,rp), &
            real(p%species,rp), real(p%ix_user,rp), real(p%ix_ele,rp), real(p%ix_branch,rp), p%spin
        enddo
      enddo
    end program
    """

    INIT = """
    program initialization
      use bmad
      use beam_utils
      use random_mod
      implicit none
      type(ele_struct) ele
      type(lat_param_struct) param
      type(beam_init_struct) bi, used
      type(bunch_struct) bunch
      logical err
      integer i, iu
      character(32) kind
      call get_command_argument(1, kind)
      call init_ele(ele, drift$)
      ele%value(p0c$) = 1.0e6_rp
      ele%value(p0c_start$) = ele%value(p0c$)
      ele%value(e_tot$) = hypot(ele%value(p0c$), m_electron)
      ele%a%beta = 3.0_rp; ele%a%alpha = 0.7_rp
      ele%b%beta = 5.0_rp; ele%b%alpha = -0.4_rp
      ele%a%eta = 0.02_rp; ele%a%etap = -0.003_rp
      ele%b%eta = -0.01_rp; ele%b%etap = 0.002_rp
      param%particle = electron$
      bi%n_particle = 2048
      bi%a_norm_emit = 2.0e-6_rp
      bi%b_norm_emit = 3.0e-6_rp
      bi%sig_z = 0.004_rp; bi%sig_pz = 0.002_rp; bi%dpz_dz = 0.2_rp
      bi%bunch_charge = 2.3e-12_rp
      bi%center = [0.001_rp, -0.0002_rp, -0.002_rp, 0.0003_rp, 0.003_rp, 0.01_rp]
      bi%random_gauss_converter = 'exact'
      bi%renorm_center = .false. ! renorm_sigma itself centers the generated random sample.
      call ran_seed_put(314159)
      call init_bunch_distribution(ele, param, bi, 0, bunch, err, beam_init_used=used)
      if (err) error stop 'initialization failed'
      open(newunit=iu, file='particles.dat', status='replace')
      do i = 1, size(bunch%particle)
        write(iu, '(9es25.16e3)') bunch%particle(i)%vec, bunch%particle(i)%charge, &
          bunch%particle(i)%beta, bunch%particle(i)%t
      enddo
      close(iu)
      write(*, '(a,5es25.16e3)') 'RESULT init ', used%a_emit, used%b_emit, used%a_norm_emit, &
        used%b_norm_emit, m_electron
    end program
    """

    TIME = """
    program initial_time
      use bmad
      use beam_utils
      implicit none
      type(lat_struct) lat
      type(beam_init_struct) bi
      type(bunch_struct) bunch
      logical err
      call bmad_parser('lat.bmad',lat)
      bi%n_particle = 1
      bi%use_particle_start = .true.
      bi%t_offset = 3.0e-9_rp
      lat%ele(0)%a%beta = 1; lat%ele(0)%b%beta = 1
      call init_bunch_distribution(lat%ele(0),lat%param,bi,0,bunch,err)
      if (err) error stop 'beam initialization failed'
      write(*,'(a,4es25.16e3)') 'RESULT time ',bunch%particle(1)%t,bunch%particle(1)%vec(5),bunch%particle(1)%beta,m_electron
    end program
    """


    def rows(log: str) -> dict[str, np.ndarray]:
        result = {}
        for line in log.splitlines():
            if line.startswith("RESULT "):
                _, label, *values = line.split()
                if label in result:
                    raise RuntimeError(f"Duplicate result {label}")
                result[label] = np.array([float(v) for v in values])
        if not result or not all(np.isfinite(v).all() for v in result.values()):
            raise RuntimeError(f"Missing or nonfinite result: {log}")
        return result


    def beam_files() -> dict[str, bytes]:
        names = ("beam.txt", "beam.h5", "patches.h5", "default_beam", "default.h5", "default.hdf5")
        result = fortran(BEAM, ["0"], outputs=names)
        if "WRITE COMPLETE" not in result.log:
            raise RuntimeError(result.log)
        with h5py.File(io.BytesIO(result.outputs["beam.h5"]), "r") as file:
            for ib in (1, 2):
                record = file[f"data/{ib:05}/particles/electron"]
                np.testing.assert_allclose(values(record, "id"), 100 * ib + np.arange(1, 4), rtol=0)
                np.testing.assert_allclose(values(record, "particleStatus"), [1, 2, 1], rtol=0)
                if values(record, "momentum/z")[2] >= 0:
                    raise RuntimeError("Writer did not preserve the negative input momentum")
        return result.outputs


    def read_beam(data: bytes, name: str = "beam.h5") -> dict[str, np.ndarray]:
        result = fortran(BEAM, ["1", name], files={name: data})
        parsed = rows(result.log)
        expected = {"constants", "b1", "b2"} | {f"b{ib}p{i}" for ib in (1, 2) for i in (1, 2, 3)}
        if parsed.keys() != expected:
            raise RuntimeError(result.log)
        np.testing.assert_array_equal(parsed["constants"], [-1, 2])
        return parsed


    def values(record, name: str) -> np.ndarray:
        obj = record[name]
        if isinstance(obj, h5py.Dataset):
            return np.asarray(obj[()])
        if isinstance(obj, h5py.Group):
            return np.full(tuple(obj.attrs["shape"]), obj.attrs["value"])
        raise TypeError(name)


    def document_claim(path: str, phrase: str) -> bool:
        """Print whether the cited wording remains in the supplied source tree."""
        text = re.sub(r"\s+", " ", (source() / path).read_text())
        present = phrase in text
        print(f"document {path}: {phrase!r}; present={present}")
        return present

    ```

## `_s3_14_optics_matrix_radiation.py` { #_s3_14_optics_matrix_radiation }

[Download _s3_14_optics_matrix_radiation.py](_s3_14_optics_matrix_radiation.py)

??? example "Source: _s3_14_optics_matrix_radiation.py"

    ```python
    """Shared parsing and analytic quadrupole maps for the optics/matrix/radiation cases."""

    import re

    import numpy as np
    from _bmad import compare


    def rows(log: str) -> dict[str, np.ndarray]:
        """Require unique finite RESULT records and reject upstream error diagnostics."""
        if re.search(r"\[(?:ERROR|FATAL|ABORT|SEVERE)\b|Fortran runtime error", log):
            raise RuntimeError(log)
        result = {}
        for line in log.splitlines():
            fields = line.split()
            if not fields or fields[0] != "RESULT":
                continue
            name = fields[1]
            if name in result or len(fields) < 3:
                raise RuntimeError(f"Invalid result record: {line}")
            values = np.array([float(s.replace("D", "E")) for s in fields[2:]])
            if not np.isfinite(values).all():
                raise RuntimeError(f"Nonfinite result: {line}")
            result[name] = values
        if not result:
            raise RuntimeError(f"No result records:\n{log}")
        return result


    def control(label, observed, expected, *, atol=2e-14, rtol=2e-13):
        """A failed prerequisite is a failed run, never a defect verdict."""
        if compare(label, observed, expected, atol=atol, rtol=rtol):
            raise RuntimeError(f"Control failed: {label}")


    def quad(k, length):
        """Thick-lens solution of x'=p, p'=-k*x in slope coordinates."""
        root = np.sqrt(abs(k))
        phase = root * length
        if k > 0:
            c, s = np.cos(phase), np.sin(phase)
            return np.array([[c, s / root], [-root * s, c]])
        c, s = np.cosh(phase), np.sinh(phase)
        return np.array([[c, s / root], [root * s, c]])

    ```

## `_s3_19_cylindrical.py` { #_s3_19_cylindrical }

[Download _s3_19_cylindrical.py](_s3_19_cylindrical.py)

??? example "Source: _s3_19_cylindrical.py"

    ```python
    """Cylindrical-map fixtures and analytic curl for s3-19."""

    import cmath
    import math
    from typing import TypedDict

    import numpy as np

    C = 299792458.0


    class Mode(TypedDict):
        frequency: float
        index: int
        m: int
        family: str
        coefficient: complex
        dz: float
        theta: float
        phase: float
        terms: int


    class SampleGroup(TypedDict):
        ie: int
        point: int
        pos: np.ndarray
        center: int
        stencils: list[list[tuple[float, list[int]]]]


    def modes_and_samples():
        modes: list[Mode] = []
        for frequency in (8e7, 7e8):
            for index in (1, 7):
                for m in (1, 2, 4):
                    for family in ("b", "e"):
                        modes.append(
                            dict(
                                frequency=frequency,
                                index=index,
                                m=m,
                                family=family,
                                coefficient=complex(2e5, -1.2e5) * 0.01 ** (1 - m),
                                dz=0.11,
                                theta=0.43,
                                phase=0.11,
                                terms=8,
                            )
                        )
                        if family == "b" and m == 2:
                            scaled = modes[-1].copy()
                            scaled["coefficient"] *= -0.37
                            modes.append(scaled)
        lines = [
            "parameter[particle]=electron",
            "parameter[p0c]=8e8",
            "parameter[geometry]=open",
            "beginning[beta_a]=10",
            "beginning[beta_b]=10",
        ]
        for ie, mode in enumerate(modes, 1):
            arrays = {key: [0.0] * 8 for key in ("e_coef_re", "e_coef_im", "b_coef_re", "b_coef_im")}
            arrays[mode["family"] + "_coef_re"][mode["index"]] = mode["coefficient"].real
            arrays[mode["family"] + "_coef_im"][mode["index"]] = mode["coefficient"].imag
            attrs = ", ".join(
                f"{key}=(" + ",".join(map(str, value)) + ")" for key, value in arrays.items()
            )
            lines.append(
                f"f{ie}: em_field, l=.7, field_calc=fieldmap, rf_frequency={mode['frequency']}, "
                f"cylindrical_map={{m={mode['m']}, harmonic=1, dz={mode['dz']}, "
                f"theta0_azimuth={mode['theta']}, phi0_fieldmap={mode['phase']}, "
                f"ele_anchor_pt=beginning, {attrs}}}"
            )
        lines.extend(
            ["lat: line=(" + ",".join(f"f{i}" for i in range(1, len(modes) + 1)) + ")", "use,lat"]
        )
        lattice = "\n".join(lines) + "\n"
        samples = []
        groups: list[SampleGroup] = []
        for ie, mode in enumerate(modes, 1):
            for ip, xyz in enumerate(
                ((0.011, -0.007, 0.19), (-0.008, 0.014, 0.37), (0.017, 0.009, 0.51))
            ):
                pos = np.array([*xyz, (0.173 + ip * 0.079) / mode["frequency"]])
                center = len(samples)
                samples.append((ie, pos.copy()))
                group: SampleGroup = dict(ie=ie, point=ip, pos=pos, center=center, stencils=[])
                for divisor in (1, 2, 4):
                    stencils = []
                    for coordinate in range(4):
                        h = (2e-5 if coordinate < 3 else 1e-4 / mode["frequency"]) / divisor
                        ids = []
                        for offset in (-2, -1, 1, 2):
                            shifted = pos.copy()
                            shifted[coordinate] += offset * h
                            ids.append(len(samples))
                            samples.append((ie, shifted))
                        stencils.append((h, ids))
                    group["stencils"].append(stencils)
                groups.append(group)
        sample_file = "".join(
            f"{ident} {ie} " + " ".join(f"{v:.17g}" for v in pos) + "\n"
            for ident, (ie, pos) in enumerate(samples)
        )
        return modes, samples, groups, {"fields.bmad": lattice, "field_samples.dat": sample_file}


    def tilde_bessel(m, kappa_squared, radius):
        # I_m(kappa*r)/kappa**m as an entire series in kappa**2.
        term = complex((radius / 2) ** m / math.factorial(m))
        total = term
        for j in range(1, 80):
            term *= kappa_squared * radius**2 / (4 * j * (m + j))
            total += term
            if abs(term) < 1e-17 * abs(total):
                return total
        raise ArithmeticError("Bessel series did not converge")


    def oracle(mode, pos):
        x, y, z, time = pos
        radius, phi = math.hypot(x, y), math.atan2(y, x)
        m, frequency = mode["m"], mode["frequency"]
        fourier_index = mode["index"]
        if 2 * fourier_index >= mode["terms"]:
            fourier_index -= mode["terms"]
        k = 2 * math.pi * fourier_index / (mode["terms"] * mode["dz"])
        omega = 2 * math.pi * frequency
        k2 = k * k - (omega / C) ** 2
        f, g, h = [tilde_bessel(order, k2, radius) for order in (m, m - 1, m + 1)]
        df, dg, dh = g - m * f / radius, k2 * f + (m - 1) * g / radius, f - (m + 1) * h / radius
        e = mode["coefficient"] if mode["family"] == "e" else 0j
        b = mode["coefficient"] if mode["family"] == "b" else 0j
        cosine, sine = math.cos(m * phi - mode["theta"]), math.sin(m * phi - mode["theta"])
        er = -1j * (k * e * h + b * f / radius) * cosine
        ep = -1j * (k * e * h + b * (f / radius - g / m)) * sine
        ez = e * f * cosine
        # Cylindrical curl of the documented E; curl(E)=i*omega*B for exp(-i*omega*t).
        dep_dr = -1j * (k * e * dh + b * (df / radius - f / radius**2 - dg / m)) * sine
        der_dphi = 1j * m * (k * e * h + b * f / radius) * sine
        curl = np.array(
            [
                -m * e * f * sine / radius - 1j * k * ep,
                1j * k * er - e * df * cosine,
                ep / radius + dep_dr - der_dphi / radius,
            ]
        )
        bcyl = curl / (1j * omega)
        phase = cmath.exp(1j * (k * z - omega * time - 2 * math.pi * mode["phase"]))
        cp, sp = math.cos(phi), math.sin(phi)
        rotation = np.array([[cp, -sp, 0], [sp, cp, 0], [0, 0, 1]])
        expected_e = (rotation @ np.array([er, ep, ez]) * phase).real
        expected_b = (rotation @ bcyl * phase).real
        extra_br = (1j * b * k * (m - 1) * f / radius * sine / omega * phase).real
        return expected_e, expected_b, extra_br * np.array([cp, sp, 0])


    PROGRAM = r"""program field_check
    use bmad
    implicit none
    type(lat_struct), target :: lat
    type(coord_struct) :: orb
    type(em_field_struct) :: field
    real(rp) :: pos(4), v(6)
    integer :: ident, ie
    logical :: err
    call bmad_parser('fields.bmad', lat, err_flag=err)
    if(err) error stop 'parse failed'
    open(10, file='field_samples.dat', status='old', action='read')
    do
      read(10, *, end=100) ident, ie, pos
      v = 0
      v(1) = pos(1)
      v(3) = pos(2)
      call init_coord(orb, v, lat%ele(ie), upstream_end$)
      call em_field_calc(lat%ele(ie), lat%param, pos(3), orb, .true., field, &
                        err_flag=err, rf_time=pos(4), use_overlap=.false.)
      if (err) error stop 'field error'
      write(*,'(a,2i8,*(es25.16e3,1x))') 'SAMPLE', ident,ie,pos,field%E,field%B
    enddo
    100 close(10)
    print '(a)', 'DONE'
    end program
    """

    ```

## `_s3_19_fields_photons_ptc.py` { #_s3_19_fields_photons_ptc }

[Download _s3_19_fields_photons_ptc.py](_s3_19_fields_photons_ptc.py)

??? example "Source: _s3_19_fields_photons_ptc.py"

    ```python
    """Inline field and photon fixtures for the s3-19 conversion reproductions."""

    import cmath
    import math

    import numpy as np
    from _bmad import fortran


    def execute(program, args, files):
        output = fortran(program, args, files=files).log
        if "DONE" not in output.splitlines():
            raise RuntimeError(output)
        return output


    def records(output, prefixes):
        result = {}
        for line in output.splitlines():
            tokens = line.split()
            if tokens and tokens[0].startswith(prefixes):
                if tokens[0] in result:
                    raise RuntimeError(f"Duplicate record {tokens[0]}")
                values = np.array(list(map(float, tokens[1:])))
                if not np.isfinite(values).all():
                    raise RuntimeError(f"Nonfinite record {line}")
                result[tokens[0]] = values
        if not result:
            raise RuntimeError(output)
        return result


    def control(actual, expected, atol=2e-14):
        actual, expected = np.asarray(actual), np.asarray(expected)
        if (
            actual.shape != expected.shape
            or not actual.size
            or not np.isfinite(actual).all()
            or not np.isfinite(expected).all()
            or not math.isfinite(atol)
            or atol < 0
        ):
            raise AssertionError((actual, expected))
        error = float(np.max(np.abs(actual - expected)))
        if error > atol:
            raise AssertionError((error, atol, actual, expected))
        return error


    def field_data(case):
        return records(execute(FIELD, [str(case)], {"verify.bmad": make_lattice()}), ("G_", "F_", "W_"))


    def photon_data(case):
        return execute(PHOTON, [str(case)], {"checks.bmad": PHOTON_LATTICE})


    def rotate(v, angle):
        c, s = math.cos(angle), math.sin(angle)
        return np.array([c * v[0] - s * v[1], s * v[0] + c * v[1], v[2]])


    def rows(output, label):
        result = [
            line.split()[1:]
            for line in output.splitlines()
            if line.split() and line.split()[0] == label
        ]
        if not result:
            raise RuntimeError(f"Missing {label} records")
        return result


    def fresnel_stack(wavelength, chi1, chi2, d1, d2, count, angle, pol):
        """Match tangential E,H at each interface, recursively from a semi-infinite bottom layer.

        Y=kz/epsilon for pi and Y=kz for sigma. At interface i,j,
        E_ref/E_inc=(Y_i-Y_j)/(Y_i+Y_j). Each interior round trip adds exp(2 i kz d).
        This uses individual interfaces, independent of Bmad's analytic periodic-cell formula.
        """
        chi = [0, chi1, chi2]
        kz = [2 * math.pi / wavelength * cmath.sqrt(math.sin(angle) ** 2 + c) for c in chi]
        admittance = [kz[i] / (1 + chi[i]) if pol == "pi" else kz[i] for i in range(3)]
        media = [0] + [2, 1] * count
        thickness = [0, d1, d2]
        reflected = 0j
        for pos in range(len(media) - 2, -1, -1):
            i, j = media[pos : pos + 2]
            interface = (admittance[i] - admittance[j]) / (admittance[i] + admittance[j])
            from_deeper = reflected * cmath.exp(2j * kz[j] * thickness[j])
            reflected = (interface + from_deeper) / (1 + interface * from_deeper)
        return reflected


    def make_lattice():
        def grid(geometry, spacing, points):
            return (
                "grid_field={geometry="
                + geometry
                + ",field_type=magnetic,interpolation_order=1,dr=("
                + spacing
                + "),{\n"
                + ",\n".join(points)
                + "}}"
            )

        a = grid(
            "xyz",
            "1,1,1",
            [f"{x} {y} {z}: 0.7 -0.4 0.2" for x in (-1, 0, 1) for y in (-1, 0, 1) for z in (0, 1)],
        )
        b = grid(
            "rotationally_symmetric_rz",
            "1,1",
            [f"{r} {z}: 0 0 1.3" for r in (0, 1) for z in (0, 1)],
        )
        elements = [
            f"ga: quadrupole,l=1,field_calc=fieldmap,{a}",
            f"gb: quadrupole,l=1,field_calc=fieldmap,{b}",
            f"gab: quadrupole,l=1,field_calc=fieldmap,{a},{b}",
            f"gba: quadrupole,l=1,field_calc=fieldmap,{b},{a}",
        ]
        angles = ("0", "pi/6", "-pi/6")
        for group, attributes in (
            ("qe", "k1=0,b1_elec=1e5,scale_multipoles=F"),
            ("qk", "k1=0,hkick=.001"),
            ("qm", "k1=.5"),
        ):
            elements.extend(
                f"{group}{j}: quadrupole,l=1,tilt={angle},{attributes},fringe_type=none"
                for j, angle in enumerate(angles)
            )
        elements.extend(
            [
                "w0: wiggler,l=1,l_period=.2,b_max=1,kx=0,field_calc=planar_model",
                "w1: wiggler,l=1,l_period=.2,b_max=1,kx=13,field_calc=planar_model",
            ]
        )
        header = "parameter[geometry]=open\nparameter[particle]=electron\nparameter[p0c]=1e9\nbeginning[beta_a]=10\nbeginning[beta_b]=10\n"
        names = [element.split(":", 1)[0] for element in elements]
        content = header + "\n".join(elements) + "\nlat: line=(" + ",".join(names) + ")\nuse,lat\n"
        return content


    FIELD = r"""program field_check
    use bmad
    use em_field_mod
    implicit none
    type(lat_struct), target :: lat
    type(ele_struct), pointer :: e
    type(coord_struct) o, ob
    type(em_field_struct) f
    integer n, i, j, k, m
    real(rp) x, y, s, c, sn, xb, yb, h, sample_s
    real(rp), parameter :: steps(4) = [-2.0_rp,-1.0_rp,1.0_rp,2.0_rp]
    character(16) arg
    character(100) tag, prefix
    logical err
    call get_command_argument(1,arg)
    read(arg,*) n
    call bmad_parser('verify.bmad',lat,err_flag=err)
    if(err) error stop 'parse failed'
    if(n == 1) then
      do j=0,2
        x=0
        if(j==1) x=1e-9_rp
        if(j==2) x=0.001_rp
        do i=1,4
          e=>lat%ele(i)
          call init_coord(o,[x,0.0_rp,0.0_rp,0.0_rp,0.0_rp,0.0_rp],e,upstream_end$)
          call em_field_calc(e,lat%param,0.37_rp,o,.true.,f,err_flag=err)
          if(err) error stop 'grid field error'
          write(tag,'(a,i0,a,i0)') 'G_',i,'_',j
          call emit(tag,f%b)
        enddo
      enddo
    else if(n==4 .or. n==5) then
      x=0.003_rp; y=0.002_rp; s=0.37_rp
      do j=0,2
        i=5+j
        if(n==5) i=8+j
        e=>lat%ele(i)
        call init_coord(o,[x,0.0_rp,y,0.0_rp,0.0_rp,0.0_rp],e,upstream_end$)
        call em_field_calc(e,lat%param,s,o,.false.,f,err_flag=err)
        if(err) error stop 'lab field error'
        write(prefix,'(a,i0)') 'F_',j
        call emit(trim(prefix)//'_PARAM',[e%value(tilt_tot$),e%value(p0c$),e%value(l$),real(charge_of(e%ref_species),rp),c_light])
        call emit(trim(prefix)//'_LABE',f%e)
        call emit(trim(prefix)//'_LABB',f%b)
        c=cos(e%value(tilt_tot$)); sn=sin(e%value(tilt_tot$))
        xb=c*x+sn*y; yb=-sn*x+c*y
        ob=o; ob%vec(1)=xb; ob%vec(3)=yb
        call em_field_calc(e,lat%param,s,ob,.true.,f,err_flag=err)
        if(err) error stop 'body field error'
        call emit(trim(prefix)//'_BODYE',f%e)
        call emit(trim(prefix)//'_BODYB',f%b)
        ! Independent magnetic quadrupole control at exactly the same angle and point.
        e=>lat%ele(11+j)
        call em_field_calc(e,lat%param,s,o,.false.,f,err_flag=err)
        if(err) error stop 'magnetic control error'
        call emit(trim(prefix)//'_MAG',f%b)
      enddo
    else if(n==6) then
      x=0.003_rp; y=0.002_rp; s=0.37_rp; h=1e-5_rp
      do i=14,15
        e=>lat%ele(i)
        call init_coord(o,[x,0.0_rp,y,0.0_rp,0.0_rp,0.0_rp],e,upstream_end$)
        call em_field_calc(e,lat%param,s,o,.true.,f,calc_dfield=.true.,err_flag=err)
        if(err) error stop 'wiggler error'
        write(prefix,'(a,i0)') 'W_',i-14
        call emit(trim(prefix)//'_PARAM',[e%value(kx$),e%value(l_period$),e%value(b_max$),e%value(l$),x,y,s,h])
        call emit(trim(prefix)//'_B',f%b)
        do j=1,3
          write(tag,'(a,a,i0)') trim(prefix),'_D_',j
          call emit(tag,f%db(:,j))
        enddo
        do j=1,3
          do k=1,4
            ob=o; sample_s=s
            if(j<3) then
              m=2*j-1
              ob%vec(m)=ob%vec(m)+steps(k)*h
            else
              sample_s=s+steps(k)*h
            endif
            call em_field_calc(e,lat%param,sample_s,ob,.true.,f,err_flag=err)
            if(err) error stop 'finite difference sample error'
            write(tag,'(a,a,i0,a,i0)') trim(prefix),'_S_',j,'_',k
            call emit(tag,f%b)
          enddo
        enddo
      enddo
    else
      error stop 'unknown field case'
    endif
    print '(a)', 'DONE'
    contains
    subroutine emit(label,values)
    character(*) label
    real(rp) values(:)
    write(*,'(a,1x,*(es25.16e3,1x))') trim(label),values
    end subroutine
    end program
    """

    PHOTON = r"""program verify_photons
    use bmad
    use track1_photon_mod
    use xraylib, only: crystal_struct, Crystal_GetCrystal
    implicit none
    type(lat_struct), target :: lat
    type(ele_struct), pointer :: ele
    type(coord_struct) :: ray, result_ray, surface_ray, origin
    type(crystal_param_struct) :: cp
    type(crystal_struct), pointer :: silicon
    real(rp) :: z, dz(2), h, vals(4), fd, x, y, sx, sy, zc, ab, ph, amp, phase, p_factor
    real(rp) :: w(3,3), wavelength, theta, xi_re, drp(3), drs(3), fields(2), phases(2), hbar(3), incoming(2)
    real(rp) :: mean_re, mean_im, eta_delta
    complex(rp) :: chi1, chi2
    integer :: i, j, k, n, state, count_pos, count_neg
    logical :: err
    character(32) :: test_case
    call get_command_argument(1,test_case)
    call bmad_parser('checks.bmad',lat,err_flag=err)
    if(err) error stop 'parse failed'
    select case(trim(test_case))
    case('1')
      ele=>lat%ele(8)
      wavelength=c_light*h_planck/9000.0_rp
      chi1=-conjg(ele%photon%material%f0_m1)*r_e*wavelength**2/(pi*ele%value(v1_unitcell$))
      chi2=-conjg(ele%photon%material%f0_m2)*r_e*wavelength**2/(pi*ele%value(v2_unitcell$))
      print '(a,7es26.17)', 'ML_CONFIG ',wavelength,chi1,chi2,ele%value(d1_thickness$),ele%value(d2_thickness$)
      do n=1,3
        ele%value(n_cell$)=real(2**n-1,rp)
        do j=1,3
          if(j==1) theta=.02_rp
          if(j==2) theta=.25_rp
          if(j==3) theta=pi/4
          call init_coord(ray,[0.0_rp,0.0_rp,0.0_rp,0.0_rp,0.0_rp,1.0_rp],ele,upstream_end$,photon$)
          ! Inverse of the entrance surface rotation: incidence is exactly theta.
          ray%vec(2)=sin(theta-ele%value(graze_angle$))
          ray%vec(6)=cos(theta-ele%value(graze_angle$))
          ray%field=1; ray%phase=0
          call track1(ray,ele,lat%param,result_ray)
          print '(a,2i4,5es26.17,i4)', 'ML ',nint(ele%value(n_cell$)),j,theta,result_ray%field,result_ray%phase,result_ray%state
        enddo
      enddo
    case('2','6')
      ! Establish recognized material independently of the helper's error flag.
      silicon=>Crystal_GetCrystal('Si')
      if(.not.associated(silicon)) error stop 'silicon crystal lookup failed'
      if(silicon%volume<=0) error stop 'invalid silicon crystal volume'
      print '(a)', 'SILICON_RECOGNIZED'
      do i=5,7
        ele=>lat%ele(i)
        call photon_absorption_and_phase_shift(ele%wall3d(1)%section(1)%material,9000.0_rp,ab,ph,err)
        print '(a,i4,l2,2es26.17)', 'ABS ',i,err,ab,ph
        do j=1,2
          call init_coord(ray,[0.0_rp,0.0_rp,0.0_rp,0.0_rp,0.0_rp,1.0_rp],ele,upstream_end$,photon$)
          if(j==2) ray%vec(2:6:2)=[.6_rp,0.0_rp,.8_rp]
          ray%field=[.6_rp,.8_rp]; ray%phase=0
          surface_ray=ray
          call offset_photon(ele,surface_ray,set$)
          call track1(ray,ele,lat%param,result_ray)
          print '(a,3i4,5es26.17)', 'MASK ',i,j,result_ray%state,surface_ray%vec(6),result_ray%field,result_ray%phase
        enddo
      enddo
    case('3')
      ele=>lat%ele(4)
      x=.093_rp; y=-.101_rp
      z=checked_z(x,y,dz)
      print '(a,5es26.17)', 'SEG ',x,y,z,dz
      associate(pt=>ele%photon%segmented%pt(1,-1))
        print '(a,5es26.17)', 'PLANE ',pt%x0,pt%y0,pt%z0,pt%dz_dx,pt%dz_dy
      end associate
      h=1e-5_rp
      sx=(checked_z(x+h,y)-checked_z(x-h,y))/(2*h)
      sy=(checked_z(x,y+h)-checked_z(x,y-h))/(2*h)
      print '(a,2es26.17)', 'SEG_FD ',sx,sy
      do i=-1,1,2
        do j=-1,1,2
          x=.08_rp+i*(.04_rp-1e-12_rp); y=-.12_rp+j*(.06_rp-1e-12_rp)
          z=checked_z(x,y)
          print '(a,3es26.17)', 'CORNER ',x,y,z
        enddo
      enddo
      call init_coord(ray,[.093_rp,0.0_rp,-.101_rp,0.0_rp,-.1_rp,1.0_rp],ele,upstream_end$,photon$)
      call track_to_surface(ele,ray,lat%param,w)
      print '(a,i4,6es26.17)', 'SEG_HIT ',ray%state,ray%vec
    case('4')
      do i=1,3
        ele=>lat%ele(i)
        x=.17_rp; h=1e-4_rp
        z=checked_z(x,0.0_rp,dz)
        do j=1,4
          k=j-3
          if(j>2) k=j-2
          vals(j)=checked_z(x+k*h,0.0_rp)
        enddo
        fd=(vals(1)-8*vals(2)+8*vals(3)-vals(4))/(12*h)
        call init_coord(ray,[x,0.0_rp,0.0_rp,0.0_rp,0.0_rp,1.0_rp],ele,upstream_end$,photon$)
        call to_surface_coords(ray,ele,result_ray)
        if(result_ray%state/=alive$) error stop 'detector coordinate control lost photon'
        print '(a,i4,5es26.17)', 'DETECTOR ',i,x,z,dz(1),fd,result_ray%vec(1)
      enddo
    case('5')
      ele=>lat%ele(9)
      cp%wavelength=c_light*h_planck/9000.0_rp
      cp%cap_gamma=r_e*cp%wavelength**2/(pi*ele%value(v_unitcell$))
      theta=asin(cp%wavelength/(2*ele%value(d_spacing$)))
      cp%old_vvec=[sin(theta),0.0_rp,cos(theta)]
      cp%new_vvec=[-sin(theta),0.0_rp,cos(theta)]
      cp%b_eff=1; cp%dtheta_sin_2theta=0
      lat%photon_type=coherent$
      count_pos=0; count_neg=0; mean_re=0; mean_im=0
      do i=1,128
        state=alive$
        call crystal_diffraction_field_calc(cp,ele,0.0_rp,lat%param,1.0_rp,.true.,amp,phase,state,drs)
        if(state/=alive$) error stop 'zero-thickness helper lost photon'
        print '(a,i4,2es26.17)', 'ZERO_SAMPLE ',i,amp,phase
        if(amp*cos(phase)>0) count_pos=count_pos+1
        if(amp*cos(phase)<0) count_neg=count_neg+1
        mean_re=mean_re+amp*cos(phase)/128
        mean_im=mean_im+amp*sin(phase)/128
      enddo
      print '(a,2i5,2es26.17)', 'ZERO_LIMIT ',count_pos,count_neg,mean_re,mean_im
      do i=1,12
        call init_coord(ray,[0.0_rp,0.0_rp,0.0_rp,0.0_rp,0.0_rp,1.0_rp],ele,upstream_end$,photon$)
        ray%field=[0.0_rp,1.0_rp]; ray%phase=0
        call track1(ray,ele,lat%param,result_ray)
        print '(a,i3,3es26.17)', 'FINITE ',result_ray%state,result_ray%field(2),result_ray%phase(2),result_ray%field(2)*cos(result_ray%phase(2))
      enddo
      lat%photon_type=incoherent$
      state=alive$
      call crystal_diffraction_field_calc(cp,ele,0.0_rp,lat%param,1.0_rp,.true.,amp,phase,state,drs)
      if(state/=alive$) error stop 'incoherent helper lost photon'
      print '(a,2es26.17)', 'INCOHERENT_ZERO ',amp,phase
    case('7')
      ele=>lat%ele(10)
      do k=1,3
        call init_coord(origin,[0.0_rp,real(k-2,rp)*2e-5_rp,0.0_rp,0.0_rp,0.0_rp,sqrt(1.0_rp-(real(k-2,rp)*2e-5_rp)**2)],ele,upstream_end$,photon$)
        origin%field=[0.0_rp,1.0_rp]; origin%phase=0
        surface_ray=origin
        call offset_photon(ele,origin,set$)
        cp%wavelength=c_light*h_planck/origin%p0c
        cp%cap_gamma=r_e*cp%wavelength**2/(pi*ele%value(v_unitcell$))
        hbar=ele%photon%material%h_norm*cp%wavelength/ele%value(d_spacing$)
        cp%old_vvec=origin%vec(2:6:2)
        cp%new_vvec=cp%old_vvec+hbar
        cp%new_vvec(3)=sqrt(1-sum(cp%new_vvec(1:2)**2))
        cp%b_eff=cp%old_vvec(3)/cp%new_vvec(3)
        cp%dtheta_sin_2theta=-dot_product(hbar+2*cp%old_vvec,hbar)/2
        p_factor=cos(ele%value(bragg_angle_in$)+ele%value(bragg_angle_out$))
        print '(a,i3,6es26.17)', 'CP ',k,cp%wavelength,cp%cap_gamma,cp%b_eff,cp%dtheta_sin_2theta,p_factor,ele%value(thickness$)
        print '(a,i3,6es26.17)', 'FACTORS ',k,ele%photon%material%f_0,ele%photon%material%f_hkl,ele%photon%material%f_hbar
        print '(a,i3,6es26.17)', 'VECTORS ',k,cp%old_vvec,cp%new_vvec
        state=alive$
        call crystal_diffraction_field_calc(cp,ele,ele%value(thickness$),lat%param,p_factor,.true.,fields(1),phases(1),state,drp)
        call crystal_diffraction_field_calc(cp,ele,ele%value(thickness$),lat%param,1.0_rp,.false.,fields(2),phases(2),state,drs)
        if(state/=alive$) error stop 'polarization helper lost photon'
        print '(a,i3,2es26.17)', 'RESPONSE ',k,fields
        print '(a,i3,6es26.17)', 'DR ',k,drp,drs
        do i=1,3
          incoming=0
          if(i==1) incoming=[1.0_rp,0.0_rp]
          if(i==2) incoming=[0.0_rp,1.0_rp]
          if(i==3) incoming=[.6_rp,.8_rp]
          ray=origin; ray%field=incoming
          call track1_crystal(ele,lat%param,ray)
          if(ray%state/=alive$) error stop 'polarization tracking lost photon'
          print '(a,2i3,5es26.17)', 'POL ',k,i,ray%vec(1:5:2)-origin%vec(1:5:2),ray%field
          if(i<3) then
            ray=surface_ray; ray%field=incoming
            call track1(ray,ele,lat%param,result_ray)
            if(result_ray%state/=alive$) error stop 'public polarization tracking lost photon'
            ray=origin; ray%field=incoming*fields; ray%phase=phases
            if(i==1) ray%vec(1:5:2)=ray%vec(1:5:2)+drp
            if(i==2) ray%vec(1:5:2)=ray%vec(1:5:2)+drs
            ray%vec(2:6:2)=cp%new_vvec
            call offset_photon(ele,ray,unset$)
            print '(a,2i3,2es26.17)', 'PUBLIC ',k,i,result_ray%vec(1),ray%vec(1)
          endif
        enddo
      enddo
    case default
      error stop 'Unknown verification case'
    end select
    print '(a)', 'DONE'
    contains
    function checked_z(xx,yy,slopes) result(height)
    real(rp) xx,yy,height
    real(rp), optional :: slopes(2)
    logical bad
    height=z_at_surface(ele,xx,yy,bad,dz_dxy=slopes)
    if(bad) error stop 'invalid surface evaluation'
    end function
    end program verify_photons
    """

    PHOTON_LATTICE = """parameter[geometry] = open
    parameter[particle] = photon
    parameter[e_tot] = 9000
    parameter[no_end_marker] = T
    parameter[ran_seed] = 71439
    no_digested
    dp: detector, aperture_type=rectangular, curvature={x2y0=.7}
    de: detector, aperture_type=rectangular, curvature={elliptical_x=.8, elliptical_z=1.3}
    dm: detector, aperture_type=rectangular, curvature={x2y0=.7, elliptical_x=.8, elliptical_z=1.3}
    sg: mirror, graze_angle=.2, curvature={x2y0=1.7, x0y2=.6}, segmented={dr=(.08,.12), ix_bounds=(-3,3), iy_bounds=(-3,3)}
    si: mask, wall={section={type=clear, material=Si, thickness=2e-5, v(1)={0,0,.2}}}
    ti: mask, wall={section={type=clear, material=Ti, thickness=2e-5, v(1)={0,0,.2}}}
    tip: ti, y_pitch=acos(.8)
    ml: multilayer_mirror, material_type="W:BORON_CARBIDE", n_cell=1, d1_thickness=1.2e-9, d2_thickness=1.8e-9
    cf: crystal, crystal_type="Si(111)", b_param=1, thickness=1e-13, ref_orbit_follows=undefracted
    cd: crystal, crystal_type="Si(111)", b_param=.55, thickness=2e-5
    test: line=(dp,de,dm,sg,si,ti,tip,ml,cf,cd)
    use,test
    """


    def polarization_paths(k, crystal_parameters, factors, vectors, displacements):
        lam, gamma, b_eff, angular_delta, pi_factor, thick = crystal_parameters[k]
        f0_re, f0_im, fh_re, fh_im, fbar_re, fbar_im = factors[k]
        f0, fh, fbar = (
            complex(f0_re, f0_im),
            complex(fh_re, fh_im),
            complex(fbar_re, fbar_im),
        )
        old, new = np.asarray(vectors[k]).reshape(2, 3)
        independent_paths = []
        for pol_factor in [pi_factor, 1.0]:
            # Solve the two-beam dispersion equation as a 2x2 eigenproblem,
            # rather than Bmad's eta-based closed form. q=xi_a+xi_b;
            # xi_a*xi_b=-g^2, xi_H=g^2/(b*xi_0).
            q = b_eff * angular_delta + gamma * f0 * (1 - b_eff) / 2
            coupling = gamma * pol_factor * math.sqrt(b_eff) * fh / 2
            eigenvalues = np.linalg.eigvals(np.array([[q, coupling], [coupling, 0]], dtype=complex))
            branch_paths, weights = [], []
            for eigenvalue in eigenvalues:
                xi_h = coupling**2 / (b_eff * eigenvalue)
                delta_0 = 1 - gamma * f0.real / 2 + eigenvalue.real
                delta_h = 1 - gamma * f0.real / 2 + xi_h.real
                k0 = np.r_[old[:2], math.sqrt(delta_0**2 - sum(old[:2] ** 2))] / lam
                kh = np.r_[new[:2], math.sqrt(delta_h**2 - sum(new[:2] ** 2))] / lam
                field_ratio = -2 * eigenvalue / (pol_factor * gamma * fbar)
                energy_flow = k0 + abs(field_ratio) ** 2 * kh
                branch_paths.append(thick * energy_flow / energy_flow[2])
                attenuation_k = (delta_h / lam) ** 2 * (gamma * f0.imag / 2 - eigenvalue.imag) / k0[2]
                weights.append(math.exp(-2 * math.pi * attenuation_k * thick))
            independent_paths.append(np.average(branch_paths, axis=0, weights=weights))
        independent_paths = np.array(independent_paths)
        assert np.max(np.abs(independent_paths - np.asarray(displacements[k]).reshape(2, 3))) < 8e-20
        return independent_paths

    ```

## `_s3_24_converter.py` { #_s3_24_converter }

[Download _s3_24_converter.py](_s3_24_converter.py)

??? example "Source: _s3_24_converter.py"

    ```python
    """Synthetic converter distribution and build-derived diagnostic runner for XLT-01–04."""

    from pathlib import Path

    import numpy as np
    from _bmad import _case, _execute, _recipe, _scratch, fortran, source

    PROGRAM = "program converter_check\nuse bmad\nuse random_mod\nimplicit none\ntype(lat_struct), target :: lat\ntype(ele_struct), pointer :: e\ntype(coord_struct) :: a,b,d\nreal(rp) :: zexpected\ninteger :: i\nlogical :: err\ncharacter(100) :: mode\ncall get_command_argument(1,mode)\ncall bmad_parser('case.bmad',lat,err_flag=err)\nif(err) error stop 'lattice parsing failed'\n  e=>lat%ele(1)\n  if(trim(mode)=='direct'.or.trim(mode)=='global') bmad_com%spin_tracking_on=.true.\n  call init_coord(a,[0.0_rp,0.0_rp,0.0_rp,0.0_rp,0.023_rp,0.0_rp],e,upstream_end$)\n  a%spin=[0,0,1]\n  d=a\n  call track_a_drift(d,e%value(l$))\n  call ran_seed_put(46731)\n  do i=1,128\n    if(trim(mode)=='direct') then\n      b=a\n      call track_a_converter(b,e,lat%param)\n    else\n      call track1(a,e,lat%param,b)\n    endif\n    if(b%state/=alive$) error stop 'converter lost particle'\n    if(i==1) then\n      print '(a,l2)', 'GLOBAL_SPIN_ENABLED ',bmad_com%spin_tracking_on\n      zexpected=d%vec(5)*b%beta/d%beta\n      print '(a,8es25.16)', 'CONVERTER_CLOCK ',d%beta,b%beta,d%t,b%t,d%vec(5), &\n           b%vec(5),zexpected,b%vec(5)/b%beta-d%vec(5)/d%beta\n      print '(a,3es25.16)', 'CONVERTER_ABSOLUTE_CLOCK ',e%ref_time,-b%beta*c_light*(b%t-e%ref_time), &\n           zexpected+b%beta*c_light*(b%t-e%ref_time)\n      if(abs(zexpected+b%beta*c_light*(b%t-e%ref_time))>2e-17_rp) error stop 'converter absolute clock oracle failed'\n      print '(a,2es25.16)', 'SPIN_TABLE ',minval(e%converter%dist(1)%sub_dist(1)%prob_pc_r%spin_z), &\n           maxval(e%converter%dist(1)%sub_dist(1)%prob_pc_r%spin_z)\n    endif\n    print '(a,i4,3es25.16)', 'CONVERTER_SAMPLE ',i,b%p0c*(1+b%vec(6)),b%charge,b%spin(3)\n  enddo\nprint '(a)', 'COMPLETED'\nend program\n"


    def lattice(tail: str = "") -> str:
        """Unit-integral outgoing density, two incoming-momentum tables and fixed spin transfers."""
        header = "parameter[geometry]=open\nbeginning[beta_a]=10\nbeginning[beta_b]=10\n"
        coefficients = {
            "alpha_x": 0,
            "alpha_y": 2,
            "beta": 0,
            "c_x": 0,
            "dxds_min": -0.8,
            "dxds_max": 0.8,
            "dyds_max": 0.6,
        }
        direction = ",\n".join(
            f"{key} = {{"
            + ("C=0, " if key == "dxds_min" else "")
            + f"fit_1d_r={{pc_out=2e5, poly=[{value}]}}, "
            f"fit_1d_r={{pc_out=1e6, poly=[{value}]}}, "
            f"fit_2d_pc={{k=0, poly=[1]}}, fit_2d_r={{k=0, poly=[{value}]}}}}"
            for key, value in coefficients.items()
        )

        def sub(inc, spin):
            prob = ",\n".join(
                f"row={{pc_out={pc},prob=[0,0.0003125,0.000625]}}" for pc in (200000, 600000, 1000000)
            )
            spins = ",\n".join(
                f"row={{pc_out={pc},spin_z=[{spin},{spin}]}}" for pc in (200000, 600000, 1000000)
            )
            return (
                f"sub_distribution={{pc_in={inc},spin_in=[0,0,1],\n"
                f"prob_pc_r={{r_values=[0,0.002,0.004],{prob}}},\n"
                f"spin_z_out={{r_values=[0.002,0.004],{spins}}},\n"
                f"direction_out={{{direction}}}}}"
            )

        base = header + (
            "parameter[particle]=electron\nparameter[p0c]=4e6\n"
            "cv: converter,l=0.0025,p0c=6e5,species_out=positron,distribution={material=tungsten,species_out=positron,thickness=0.0025,\n"
            + sub(3000000, 0.25)
            + ",\n"
            + sub(5000000, 0.75)
            + "}\n"
            "channel: line=(cv)\nuse,channel\n"
        )
        return base + tail


    def samples(
        tail: str = "", mode: str = "converter", diagnostic: bool = False, assigned: bool = False
    ):
        """Return 128 samples, clock records and source-matched diagnostic polarization if requested."""
        if diagnostic:
            code = (source() / "bmad/low_level/track_a_converter.f90").read_text()
            anchor = "out%weight = rsd2 * dist%sub_dist(ix_sd)%prob_pc_r%integrated_prob + rsd * dist%sub_dist(ix_sd+1)%prob_pc_r%integrated_prob"
            assignment = "out%rel_spin_z = rsd2*out1%rel_spin_z + rsd*out2%rel_spin_z"
            if assigned:
                if code.count(anchor) != 1:
                    raise RuntimeError("Missing unique incoming-momentum interpolation anchor")
                code = code.replace(anchor, anchor + "\n" + assignment)
            compile_argv, link_argv = _recipe(False)
            with _scratch() as tmp:
                work = Path(tmp)
                (work / "probe.f90").write_text(PROGRAM)
                (work / "converter.f90").write_text(code)
                _execute(compile_argv, work, "", 290)
                diagnostic_argv = [
                    x.replace("probe.f90", "converter.f90").replace("probe.o", "converter.o")
                    for x in compile_argv
                ]
                diagnostic_argv.extend(["-O0", "-finit-derived", "-finit-real=nan"])
                _execute(diagnostic_argv, work, "", 290)
                link_argv.insert(link_argv.index("probe.o") + 1, "converter.o")
                _execute(link_argv, work, "", 290)
                result = _case(
                    [str(work / "probe"), mode], work / "run", {"case.bmad": lattice(tail)}, (), "", 290
                )
        else:
            result = fortran(PROGRAM, [mode], files={"case.bmad": lattice(tail)})
        if "COMPLETED" not in result.log:
            raise RuntimeError(result.log)
        rows = np.array(
            [
                [float(x) for x in line.split()[2:]]
                for line in result.log.splitlines()
                if line.startswith("CONVERTER_SAMPLE")
            ]
        )
        clock = np.array(
            [
                float(x)
                for x in next(
                    line for line in result.log.splitlines() if line.startswith("CONVERTER_CLOCK")
                ).split()[1:]
            ]
        )
        absolute = np.array(
            [
                float(x)
                for x in next(
                    line
                    for line in result.log.splitlines()
                    if line.startswith("CONVERTER_ABSOLUTE_CLOCK")
                ).split()[1:]
            ]
        )
        if rows.shape != (128, 3) or not np.isfinite(rows[:, :2]).all():
            raise RuntimeError("Invalid converter sample records")
        if not np.all(rows[:, 1] == rows[0, 1]):
            raise RuntimeError("Identical inputs received different physical weights")
        return rows, clock, absolute

    ```

## `_s3_24_translations.py` { #_s3_24_translations }

[Download _s3_24_translations.py](_s3_24_translations.py)

??? example "Source: _s3_24_translations.py"

    ```python
    """Isolated translation and tracking fixtures for the SAD/multipole conversion chunk."""

    import cmath
    import math
    import os
    import re
    import sys
    from pathlib import Path

    import numpy as np
    from _bmad import RunFailure, _case, _execute, _scratch, fortran, run, source

    HEADER = """parameter[geometry]=open
    parameter[particle]=electron
    parameter[p0c]=1e9
    beginning[beta_a]=10
    beginning[beta_b]=10
    """
    PROGRAM = """program check_lattice
    use bmad
    implicit none
    type(lat_struct) :: lat
    type(coord_struct) :: a,b
    real(rp) :: v(6)
    integer :: i
    logical :: err
    global_com%exit_on_error=.false.
    call bmad_parser('case.bmad',lat,err_flag=err)
    if(err) then
      print '(a)', 'PARSE_REJECTED'
      stop
    endif
    open(11,file='start.txt'); read(11,*) v; close(11)
    call init_coord(a,v,lat%ele(0),downstream_end$)
    do i=1,lat%n_ele_track
      call track1(a,lat%ele(i),lat%param,b)
      if(b%state/=alive$) error stop 'Particle lost'
      if(i==1) print '(a,6es25.16)', 'FIRST ',b%vec
      a=b
    enddo
    print '(a,6es25.16)', 'ORBIT ',b%vec
    print '(a,3es25.16)', 'FLOOR ',lat%ele(lat%n_ele_track)%floor%r
    print '(a,5es25.16)', 'ATTR ',lat%ele(1)%value(l$),lat%ele(1)%value(voltage$), &
      lat%ele(1)%value(gradient$),lat%ele(1)%value(p0c_start$),lat%ele(1)%value(p0c$)
    print '(a)', 'COMPLETED'
    end program
    """


    def lattice(element: str, tail: str = "", line: str = "m") -> str:
        return HEADER + f"m: {element}\n{tail}\nlat: line=({line})\nuse,lat\n"


    def track(text: str, start=(0.001, 0, 0.002, 0, 0, 0), *, allow_diagnostics=False):
        """Return named native records; parser rejection carries its exact diagnostic log."""
        result = fortran(PROGRAM, files={"case.bmad": text, "start.txt": " ".join(map(str, start))})
        if not allow_diagnostics and re.search(r"\[(?:ERROR|FATAL|ABORT)|ERROR IN ", result.log):
            raise RuntimeError(result.log)
        if "PARSE_REJECTED" in result.log:
            return {}, result.log
        if "COMPLETED" not in result.log:
            raise RuntimeError(result.log)
        records = {}
        for line in result.log.splitlines():
            tokens = line.split()
            if tokens and tokens[0] in ("FIRST", "ORBIT", "FLOOR", "ATTR"):
                records[tokens[0]] = np.array([float(x) for x in tokens[1:]])
        if set(records) != {"FIRST", "ORBIT", "FLOOR", "ATTR"}:
            raise RuntimeError(result.log)
        if not all(np.isfinite(x).all() for x in records.values()):
            raise RuntimeError("Nonfinite native tracking records")
        return records, result.log


    def export(text: str, kind: str) -> str:
        suffix = {"madx": "madx", "mad8": "mad8", "sad": "sad", "elegant": "lte"}[kind]
        result = run(
            "bmad_to_mad_sad_elegant",
            [f"-{kind}", "case.bmad"],
            files={"case.bmad": text},
            outputs=[f"case.{suffix}"],
        )
        if "ERROR IN" in result.log or "FATAL" in result.log:
            raise RuntimeError(result.log)
        return result.outputs[f"case.{suffix}"].decode()


    def translate_sad(deck: str) -> str:
        """Execute the upstream importer selected by BMAD_SOURCE with inline parameters."""
        params = f"""sad_lattice_file = "case.sad"
    bmad_lattice_file = "case.bmad"
    lattice_geometry = "open"
    patch_for_fshift = "FALSE"
    sad_to_bmad_postprocess_exe = "sad_to_bmad_postprocess"
    calc_fshift_for = "bmad"
    ignore_marker_offsets = False
    header_lines = {HEADER!r}
    footer_lines = ""
    """
        with _scratch() as tmp:
            result = _case(
                [
                    sys.executable,
                    "-I",
                    str(source() / "util_programs/sad_to_bmad/sad_to_bmad.py"),
                    "case.params",
                    "case.sad",
                ],
                Path(tmp) / "run",
                {"case.params": params, "case.sad": deck},
                ["case.bmad"],
                "",
                290,
            )
        return result.outputs["case.bmad"].decode()


    def sad_deck(element: str) -> str:
        return f"MOMENTUM = 1 GEV;\nDRIFT D = (L = 1);\n{element}\nLINE RING = (M D);\nUSE RING;\n"


    def sad_kick(n: int, k: float, rotate: float, x=0.001, y=0.002):
        # SAD TENT rotates lab coordinates by +ROTATE. Its thin kick uses
        # Kn/n!*(xb-i*yb)^n; TEXIT undoes that frame rotation for the momenta.
        c, s = math.cos(rotate), math.sin(rotate)
        w = complex(c * x - s * y, -(s * x + c * y)) ** n * k / math.factorial(n)
        return np.array([-c * w.real - s * w.imag, s * w.real - c * w.imag])


    def bmad_kick(n: int, kn: float, tilt: float, skew=0.0):
        # Bmad: -dpx+i*dpy=(KnL+i*KnSL)/n! * exp[-i(n+1)T]*(x+i*y)^n.
        g = (
            complex(kn, skew)
            / math.factorial(n)
            * cmath.exp(-1j * (n + 1) * tilt)
            * complex(0.001, 0.002) ** n
        )
        return np.array([-g.real, g.imag])


    def madx_track(text: str, *, survey: bool = False, ptc: bool = False):
        from cpymad.madx import Madx

        with _scratch() as tmp:
            m = Madx(stdout=False, cwd=tmp)
            try:
                m.input(text)
                if survey:
                    table = m.survey()
                    result = np.array([table.x[-1], table.y[-1]])
                elif ptc:
                    m.input(
                        "ptc_create_universe; ptc_create_layout, model=2, method=6, nst=10, exact=true; ptc_start,x=.001,y=.002; ptc_track,icase=4,element_by_element,onetable,dump=false; ptc_track_end; ptc_end;"
                    )
                    result = np.array([m.table.trackone.px[-1], m.table.trackone.py[-1]])
                else:
                    m.input("track,onepass,onetable; start,x=.001,y=.002; run,turns=1; endtrack;")
                    result = np.array([m.table.trackone.px[-1], m.table.trackone.py[-1]])
            finally:
                m.quit()
        if not np.isfinite(result).all():
            raise RuntimeError("MAD-X returned nonfinite results")
        return result


    def statements(text: str):
        content = "\n".join(line for line in text.splitlines() if not line.startswith("!"))
        return [" ".join(s.split()) for s in content.replace("&\n", " ").splitlines() if s.strip()]


    def undefined_names(text: str):
        items = statements(text)
        defined = {s.split(":")[0].strip().upper() for s in items if ":" in s}
        beamline = next(s for s in items if s.upper().startswith("LINE_1:"))
        match = re.search(r"\((.*)\)", beamline)
        if match is None:
            raise RuntimeError("No exported beamline")
        used = match[1].replace(" ", "").split(",")
        return sorted({n for n in used if n.upper() not in defined}), beamline


    def elegant_track(text: str, *, x=0.001, y=0.002, rejected: str | None = None):
        """Track one particle, or recognize exactly the expected undefined-name rejection."""
        import pysdds

        executable = os.environ.get("ELEGANT_BIN")
        rpn = os.environ.get("RPN_DEFNS")
        if not executable or not rpn:
            raise RuntimeError(
                "Set ELEGANT_BIN to the Elegant executable and RPN_DEFNS to its definitions"
            )
        p0 = 1957.0
        with _scratch() as tmp:
            work = Path(tmp)
            (work / "lattice.lte").write_text(text)
            beam = pysdds.SDDSFile(add_data_nm=True)
            beam.mode = "binary"
            beam.n_pages = 1
            for name, val, unit in (
                ("x", x, "m"),
                ("xp", 0.0, ""),
                ("y", y, "m"),
                ("yp", 0.0, ""),
                ("t", 0.0, "s"),
                ("p", p0, "m$be$nc"),
            ):
                beam.add_column(name, "double", units=unit, data=[np.array([val], dtype=float)])
            pysdds.write(beam, work / "beam.sdds")
            (work / "run.ele").write_text(
                f"&run_setup lattice=lattice.lte, use_beamline=LAT, p_central={p0}, output=out.sdds, default_order=2 &end\n&run_control n_steps=1,n_passes=1 &end\n&sdds_beam input=beam.sdds &end\n&track &end\n"
            )
            try:
                _execute([executable, "run.ele", f"-rpnDefns={rpn}"], work, "", 290)
            except RunFailure as error:
                if rejected and f"no expansion for entity {rejected}" in error.log:
                    print(
                        f"Elegant rejected undefined name {rejected}: expected a complete beamline definition"
                    )
                    return None
                raise
            if rejected:
                # If a future writer defines every name, successful tracking is a valid result.
                print(f"Elegant accepted beamline (previously undefined name {rejected})")
            data = pysdds.read(work / "out.sdds")
            cols = [np.asarray(data[0, c]) for c in ("x", "xp", "y", "yp", "p")]
            if any(c.shape != (1,) for c in cols):
                raise RuntimeError("Expected exactly one surviving particle")
            xx, xp, yy, yp, p = [float(c[0]) for c in cols]
            # Elegant stores slopes; Bmad uses canonical momenta normalized by P0.
            scale = p / p0 / math.sqrt(1 + xp * xp + yp * yp)
            result = np.array([xx, xp * scale, yy, yp * scale])
            if not np.isfinite(result).all():
                raise RuntimeError("Nonfinite Elegant orbit")
            return result

    ```

## `_s4_17_translations.py` { #_s4_17_translations }

[Download _s4_17_translations.py](_s4_17_translations.py)

??? example "Source: _s4_17_translations.py"

    ```python
    """Strict process checks for the reviewed translator examples.

    The diagnostic tracker deliberately lets Bmad report parser errors. Ordinary
    controls must reject those diagnostics even if Bmad also returns finite records.
    """

    import re

    from _bmad import run
    from _s3_24_translations import track as _diagnostic_track

    ERROR = r"\[(?:ERROR|FATAL|ABORT)|ERROR IN "


    def diagnostic_track(text, start=(0.001, 0, 0.002, 0, 0, 0), *, expected_error=None):
        """Allow only the named diagnostic; a successful future translation is valid."""
        result, log = _diagnostic_track(text, start=start, allow_diagnostics=True)
        messages = re.findall(
            r"(?ms)^(?:\[(?:ERROR|FATAL|ABORT)[^\n]*|ERROR IN [^\n]*).*?"
            r"(?=^\[|^ERROR IN |^PARSE_REJECTED|^FIRST |^ORBIT |^COMPLETED|\Z)",
            log,
        )
        if any(expected_error is None or expected_error not in message for message in messages):
            raise RuntimeError(log)
        if not result and not messages:
            raise RuntimeError(log)
        return result, log


    def track(text, start=(0.001, 0, 0.002, 0, 0, 0)):
        """Require successful parsing/tracking with no fatal or error diagnostics."""
        result, log = diagnostic_track(text, start=start)
        if not result or re.search(ERROR, log):
            raise RuntimeError(log)
        return result, log


    def export(text: str, kind: str) -> str:
        """Require a completed production export and reject parse/tracking errors."""
        suffix = {"madx": "madx", "mad8": "mad8", "sad": "sad", "elegant": "lte"}[kind]
        result = run(
            "bmad_to_mad_sad_elegant",
            [f"-{kind}", "case.bmad"],
            files={"case.bmad": text},
            outputs=[f"case.{suffix}"],
        )
        if "Written" not in result.log or re.search(ERROR, result.log):
            raise RuntimeError(result.log)
        return result.outputs[f"case.{suffix}"].decode()


    def tao(text: str, commands: list[str]) -> dict[str, list[str]]:
        """Read Tao pipe records only after successful initialization and completion."""
        result = run(
            "tao",
            ["-noplot", "-noinit", "-lat", "case.bmad", "-startup", "run.tao"],
            files={"case.bmad": text, "run.tao": "\n".join([*commands, "quit", ""])},
        )
        if re.search(ERROR, result.log):
            raise RuntimeError(result.log)
        sections = {}
        current = None
        for line in result.log.splitlines():
            if line.startswith("Tao: "):
                current = sections.setdefault(line[5:].strip(), [])
            elif current is not None and line.strip():
                current.append(line)
        if "quit" not in sections or any(not sections.get(command) for command in commands):
            raise RuntimeError(result.log)
        if any("ERROR" in line for command in commands for line in sections[command]):
            raise RuntimeError(result.log)
        return sections

    ```

## `_srdt_wiggler_maps.py` { #_srdt_wiggler_maps }

[Download _srdt_wiggler_maps.py](_srdt_wiggler_maps.py)

??? example "Source: _srdt_wiggler_maps.py"

    ```python
    """Inline Bmad drivers and analytic maps for the SRDT, wiggler and map reproducers."""

    import math

    import numpy as np
    from _bmad import fortran

    SRDT_PROGRAM = """program srdt_probe
    use bmad
    use srdt_mod
    implicit none
    type(lat_struct) lat
    type(coord_struct), allocatable :: orb(:)
    type(summation_rdt_struct) s
    integer status
    logical err
    call bmad_parser('lat.bmad',lat,err_flag=err)
    if(err) error stop 'Parsing failed'
    call twiss_and_track(lat,orb,status)
    if(status /= ok$) error stop 'Twiss failed'
    call srdt_calc(lat,s,1)
    write(*,'(a,5es25.16)') 'TWISS ',lat%ele(1)%a%beta,lat%ele(1)%b%beta, &
     lat%ele(1)%x%eta,lat%ele(1)%a%phi,lat%ele(1)%b%phi
    write(*,'(a,8es25.16)') 'SRDT ',s%h21000,s%h10110,s%h11001,s%h00111
    end program
    """

    SRDT_RING = """parameter[geometry] = closed
    parameter[taylor_order] = 3
    beginning[e_tot] = 1e9
    qf: quadrupole, l = 0.5, k1 = 0.8
    qd: quadrupole, l = 0.5, k1 = -0.8
    b: sbend, l = 2, angle = 2*pi/16
    d: drift, l = 0.5
    cell: line = (qf, d, b, d, qd, d, b, d)
    s: {spec}
    ring: line = (s, 8*cell)
    use, ring
    """

    LSQ_PROGRAM = """program probe

    use bmad
    use srdt_mod

    implicit none

    type (lat_struct) lat
    type (coord_struct), allocatable :: orb(:)
    type (summation_rdt_struct) s0, s1, tot1, tot2
    type (summation_rdt_struct), allocatable :: pe1(:), pe2(:)
    complex(rp), allocatable :: cache(:,:,:)
    real(rp), allocatable :: ls(:), val(:)
    real(rp) q0(18), q1(18), cx, cy
    integer, allocatable :: vars(:)
    integer i, j, n, status, iu
    character(200) file, mode, arg

    call get_command_argument(1, file)
    call get_command_argument(2, mode)
    call bmad_parser (file, lat)
    call twiss_and_track (lat, orb, status)
    if (status /= ok$) error stop 'Twiss failed'

    n = 0
    do i = 1, lat%n_ele_track
      if (lat%ele(i)%name(1:1) == 'S') n = n + 1
    enddo
    allocate (vars(n), val(n))
    n = 0
    do i = 1, lat%n_ele_track
      if (lat%ele(i)%name(1:1) == 'S') then
        n = n + 1
        vars(n) = i
      endif
    enddo

    select case (mode)
    case ('columns')
      call srdt_calc (lat, s0, 1)
      call to_q(s0, q0)
      write (*, '(a, 18es24.15)') 'B ', q0
      do j = 1, n
        val = 0;  val(j) = 1
        call set_vars(val)
        call srdt_calc (lat, s1, 1)
        call to_q(s1, q1)
        write (*, '(a, i3, 18es24.15)') 'A ', j, q1 - q0
      enddo

    case ('lsq')
      call get_command_argument(3, arg);  read (arg, *) cx
      call get_command_argument(4, arg);  read (arg, *) cy
      select case (mode)
      case ('lsq');  call srdt_lsq_solution (lat, vars, ls, chrom_set_x_opt = cx, chrom_set_y_opt = cy)
      end select
      write (*, '(a, 24es24.15)') 'SOLN ', ls
      call set_vars(ls)
      call evaluate()

    case ('apply', 'perele')
      call get_command_argument(3, arg)
      open (newunit = iu, file = arg, status = 'old')
      read (iu, *) val
      close (iu)
      call set_vars(val)
      if (mode == 'apply') then
        call evaluate()
      else
        call srdt_calc (lat, s1, 2, per_ele_out = pe1)
        call srdt_calc_with_cache (lat, s0, 2, cache = cache, per_ele_out = pe2)
        tot1 = sum_ele(pe1);  tot2 = sum_ele(pe2)
        write (*, '(a)') 'term      srdt_sums (srdt_calc)               sum(per_ele_out) srdt_calc        sum(per_ele_out) with_cache'
        call row('h11001', s1%h11001, tot1%h11001, tot2%h11001)
        call row('h00111', s1%h00111, tot1%h00111, tot2%h00111)
        call row('h20001', s1%h20001, tot1%h20001, tot2%h20001)
        call row('h00201', s1%h00201, tot1%h00201, tot2%h00201)
        call row('h10002', s1%h10002, tot1%h10002, tot2%h10002)
        call row('h21000', s1%h21000, tot1%h21000, tot2%h21000)
        call row('h30000', s1%h30000, tot1%h30000, tot2%h30000)
        call row('h10110', s1%h10110, tot1%h10110, tot2%h10110)
        call row('h10020', s1%h10020, tot1%h10020, tot2%h10020)
        call row('h10200', s1%h10200, tot1%h10200, tot2%h10200)
        call row('h22000', s1%h22000, tot1%h22000, tot2%h22000)
        call row('h31000', s1%h31000, tot1%h31000, tot2%h31000)
        call row('h20020', s1%h20020, tot1%h20020, tot2%h20020)
        write (*, '(a, 2es24.15)') 'with_cache sums h22000 ', s0%h22000
      endif
    end select

    contains

    subroutine set_vars(v)
    real(rp) v(:)
    character(60) str
    logical err
    integer k
    do k = 1, n
      if (lat%ele(vars(k))%key == sextupole$) then
        write (str, '(a, es24.16)') 'K2 = ', v(k)
      else
        write (str, '(a, es24.16)') 'K2L = ', v(k)
      endif
      call set_ele_attribute (lat%ele(vars(k)), str, err)
      if (err) error stop 'set_ele_attribute failed'
    enddo
    call lattice_bookkeeper (lat)
    call twiss_and_track (lat, orb, status)
    if (status /= ok$) error stop 'Twiss failed'
    end subroutine

    subroutine to_q(s, q)
    type (summation_rdt_struct) s
    real(rp) q(18)
    q = [real(s%h11001), real(s%h00111), real(s%h20001), aimag(s%h20001), real(s%h00201), aimag(s%h00201), &
         real(s%h10002), aimag(s%h10002), real(s%h21000), aimag(s%h21000), real(s%h30000), aimag(s%h30000), &
         real(s%h10110), aimag(s%h10110), real(s%h10020), aimag(s%h10020), real(s%h10200), aimag(s%h10200)]
    end subroutine

    subroutine evaluate()
    real(rp) chrom_a, chrom_b, de
    logical err
    call srdt_calc (lat, s1, 1)
    call to_q(s1, q1)
    de = 1e-6
    call chrom_calc (lat, de, chrom_a, chrom_b, err)
    if (err) error stop 'Chromaticity failed'
    write (*, '(a, 18es24.15)') 'Q ', q1
    write (*, '(a, 2es24.15)') 'CHROM ', chrom_a, chrom_b
    end subroutine

    function sum_ele(pe) result (t)
    type (summation_rdt_struct) pe(:), t
    t%h11001 = sum(pe%h11001); t%h00111 = sum(pe%h00111); t%h20001 = sum(pe%h20001)
    t%h00201 = sum(pe%h00201); t%h10002 = sum(pe%h10002); t%h21000 = sum(pe%h21000)
    t%h30000 = sum(pe%h30000); t%h10110 = sum(pe%h10110); t%h10020 = sum(pe%h10020)
    t%h10200 = sum(pe%h10200); t%h22000 = sum(pe%h22000); t%h31000 = sum(pe%h31000)
    t%h20020 = sum(pe%h20020)
    end function

    subroutine row(name, a, b, c)
    character(*) name
    complex(rp) a, b, c
    write (*, '(a8, 3(2es25.16, 4x))') name, a, b, c
    end subroutine

    end program"""

    THICK_RING = """parameter[geometry] = closed
    beginning[e_tot] = 1e9
    qf: quadrupole, l = 0.5, k1 = 0.8
    qd: quadrupole, l = 0.5, k1 = -0.8
    b: sbend, l = 2, angle = 2*pi/16
    d: drift, l = 0.5
    da: drift, l = 0.2
    sf1: sextupole, l = 0.1, k2 = 0
    sd1: sextupole, l = 0.1, k2 = 0
    sx1: sextupole, l = 0.1, k2 = 0
    c1: line = (qf, da, sf1, da, b, d, qd, da, sd1, da, b, da, sx1, da)
    sf2: sextupole, l = 0.1, k2 = 0
    sd2: sextupole, l = 0.1, k2 = 0
    sx2: sextupole, l = 0.1, k2 = 0
    c2: line = (qf, da, sf2, da, b, d, qd, da, sd2, da, b, da, sx2, da)
    sf3: sextupole, l = 0.1, k2 = 0
    sd3: sextupole, l = 0.1, k2 = 0
    sx3: sextupole, l = 0.1, k2 = 0
    c3: line = (qf, da, sf3, da, b, d, qd, da, sd3, da, b, da, sx3, da)
    sf4: sextupole, l = 0.1, k2 = 0
    sd4: sextupole, l = 0.1, k2 = 0
    sx4: sextupole, l = 0.1, k2 = 0
    c4: line = (qf, da, sf4, da, b, d, qd, da, sd4, da, b, da, sx4, da)
    sf5: sextupole, l = 0.1, k2 = 0
    sd5: sextupole, l = 0.1, k2 = 0
    sx5: sextupole, l = 0.1, k2 = 0
    c5: line = (qf, da, sf5, da, b, d, qd, da, sd5, da, b, da, sx5, da)
    sf6: sextupole, l = 0.1, k2 = 0
    sd6: sextupole, l = 0.1, k2 = 0
    sx6: sextupole, l = 0.1, k2 = 0
    c6: line = (qf, da, sf6, da, b, d, qd, da, sd6, da, b, da, sx6, da)
    sf7: sextupole, l = 0.1, k2 = 0
    sd7: sextupole, l = 0.1, k2 = 0
    sx7: sextupole, l = 0.1, k2 = 0
    c7: line = (qf, da, sf7, da, b, d, qd, da, sd7, da, b, da, sx7, da)
    sf8: sextupole, l = 0.1, k2 = 0
    sd8: sextupole, l = 0.1, k2 = 0
    sx8: sextupole, l = 0.1, k2 = 0
    c8: line = (qf, da, sf8, da, b, d, qd, da, sd8, da, b, da, sx8, da)
    ring: line = (c1, c2, c3, c4, c5, c6, c7, c8)
    use, ring
    """

    THIN_RING = """parameter[geometry] = closed
    beginning[e_tot] = 1e9
    qf: quadrupole, l = 0.5, k1 = 0.8
    qd: quadrupole, l = 0.5, k1 = -0.8
    b: sbend, l = 2, angle = 2*pi/16
    d: drift, l = 0.5
    da: drift, l = 0.2
    db: drift, l = 0.3
    sf1: multipole, k2l = 0
    sd1: multipole, k2l = 0
    sx1: multipole, k2l = 0
    c1: line = (qf, da, sf1, db, b, d, qd, da, sd1, db, b, da, sx1, db)
    sf2: multipole, k2l = 0
    sd2: multipole, k2l = 0
    sx2: multipole, k2l = 0
    c2: line = (qf, da, sf2, db, b, d, qd, da, sd2, db, b, da, sx2, db)
    sf3: multipole, k2l = 0
    sd3: multipole, k2l = 0
    sx3: multipole, k2l = 0
    c3: line = (qf, da, sf3, db, b, d, qd, da, sd3, db, b, da, sx3, db)
    sf4: multipole, k2l = 0
    sd4: multipole, k2l = 0
    sx4: multipole, k2l = 0
    c4: line = (qf, da, sf4, db, b, d, qd, da, sd4, db, b, da, sx4, db)
    sf5: multipole, k2l = 0
    sd5: multipole, k2l = 0
    sx5: multipole, k2l = 0
    c5: line = (qf, da, sf5, db, b, d, qd, da, sd5, db, b, da, sx5, db)
    sf6: multipole, k2l = 0
    sd6: multipole, k2l = 0
    sx6: multipole, k2l = 0
    c6: line = (qf, da, sf6, db, b, d, qd, da, sd6, db, b, da, sx6, db)
    sf7: multipole, k2l = 0
    sd7: multipole, k2l = 0
    sx7: multipole, k2l = 0
    c7: line = (qf, da, sf7, db, b, d, qd, da, sd7, db, b, da, sx7, db)
    sf8: multipole, k2l = 0
    sd8: multipole, k2l = 0
    sx8: multipole, k2l = 0
    c8: line = (qf, da, sf8, db, b, d, qd, da, sd8, db, b, da, sx8, db)
    ring: line = (c1, c2, c3, c4, c5, c6, c7, c8)
    use, ring
    """

    MAP_PROGRAM = """program maps
    use bmad
    implicit none
    type(lat_struct), target :: lat
    type(ele_struct), pointer :: e
    type(coord_struct) :: a,b,plus,minus,bp,bm
    real(rp) :: m(6,6),fd(6,6),bd(4,4),rot(4,4),expect(4,4),k,l,q,c,s
    integer i,j
    logical err
    call bmad_parser('lat.bmad',lat,err_flag=err)
    if(err) error stop 'Parsing failed'
      do i=1,3
        e=>lat%ele(i)
        call init_coord(a,[0.0_rp,0.0_rp,0.0_rp,0.0_rp,0.0_rp,0.0_rp],e,upstream_end$)
        call make_mat6_taylor(e,a,b,err)
        if(err) error stop 'Taylor matrix failed'
        m=e%mat6
        do j=1,6
          plus=a; minus=a
          plus%vec(j)=plus%vec(j)+2e-8_rp
          minus%vec(j)=minus%vec(j)-2e-8_rp
          call track1(plus,e,lat%param,bp,err_flag=err)
          if(err.or.bp%state/=alive$) error stop 'Stencil failed'
          call track1(minus,e,lat%param,bm,err_flag=err)
          if(err.or.bm%state/=alive$) error stop 'Stencil failed'
          fd(:,j)=(bp%vec-bm%vec)/4e-8_rp
        enddo
        k=e%value(k1$); l=e%value(l$); q=sqrt(k)
        bd=0
        bd(1,1)=cos(q*l); bd(1,2)=sin(q*l)/q
        bd(2,1)=-q*sin(q*l); bd(2,2)=cos(q*l)
        bd(3,3)=cosh(q*l); bd(3,4)=sinh(q*l)/q
        bd(4,3)=q*sinh(q*l); bd(4,4)=cosh(q*l)
        c=cos(e%value(tilt$)); s=sin(e%value(tilt$))
        rot=0
        rot(1,1)=c; rot(1,3)=s; rot(2,2)=c; rot(2,4)=s
        rot(3,1)=-s; rot(3,3)=c; rot(4,2)=-s; rot(4,4)=c
        expect=matmul(transpose(rot),matmul(bd,rot))
        print '(a,i3,5es25.16)', 'TAYLOR ',i,m(2,3),fd(2,3),expect(2,3), &
            maxval(abs(fd(1:4,1:4)-expect)),maxval(abs(m(1:4,1:4)-expect))
        if(maxval(abs(fd(1:4,1:4)-expect))>2e-13_rp) error stop 'quadrupole oracle failed'
        if(i>1.and.maxval(abs(m(1:4,1:4)-expect))>2e-13_rp) error stop 'Taylor control failed'
      enddo

    end program
    """

    MAP_LATTICE = """parameter[geometry]=open
    beginning[beta_a]=10
    beginning[beta_b]=10
    parameter[particle]=proton
    parameter[p0c]=1.7e9
    parameter[taylor_order]=1
    q1: quadrupole, l=0.8, k1=0.7, tilt=0.31, fringe_type=none, &
        tracking_method=taylor, mat6_calc_method=taylor, taylor_map_includes_offsets=F
    q2: q1, taylor_map_includes_offsets=T
    q3: q1, tilt=0
    channel: line=(q1,q2,q3)
    use,channel
    """

    MATRIX_PROGRAM = """program verify
    use bmad
    implicit none
    type(lat_struct), target :: lat
    type(ele_struct), pointer :: e
    type(coord_struct) :: a, b, d, q, w
    real(rp) :: v(6), m(6,6), mt(6,6), mtfine(6,6), fd(6), h, vals(6,4), extra
    integer :: id, ie, mode, shift, j, k, n, ios, default_method
    integer, parameter :: offsets(4) = [-2,-1,1,2]
    logical :: err
    character(200) :: lattice, cases
    call get_command_argument(1,lattice)
    call get_command_argument(2,cases)
    call bmad_parser(trim(lattice),lat)
    bmad_com%radiation_damping_on = .false.
    bmad_com%radiation_fluctuations_on = .false.
    bmad_com%spin_tracking_on = .false.
    bmad_com%d_orb = [1d-7,1d-7,1d-7,1d-7,1d-5,1d-4]
    write(*,'(a,2es26.17)') 'CONSTANTS ',c_light,mass_of(electron$)
    open(10,file=trim(cases),status='old')
    do
      read(10,*,iostat=ios) id,ie,mode,shift,extra,v
      if (ios < 0) exit
      if (ios /= 0) error stop 'Invalid input record'
      e => lat%ele(ie)
      if (e%tracking_method /= bmad_standard$) error stop 'Unexpected tracking method'
      if (.not. valid_mat6_calc_method(e,lat%param%particle,bmad_standard$)) error stop 'Unsupported matrix method'
      e%symplectify = .false.
      bmad_com%absolute_time_tracking = (mode == 1)
      bmad_com%absolute_time_ref_shift = (shift == 1)
      call init_coord(a,v,e,upstream_end$,lat%param%particle)
      a%t = a%t + extra
      call track1(a,e,lat%param,b,err_flag=err)
      if (err .or. b%state /= alive$) error stop 'Direct track failed'
      default_method = e%mat6_calc_method
      if (default_method /= auto$) error stop 'Unexpected parsed matrix default'
      e%mat6_calc_method = auto$
      call make_mat6(e,lat%param,a,d,err)
      if (err .or. d%state /= alive$) error stop 'Auto matrix failed'
      m = e%mat6
      e%mat6_calc_method = bmad_standard$
      call make_mat6(e,lat%param,a,d,err)
      if (err .or. d%state /= alive$) error stop 'Native matrix failed'
      if (maxval(abs(m-e%mat6)) > 0) error stop 'Auto differs from native'
      if (maxval(abs(b%vec-d%vec)) > 1d-14) error stop 'Native matrix changes orbit'
      e%mat6_calc_method = tracking$
      call make_mat6(e,lat%param,a,d,err)
      if (err .or. d%state /= alive$) error stop 'Tracking matrix failed'
      mt = e%mat6
      bmad_com%d_orb(6) = 1d-6
      call make_mat6(e,lat%param,a,d,err)
      if (err .or. d%state /= alive$) error stop 'Fine tracking matrix failed'
      mtfine = e%mat6
      bmad_com%d_orb(6) = 1d-4
      e%mat6_calc_method = default_method
      write(*,'(a,5i7,21es26.17)') 'CASE ',id,ie,mode,shift,default_method, &
        a%vec,b%vec,a%beta,a%t,e%value(ref_time_start$),e%value(p0c$),e%value(l$), &
        e%value(b_max$),e%value(l_period$),e%value(t_offset$),real(e%field_calc,rp)
      do j=1,6
        write(*,'(a,2i7,18es26.17)') 'MAT ',id,j,m(j,:),mt(j,:),mtfine(j,:)
      enddo
      do k=1,2
        do j=1,6
          h = 4d-6 / 4**(k-1)
          if (j == 5) h = 4d-4 / 4**(k-1)
          if (j == 6) h = 4d-4 / 4**(k-1)
          do n=1,4
            q = a
            q%vec(j) = q%vec(j) + offsets(n)*h
            ! Preserve the particle's reference clock while varying phase space.
            call convert_pc_to(q%p0c*(1+q%vec(6)),q%species,beta=q%beta)
            q%t = a%t - q%vec(5)/(c_light*q%beta) + a%vec(5)/(c_light*a%beta)
            call track1(q,e,lat%param,w,err_flag=err)
            if (err .or. w%state /= alive$) error stop 'Stencil track failed'
            vals(:,n) = w%vec
          enddo
          fd = (vals(:,1)-8*vals(:,2)+8*vals(:,3)-vals(:,4))/(12*h)
          write(*,'(a,3i7,7es26.17)') 'FD ',id,k,j,h,fd
        enddo
      enddo
    enddo
    close(10)
    end program
    """

    RADIATION_PROGRAM = """program independent_review
    use bmad
    use twiss_and_track_mod
    use radiation_mod, only: release_rad_int_cache
    implicit none
    type(lat_struct) lat
    type(coord_struct), allocatable :: orb(:)
    type(normal_modes_struct) modes
    type(rad_int_all_ele_struct) by_ele
    character(300) filename, action
    integer status, i, choice, cache

    call get_command_argument(1, filename)
    call get_command_argument(2, action)
    if (len_trim(filename) == 0) error stop 'Lattice required'
    global_com%exit_on_error = .true.
    call bmad_parser(trim(filename), lat)
    bmad_com%radiation_damping_on = .false.
    bmad_com%radiation_fluctuations_on = .false.
    call twiss_and_track(lat, orb, status, use_particle_start=.true.)
    if (status /= ok$) error stop 'Twiss or reference trajectory failed'
    if (any(orb(0:lat%n_ele_track)%state /= alive$)) error stop 'Reference lost'
    write(*,'(a,5es26.17)') 'META ', lat%ele(0)%value(p0c$), mass_of(lat%param%particle), &
      lat%ele(0)%value(e_tot$), c_light, classical_radius(lat%param%particle)
    write(*,'(a,6es26.17)') 'START ', orb(0)%vec
    write(*,'(a,6es26.17)') 'END ', orb(lat%n_ele_track)%vec


      do choice = 1, 3
        select case(choice)
        case(1); cache = -2
        case(2); cache = -1
        case(3); cache = 0
        end select
        call radiation_integrals(lat, orb, modes, ix_cache=cache, rad_int_by_ele=by_ele)
        write(*,'(a,2i5,7es26.17)') 'TOTAL ', choice, cache, modes%synch_int(0:3), &
           modes%lin%i2_E4, modes%lin%i3_E7, modes%lin%sig_E1
        do i = 1, lat%n_ele_track
          write(*,'(a,2i5,7es26.17)') 'ELEMENT ', choice, i, by_ele%branch(0)%ele(i)%i0, &
            by_ele%branch(0)%ele(i)%i1, by_ele%branch(0)%ele(i)%i2, by_ele%branch(0)%ele(i)%i3, &
            lat%ele(i)%value(l$), lat%ele(i)%value(g_max$), lat%ele(i)%value(num_steps$)
        enddo
        if (choice == 3) call release_rad_int_cache(cache)
      enddo
    end program
    """


    def records(log, label):
        """Require numeric records with the given first token."""
        rows = [
            list(map(float, line.split()[1:]))
            for line in log.splitlines()
            if line.startswith(label + " ")
        ]
        if not rows:
            raise RuntimeError(f"Missing {label} records:\n{log}")
        return np.asarray(rows)


    def srdt_case(spec):
        result = fortran(SRDT_PROGRAM, files={"lat.bmad": SRDT_RING.format(spec=spec)})
        twiss = records(result.log, "TWISS")
        values = records(result.log, "SRDT")
        if twiss.shape != (1, 5) or values.shape != (1, 8):
            raise RuntimeError("Incomplete SRDT output")
        return twiss[0], values[0].reshape(4, 2)


    def matrices(lattice, vectors):
        cases = "".join(
            f"{i} {ie} 0 1 0 " + " ".join(map(str, v)) + "\n" for i, (ie, v) in enumerate(vectors, 1)
        )
        log = fortran(
            MATRIX_PROGRAM, ["lat.bmad", "cases.txt"], files={"lat.bmad": lattice, "cases.txt": cases}
        ).log
        case, mat, fd = (records(log, label) for label in ("CASE", "MAT", "FD"))
        if (
            case.shape != (len(vectors), 26)
            or mat.shape != (6 * len(vectors), 20)
            or fd.shape != (12 * len(vectors), 10)
        ):
            raise RuntimeError("Incomplete matrix output")
        results = []
        for i in range(1, len(vectors) + 1):
            row = case[i - 1, 5:]
            m = mat[mat[:, 0] == i, 2:].reshape(6, 3, 6).transpose(1, 0, 2)
            differences = fd[fd[:, 0] == i, 4:].reshape(2, 6, 6).transpose(0, 2, 1)
            results.append((row, m, differences))
        return results


    def split_map(v, length, period, field, p0, helical):
        """Two cubic half kicks around the averaged Hill rotation; p = P_transverse/P0."""
        r = 1 + v[5]
        kappa = 299792458.0 * field / (math.sqrt(2) * p0)
        theta = kappa * length / r
        coefficient = -length * kappa**2 * (2 * math.pi / period) ** 2 / (3 * r)
        out = np.array(v[:4], dtype=complex)
        for i in (0, 2):
            if i == 0 and not helical:
                out[i] += length * out[i + 1] / r
                continue
            x, p = out[i : i + 2]
            p += coefficient * x**3
            if field == 0:
                x += length * p / r
            else:
                x, p = (
                    np.cos(theta) * x + np.sin(theta) * p / kappa,
                    -kappa * np.sin(theta) * x + np.cos(theta) * p,
                )
            p += coefficient * x**3
            out[i : i + 2] = [x, p]
        return out


    def split_jacobian(v, length, period, field, p0, helical):
        jac = np.empty((4, 6))
        for col in range(6):
            perturbed = np.array(v, dtype=complex)
            perturbed[col] += 1e-25j
            jac[:, col] = split_map(perturbed, length, period, field, p0, helical).imag / 1e-25
        return jac


    def radiation(lattice):
        log = fortran(RADIATION_PROGRAM, ["lat.bmad"], files={"lat.bmad": lattice}).log
        meta, total, end = (records(log, label) for label in ("META", "TOTAL", "END"))
        if meta.shape != (1, 5) or total.shape != (3, 9) or end.shape != (1, 6):
            raise RuntimeError("Incomplete radiation output")
        np.testing.assert_array_equal(total[:, 2:], np.tile(total[0, 2:], (3, 1)))
        return meta[0], total[0, 2:], end[0]

    ```

## `_tao_contracts_normal_form.py` { #_tao_contracts_normal_form }

[Download _tao_contracts_normal_form.py](_tao_contracts_normal_form.py)

??? example "Source: _tao_contracts_normal_form.py"

    ```python
    """Inputs and strict Tao record parsing for the contracts and normal-form cases."""

    import re
    from collections.abc import Mapping, Sequence

    import numpy as np
    from _bmad import Result, run

    ERROR = re.compile(
        r"\[(?:ERROR|FATAL|ABORT|SEVERE)\b|Program received signal|Fortran runtime error"
        r"|^\s*(?:ERROR(?: STOP| IN\b|:|;|\s*$)|FATAL\b|Backtrace for this error:)",
        re.IGNORECASE | re.MULTILINE,
    )
    LATTICE = """parameter[geometry] = open
    parameter[particle] = electron
    parameter[p0c] = 1e6
    beginning[beta_a] = 3
    beginning[alpha_a] = 0.2
    beginning[beta_b] = 4
    particle_start[x] = 0.001
    particle_start[y] = -0.002
    particle_start[z] = 0.03
    e: drift, l=1.7
    lat: line=(e)
    use, lat
    """
    FODO = """parameter[geometry] = closed
    parameter[particle] = electron
    parameter[p0c] = 1e9
    qf: quadrupole, l=0.2, k1=1, fringe_type=none
    qd: quadrupole, l=0.2, k1=-0.96, fringe_type=none
    d: drift, l=0.8
    lat: line=(qf,d,qd,d)
    use, lat
    """
    THIN_RING = """parameter[geometry] = closed
    parameter[particle] = electron
    parameter[p0c] = 1e9
    parameter[taylor_order] = 4
    sx: multipole, k2l={strength}
    qf: multipole, k1l=0.8
    qd: multipole, k1l=-0.7
    d: drift, l=1
    lat: line=(sx,qf,d,qd,d)
    use, lat
    """


    def tao(
        lattice: str,
        commands: Sequence[str],
        *,
        files: Mapping[str, str] | None = None,
        outputs: Sequence[str] = (),
        options: Sequence[str] | None = None,
        check: bool = True,
    ) -> Result:
        """Run an isolated command file; require clean completion for ordinary cases."""
        args = ["-noplot", *(options if options is not None else ["-lat", "lat.bmad", "-noinit"])]
        result = run(
            "tao",
            [*args, "-startup", "run.tao"],
            files={"lat.bmad": lattice, "run.tao": "\n".join([*commands, "quit", ""]), **(files or {})},
            outputs=outputs,
        )
        if check and (ERROR.search(result.log) or result.log.count("Tao: quit\n") != 1):
            raise RuntimeError(result.log)
        return result


    def block(log: str, command: str) -> str:
        marker = f"Tao: {command}\n"
        if log.count(marker) != 1:
            raise ValueError(f"Expected exactly one command {command!r}:\n{log}")
        return log.split(marker, 1)[1].split("\nTao:", 1)[0].strip()


    def record(text: str, name: str) -> list[str]:
        rows = [line.split(";") for line in text.splitlines() if line.startswith(name + ";")]
        if len(rows) != 1:
            raise ValueError(f"Expected one {name} record:\n{text}")
        return rows[0]


    def scalar(text: str, name: str) -> float:
        row = record(text, name)
        if len(row) != 4 or row[1] != "REAL":
            raise ValueError(f"Expected scalar REAL: {row}")
        value = float(row[3])
        if not np.isfinite(value):
            raise ValueError(f"Nonfinite {name}: {value}")
        return value


    def evaluated(log: str, expression: str) -> float:
        rows = block(log, "pipe evaluate " + expression).splitlines()
        if len(rows) != 1:
            raise ValueError(f"Expected one evaluated value: {rows}")
        fields = rows[0].split(";")
        if len(fields) != 2 or fields[0] != "1":
            raise ValueError(f"Expected valid evaluation: {fields}")
        value = float(fields[1])
        if not np.isfinite(value):
            raise ValueError(f"Nonfinite evaluation: {fields}")
        return value


    def thin_beta() -> float:
        """Periodic horizontal beta at SX from independent thin-lens matrices."""
        drift = np.array([[1.0, 1.0], [0.0, 1.0]])
        matrix = (
            drift @ np.array([[1.0, 0.0], [0.7, 1.0]]) @ drift @ np.array([[1.0, 0.0], [-0.8, 1.0]])
        )
        sine = np.sqrt(1 - (np.trace(matrix) / 2) ** 2)
        return float(matrix[0, 1] / sine)


    def fodo_matrix(delta: float = 0.0) -> np.ndarray:
        """Independent thick-quadrupole transverse matrix in normalized momenta."""
        momentum = 1 + delta
        drift = np.array([[1.0, 0.8 / momentum], [0.0, 1.0]])

        def quad(k: float) -> np.ndarray:
            root = np.sqrt(abs(k) / momentum)
            phase = root * 0.2
            c, s = (np.cos(phase), np.sin(phase)) if k > 0 else (np.cosh(phase), np.sinh(phase))
            return np.array([[c, s / (momentum * root)], [-np.sign(k) * momentum * root * s, c]])

        matrix = np.zeros((4, 4))
        for start, sign in ((0, 1), (2, -1)):
            matrix[start : start + 2, start : start + 2] = (
                drift @ quad(-0.96 * sign) @ drift @ quad(sign)
            )
        return matrix

    ```

## `_tao_optimization_io.py` { #_tao_optimization_io }

[Download _tao_optimization_io.py](_tao_optimization_io.py)

??? example "Source: _tao_optimization_io.py"

    ```python
    """Embedded Tao optimization/I/O cases and checked command-output parsing."""

    import re
    from collections.abc import Mapping, Sequence

    import numpy as np
    from _bmad import Result, run


    def tao(
        commands: Sequence[str],
        *,
        lattice: str,
        init: str | None = None,
        files: Mapping[str, str | bytes] | None = None,
        outputs: Sequence[str] = (),
        allow_errors: bool = False,
        allow_aborted_script: bool = False,
    ) -> Result:
        """Run one fresh Tao process; callers accepting diagnostics must identify them exactly."""
        inputs: dict[str, str | bytes] = {
            "line.bmad": lattice,
            "lat.bmad": lattice,
            "run.tao": "\n".join([*commands, "quit", ""]),
            **(files or {}),
        }
        if init is None:
            options = ["-noinit", "-lat", "lat.bmad"]
        else:
            inputs["case.init"] = init
            options = ["-init", "case.init"]
        result = run("tao", ["-noplot", *options, "-startup", "run.tao"], files=inputs, outputs=outputs)
        if not allow_aborted_script and result.log.count("Tao: quit\n") != 1:
            raise RuntimeError(f"Tao did not complete its command file:\n{result.log}")
        if not allow_errors and diagnostics(result.log):
            raise RuntimeError(result.log)
        return result


    def diagnostics(log: str) -> list[tuple[str, tuple[str, ...]]]:
        """Parse complete Tao errors and reject unmatched fatal/runtime diagnostics."""
        pattern = re.compile(
            r"^(?:Tao> )?\[(?:ERROR|FATAL|ABORT|SEVERE)\b[^\n]*\]\s+(\w+):\n"
            r"((?:[ \t]+[^\n]*\n)+)",
            re.MULTILINE,
        )
        remainder = pattern.sub("", log)
        if re.search(r"\[(?:ERROR|FATAL|ABORT|SEVERE)\b|Fortran runtime error|SIGSEGV", remainder):
            raise RuntimeError(log)
        return [
            (m[1], tuple(line.strip() for line in m[2].splitlines())) for m in pattern.finditer(log)
        ]


    def block(log: str, command: str) -> str:
        marker = f"Tao: {command}\n"
        if log.count(marker) != 1:
            raise RuntimeError(f"Expected one completed command {command!r}:\n{log}")
        return log.split(marker, 1)[1].split("\nTao:", 1)[0].strip()


    def records(text: str) -> list[list[str]]:
        return [[part.strip() for part in line.split(";")] for line in text.splitlines() if ";" in line]


    def parameters(text: str) -> dict[str, str]:
        rows = records(text)
        if not rows or any(len(row) < 4 for row in rows):
            raise ValueError(f"Expected typed parameter records: {text}")
        if len({row[0] for row in rows}) != len(rows):
            raise ValueError(f"Duplicate parameters: {text}")
        return {row[0]: row[3] for row in rows}


    def numbers(text: str) -> np.ndarray:
        rows = [line.replace(";", " ").split() for line in text.splitlines() if line.strip()]
        if not rows:
            raise ValueError(f"Expected numeric rows: {text}")
        return np.array(rows, dtype=float)


    def data_values(log: str, command: str) -> np.ndarray:
        rows = records(block(log, command))
        if len(rows) != 1 or rows[0][0] != "1":
            raise ValueError(f"Expected one data array in universe 1: {rows}")
        return np.array(rows[0][1:], dtype=float)


    def control(label: str, observed, expected, *, atol: float) -> None:
        from _bmad import compare

        if compare(label, observed, expected, atol=atol):
            raise RuntimeError(f"Control failed: {label}")


    LINE = """no_digested
    parameter[geometry] = open
    parameter[particle] = electron
    beginning[p0c] = 1e9
    beginning[beta_a] = 2
    beginning[beta_b] = 3
    beginning[dbeta_dpz_a] = 11
    beginning[dbeta_dpz_b] = 22
    beginning[dalpha_dpz_a] = 33
    beginning[dalpha_dpz_b] = 44
    particle_start[x] = 0.003
    particle_start[px] = 0.001
    d1: drift, l = 1
    q1: quadrupole, l = 0.2, k1 = 0.3
    d2: drift, l = 1
    q2: quadrupole, l = 0.2, k1 = -0.2
    fin: marker
    lin: line = (d1, q1, d2, q2, fin)
    use, lin
    """

    LINEAR_INIT = """&tao_design_lattice
      design_lattice(1)%file = 'line.bmad'
    /
    &tao_params
      global%plot_on = F
      global%optimizer = 'svd'
      global%n_opti_cycles = 1
      global%opti_write_var_file = F
    /
    &tao_d2_data
      d2_data%name = 'linear'
      n_d1_data = 1
    /
    &tao_d1_data
      ix_d1_data = 1
      d1_data%name = 'k'
      datum(1)%data_type = 'expression:ele::q1[k1]'
      datum(1)%meas = 0
      datum(1)%weight = 1
      datum(2)%data_type = 'expression:ele::q2[k1]'
      datum(2)%meas = 0
      datum(2)%weight = 1
    /
    &tao_var
      v1_var%name = 'v'
      default_step = 1e-5
      default_universe = '1'
      default_attribute = 'k1'
      ix_min_var = 1
      ix_max_var = 2
      var(1)%ele_name = 'q1'
      var(1)%weight = 0
      var(2)%ele_name = 'q2'
      var(2)%merit_type = 'target'
      var(2)%meas = 0.1
      var(2)%weight = 1
    /
    """

    WEIGHT_INIT = """&tao_design_lattice
      design_lattice(1)%file = 'line.bmad'
    /
    &tao_params
      global%plot_on = F
      global%optimizer = 'geodesic_lm'
      global%opti_write_var_file = F
    /
    &tao_d2_data
      d2_data%name = 'linear'
      n_d1_data = 1
    /
    &tao_d1_data
      ix_d1_data = 1
      d1_data%name = 'k'
      datum(1)%data_type = 'expression:ele::q2[k1]'
      datum(1)%meas = 0
      datum(1)%weight = 1
    /
    &tao_var
      v1_var%name = 'v'
      default_step = 1e-5
      default_universe = '1'
      default_attribute = 'k1'
      ix_min_var = 1
      ix_max_var = 1
      var(1)%ele_name = 'q2'
      var(1)%merit_type = 'target'
      var(1)%meas = 0.1
      var(1)%weight = 100
    /
    """

    RMS_INIT = """&tao_design_lattice
      design_lattice(1)%file = 'line.bmad'
    /
    &tao_params
      global%plot_on = F
    /
    &tao_d2_data
      d2_data%name = 'range'
      n_d1_data = 1
    /
    &tao_d1_data
      ix_d1_data = 1
      d1_data%name = 'x'
      datum(1) = 'orbit.x' '' 'beginning' 'd1' 'rms' 0.0 1.0
      datum(2) = 'orbit.x' '' 'beginning' 'd1' 'average' 0.0 1.0
      datum(3) = 'orbit.x' '' 'beginning' 'd1' 'integral' 0.0 1.0
    /
    """

    ABS_INIT = """&tao_design_lattice
      design_lattice(1)%file = 'line.bmad'
    /
    &tao_params
      global%plot_on = F
    /
    &tao_d2_data
      d2_data%name = 'dat'
      n_d1_data = 1
    /
    &tao_d1_data
      ix_d1_data = 1
      d1_data%name = 'x'
      datum(1) = 'orbit.x' 'beginning' '' 'd1' 'abs_max' 0.0 1.0
      datum(2) = 'orbit.x' 'beginning' '' 'd1' 'target' 0.0 1.0
      datum(3) = 'orbit.x' 'beginning' 'd1' 'd1' 'abs_min' 0.0 1.0
    /
    """

    LATTICE = """parameter[geometry] = open
    parameter[particle] = electron
    parameter[p0c] = 1e6
    beginning[beta_a] = 3
    beginning[alpha_a] = 0.2
    beginning[beta_b] = 4
    particle_start[x] = 0.001
    particle_start[y] = -0.002
    particle_start[z] = 0.03
    e: drift, l=1.7
    lat: line=(e)
    use, lat
    """

    RING_LATTICE = """parameter[geometry] = closed
    parameter[particle] = electron
    parameter[p0c] = 1e9
    qf: quadrupole, l=0.2, k1=1, fringe_type=none
    qd: quadrupole, l=0.2, k1=-0.96, fringe_type=none
    d: drift, l=0.8
    lat: line=(qf,d,qd,d)
    use, lat
    """

    ```

## `_tracking_contracts_materials.py` { #_tracking_contracts_materials }

[Download _tracking_contracts_materials.py](_tracking_contracts_materials.py)

??? example "Source: _tracking_contracts_materials.py"

    ```python
    """Shared parsing and inline cases for the tracking contracts and materials entries."""

    from __future__ import annotations

    import re

    import numpy as np
    from _bmad import fortran, run


    def rows(log: str) -> dict[str, np.ndarray]:
        """Read unique finite numeric RESULT records; reject native error diagnostics."""
        if re.search(r"\[(?:ERROR|FATAL|ABORT|SEVERE)\b|Fortran runtime error|ERROR STOP", log):
            raise RuntimeError(log)
        result = {}
        for line in log.splitlines():
            if line.startswith("RESULT "):
                _, name, *values = line.split()
                if name in result or not values:
                    raise ValueError(f"Duplicate or empty record: {line}")
                result[name] = np.array([float(x.replace("D", "E")) for x in values])
                if not np.isfinite(result[name]).all():
                    raise ValueError(f"Nonfinite record: {line}")
        if not result:
            raise ValueError(f"No result records:\n{log}")
        return result


    def constants() -> dict[str, np.ndarray]:
        return rows(
            fortran("""
    program constants_probe
    use bmad
    implicit none
    write(*, '(a,6es26.17e3)') 'RESULT constants ', &
     m_electron, m_proton, r_e, anomalous_mag_moment_electron, e_charge, c_light
    end program
    """).log
        )


    def lattice(element: str, *, start: str = "", p0c: float = 1e9, line: str = "e") -> str:
        return (
            f"no_digested\nparameter[geometry]=open\nparameter[particle]=electron\n"
            f"parameter[p0c]={p0c:.17g}\nbeginning[beta_a]=10\nbeginning[beta_b]=10\n"
            f"{start}\n{element}\nbl: line=({line})\nuse, bl\n"
        )


    def tao(lat: str, commands: list[str]) -> dict[str, str]:
        log = run(
            "tao",
            ["-noplot", "-noinit", "-lat", "lat.bmad", "-startup", "run.tao"],
            files={"lat.bmad": lat, "run.tao": "\n".join([*commands, "quit", ""])},
        ).log
        if "Tao: quit" not in log or re.search(r"\[(?:ERROR|FATAL|ABORT|SEVERE)\b", log):
            raise RuntimeError(log)
        blocks = {}
        for command in commands:
            marker = f"Tao: {command}\n"
            if log.count(marker) != 1:
                raise ValueError(log)
            blocks[command] = log.split(marker)[1].split("\nTao:", 1)[0]
        return blocks


    def pipe_values(block: str, name: str) -> np.ndarray:
        matches = [line for line in block.splitlines() if line.startswith(name + ";")]
        if len(matches) != 1:
            raise ValueError(f"Expected one {name} record:\n{block}")
        fields = matches[0].split(";")[3:]
        values = np.array([float(s) for s in fields if s.strip()])
        if not values.size or not np.isfinite(values).all():
            raise ValueError(matches[0])
        return values


    def orbit(block: str) -> np.ndarray:
        if "state;STR;F;Alive" not in block:
            raise ValueError(f"Particle is not alive:\n{block}")
        return np.array([pipe_values(block, k)[0] for k in ("x", "px", "y", "py", "z", "pz")])

    ```

## `_translators.py` { #_translators }

[Download _translators.py](_translators.py)

??? example "Source: _translators.py"

    ```python
    """Shared code for the lattice-translator reproducers: Bmad export, Tao records, MAD-X, Elegant.

    export() runs the production converter bmad_to_mad_sad_elegant, the program users run to write
    MAD-X, MAD8, SAD and Elegant files; import_madx() runs util_programs/mad_to_bmad/madx_to_bmad.py
    of the BMAD_SOURCE tree. tao() collects `pipe` records of a Bmad lattice. madx() runs MAD-X through
    cpymad. elegant_track() runs serial `elegant`, which must be on PATH with RPN_DEFNS naming the
    defns.rpn file of SDDS. Every case runs in fresh scratch with one numerical thread.
    """

    from __future__ import annotations

    import contextlib
    import os
    import re
    import shutil
    import subprocess
    import sys
    import tempfile
    from collections.abc import Iterator, Sequence
    from pathlib import Path

    import numpy as np
    from _bmad import TIMEOUT, RunFailure, run, source
    from cpymad.madx import Madx

    SUFFIX = {"madx": "madx", "mad8": "mad8", "sad": "sad", "elegant": "lte"}


    def lattice(
        elements: str,
        line: str,
        *,
        p0c: float = 1e9,
        particle: str = "electron",
        beta: tuple[float, float] = (10.0, 10.0),
        start: Sequence[float] | None = None,
        extra: str = "",
    ) -> str:
        """Open-geometry Bmad lattice: element definitions, beamline `lat` and the initial optics."""
        lines = [
            "no_digested",
            "parameter[geometry] = open",
            f"parameter[particle] = {particle}",
            f"parameter[p0c] = {p0c!r}",
            f"beginning[beta_a] = {beta[0]!r}",
            f"beginning[beta_b] = {beta[1]!r}",
        ]
        for name, value in zip(("x", "px", "y", "py", "z", "pz"), start or (), strict=False):
            lines.append(f"particle_start[{name}] = {value!r}")
        lines += [extra, elements, f"lat: line = ({line})", "use, lat", ""]
        return "\n".join(lines)


    def bend_exit(g: float, g_tot: float, angle: float) -> np.ndarray:
        """Exit (x, px) of an on-axis particle at reference momentum in a sector bend, edges ignored.

        The reference orbit is a circle of radius 1/g about the origin, from (1/g, 0) through the
        bend angle; the particle moves on a circle of radius 1/g_tot tangent to it at the entrance.
        Bmad and MAD-X take x outward (away from the centre) and px = sin of the angle between the
        particle and the reference direction at the exit plane.
        """
        rho, rho_p = 1 / g, 1 / g_tot
        # Particle at (rho - rho_p + rho_p cos(psi), rho_p sin(psi)); the exit plane is the ray from
        # the origin at the bend angle: -X sin(angle) + Y cos(angle) = 0, which gives
        # rho_p sin(psi - angle) = (rho - rho_p) sin(angle).
        psi = angle + np.arcsin((rho - rho_p) * np.sin(angle) / rho_p)
        big_x, big_y = rho - rho_p + rho_p * np.cos(psi), rho_p * np.sin(psi)
        x = big_x * np.cos(angle) + big_y * np.sin(angle) - rho
        return np.array([x, -np.sin(psi - angle)])


    def tao(lat: str, commands: Sequence[str]) -> dict[str, list[str]]:
        """Run Tao on a lattice and return the output lines of each command, keyed by command.

        Every command must echo and print at least one record, and the run must reach `quit`.
        """
        startup = "\n".join([*commands, "quit", ""])
        log = run(
            "tao",
            ["-noplot", "-noinit", "-lat", "lat.bmad", "-startup", "run.tao"],
            files={"lat.bmad": lat, "run.tao": startup},
        ).log
        if re.search(r"\[(?:ERROR|FATAL|ABORT)|ERROR IN ", log):
            raise RuntimeError(f"Tao initialization or command failed:\n{log[-3000:]}")
        sections: dict[str, list[str]] = {}
        current: list[str] | None = None
        for text in log.splitlines():
            if text.startswith("Tao: "):
                current = sections.setdefault(text[5:].strip(), [])
            elif current is not None and text.strip():
                current.append(text)
        if "quit" not in sections:
            raise RuntimeError(f"Tao did not reach quit:\n{log[-3000:]}")
        for command in commands:
            if not sections.get(command):
                raise RuntimeError(f"Tao printed no records for {command!r}:\n{log[-3000:]}")
            if any("ERROR" in text for text in sections[command]):
                raise RuntimeError(f"Tao reported an error for {command!r}:\n{log[-3000:]}")
        return sections


    def record(lines: Sequence[str]) -> dict[str, str]:
        """Parse `name;type;settable;value` pipe records into {name: value text}."""
        values = {}
        for text in lines:
            fields = text.split(";")
            if len(fields) >= 4:
                values[fields[0].strip()] = fields[3].strip()
        if not values:
            raise RuntimeError(f"No pipe records in {lines!r}")
        return values


    def numbers(lines: Sequence[str], keys: Sequence[str]) -> np.ndarray:
        """Named real values of one pipe record."""
        values = record(lines)
        return np.array([float(values[key]) for key in keys])


    def export(lat: str, fmt: str) -> str:
        """Translate a Bmad lattice with bmad_to_mad_sad_elegant -<fmt>; return the written file."""
        name = "lat." + SUFFIX[fmt]
        result = run(
            "bmad_to_mad_sad_elegant", [f"-{fmt}", "lat.bmad"], files={"lat.bmad": lat}, outputs=[name]
        )
        if "PROBLEM TRACKING" in result.log or "ERROR" in result.log:
            raise RuntimeError(f"Export to {fmt} reported a problem:\n{result.log[-3000:]}")
        return result.outputs[name].decode()


    def statements(text: str, fmt: str) -> dict[str, str]:
        """Element definitions of a written file by element name, continuation lines joined.

        MAD-X statements end at `;`; MAD8 and Elegant continue a line ending in `&`.
        """
        if fmt == "madx":
            joined = [" ".join(s.split()) for s in text.replace("\n", " ").split(";")]
        else:
            joined = [" ".join(s.split()) for s in text.replace("&\n", " ").splitlines()]
        found = {}
        for statement in joined:
            match = re.match(r"([A-Za-z_][\w.]*)\s*:\s*(.*)$", statement)
            if match and not statement.startswith(("!", "//")):
                found[match[1].upper()] = match[2]
        return found


    def value(statement: str, key: str, default: float | None = None) -> float:
        """Number assigned to `key` in one element statement; `default` when it is absent."""
        match = re.search(rf"\b{key}\s*=\s*([^,;\s]+)", statement)
        if match is None:
            if default is None:
                raise KeyError(f"{key} is not set in: {statement}")
            return default
        return float(match[1])


    def element_type(text: str, fmt: str, name: str) -> str:
        """Element class keyword that a written file gives to the named element."""
        return statements(text, fmt)[name.upper()].split(",")[0].strip().lower()


    @contextlib.contextmanager
    def madx(deck: str) -> Iterator[Madx]:
        """A MAD-X process (cpymad) that has read the deck, in a scratch directory."""
        with tempfile.TemporaryDirectory(prefix="madx-") as tmp:
            Path(tmp, "deck.madx").write_text(deck)
            with Madx(stdout=False, cwd=tmp) as mad:
                mad.call("deck.madx")
                yield mad


    def madx_track(mad: Madx, start: Sequence[float], *, sequence: str = "lat") -> np.ndarray:
        """One-pass MAD-X TRACK of one particle; returns (x, px, y, py, t, pt) at the end."""
        names = ("x", "px", "y", "py", "t", "pt")
        begin = ", ".join(f"{k} = {float(v)!r}" for k, v in zip(names, start, strict=True))
        mad.input(
            f"use, period = {sequence}; track, onepass, onetable; start, {begin};"
            "run, turns = 1, maxaper = {1, 1, 1, 1, 1, 1}; endtrack;"
        )
        table = mad.table.trackone
        last = table.turn == 1
        if np.count_nonzero(last) != 1:
            raise RuntimeError("MAD-X TRACK lost the particle")
        return np.array([table[k][last][0] for k in names])


    def import_madx(deck: str) -> str:
        """Translate a MAD-X deck with BMAD_SOURCE's madx_to_bmad.py; return the Bmad text."""
        script = source() / "util_programs/mad_to_bmad/madx_to_bmad.py"
        with tempfile.TemporaryDirectory(prefix="madx-import-") as tmp:
            Path(tmp, "source.madx").write_text(deck)
            done = subprocess.run(
                [sys.executable, "-I", str(script), "source.madx"],
                cwd=tmp,
                capture_output=True,
                text=True,
                timeout=TIMEOUT,
                check=False,
            )
            if done.returncode:
                raise RunFailure("madx_to_bmad.py", done.returncode, done.stdout + done.stderr)
            return Path(tmp, "source.bmad").read_text()


    def elegant_track(
        lte: str, p_central_mev: float, start: Sequence[float], *, beamline: str = "lat"
    ) -> np.ndarray:
        """Track one particle (x, xp, y, yp) through an Elegant lattice; return the final centroid.

        Returns (s, x, xp, y, yp) at the end of the beamline, from the centroid output.
        """
        import pysdds

        if "RPN_DEFNS" not in os.environ or shutil.which("elegant") is None:
            raise RuntimeError("Put elegant on PATH and set RPN_DEFNS to the defns.rpn file of SDDS.")
        centroid = ", ".join(f"centroid[{i}] = {float(v)!r}" for i, v in enumerate(start))
        commands = (
            f'&run_setup lattice = "lat.lte", use_beamline = "{beamline}", '
            f'p_central_mev = {p_central_mev!r}, centroid = "out.cen", output = "out.sdds" &end\n'
            "&run_control n_steps = 1 &end\n"
            f"&bunched_beam n_particles_per_bunch = 1, {centroid} &end\n"
            "&track &end\n"
        )
        env = dict(os.environ, OMP_NUM_THREADS="1")
        with tempfile.TemporaryDirectory(prefix="elegant-") as tmp:
            Path(tmp, "lat.lte").write_text(lte)
            Path(tmp, "run.ele").write_text(commands)
            done = subprocess.run(
                ["elegant", "run.ele"],
                cwd=tmp,
                env=env,
                capture_output=True,
                text=True,
                timeout=TIMEOUT,
                check=False,
            )
            if done.returncode:
                raise RunFailure("elegant", done.returncode, done.stdout + done.stderr)
            particles = pysdds.read(Path(tmp, "out.sdds"))
            phase = [np.asarray(particles[0, name]) for name in ("x", "xp", "y", "yp", "t", "p")]
            if any(values.shape != (1,) or not np.isfinite(values).all() for values in phase):
                raise RuntimeError("Elegant must return exactly one finite surviving particle")
            out = pysdds.read(Path(tmp, "out.cen"))
            columns = [np.asarray(out[0, name]) for name in ("s", "Cx", "Cxp", "Cy", "Cyp")]
        if any(
            column.ndim != 1 or not len(column) or not np.isfinite(column).all() for column in columns
        ):
            raise RuntimeError("Elegant returned incomplete or nonfinite centroid records")
        result = np.array([column[-1] for column in columns])
        if not np.allclose(result[1:], [values[0] for values in phase[:4]], rtol=1e-12, atol=1e-15):
            raise RuntimeError("Elegant final particle and centroid disagree")
        return result

    ```

## `_wake_tables_and_bbu.py` { #_wake_tables_and_bbu }

[Download _wake_tables_and_bbu.py](_wake_tables_and_bbu.py)

??? example "Source: _wake_tables_and_bbu.py"

    ```python
    """Inline wake-table cases and independent sums for the wake-tables-and-bbu chunk."""

    import re

    import numpy as np
    from _bmad import compare, run


    def rows(log: str) -> dict[str, np.ndarray]:
        """Read unique numeric RESULT records; absence and duplicates are harness failures."""
        result = {}
        for line in log.splitlines():
            if not line.startswith("RESULT "):
                continue
            _, name, *values = line.split()
            if name in result or not values:
                raise RuntimeError(f"Invalid RESULT record: {line}")
            result[name] = np.array([float(x.replace("D", "E")) for x in values])
        if not result:
            raise RuntimeError(f"No RESULT records:\n{log}")
        return result


    def require(label, observed, expected, *, atol=0.0, rtol=0.0):
        if compare(label, observed, expected, atol=atol, rtol=rtol):
            raise RuntimeError(f"Control failed: {label}")


    def table_lattice():
        z = np.arange(-5, 6) * 0.01
        table = 1e18 * (1 + z / 0.05) ** 2
        text = ", ".join(f"{u:.17g} {w:.17g}" for u, w in zip(z, table, strict=True))
        lattice = (
            "parameter[particle]=electron\nparameter[p0c]=1e12\n"
            "w: lcavity,l=0.6,sr_wake={scale_with_length=F,z_long={w={"
            + text
            + ", }}}\nl: line=(w)\nuse,l\n"
        )
        return lattice, table


    def table_kicks(z, charge, table, center):
        """Direct cloud-in-cell source/witness sum, zero outside the supplied wake table."""
        table = table.copy()
        half = len(table) // 2
        table[half] *= 0.5  # Fundamental theorem of beam loading: half the self wake.
        grid = (np.asarray(z) - center) / 0.01 + half
        low = np.floor(grid).astype(int)
        fraction = grid - low
        weights = np.stack([1 - fraction, fraction], axis=1)
        result = np.zeros(len(z))
        for i in range(len(z)):
            for j in range(len(z)):
                for a in (0, 1):
                    for b in (0, 1):
                        index = low[i] + a - low[j] - b + half
                        if 0 <= index < len(table):
                            result[i] -= weights[i, a] * weights[j, b] * charge[j] * table[index] / 1e12
        return result


    def bbu_control(init="control.init", label="BBU control HOM voltage gain"):
        result = run("bbu", [init], files=BBU_FILES, outputs=["for_py.txt"])
        text = result.outputs["for_py.txt"].decode()
        match = re.search(r"(?im)^\s*v_gain\s*=\s*([\d.eEdD+\-]+)", text)
        if match is None:
            raise RuntimeError(f"Missing BBU completion record: {text}\n{result.log}")
        if "lostbool = False" not in text:
            raise RuntimeError(f"BBU control lost a particle: {text}\n{result.log}")
        require(label, float(match[1].replace("D", "E")), 1.0, atol=2e-15)
        return result


    TABLE_WAKE_SOURCE = r"""
    program table_wake
      use bmad
      use wake_mod
      implicit none
      type(lat_struct) lat
      type(bunch_struct) bunch
      integer i
      logical err
      call bmad_parser('lat.bmad', lat, err_flag=err)
      if (err) error stop 'Parser failed'
      call reallocate_bunch(bunch, 3)
      do i = 1, 3
        call init_coord(bunch%particle(i), [0.0_rp,0.0_rp,0.0_rp,0.0_rp,(i-2)*0.01_rp,0.0_rp], &
          lat%ele(1), upstream_end$, electron$)
        bunch%particle(i)%charge = 0
      enddo
      bunch%particle(2)%charge = 2.0e-12_rp
      call sr_z_long_wake(lat%ele(1), bunch, 0.0_rp)
      write(*, '(a,3es25.16e3)') 'RESULT table ', bunch%particle%vec(6)
      write(*, '(a,9es25.16e3)') 'RESULT raw ', lat%ele(1)%wake%sr%z_long%w
    end program
    """


    WAKE_SUPPORT_SOURCE = r"""
    program wake_support
      use bmad
      use wake_mod
      use output_mod, only: output_direct
      implicit none
      type(lat_struct) lat
      type(bunch_struct) bunch
      logical err
      integer i,j,which,perm(3),n
      real(rp) physical_z(3), physical_kick(3)
      character(16) arg,name
      call get_command_argument(1,arg); read(arg,*) which
      call bmad_parser('lat.bmad',lat,err_flag=err)
      if (err) error stop 'Parser failed'
      bmad_com%sr_wakes_on = .true.
      select case(which)
      case(1)
        physical_z = [0.013_rp,0.002_rp,-0.014_rp]
        do j=0,2
          call prepare(3)
          perm=[1,2,3]
          if (j==2) perm=[1,3,2]
          do i=1,3
            bunch%particle(i)%vec(5)=physical_z(perm(i))
            if (perm(i)==1) bunch%particle(i)%charge=2.0e-12_rp
          enddo
          call order_particles_in_z(bunch)
          if (j==0) then
            call sr_z_long_wake(lat%ele(1),bunch,-0.0005_rp)
          else
            call track1_sr_wake(bunch,lat%ele(1))
          endif
          do i=1,3
            physical_kick(perm(i))=bunch%particle(i)%vec(6)
          enddo
          write(name,'(a,i0)') 'permutation',j
          write(*,'(3a,3es25.16e3)') 'RESULT ',trim(name),' ',physical_kick
        enddo
      case(2)
        call prepare(2)
        bunch%particle(1)%vec(5)=0.09_rp
        bunch%particle(2)%charge=2.0e-12_rp
        open(unit=51,file='diagnostic.log',status='replace')
        call output_direct(file_unit=51,print_and_capture=.false.,min_level=s_error$,max_level=s_error$)
        call sr_z_long_wake(lat%ele(1),bunch,0.0_rp)
        close(51)
        write(*,'(a,2i4)') 'RESULT lost ',(merge(1,0,bunch%particle(i)%state==lost$),i=1,2)
        write(*,'(a,2es25.16e3)') 'RESULT kicks ',bunch%particle%vec(6)
      case(3)
        call prepare(101)
        bunch%particle(1)%charge=2.0e-12_rp
        bunch%particle(101)%vec(5)=-0.065_rp
        call sr_z_long_wake(lat%ele(1),bunch,0.0_rp)
        write(*,'(a,100es25.16e3)') 'RESULT core ',bunch%particle(1:100)%vec(6)
        write(*,'(a,es25.16e3)') 'RESULT outlier ',bunch%particle(101)%vec(6)
        write(*,'(a,i4)') 'RESULT live ',count(bunch%particle%state==alive$)
      end select
    contains
      subroutine prepare(count_particles)
        integer count_particles,k
        call reallocate_bunch(bunch,count_particles)
        do k=1,count_particles
          call init_coord(bunch%particle(k),[0.0_rp,0.0_rp,0.0_rp,0.0_rp,0.0_rp,0.0_rp], &
            lat%ele(1),upstream_end$,electron$)
          bunch%particle(k)%charge=0
        enddo
      end subroutine
    end program
    """

    BBU_FILES = {
        "cutoff.bmad": "no_digested\nparameter[particle]=electron\nparameter[p0c]=1e9\nparameter[geometry]=open\nbeginning[beta_a]=10\nbeginning[beta_b]=10\nw: lcavity, l=1, rf_frequency=1.3e9, lr_wake={ &\n  mode={1e9,100,0,0,1,unpolarized}, mode={2e9,100,0,0,2,unpolarized}}\nln: line=(w)\nuse,ln\n",
        "cutoff_all.bmad": "no_digested\nparameter[particle]=electron\nparameter[p0c]=1e9\nparameter[geometry]=open\nbeginning[beta_a]=10\nbeginning[beta_b]=10\nw1: lcavity, l=1, rf_frequency=1.3e9, lr_wake={mode={1e9,100,0,0,1,unpolarized}}\nw2: lcavity, l=1, rf_frequency=1.3e9, lr_wake={mode={2e9,100,0,0,2,unpolarized}}\nln: line=(w1,w2)\nuse,ln\n",
        "control.init": "&bbu_params\nbbu_param%lat_filename='cutoff.bmad'\nbbu_param%hom_order_cutoff=2\nbbu_param%hybridize=False\nbbu_param%simulation_turns_max=12\nbbu_param%ran_seed=123\n/\n",
        "ramp.init": "&bbu_params\nbbu_param%lat_filename='cutoff.bmad'\nbbu_param%ramp_on=True\nbbu_param%ramp_pattern=1,2,3\nbbu_param%hybridize=False\nbbu_param%simulation_turns_max=12\nbbu_param%ran_seed=123\n/\n",
        "cutoff.init": "&bbu_params\nbbu_param%lat_filename='cutoff.bmad'\nbbu_param%hom_order_cutoff=1\nbbu_param%hybridize=False\nbbu_param%simulation_turns_max=12\nbbu_param%ran_seed=123\n/\n",
        "cutoff_all.init": "&bbu_params\nbbu_param%lat_filename='cutoff_all.bmad'\nbbu_param%hom_order_cutoff=1\nbbu_param%hybridize=False\nbbu_param%simulation_turns_max=12\nbbu_param%ran_seed=123\n/\n",
    }

    # Match the two-cavity complete-removal case while retaining every mode.
    BBU_FILES["control_all.init"] = BBU_FILES["control.init"].replace("cutoff.bmad", "cutoff_all.bmad")

    ```
