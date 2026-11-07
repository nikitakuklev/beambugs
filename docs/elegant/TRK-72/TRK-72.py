"""TRK-72: CSRCSBEND aborts on internal obstruction checks and global aperture-loss conversion

A nonzero-angle CSRCSBEND using symplectic integration aborts when active obstruction data trigger an internal global-coordinate check, even when every obstruction is far from the beam and global loss output is disabled. Both bend signs fail because the shared converter has no CSRCSBEND case. A 0.6000 m bend with 512 particles and 32 bins exits with an unsupported-global-coordinates error instead of retaining all particles; an independent circular-orbit calculation establishes zero obstruction hits. Internal aperture losses also abort when global loss coordinates are requested; zero-angle and LINEARIZE=1 controls complete normally. Three-rank tracking prints the error and hangs in collective error handling.

Track 512 fixed particles through both bend signs with a distant obstruction, controls and internal MAXAMP losses. Compare unobstructed trajectories with exact circle intersections, and require the specific unsupported-conversion diagnostic for failed cases. With --mpi, check a three-rank control and terminate the failing process group after 290 seconds, requiring that diagnostic as well as the hang.
"""

import os
import signal
import subprocess
import tempfile
from pathlib import Path

from _diagnostics import run

# isort: split
import _elegant as native
import numpy as np
import pysdds

native.TIMEOUT = 290  # ty: ignore[invalid-assignment]  # Per-native-process deadline.
N = 512
P0 = 1500.0
LENGTH = 0.6
FATAL = "Error: at present, can't provide global loss coordinates for CSRCSBEND element"


def fixture(
    theta,
    *,
    obstruction=True,
    disabled=False,
    global_coords=False,
    linear=False,
    aperture=False,
    csr=False,
):
    prefix = "A:MAXAMP,X_MAX=0.0002,Y_MAX=0.1\n" if aperture else ""
    if csr:
        prefix += "C:CHARGE,TOTAL=1e-12\n"
    lattice = prefix + (
        f"B:CSRCSBEND,L={LENGTH},ANGLE={theta},N_SLICES=16,INTEGRATION_ORDER=6,"
        f"CSR={int(csr)},BINS=32,STEADY_STATE=1,LINEARIZE={int(linear)},"
        "SYNCH_RAD=0,ISR=0,EDGE1_EFFECTS=0,EDGE2_EFFECTS=0\n"
        f"BL:LINE=({'A,' if aperture else ''}{'C,' if csr else ''}B)\n"
    )
    commands = (
        "&run_setup lattice=line.lte,use_beamline=BL,output=out.sdds,losses=lost.sdds,"
        f"p_central={P0},losses_include_global_coordinates={int(global_coords)},"
        "random_number_seed=71643 &end\n&floor_coordinates filename=floor.sdds &end\n"
    )
    if obstruction:
        commands += f"&obstruction_data input=obs.sdds,disable={int(disabled)} &end\n"
    commands += (
        "&run_control n_steps=1,n_passes=1 &end\n"
        "&sdds_beam input=beam.sdds &end\n&track soft_failure=0 &end\n"
    )
    files = {"line.lte": lattice, "run.ele": commands}
    if obstruction:
        # The no-go rectangle is 10 m away; the entire 0.6 m orbit is disjoint from it.
        files["obs.sdds"] = (
            "SDDS1\n&parameter name=ZCenter,type=double,units=m,fixed_value=0 &end\n"
            "&parameter name=XCenter,type=double,units=m,fixed_value=0 &end\n"
            "&parameter name=Superperiodicity,type=long,fixed_value=1 &end\n"
            "&column name=Z,type=double,units=m &end\n"
            "&column name=X,type=double,units=m &end\n"
            "&data mode=ascii &end\n5\n10 10\n11 10\n11 11\n10 11\n10 10\n"
        )
    beam = {
        "x": np.zeros(N) if aperture else np.linspace(-1e-4, 1e-4, N),
        "xp": np.full(N, 0.001 if aperture else 0),
        "t": np.linspace(-2e-12, 2e-12, N),
        "p": np.full(N, P0),
        "particleID": np.arange(1, N + 1),
    }
    return files, beam


def serial(theta, **options):
    files, beam = fixture(theta, **options)
    try:
        result = run(files, "serial", {"beam.sdds": beam}, ("out.sdds",))
    except native.RunFailure as exc:
        if exc.status != 1 or FATAL not in exc.log:
            raise
        return None
    return native.particles(result["out.sdds"])


def survivors(data, expected):
    if data is None or not np.array_equal(data["particleID"], expected):
        raise RuntimeError("Unexpected survivor IDs")


def mpi(obstruction):
    files, beam = fixture(-0.04, obstruction=obstruction)
    with tempfile.TemporaryDirectory(prefix="csrcsbend-mpi-") as tmp:
        work = Path(tmp)
        for name, text in files.items():
            (work / name).write_text(text)
        native.write_beam(work / "beam.sdds", beam)
        proc = subprocess.Popen(
            [
                "mpirun",
                "--host",
                "localhost:1",
                "--oversubscribe",
                "--bind-to",
                "none",
                "-n",
                "3",
                "Pelegant",
                "run.ele",
            ],
            cwd=work,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            start_new_session=True,
        )
        timeout = False
        try:
            stdout, stderr = proc.communicate(timeout=290)
        except subprocess.TimeoutExpired:
            timeout = True
            # Killing the group also stops MPI workers; a timeout alone is never evidence.
            os.killpg(proc.pid, signal.SIGTERM)
            stdout, stderr = proc.communicate(timeout=290)
        log = stdout + stderr
        if timeout:
            if not obstruction or FATAL not in log:
                raise RuntimeError(
                    f"MPI timeout without the required conversion diagnostic:\n{log}"
                )
            print(
                "MPI obstruction: diagnostic=CSRCSBEND unsupported-global-coordinates; timeout=290 s; expected=512 survivors and normal completion"
            )
            return True
        if proc.returncode:
            raise native.RunFailure("Pelegant", proc.returncode, log)
        survivors(native.particles(pysdds.read(work / "out.sdds")), beam["particleID"])
        print(f"MPI obstruction={int(obstruction)}: survivors=512 expected=512 (distant rectangle)")
        return False


def main():
    modes = native.modes()
    faults = []
    for theta in (0.04, -0.04):
        control = serial(theta, obstruction=False)
        survivors(control, np.arange(1, N + 1))
        # Exact circle intersected with the exit radial plane. Cancellation-free 1-cos(a).
        x0 = np.linspace(-1e-4, 1e-4, N)
        k = theta / LENGTH
        a = np.arcsin(k * x0 * np.sin(theta))
        expected = np.column_stack(
            [
                x0 * np.cos(theta) - 2 * np.sin(a / 2) ** 2 / k,
                -np.tan(a),
                np.zeros((N, 2)),
            ]
        )
        observed = np.column_stack([control[n] for n in ("x", "xp", "y", "yp")])
        np.testing.assert_allclose(observed, expected, rtol=0, atol=2e-13)
        error = np.max(np.abs(observed - expected))
        print(
            f"circle angle={theta:+.2f}: survivors=512 expected=512; max coordinate residual={error:.3e} (exact radial-plane intersection); obstruction hits=0"
        )
        for glob in (False, True):
            out = serial(theta, global_coords=glob)
            faults.append(out is None)
            if out is not None:
                survivors(out, np.arange(1, N + 1))
            print(
                f"obstruction angle={theta:+.2f} global={int(glob)}: {'exit=1, diagnostic=CSRCSBEND unsupported-global-coordinates' if out is None else 'survivors=512'}; expected=512 survivors (distant rectangle)"
            )
        for label, angle, options in (
            ("disabled", theta, {"disabled": True}),
            ("linear", theta, {"linear": True}),
            ("zero angle", 0.0, {}),
        ):
            survivors(serial(angle, **options), np.arange(1, N + 1))
            print(
                f"control {label} angle={angle:+.2f}: survivors=512 expected=512 (distant rectangle)"
            )
        for glob in (False, True):
            out = serial(theta, obstruction=False, aperture=True, global_coords=glob)
            if glob:
                faults.append(out is None)
            if out is not None:
                survivors(out, np.array([], dtype=int))
            elif not glob:
                raise RuntimeError("Local-aperture control failed")
            print(
                f"MAXAMP angle={theta:+.2f} global={int(glob)}: {'exit=1, diagnostic=CSRCSBEND unsupported-global-coordinates' if out is None else 'survivors=0'}; expected=0 survivors and normal completion (1 mrad ray crosses 0.2 mm wall)"
            )
    charged = serial(0.04, csr=True)
    faults.append(charged is None)
    if charged is not None:
        survivors(charged, np.arange(1, N + 1))
    print(
        f"CSR=1, charge=1 pC: {'exit=1, diagnostic=CSRCSBEND unsupported-global-coordinates' if charged is None else 'survivors=512'}; expected=512 survivors (distant rectangle)"
    )
    if "mpi" in modes:
        mpi(False)
        faults.append(mpi(True))
    native.verdict(any(faults))


if __name__ == "__main__":
    main()
