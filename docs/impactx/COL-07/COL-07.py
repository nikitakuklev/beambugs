"""COL-07: ISR omits the particle path through the bend field

With incoherent synchrotron radiation enabled in bends, mean energy loss and diffusion use the reference slice length instead of each particle's path through the magnetic field. A 1.000 m, 100.0 mrad bend with 30.00 mrad face angles has zero relative loss slope with horizontal displacement instead of 0.04003 m⁻¹. At 3.000 GeV and a 0.1000 mm offset, this omits 4.003 ppm of the bend's energy loss, about 0.04565 eV per pass. Refining or explicitly splitting the body preserves the error; the rectangular-face displacement term cancels physically.

Setup: run pinned to one CPU (`taskset -c <n>`); the shared `_slicing` helper refuses to run on more, and child processes inherit that CPU.

Paired cold electron bunches at opposite offsets share the AMReX random stream, resolving their radiation-path response without independent sampling errors. A Cartesian Lorentz integration between physical face planes and its analytic first variation provide independent path lengths, while on-axis normalization, radiation-off and rectangular-face cases supply controls. Fresh native processes cover body refinement, explicit splitting, three bend types, quantum orders and reference loss.
"""

import math

from _slicing import environment, run
from scipy.integrate import solve_ivp


def ode_path(x, edge, tolerance):
    # Independent Cartesian Lorentz orbit, parametrized by physical distance.
    # Uniform vertical field gives da/dl=1/rho and no energy change.
    # Start on the actual entrance face, Z=X*tan(edge); stop on the exit face.
    rho, theta = 10.0, 0.1
    end = (rho * (math.cos(theta) - 1), rho * math.sin(theta))
    normal = (-math.sin(theta - edge), math.cos(theta - edge))

    def rhs(_length, state):
        return [-math.sin(state[2]), math.cos(state[2]), 1 / rho]

    class ExitFace:
        terminal = True
        direction = 1

        def __call__(self, _length, state):
            return float(normal[0] * (state[0] - end[0]) + normal[1] * (state[1] - end[1]))

    exit_face = ExitFace()
    sol = solve_ivp(
        rhs,
        (0, 2),
        [x, x * math.tan(edge), 0],
        method="DOP853",
        events=exit_face,
        rtol=tolerance,
        atol=tolerance / 10,
        max_step=0.025,
    )
    if not sol.success or len(sol.t_events[0]) != 1:
        raise RuntimeError(sol.message)
    return float(sol.t_events[0][0])


def main():
    environment()
    failures = 0
    for edge, body, n, pieces, strang, order, ref_loss, enabled in [
        (0.0, "Sbend", 1, 1, False, 1, False, True),
        (0.03, "Sbend", 1, 1, False, 1, False, True),
        (0.03, "Sbend", 32, 1, False, 1, False, True),
        (0.03, "Sbend", 1, 8, True, 1, False, True),
        (0.03, "ExactSbend", 8, 1, True, 1, False, True),
        (0.03, "CFbend", 8, 1, False, 1, False, True),
        (0.03, "Sbend", 8, 1, False, 2, False, True),
        (0.03, "Sbend", 8, 1, False, 3, True, True),
        (0.05, "Sbend", 8, 1, False, 1, False, True),
        (0.03, "Sbend", 8, 1, False, 1, False, False),
    ]:
        config = dict(
            kind="radiation",
            edge=edge,
            body=body,
            n=n,
            pieces=pieces,
            strang=strang,
            order=order,
            ref_loss=ref_loss,
            enabled=enabled,
            gamma=2000.0,
        )
        pair = [run(dict(config, x=x)) for x in (-0.0005, 0.0005)]
        paths = [ode_path(x, edge, 3e-13) for x in (-0.0005, 0, 0.0005)]
        fine = [ode_path(x, edge, 3e-14) for x in (-0.0005, 0, 0.0005)]
        expected_slope = (fine[2] - fine[0]) / 0.001
        coarse_slope = (paths[2] - paths[0]) / 0.001
        # First variation of a uniform-field orbit between the physical face planes.
        analytic = math.sin(0.1 - 2 * edge) / math.cos(edge) ** 2
        assert abs(expected_slope - coarse_slope) < 2e-10
        assert abs(expected_slope - analytic) < 2e-10
        assert abs(fine[1] - 1) < 2e-14
        identical = pair[0]["pt_sha256"] == pair[1]["pt_sha256"]
        if not enabled:
            assert pair[0]["mean_pt"] == pair[0]["var_pt"] == 0
            assert pair[1]["mean_pt"] == pair[1]["var_pt"] == 0
            mean_slope = var_slope = 0.0
        else:
            # Recover absolute energy loss when the reference loses energy too.
            # pt is relative to the final reference momentum, in initial-P0 units here.
            means = [
                (config["gamma"] - row["ref_gamma"]) / row["bg"]
                + math.sqrt(row["ref_gamma"] ** 2 - 1) / row["bg"] * row["mean_pt"]
                for row in pair
            ]
            mean_slope = (means[1] - means[0]) / (0.001 * sum(means) / 2)
            var_slope = (pair[1]["var_pt"] - pair[0]["var_pt"]) / (
                0.001 * (pair[1]["var_pt"] + pair[0]["var_pt"]) / 2
            )
            if abs(analytic) > 0.01:
                failures += identical and mean_slope == var_slope == 0
            else:
                assert abs(mean_slope) < 2e-10 and abs(var_slope) < 2e-10
        print(
            "RADIATION",
            body,
            "faces",
            edge,
            "slices",
            n,
            "pieces",
            pieces,
            "Strang",
            strang,
            "order",
            order,
            "ref_loss",
            ref_loss,
            "enabled",
            enabled,
            "mean_pt",
            pair[0]["mean_pt"],
            "var_pt",
            pair[0]["var_pt"],
            "paired_arrays_identical",
            identical,
            "mean_var_slopes",
            mean_slope,
            var_slope,
            "expected",
            expected_slope if enabled else 0,
            "analytic",
            analytic if enabled else 0,
            "ODE_refinement_error",
            abs(expected_slope - coarse_slope),
            "oracle=physical_field_path",
        )
    control = run(dict(kind="radiation", x=0.0, edge=0.0, n=1, gamma=2000.0))
    # Classical mean fractional momentum loss tau; convert to pt with beta.
    # Diffusion variance is tau*55*chi/(16*sqrt(3)); pt variance adds beta^2.
    tau = 2 / 3 * 2.8179403227e-15 * control["bg"] ** 3 / 100
    chi = 1.054571817e-34 / (control["mass"] * 299792458) * control["bg"] ** 2 / 10
    expected_mean = control["beta"] * tau
    expected_var = control["beta"] ** 2 * tau * 55 / (16 * math.sqrt(3)) * chi
    zm = (control["mean_pt"] - expected_mean) / math.sqrt(expected_var / 8192)
    zv = (control["var_pt"] - expected_var) / (expected_var * math.sqrt(2 / 8191))
    assert abs(zm) < 6 and abs(zv) < 6
    print(
        "ON_AXIS",
        "mean_pt",
        control["mean_pt"],
        "expected",
        expected_mean,
        "var_pt",
        control["var_pt"],
        "expected",
        expected_var,
        "sampling_z",
        zm,
        zv,
        "oracle=classical_mean/quantum_diffusion",
    )
    gamma = 3000 / 0.51099895
    loss = 2 / 3 * 2.8179403227e-15 * (gamma**2 - 1) ** 1.5 / 100
    relative_error = math.sin(0.04) / math.cos(0.03) ** 2 * 0.0001
    print(
        "PRACTICAL",
        "energy_GeV",
        3.0,
        "offset_mm",
        0.1,
        "expected_fractional_loss",
        loss,
        "missing_relative_loss",
        relative_error,
        "missing_eV",
        3e9 * loss * relative_error,
        "oracle=loss_times_path_variation",
    )
    print("DEFECT REPRODUCED" if failures else "NOT REPRODUCED")


if __name__ == "__main__":
    main()
