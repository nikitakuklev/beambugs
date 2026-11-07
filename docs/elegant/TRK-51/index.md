---
title: "TRK-51: DQCOR tracking applies steering calibration twice"
hide:
  - toc
---

# TRK-51 — DQCOR tracking applies steering calibration twice

| | |
|---|---|
| Code | [Elegant](../index.md) · Tracking |
| Priority | P3 |
| Severity | Wrong physics |
| Evidence | Reproduced, independently rechecked |
| Last reproduced | 2026-11-06 on Elegant master (`0355dd6`, committed 2026-11-03) |
| Tags | <span class="bb-tag">partly-source-only</span> [?](../../methodology.md#tags) |
| Source | [src/compute_matrices.c:60–72](https://github.com/rtsoliday/elegant/blob/0355dd6aca12c861971fbf8f3ae78a0ded991734/src/compute_matrices.c#L60-L72)<br>[src/multipole.c:1007–1071](https://github.com/rtsoliday/elegant/blob/0355dd6aca12c861971fbf8f3ae78a0ded991734/src/multipole.c#L1007-L1071)<br>[src/quad_matrix.c:1429–1456](https://github.com/rtsoliday/elegant/blob/0355dd6aca12c861971fbf8f3ae78a0ded991734/src/quad_matrix.c#L1429-L1456)<br>[src/compute_matrices.c:1687–1701](https://github.com/rtsoliday/elegant/blob/0355dd6aca12c861971fbf8f3ae78a0ded991734/src/compute_matrices.c#L1687-L1701)<br>[src/gpu/gpu_stub.c:11232–11259](https://github.com/rtsoliday/elegant/blob/0355dd6aca12c861971fbf8f3ae78a0ded991734/src/gpu/gpu_stub.c#L11232-L11259)<br>[src/gpu/gpu_cuda_runtime.cu:1890–1891](https://github.com/rtsoliday/elegant/blob/0355dd6aca12c861971fbf8f3ae78a0ded991734/src/gpu/gpu_cuda_runtime.cu#L1890-L1891)<br>[src/track_data.c:4095–4128](https://github.com/rtsoliday/elegant/blob/0355dd6aca12c861971fbf8f3ae78a0ded991734/src/track_data.c#L4095-L4128)<br>[src/track_data.c:4161–4161](https://github.com/rtsoliday/elegant/blob/0355dd6aca12c861971fbf8f3ae78a0ded991734/src/track_data.c#L4161-L4161)<br>[src/track_data.c:4313–4313](https://github.com/rtsoliday/elegant/blob/0355dd6aca12c861971fbf8f3ae78a0ded991734/src/track_data.c#L4313-L4313) |

Tracking a DQCOR with nonunit calibration applies the factors before rotating the steering vector and again in the quadrupole frame. A zero-gradient element with a 230.0 µrad horizontal kick and calibration 1.080 gives horizontal momentum divided by reference momentum of 2.683×10⁻⁴ instead of 2.484×10⁻⁴, an 8.000% excess. With skew gradient, unequal factors can also create cross-plane steering: an input with zero vertical kick acquires a 17.39 µrad-equivalent vertical impulse. Analytic matrices apply calibration once, while tracking-derived matrices inherit the wrong orbit. Independent constant-field impulse calculations and numerical integration establish the discrepancy in serial and MPI tracking.

Run `python TRK-51.py` with [_elegant.py](../helpers/_elegant.py) and [_slicing.py](../helpers/_slicing.py) in the same directory.

??? example "Reproducer: TRK-51.py"

    [Download TRK-51.py](TRK-51.py)

    ```python
    """TRK-51: DQCOR tracking applies steering calibration twice

    Tracking a DQCOR with nonunit calibration applies the factors before rotating the steering vector and again in the quadrupole frame. A zero-gradient element with a 230.0 µrad horizontal kick and calibration 1.080 gives horizontal momentum divided by reference momentum of 2.683×10⁻⁴ instead of 2.484×10⁻⁴, an 8.000% excess. With skew gradient, unequal factors can also create cross-plane steering: an input with zero vertical kick acquires a 17.39 µrad-equivalent vertical impulse. Analytic matrices apply calibration once, while tracking-derived matrices inherit the wrong orbit. Independent constant-field impulse calculations and numerical integration establish the discrepancy in serial and MPI tracking.

    The script compares tracked impulses with a once-calibrated constant-field oracle and integrates forced normal/skew quadrupoles independently. It checks analytic and tracking-derived affine maps, calibrated-equivalent elements and fixed off-momentum rays; --mpi also runs three-rank Pelegant.
    """

    import numpy as np
    from _elegant import modes, verdict
    from _slicing import canonical, close, discrepancy, track
    from scipy.integrate import solve_ivp
    from scipy.linalg import expm

    P0 = 300.0
    LENGTH = 0.42
    KICKS = np.array([2.3e-4, -1.4e-4])
    CAL = np.array([1.08, 0.94])


    def run(
        label,
        *,
        kicks=KICKS,
        cal=CAL,
        k1=0.0,
        j1=0.0,
        order=4,
        slices=1,
        tracking_matrix=0,
        rays=None,
        mode="serial",
        reverse=False,
        extra="",
        kind="DQCOR",
    ):
        if rays is None:
            rays = np.zeros((8, 6))
        params = dict(
            L=LENGTH,
            K1=k1,
            HKICK=kicks[0],
            VKICK=kicks[1],
            HCALIBRATION=cal[0],
            VCALIBRATION=cal[1],
            N_SLICES=slices,
            INTEGRATION_ORDER=order,
            SYNCH_RAD=0,
            ISR=0,
        )
        if kind == "DQCOR":
            params.update(J1=j1, TRACKING_MATRIX=tracking_matrix)
        elif kind == "KSEXT":
            params.update(K2=0)
        lattice = f"Q: {kind}," + ",".join(f"{k}={v:.17g}" for k, v in params.items())
        lattice += extra + "\nBL: LINE=(Q)\n"
        return track(
            label, lattice, rays, p0=P0, reverse=reverse, matrix=True, mode=mode, default_order=3
        )


    def ode(k1, j1, target, steps):
        # Independently solve the driven normal/skew quadrupole. The linear on-axis
        # response to a constant force fixes that force from the requested exit kick.
        # This uses a matrix exponential, not elegant's trig steering helper.
        gradient = np.array([[-k1, -j1], [-j1, k1]])
        augmented = np.zeros((6, 6))
        augmented[:2, 2:4] = np.eye(2)
        augmented[2:4, :2] = gradient
        augmented[2:4, 4:6] = np.eye(2)
        response = expm(LENGTH * augmented)
        force = np.linalg.solve(response[2:4, 4:6], target)

        def rhs(_s, state):
            position, q = state[:2], state[2:]
            return np.concatenate((q / np.sqrt(1 - q @ q), gradient @ position + force))

        result = solve_ivp(
            rhs,
            (0, LENGTH),
            np.zeros(4),
            method="DOP853",
            rtol=3e-14,
            atol=1e-17,
            max_step=LENGTH / steps,
        )
        if not result.success:
            raise RuntimeError(result.message)
        return result.y[:, -1]


    def phase_space(result):
        return np.column_stack((result["q"][:, [0, 2]], canonical(result["q"])))


    def main():
        failures = []
        wanted = KICKS * CAL
        doubled = wanted * CAL
        for mode in modes():
            for order in (2, 4, 6):
                for slices in (1, 16):
                    result = run("zero gradient", order=order, slices=slices, mode=mode)
                    impulse = canonical(result["q"])
                    failures.append(discrepancy(impulse, wanted, doubled, 3e-16))
                    # The analytic zero-gradient model treats its inputs as angles;
                    # keep its cubic angle/momentum distinction at this tolerance.
                    tangent = np.tan(wanted)
                    analytic = tangent / np.sqrt(1 + tangent @ tangent)
                    close(result["C"][[1, 3]], analytic, 3e-16, "Once-calibrated analytic matrix")
                    print(
                        f"{mode} order={order} slices={slices}: impulse={impulse[0].tolist()} "
                        f"expected={wanted.tolist()} (D*k); analytic C2={result['C'][1]:.16g}"
                    )
            control = run(
                "absorbed calibration", kicks=wanted, cal=np.ones(2), order=6, slices=16, mode=mode
            )
            close(canonical(control["q"]), wanted, 3e-16, "Constant-force impulse")
            # Exact integral of x'=q/sqrt(1-q^2), q(s)=s*D*k/L.
            position = LENGTH * wanted / (1 + np.sqrt(1 - wanted @ wanted))
            error = close(control["q"][:, [0, 2]], position, 3e-16, "Constant-force position")
            print(
                f"{mode} absorbed calibration: impulse={canonical(control['q'])[0].tolist()} "
                f"expected={wanted.tolist()} (D*k); position error={error:.4g}"
            )
            for label, cal, kicks in (
                ("unit", [1.0, 1.0], KICKS),
                ("zero calibration", [0.0, 0.0], KICKS),
                ("horizontal only", [1.08, 1.0], KICKS),
                ("vertical only", [1.0, 0.94], KICKS),
                ("negative", [-1.0, 1.0], KICKS),
                ("one percent", [1.01, 0.99], KICKS),
                ("no steering", CAL, np.zeros(2)),
            ):
                cal = np.asarray(cal)
                result = run(label, cal=cal, kicks=kicks, mode=mode)
                failures.append(discrepancy(canonical(result["q"]), kicks * cal, kicks * cal**2, 3e-16))
                print(
                    f"{mode} {label}: impulse={canonical(result['q'])[0].tolist()} "
                    f"expected={(kicks * cal).tolist()} (D*k)"
                )
            rays = np.zeros((8, 6))
            rays[:, 0] = np.linspace(-1e-3, 2e-3, 8)
            rays[:, 1] = np.linspace(-4e-4, 3e-4, 8)
            rays[:, 2] = -rays[:, 0] / 2
            rays[:, 3] = rays[::-1, 1] / 3
            rays[:, 5] = np.linspace(-0.03, 0.04, 8)
            for label, kwargs in (
                ("off momentum", {}),
                ("backward", {"reverse": True}),
                ("steering flags", {"extra": ",HSTEERING=1,VSTEERING=1"}),
            ):
                result = run(label, rays=rays, mode=mode, **kwargs)
                impulse = canonical(result["q"]) - canonical(rays)
                # Reverse steering has its own sign defect. Magnitudes isolate the
                # second calibration independently of that traversal-sign error.
                if label == "backward":
                    impulse = np.abs(impulse)
                    expected, faulty = np.abs(wanted), np.abs(doubled)
                    quantity = "impulse magnitude"
                else:
                    expected, faulty = wanted, doubled
                    quantity = "impulse"
                failures.append(discrepancy(impulse, expected, faulty, 3e-16))
                print(
                    f"{mode} {label} first ray: {quantity}={impulse[0].tolist()} "
                    f"expected={expected.tolist()} (once-calibrated force; all eight rays checked)"
                )
            for kind in ("KQUAD", "KSEXT"):
                result = run(kind, kind=kind, mode=mode)
                close(canonical(result["q"]), wanted, 3e-16, kind + " calibration control")
            for matrix_order in (1, 2, 3):
                result = run("tracking matrix", tracking_matrix=matrix_order, mode=mode)
                close(result["C"][[1, 3]], result["q"][0, [1, 3]], 3e-16, "Tracking affine orbit")
                expected = wanted / np.sqrt(1 - wanted @ wanted)
                faulty = doubled / np.sqrt(1 - doubled @ doubled)
                failures.append(discrepancy(result["C"][[1, 3]], expected, faulty, 3e-16))
                print(
                    f"{mode} tracking_matrix={matrix_order}: affine slopes={result['C'][[1, 3]].tolist()} "
                    f"expected={expected.tolist()} (once-calibrated canonical orbit)"
                )
            for label, k1, j1, kicks, cal in (
                ("normal", 0.7, 0.0, KICKS, CAL),
                ("negative normal", -0.7, 0.0, KICKS, CAL),
                ("skew", 0.0, 0.9, KICKS, CAL),
                ("combined", 0.7, 0.9, KICKS, CAL),
                ("skew equal calibration", 0.0, 0.9, KICKS, np.array([1.08, 1.08])),
                ("skew horizontal only", 0.0, 0.9, np.array([2.3e-4, 0.0]), CAL),
                ("skew zero horizontal calibration", 0.0, 0.9, KICKS, np.array([0.0, 1.0])),
            ):
                phi = np.arctan2(j1, k1) / 2
                rotation = np.array([[np.cos(phi), np.sin(phi)], [-np.sin(phi), np.cos(phi)]])
                target = kicks * cal
                # The second calibration acts in the rotated quadrupole frame.
                wrong_target = rotation.T @ (cal * (rotation @ target))
                expected = ode(k1, j1, target, 16)
                close(expected, ode(k1, j1, target, 32), 3e-16, "ODE refinement")
                wrong = ode(k1, j1, wrong_target, 32)
                result = run(label, k1=k1, j1=j1, kicks=kicks, cal=cal, order=6, slices=32, mode=mode)
                control = run(
                    label, k1=k1, j1=j1, kicks=target, cal=np.ones(2), order=6, slices=32, mode=mode
                )
                error = close(phase_space(control), expected, 1e-16, "ODE control")
                failures.append(discrepancy(phase_space(result), expected, wrong, 1e-16))
                close(result["C"][[1, 3]], target, 3e-16, "Finite-gradient analytic affine")
                print(
                    f"{mode} {label}: impulse={canonical(result['q'])[0].tolist()} "
                    f"expected={expected[2:].tolist()} (independent DOP853); control error={error:.4g}"
                )
            print(
                f"{mode} controls: analytic matrices, unit/zero calibration, zero steering, "
                "KQUAD/KSEXT and independent force integrals pass"
            )
        verdict(any(failures))


    if __name__ == "__main__":
        main()

    ```

## Provenance

| Role | Model | Effort | Date | Code examined |
|---|---|---|---|---|
| Found | `gpt-6.1-sol` | xhigh | 2026-11-06 | master (`0355dd6aca12`), source with SDDS a3f730a14c49, CPU, double, serial, one thread |
| Confirmed | `gpt-6-astra` | xhigh | 2026-11-06 | master (`0355dd6aca12`), source with SDDS a3f730a14c49, CPU, double, serial and 3-rank MPI, one thread |

