---
title: "TRK-61: Canceled dipole strength creates NaN bend transport and matrices"
hide:
  - toc
---

# TRK-61 — Canceled dipole strength creates NaN bend transport and matrices

| | |
|---|---|
| Code | [Elegant](../index.md) · Tracking |
| Priority | P3 |
| Severity | Wrong physics |
| Evidence | Reproduced, independently rechecked |
| Last reproduced | 2026-11-06 on Elegant commit [`0355dd6`](https://github.com/rtsoliday/elegant/commit/0355dd6aca12), committed 2026-11-03 |
| Tags | <span class="bb-tag">partly-source-only</span> [?](../../methodology.md#tags) |
| Source | [src/csbend.c:1077–1081](https://github.com/rtsoliday/elegant/blob/0355dd6aca12c861971fbf8f3ae78a0ded991734/src/csbend.c#L1077-L1081)<br>[src/csbend.c:1111–1116](https://github.com/rtsoliday/elegant/blob/0355dd6aca12c861971fbf8f3ae78a0ded991734/src/csbend.c#L1111-L1116)<br>[src/csbend.c:1192–1199](https://github.com/rtsoliday/elegant/blob/0355dd6aca12c861971fbf8f3ae78a0ded991734/src/csbend.c#L1192-L1199)<br>[src/csbend.c:1493–1503](https://github.com/rtsoliday/elegant/blob/0355dd6aca12c861971fbf8f3ae78a0ded991734/src/csbend.c#L1493-L1503)<br>[src/csbend.c:1019–1022](https://github.com/rtsoliday/elegant/blob/0355dd6aca12c861971fbf8f3ae78a0ded991734/src/csbend.c#L1019-L1022)<br>[src/bend_matrix.c:82–88](https://github.com/rtsoliday/elegant/blob/0355dd6aca12c861971fbf8f3ae78a0ded991734/src/bend_matrix.c#L82-L88)<br>[src/bend_matrix.c:629–630](https://github.com/rtsoliday/elegant/blob/0355dd6aca12c861971fbf8f3ae78a0ded991734/src/bend_matrix.c#L629-L630)<br>[src/compute_matrices.c:1450–1470](https://github.com/rtsoliday/elegant/blob/0355dd6aca12c861971fbf8f3ae78a0ded991734/src/compute_matrices.c#L1450-L1470)<br>[src/compute_matrices.c:1798–1832](https://github.com/rtsoliday/elegant/blob/0355dd6aca12c861971fbf8f3ae78a0ded991734/src/compute_matrices.c#L1798-L1832)<br>[src/compute_matrices.c:1878–1896](https://github.com/rtsoliday/elegant/blob/0355dd6aca12c861971fbf8f3ae78a0ded991734/src/compute_matrices.c#L1878-L1896)<br>[src/csbend.c:2794–2825](https://github.com/rtsoliday/elegant/blob/0355dd6aca12c861971fbf8f3ae78a0ded991734/src/csbend.c#L2794-L2825)<br>[src/csbend.c:2847–2852](https://github.com/rtsoliday/elegant/blob/0355dd6aca12c861971fbf8f3ae78a0ded991734/src/csbend.c#L2847-L2852)<br>[src/multipole.c:788–796](https://github.com/rtsoliday/elegant/blob/0355dd6aca12c861971fbf8f3ae78a0ded991734/src/multipole.c#L788-L796)<br>[doc/elegant.tex:11973–11974](https://github.com/rtsoliday/elegant/blob/0355dd6aca12c861971fbf8f3ae78a0ded991734/doc/elegant.tex#L11973-L11974)<br>[src/track_data.c:1961–1963](https://github.com/rtsoliday/elegant/blob/0355dd6aca12c861971fbf8f3ae78a0ded991734/src/track_data.c#L1961-L1963)<br>[doc/csbend.tex:121–133](https://github.com/rtsoliday/elegant/blob/0355dd6aca12c861971fbf8f3ae78a0ded991734/doc/csbend.tex#L121-L133)<br>[doc/csbend.tex:1–11](https://github.com/rtsoliday/elegant/blob/0355dd6aca12c861971fbf8f3ae78a0ded991734/doc/csbend.tex#L1-L11) |

Canceling a CSBEND’s physical dipole field with FSE = −1 or FSE + FSE_DIPOLE = −1 produces NaN transverse coordinates and loses the beam, including a pure dipole with edges disabled. A 0.4000 m bend with reference angle 0.02000 rad at momentum 3.066 GeV/c loses all five rays; its central ray should emerge at x = 4.001 mm with slope x′ = 0.02000. Independent Cartesian ray intersection and Lorentz integration give finite transport. SBEN/RBEN tracking and analytic CSBEND/CSRCSBEND matrices have the same zero-denominator failure. Native CSRCSBEND tracking with radiation and CSR disabled remains finite.

Run `python TRK-61.py` with [_elegant.py](../helpers/_elegant.py) in the same directory.

??? example "Reproducer: TRK-61.py"

    [Download TRK-61.py](TRK-61.py)

    ```python
    """TRK-61: Canceled dipole strength creates NaN bend transport and matrices

    Canceling a CSBEND’s physical dipole field with FSE = −1 or FSE + FSE_DIPOLE = −1 produces NaN transverse coordinates and loses the beam, including a pure dipole with edges disabled. A 0.4000 m bend with reference angle 0.02000 rad at momentum 3.066 GeV/c loses all five rays; its central ray should emerge at x = 4.001 mm with slope x′ = 0.02000. Independent Cartesian ray intersection and Lorentz integration give finite transport. SBEN/RBEN tracking and analytic CSBEND/CSRCSBEND matrices have the same zero-denominator failure. Native CSRCSBEND tracking with radiation and CSR disabled remains finite.

    Track five fixed rays through canceled-field bends and compare with extended-precision Cartesian ray intersection and an independent Lorentz integration. Exercise finite-field controls, integration and edge options, SBEN/RBEN and analytic matrices, with passing CSRCSBEND and KQUAD tracking controls. --mpi adds sequential three-rank CSBEND and CSRCSBEND checks.
    """

    from __future__ import annotations

    import os

    for _name in (
        "OMP_NUM_THREADS",
        "MKL_NUM_THREADS",
        "OPENBLAS_NUM_THREADS",
        "NUMEXPR_NUM_THREADS",
        "NPROC",
        "PJRT_NPROC",
    ):
        os.environ[_name] = "1"

    import _elegant
    import numpy as np
    from scipy.integrate import solve_ivp

    _elegant.TIMEOUT = 290  # ty: ignore[invalid-assignment]  # Per-native-process deadline.
    C = 299792458.0
    NAMES = ("x", "xp", "y", "yp", "t", "p")
    RAYS = np.array(
        [
            [0, 0, 0, 0, 0, 1],
            [0.0007, -0.0013, 0.0004, 0.0008, 1e-12, 0.98],
            [-0.0011, 0.0021, -0.0009, -0.0012, -2e-12, 1.02],
            [0.0017, 0.0006, -0.0012, 0.0016, 3e-12, 1.01],
            [-0.0004, -0.0017, 0.0013, -0.0005, -1e-12, 0.99],
        ]
    )


    def frame(length: float, angle: float, dtype=np.longdouble):
        """Survey bends toward lab -x; local x points away from curvature center."""
        a, ell = dtype(angle), dtype(length)
        sn, cs = np.sin(a), np.cos(a)
        # The half-angle identity also avoids subtracting nearly equal cosines.
        centre = np.array(
            [
                -2 * ell / a * np.sin(a / 2) ** 2 if a else 0,
                0,
                ell / a * sn if a else ell,
            ],
            dtype=dtype,
        )
        normal = np.array([-sn, 0, cs], dtype=dtype)
        ex = np.array([cs, 0, sn], dtype=dtype)
        return centre, normal, ex


    def straight_ray(row, length: float, angle: float):
        r, normal, ex = frame(length, angle)
        x, xp, y, yp, t, p = map(np.longdouble, row)
        initial = np.array([x, y, 0], dtype=np.longdouble)
        direction = np.array([xp, yp, 1], dtype=np.longdouble)
        direction /= np.sqrt(direction @ direction)
        distance = ((r - initial) @ normal) / (direction @ normal)
        assert distance > 0
        end = initial + distance * direction
        # Slopes are momentum ratios in the survey frame, not canonical momenta.
        beta = p / np.sqrt(1 + p * p)
        return np.array(
            [
                (end - r) @ ex,
                (direction @ ex) / (direction @ normal),
                end[1],
                direction[1] / (direction @ normal),
                t * C + distance / beta,
                p,
            ],
            dtype=float,
        )


    def lorentz_ode(row, length: float, angle: float, p0: float, field: float):
        r, normal, ex = frame(length, angle, float)
        x, xp, y, yp, t, p = row
        u = np.array([xp, yp, 1.0])
        u /= np.linalg.norm(u)
        curvature = angle / length * field * p0 / p

        def rhs(_ell, state):
            ux, uy, uz = state[3:]
            return [ux, uy, uz, -curvature * uz, 0.0, curvature * ux]

        class ExitPlane:
            terminal = True
            direction = 1

            def __call__(self, _ell, state):
                return normal @ (state[:3] - r)

        result = solve_ivp(
            rhs,
            (0, 2 * length),
            [x, y, 0, *u],
            events=ExitPlane(),
            method="DOP853",
            rtol=3e-14,
            atol=2e-15,
            max_step=length / 8,
        )
        if not result.success or len(result.t_events[0]) != 1:
            raise RuntimeError("Lorentz integration did not reach the exit plane")
        end = result.y_events[0][0]
        distance = result.t_events[0][0]
        u = end[3:]
        beta = p / np.sqrt(1 + p * p)
        return np.array(
            [
                (end[:3] - r) @ ex,
                (u @ ex) / (u @ normal),
                end[1],
                u[1] / (u @ normal),
                t * C + distance / beta,
                p,
            ]
        )


    def csbend(fse=-1.0, dipole=0.0, angle=0.02, order=4, edges=(0, 0), extra=""):
        return (
            f"CSBEND,L=.4,ANGLE={angle:.17g},FSE={fse:.17g},"
            f"FSE_DIPOLE={dipole:.17g},INTEGRATION_ORDER={order},N_SLICES=64,"
            f"EDGE1_EFFECTS={edges[0]},EDGE2_EFFECTS={edges[1]},SYNCH_RAD=0,ISR=0,"
            "REFERENCE_CORRECTION=0,FSE_CORRECTION=0" + extra
        )


    def read_rows(data, *, lost=False):
        if data.n_pages != 1:
            raise RuntimeError("Expected exactly one native output page")
        names = ("x", "xp", "y", "yp", "s", "p") if lost else NAMES
        rows = np.array([data[0, name] for name in names], dtype=float).T
        ids = np.asarray(data[0, "particleID"], dtype=int)
        order = np.argsort(ids)
        rows, ids = rows[order], ids[order]
        if not lost:
            rows[:, 4] *= C
            if not np.isfinite(rows).all():
                raise RuntimeError("Nonfinite survivor coordinates")
        return rows, ids


    def native(
        label,
        element,
        *,
        angle=0.02,
        p0=6000.0,
        field=0.0,
        matrix=False,
        loss_case=False,
        mode="serial",
        map_tracking=False,
    ):
        rays = RAYS.copy()
        rays[:, 5] *= p0
        expected = np.array([straight_ray(row, 0.4, angle) for row in rays])
        ode = np.array([lorentz_ode(row, 0.4, angle, p0, field) for row in rays])
        oracle_error = float(np.max(abs(ode - expected))) if field == 0 else 0.0
        if field == 0 and oracle_error > 5e-13:
            raise RuntimeError("Cartesian intersection and Lorentz integration disagree")
        if field != 0:
            expected = ode
        commands = (
            f"&run_setup lattice=lattice.lte,use_beamline=BL,p_central={p0:.17g},"
            "output=out.sdds,losses=loss.sdds,default_order=2,random_number_seed=271828 &end\n"
            "&run_control n_steps=1,n_passes=1 &end\n"
        )
        if matrix:
            commands += "&matrix_output SDDS_output=matrix.sdds,SDDS_output_order=1 &end\n"
        commands += "&sdds_beam input=beam.sdds &end\n&track soft_failure=0 &end\n"
        result = _elegant.run(
            {"lattice.lte": f"B: {element}\nBL: LINE=(B)\n", "run.ele": commands},
            mode=mode,
            beams={"beam.sdds": dict(zip(NAMES, rays.T, strict=True))},
            outputs=("out.sdds", "loss.sdds", "matrix.sdds") if matrix else ("out.sdds", "loss.sdds"),
        )
        out, ids = read_rows(result["out.sdds"])
        loss, lost_ids = read_rows(result["loss.sdds"], lost=True)
        if not np.array_equal(np.sort(np.r_[ids, lost_ids]), np.arange(1, 6)):
            raise RuntimeError("Native output does not account for each fixed input ray exactly once")
        nan_rows = int(np.count_nonzero(np.isnan(loss[:, :4]).all(axis=1)))
        hit = False
        if loss_case and len(out) == 0 and len(loss) == 5 and nan_rows == 5:
            hit = True
        elif len(out) != 5 or len(loss) != 0:
            raise RuntimeError("Unexpected loss pattern")
        error = float(np.max(abs(out - expected))) if len(out) == 5 else None
        if error is not None and not map_tracking and error > 2e-12:
            raise RuntimeError(f"Finite-transport comparison failed: {error}")
        if not loss_case and hit:
            raise RuntimeError("Control lost particles")
        if not np.isfinite(loss[:, 5]).all():
            raise RuntimeError("Invalid lost-particle momentum")
        matrix_text = ""
        if matrix:
            data = result["matrix.sdds"]
            if data.n_pages != 1:
                raise RuntimeError("Expected one matrix page")
            c1 = float(data[0, "C1"][-1])
            r11 = float(data[0, "R11"][-1])
            matrix_hit = np.isnan(c1) and np.isnan(r11)
            if not matrix_hit and not np.isfinite([c1, r11]).all():
                raise RuntimeError("Unexpected invalid matrix pattern")
            hit |= bool(matrix_hit)
            matrix_text = f"; C1={c1:.9g} R11={r11:.9g} expected=finite (ray-plane Jacobian)"
        detail = (
            f"max_coordinate_error={error:.3e} (m, slope, ct in m, mc)"
            if error is not None
            else f"lost_all_four_transverse_NaN={nan_rows}"
        )
        method = "Lorentz DOP853" if field else "Cartesian ray-plane intersection"
        print(
            f"{mode}/{label}: survivors={len(out)} expected=5; {detail}; "
            f"expected_central_x={expected[0, 0]:.17g} m xp={expected[0, 1]:.17g} "
            f"ct={expected[0, 4]:.17g} m ({method}); oracle_delta={oracle_error:.3e}" + matrix_text,
            flush=True,
        )
        return hit


    def csr():
        return (
            "CSRCSBEND,L=.4,ANGLE=.02,FSE=-1,CSR=0,ISR=0,SYNCH_RAD=0,"
            "N_SLICES=64,BINS=128,SG_HALFWIDTH=1,INTEGRATION_ORDER=4,"
            "EDGE1_EFFECTS=0,EDGE2_EFFECTS=0"
        )


    def main():
        modes = _elegant.modes()
        hits = [native("csbend_zero", csbend(), loss_case=True)]
        for fse in (0.0, -0.9, -1 + 1e-9, -1 - 1e-9):
            hits.append(native(f"csbend_control_FSE={fse:.12g}", csbend(fse), field=1 + fse))
        hits.append(native("csrcsbend_zero", csr()))
        hits.append(
            native(
                "kquad_zero",
                "KQUAD,L=.4,K1=1.7,FSE=-1,N_SLICES=16,"
                "INTEGRATION_ORDER=4,EDGE1_EFFECTS=0,EDGE2_EFFECTS=0",
                angle=0,
            )
        )
        for kind in ("SBEN", "RBEN"):
            hits.append(
                native(
                    kind,
                    f"{kind},L=.4,ANGLE=.02,FSE=-1,EDGE1_EFFECTS=0,EDGE2_EFFECTS=0",
                    matrix=True,
                    loss_case=True,
                    map_tracking=True,
                )
            )
        for order in (2, 6):
            hits.append(native(f"csbend_order={order}", csbend(order=order), loss_case=True))
        for edge in (1, 3, 4):
            hits.append(
                native(
                    f"csbend_edges={edge}",
                    csbend(edges=(edge, edge), extra=",E1=.01,E2=.01,HGAP=.01"),
                    loss_case=True,
                )
            )
        for label, extra in (
            ("linear_field", ",NONLINEAR=0"),
            ("expanded", ",EXPAND_HAMILTONIAN=1"),
            ("use_bn", ",USE_BN=1,B1=0,B2=0"),
            ("finite_higher_off", ",K1=1.7,K2=8,K3=20"),
            ("sector_fringe", ",HGAP=.01"),
        ):
            hits.append(native(label, csbend(extra=extra), loss_case=True))
        for label, fse, dipole in (("dipole_only_off", 0, -1), ("split_off", -0.125, -0.875)):
            hits.append(native(label, csbend(fse, dipole), loss_case=True))
        hits.append(native("negative_angle", csbend(angle=-0.02), angle=-0.02, loss_case=True))
        hits.append(native("finite_speed", csbend(angle=0.1), angle=0.1, p0=12, loss_case=True))
        hits.append(native("zero_angle", csbend(angle=0), angle=0))
        # This is a dispatch control: the existing large-radius branch substitutes a straight drift.
        hits.append(native("large_radius_straight_dispatch", csbend(angle=1e-8), angle=0))
        hits.append(
            native(
                "csbend_analytic_matrix",
                csbend(extra=",TRACKING_MATRIX=0"),
                matrix=True,
                loss_case=True,
            )
        )
        hits.append(native("csrcsbend_analytic_matrix", csr(), matrix=True))
        if "mpi" in modes:
            hits.append(native("csbend_zero", csbend(), loss_case=True, mode="mpi"))
            hits.append(native("csrcsbend_zero", csr(), mode="mpi"))
        _elegant.verdict(any(hits))


    if __name__ == "__main__":
        main()

    ```

## Provenance

| Role | Model | Effort | Date | Code examined |
|---|---|---|---|---|
| Found | `gpt-6-astra` | xhigh | 2026-11-06 | master (`0355dd6aca12`), source with SDDS a3f730a14c49, CPU, double, serial, one thread |
| Confirmed | `gpt-6.1-sol` | xhigh | 2026-11-06 | master (`0355dd6aca12`), source with SDDS a3f730a14c49, CPU, double, serial and 3-rank Pelegant, one thread |

