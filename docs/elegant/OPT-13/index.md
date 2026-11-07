---
title: "OPT-13: Quick second chromaticity translates a supplied closed orbit twice"
hide:
  - toc
---

# OPT-13 — Quick second chromaticity translates a supplied closed orbit twice

| | |
|---|---|
| Code | [Elegant](../index.md) · Optics |
| Priority | P2 |
| Severity | Wrong physics |
| Evidence | Reproduced, independently rechecked |
| Last reproduced | 2026-11-04 on Elegant master (`0355dd6`, head build) |
| Tags | <span class="bb-tag">partly-source-only</span> [?](../../methodology.md#tags) |
| Source | [`compute_periodic_twiss`, twiss.cc:173–190](https://github.com/rtsoliday/elegant/blob/0355dd6aca12c861971fbf8f3ae78a0ded991734/src/twiss.cc#L173-L190)<br>[`compute_twiss_parameters`, twiss.cc:2289–2293](https://github.com/rtsoliday/elegant/blob/0355dd6aca12c861971fbf8f3ae78a0ded991734/src/twiss.cc#L2289-L2293)<br>[`computeHigherOrderChromaticities`, chrom.c:928–947](https://github.com/rtsoliday/elegant/blob/0355dd6aca12c861971fbf8f3ae78a0ded991734/src/chrom.c#L928-L947)<br>[`accumulate_matrices`, compute_matrices.c:214–294](https://github.com/rtsoliday/elegant/blob/0355dd6aca12c861971fbf8f3ae78a0ded991734/src/compute_matrices.c#L214-L294)<br>[`concat_matrices`, concat_mat.c:31–132](https://github.com/rtsoliday/elegant/blob/0355dd6aca12c861971fbf8f3ae78a0ded991734/src/concat_mat.c#L31-L132)<br>[Twiss parameters on a closed orbit, elegant.tex:8084–8085](https://github.com/rtsoliday/elegant/blob/0355dd6aca12c861971fbf8f3ae78a0ded991734/doc/elegant.tex#L8084-L8085)<br>[quick mode as a quicker way to the second-order chromaticity, elegant.tex:8147–8148](https://github.com/rtsoliday/elegant/blob/0355dd6aca12c861971fbf8f3ae78a0ded991734/doc/elegant.tex#L8147-L8148)<br>[`MATR` map file format, matr.tex:1–20](https://github.com/rtsoliday/elegant/blob/0355dd6aca12c861971fbf8f3ae78a0ded991734/doc/matr.tex#L1-L20)<br>[the `T_MATR` case, compute_matrices.c:1576–1601](https://github.com/rtsoliday/elegant/blob/0355dd6aca12c861971fbf8f3ae78a0ded991734/src/compute_matrices.c#L1576-L1601) |

With `quick_higher_order_chromaticity=1`, Elegant shifts a one-turn map by the closed orbit after it has already been expanded about that orbit, corrupting the second chromaticity d²ν/dδ², where ν is tune and δ is relative momentum offset. For a prescribed cubic map with a closed orbit 10 mm off axis and a momentum-independent Jacobian there, the exact horizontal derivative is 0; quick mode gives −0.08750 and full mode −3.198×10⁻¹⁰. In this map, halving the displacement reduces the error fourfold, while an on-axis orbit or removal of the cubic term leaves only fitting noise.

Run `python OPT-13.py` with [_elegant.py](../helpers/_elegant.py) and [_matr.py](../helpers/_matr.py) in the same directory.

??? example "Reproducer: OPT-13.py"

    [Download OPT-13.py](OPT-13.py)

    ```python
    """OPT-13: Quick second chromaticity translates a supplied closed orbit twice

    With `quick_higher_order_chromaticity=1`, Elegant shifts a one-turn map by the closed orbit after it
    has already been expanded about that orbit, corrupting the second chromaticity d²ν/dδ², where ν is
    tune and δ is relative momentum offset. For a prescribed cubic map with a closed orbit 10 mm off axis
    and a momentum-independent Jacobian there, the exact horizontal derivative is 0; quick mode gives
    −0.08750 and full mode −3.198×10⁻¹⁰. In this map, halving the displacement reduces the error
    fourfold, while an on-axis orbit or removal of the cubic term leaves only fitting noise.

    The script prescribes a stable cubic one-turn map with a fixed point at x = b for every δ and a
    Jacobian there that is the same rotation for every δ, so the tunes do not depend on δ. It runs
    `twiss_output` on the closed orbit in quick and in full mode, and compares the written d²νx/dδ²
    with 0 and, for quick mode, with the value the double translation predicts. Tracked particle pairs
    confirm the fixed point and the Jacobian; full mode, zero displacement and zero cubic term are the
    controls.
    """

    import sys

    import numpy as np
    from _elegant import modes, particles, run, verdict
    from _matr import TUNES, matrix_text, rotations

    P0 = 1000.0  # reference momentum, units of m_e c
    LENGTH = 10.0  # m
    BETA_X = 1.0  # m, horizontal beta function of the rotation A (alpha = 0), in slope coordinates
    DELTAS = (-0.01, 0.0, 0.01)
    PROBE = 1e-5  # half-step of the Jacobian probes
    COPIES = 2  # copies of each particle, so that each MPI worker tracks some
    ATOL = 2e-8  # trace-fit floor of the full calculation is 3.2e-10
    PREDICTION_ATOL = 1e-8

    COMMANDS = """\
    &run_setup lattice=lattice.lte, use_beamline=BL, p_central=1000, default_order=3,
     output=out.sdds &end
    &run_control n_steps=1, n_passes=1 &end
    &twiss_output filename=twiss.sdds, output_at_each_step=1, concat_order=3,
     higher_order_chromaticity=1, higher_order_chromaticity_points=11,
     higher_order_chromaticity_range=0.01, quick_higher_order_chromaticity={quick} &end
    &closed_orbit output=orbit.sdds, output_monitors_only=0, closed_orbit_accuracy=1e-14,
     closed_orbit_accuracy_requirement=1e-12, iteration_fraction=1 &end
    &sdds_beam input=beam.sdds &end
    &track soft_failure=0 &end
    """


    def prescribed_map(b, a):
        """MATR file of F(u, v, delta) = (b, 0) + A (u - b, v + a (u - b)^2 delta) in the x plane.

        A is the rotation by 2 pi nu_x (betatron tune 0.17) in (x, x') with beta_x = BETA_X = 1 m and
        alpha_x = 0, a = 100 1/m^2 gives the kick a (u - b)^2 delta in the slope v, and the y plane is
        the rotation with tune 0.23. Expanding (u - b)^2 = u^2 - 2 b u + b^2 gives the C, R, T and Q coefficients below. At
        u = b the map returns (b, 0) for every delta, and its Jacobian there is exactly A for every
        delta, so the tunes do not depend on delta.
        """
        r = rotations()
        c, s = r[0, 0], r[0, 1]
        constants = np.array([(1 - c) * b, s * b, 0, 0, LENGTH, 0.0])
        t = np.zeros((6, 6, 6))
        q = np.zeros((6, 6, 6, 6))
        for i, factor in ((0, s), (1, c)):  # rows of A acting on the kick a (u - b)^2 delta
            r[i, 5] = a * b * b * factor
            t[i, 5, 0] = -2 * a * b * factor
            q[i, 5, 0, 0] = a * factor
        return matrix_text(constants, r, t, q), rotations()[:2, :2]


    def probe_beam(b):
        """Four particles per delta at +-PROBE in x and in xp about the fixed point (b, 0)."""
        rows = []
        for delta in DELTAS:
            for dx, dxp in ((PROBE, 0), (-PROBE, 0), (0, PROBE), (0, -PROBE)):
                rows.append((b + dx, dxp, P0 * (1 + delta)))
        x, xp, p = np.repeat(np.array(rows), COPIES, axis=0).T
        return {"x": x, "xp": xp, "p": p, "particleID": np.arange(1, len(x) + 1)}


    def case(mode, b, a, quick):
        """(written d2nu_x/ddelta2, closed-orbit x, largest tracked Jacobian error)."""
        text, jacobian = prescribed_map(b, a)
        # Elegant upper-cases names in the lattice, so the map file must have an upper-case name.
        result = run(
            {
                "lattice.lte": "E: MATR,L=10,FILENAME=MAP.TXT,ORDER=3\nBL: LINE=(E)\n",
                "run.ele": COMMANDS.format(quick=quick),
                "MAP.TXT": text,
            },
            mode,
            {"beam.sdds": probe_beam(b)},
            outputs=("twiss.sdds", "orbit.sdds", "out.sdds"),
        )
        orbit = result["orbit.sdds"]
        if int(orbit["failed"][0]):
            sys.exit(f"{mode}: the closed-orbit search failed")
        out = particles(result["out.sdds"])
        if not (np.array_equal(out["x"], np.repeat(out["x"][::COPIES], COPIES))):
            sys.exit("Identical particles left the element differently.")
        # Symmetric differences about the fixed point give the transverse Jacobian at each delta.
        x, xp = (out[name][::COPIES].reshape(len(DELTAS), 4) for name in ("x", "xp"))
        tracked = np.empty((len(DELTAS), 2, 2))
        for row, column in enumerate((x, xp)):
            tracked[:, row, 0] = (column[:, 0] - column[:, 1]) / (2 * PROBE)  # d/dx
            tracked[:, row, 1] = (column[:, 2] - column[:, 3]) / (2 * PROBE)  # d/dxp
        return (
            float(result["twiss.sdds"]["dnux/dp2"][0]),
            float(orbit[0, "x"][0]),
            float(np.max(np.abs(tracked - jacobian))),
        )


    def compare(label, observed, expected, atol):
        """Print one case; return True when observed and expected disagree."""
        if not (np.isfinite(observed) and np.isfinite(expected)):
            sys.exit(f"{label}: nonfinite value, the comparison is not valid.")
        bad = abs(observed - expected) > atol
        print(f"{label}: observed={observed:.16g} expected={expected:.16g}")
        return bad


    defect = False
    for mode in modes():
        for b, a, quick in (
            (0.01, 100.0, 0),  # control: full calculation
            (0.0, 100.0, 1),  # control: closed orbit at the origin
            (0.01, 0.0, 1),  # control: no cubic term
            (0.005, 100.0, 1),  # half the displacement
            (0.01, 100.0, 1),
        ):
            label = f"{mode} b={b:g} a={a:g} {'quick' if quick else 'full'}"
            written, orbit_x, jacobian_error = case(mode, b, a, quick)
            if not np.isfinite([written, orbit_x, jacobian_error]).all():
                sys.exit(f"{label}: nonfinite written derivative, orbit or Jacobian error.")
            # The fixed point (b, 0) and the constant Jacobian are checked by tracking, independently of
            # the Twiss calculation.
            print(
                f"{label}: closed orbit x={orbit_x:.16g}, tracked Jacobian error={jacobian_error:.3g}"
            )
            if abs(orbit_x - b) > 1e-12 or jacobian_error > 3e-11:
                sys.exit("The prescribed map is not realized by tracking: invalid comparison.")
            # Exact answer: the tunes do not depend on delta, so d2nu/ddelta2 = 0. The double
            # translation evaluates the Jacobian at u = 2b. With chi = a beta_x b (the kick in the
            # slope coordinate, normalized by beta_x), trace = 2 cos(mu) + 2 chi sin(mu) delta
            # + O(delta^2) with mu = 2 pi nu_x; differentiating nu = arccos(trace / 2) / (2 pi) twice
            # gives d2nu/ddelta2 = -chi^2 cot(mu) / (2 pi).
            mu = 2 * np.pi * TUNES[0]
            chi = a * BETA_X * b
            shifted = -(chi**2) / np.tan(mu) / (2 * np.pi) if quick else 0.0
            bad = compare(f"{label} d2nu_x/ddelta2", written, 0.0, ATOL)
            if not (quick and b and a):
                if bad:
                    sys.exit("A control has nonzero second chromaticity: invalid comparison.")
            else:
                defect |= bad
                print(f"{label}: double translation predicts {shifted:.16g}")
                # A value that agrees with the exact zero is not this defect; only a nonzero value
                # must follow the double translation.
                if bad and abs(written - shifted) > PREDICTION_ATOL:
                    sys.exit("The written value does not follow the double translation.")
    print(f"comparison: absolute tolerance {ATOL:g}")
    verdict(defect)

    ```

## Provenance

| Role | Model | Effort | Date | Code examined |
|---|---|---|---|---|
| Found | `gpt-6.1-sol` | xhigh | 2026-10-15 | 2026.4.0 (`d177ff5347c7`), source reading, not executed for this claim |
| Confirmed | `gpt-6-astra` | xhigh | 2026-10-15 | 2026.4.0 (`d177ff5347c7`), source with SDDS 3fa650ee9a1c, CPU, serial |
| Confirmed | `gpt-6.1-sol` | xhigh | 2026-10-15 | 2026.4.0 (`d177ff5347c7`), source with SDDS 3fa650ee9a1c, CPU, serial |
| Confirmed | `gpt-6.1-sol` | xhigh | 2026-11-03 | master (`0355dd6aca12`), source with SDDS a3f730a14c49, CPU, serial and 3-rank Pelegant |

