---
title: "TRK-14: POLAR random-direction modes give NaN spins along a longitudinal axis"
hide:
  - toc
---

# TRK-14 — POLAR random-direction modes give NaN spins along a longitudinal axis

| | |
|---|---|
| Code | [Elegant](../index.md) · Tracking |
| Priority | P3 |
| Severity | Wrong physics |
| Evidence | Reproduced, independently rechecked |
| Last reproduced | 2026-11-04 on Elegant commit [`0355dd6`](https://github.com/rtsoliday/elegant/commit/0355dd6aca12), committed 2026-11-03 |
| Tags | <span class="bb-tag">partly-source-only</span> [?](../../methodology.md#tags) |
| Source | [`polarizeBeam`, generate_bunch.c:1367–1508](https://github.com/rtsoliday/elegant/blob/0355dd6aca12c861971fbf8f3ae78a0ded991734/src/generate_bunch.c#L1367-L1508)<br>[axis normalization and two-state routing, generate_bunch.c:1381–1391](https://github.com/rtsoliday/elegant/blob/0355dd6aca12c861971fbf8f3ae78a0ded991734/src/generate_bunch.c#L1381-L1391)<br>[narrow-cone rotation, generate_bunch.c:1451–1453](https://github.com/rtsoliday/elegant/blob/0355dd6aca12c861971fbf8f3ae78a0ded991734/src/generate_bunch.c#L1451-L1453)<br>[wide-cone rotation, generate_bunch.c:1472–1474](https://github.com/rtsoliday/elegant/blob/0355dd6aca12c861971fbf8f3ae78a0ded991734/src/generate_bunch.c#L1472-L1474)<br>[von Mises–Fisher rotation, generate_bunch.c:1491–1493](https://github.com/rtsoliday/elegant/blob/0355dd6aca12c861971fbf8f3ae78a0ded991734/src/generate_bunch.c#L1491-L1493)<br>[`performQuaternionRotation`, spinUpdate.c:127–163](https://github.com/rtsoliday/elegant/blob/0355dd6aca12c861971fbf8f3ae78a0ded991734/src/spinUpdate.c#L127-L163)<br>[POLAR manual, polar.tex:1–19](https://github.com/rtsoliday/elegant/blob/0355dd6aca12c861971fbf8f3ae78a0ded991734/doc/polar.tex#L1-L19)<br>[POLAR dispatch, do_tracking.c:2001–2004](https://github.com/rtsoliday/elegant/blob/0355dd6aca12c861971fbf8f3ae78a0ded991734/src/do_tracking.c#L2001-L2004) |

With spin tracking enabled, POLAR's narrow-cone, wide-cone and von Mises–Fisher modes return NaN (not-a-number) spin components for every tested particle when the polarization axis is (0, 0, ±1). At requested polarizations P = 0.25 and 0.8, all 12 spins become nonfinite without warning; the tilted axis (0.6, 0, 0.8) gives 12 finite unit spins. Rotations should preserve unit spin length, and a narrow cone should satisfy S·v = P, where S is the spin and v the unit axis. The rotation these modes share is singular on that axis, so by source inspection the failure extends to every 0 < P < 1.

Run `python TRK-14.py` with [_elegant.py](../helpers/_elegant.py) in the same directory.

??? example "Reproducer: TRK-14.py"

    [Download TRK-14.py](TRK-14.py)

    ```python
    """TRK-14: POLAR random-direction modes give NaN spins along a longitudinal axis

    With spin tracking enabled, POLAR's narrow-cone, wide-cone and von Mises–Fisher modes return NaN
    (not-a-number) spin components for every tested particle when the polarization axis is (0, 0, ±1).
    At requested polarizations P = 0.25 and 0.8, all 12 spins become nonfinite without warning; the
    tilted axis (0.6, 0, 0.8) gives 12 finite unit spins. Rotations should preserve unit spin length,
    and a narrow cone should satisfy S·v = P, where S is the spin and v the unit axis. The rotation
    these modes share is singular on that axis, so by source inspection the failure extends to every
    0 < P < 1.

    The script tracks 12 identical electrons with spin tracking enabled through a POLAR element for each
    mode, P of 0.25 and 0.8, and the axes (0, 0, 1) and (0, 0, -1), and counts the particles whose three
    spin components are finite and of unit length; a narrow cone must also satisfy S·v = P. The controls
    are the tilted axis (0.6, 0, 0.8) in every mode, and the longitudinal axis with the two-state mode,
    with P = 0 and with P = 1, which bypass the cone rotation.
    """

    import sys

    import numpy as np
    from _elegant import modes, particles, run, verdict

    N = 12  # particles, enough for every MPI worker to hold some
    P0 = 30.0  # reference momentum, units of m_e c; POLAR does not change the orbit
    TOL = 5e-15  # spin norm and S.v - P; a quaternion rotation is exact to rounding


    def fmt(values):
        return "[" + ", ".join(f"{v:.16g}" for v in np.atleast_1d(values)) + "]"


    def polarize(mode, kind, polarization, axis):
        """Spins of N particles after a POLAR element; initial spins are (0, 0, 1)."""
        vx, vy, vz = axis
        lattice = (
            f'P: POLAR,POLARIZATION={polarization!r},VX={vx!r},VY={vy!r},VZ={vz!r},MODE="{kind}"\n'
            "BL: LINE=(P)\n"
        )
        commands = (
            f"&run_setup lattice=lattice.lte, use_beamline=BL, p_central={P0!r}, output=out.sdds,"
            " spin_tracking=1, random_number_seed=64783 &end\n"
            "&run_control n_steps=1, n_passes=1 &end\n"
            "&sdds_beam input=beam.sdds &end\n"
            "&track &end\n"
        )
        beam = {
            "p": np.full(N, P0),
            "spx": np.zeros(N),
            "spy": np.zeros(N),
            "spz": np.ones(N),
        }
        out = particles(
            run({"lattice.lte": lattice, "run.ele": commands}, mode, {"beam.sdds": beam})["out.sdds"]
        )
        if len(out["p"]) != N:
            sys.exit(f"{mode}: {N - len(out['p'])} particles lost")
        return np.column_stack([out["spx"], out["spy"], out["spz"]])


    def check(label, spins, axis, polarization, cone):
        """Print finite count and spin norm deviation (and S.v - P for a narrow cone).

        Expected: every spin finite (12 of 12), |S| = 1, and for a narrow cone S.v = P exactly,
        because the sampled polar angle satisfies cos(theta) = P and the rotation to the axis keeps it.
        Returns False when the invariants hold and True when every component of every spin is NaN,
        the recorded defect; any other state is a failed run, not a verdict.
        """
        finite = int(np.count_nonzero(np.all(np.isfinite(spins), axis=1)))
        norm_deviation = np.max(np.abs(np.linalg.norm(spins, axis=1) - 1))
        observed, expected = [finite, norm_deviation], [N, 0.0]
        if cone:
            v = np.asarray(axis) / np.linalg.norm(axis)
            observed.append(np.max(np.abs(spins @ v - polarization)))
            expected.append(0.0)
        print(f"{label}: observed={fmt(observed)} expected={fmt(expected)}")
        if finite == N and np.all(np.abs(np.array(observed[1:])) <= TOL):
            return False
        if np.all(np.isnan(spins)):
            return True
        sys.exit(f"{label}: the spins are neither valid nor all NaN")


    defect = []
    print("columns: finite spins, max | |S| - 1 |" + ", max |S.v - P| (narrow cone)")
    for mode in modes():
        for kind in ("narrowcone", "widecone", "vonmises-fisher"):
            for polarization in (0.25, 0.8):
                for axis in ((0.0, 0.0, 1.0), (0.0, 0.0, -1.0), (0.6, 0.0, 0.8)):
                    label = f"{mode} {kind} P={polarization} axis={axis}"
                    spins = polarize(mode, kind, polarization, axis)
                    bad = check(label, spins, axis, polarization, kind == "narrowcone")
                    if axis[0]:
                        # The tilted axis has VX != 0, so the rotation is regular.
                        if bad:
                            sys.exit("The tilted-axis control fails: the comparison is not valid.")
                    else:
                        defect.append(bad)
        # Longitudinal axis, but the cone rotation is bypassed: P = 0 or P = 1 select the two-state
        # branch, and mode "twostate" never rotates.
        for kind, polarization in (("twostate", 0.5), ("narrowcone", 0.0), ("widecone", 1.0)):
            spins = polarize(mode, kind, polarization, (0.0, 0.0, 1.0))
            label = f"{mode} control {kind} P={polarization} axis=(0.0, 0.0, 1.0)"
            if check(label, spins, (0.0, 0.0, 1.0), polarization, False):
                sys.exit("A longitudinal control fails: the comparison is not valid.")
    print(f"comparison: unit norm and S.v = P to {TOL:g}; finite count")
    verdict(any(defect))

    ```

## Provenance

| Role | Model | Effort | Date | Code examined |
|---|---|---|---|---|
| Found | `gpt-6.1-sol` | high | 2026-10-25 | master (`75a2e1a1a58b`), source reading, not executed for this claim |
| Confirmed | `gpt-6-astra` | xhigh | 2026-11-03 | master (`0355dd6aca12`), source with SDDS a3f730a14c49, CPU, serial and 3-rank Pelegant |

