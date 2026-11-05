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

Download these shared helpers into the same directory as the reproducer scripts:

- [Download _elegant.py](../helpers/_elegant.py) · [View source](../helpers/index.md#_elegant)
- [Download _sole.py](../helpers/_sole.py) · [View source](../helpers/index.md#_sole)

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

## Provenance

| Role | Model | Effort | Date | Code examined |
|---|---|---|---|---|
| Found | `gpt-6-astra` | xhigh | 2026-11-04 | master (`0355dd6aca12`), source with SDDS a3f730a14c49, CPU, serial |
| Confirmed | `gpt-6.1-sol` | xhigh | 2026-11-04 | master (`0355dd6aca12`), source with SDDS a3f730a14c49, CPU, serial |

