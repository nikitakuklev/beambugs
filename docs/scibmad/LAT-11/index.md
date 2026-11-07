---
title: "LAT-11: Aperture kernels overwrite the state of particles that are not alive"
hide:
  - toc
---

# LAT-11 — Aperture kernels overwrite the state of particles that are not alive

| | |
|---|---|
| Code | [SciBmad](../index.md) · Lattice and API |
| Priority | P3 |
| Severity | Silent wrong result |
| Evidence | Probe-pinned |
| Last reproduced | 2026-10-31 on BeamTracking main (`0bf3585`, head build) |
| Tags | <span class="bb-tag">partly-source-only</span> [?](../../methodology.md#tags) |
| Source | [`BeamTracking.jl/src/kernels/aperture.jl:4-7`](https://github.com/bmad-sim/BeamTracking.jl/blob/0bf3585fdc9c5b41859b23b92854376b933a4fe6/src/kernels/aperture.jl#L4-L7)<br>[`BeamTracking.jl/src/kernels/aperture.jl:21-25`](https://github.com/bmad-sim/BeamTracking.jl/blob/0bf3585fdc9c5b41859b23b92854376b933a4fe6/src/kernels/aperture.jl#L21-L25)<br>[`SciBmad.jl/docs/src/track.md:72-73`](https://github.com/bmad-sim/SciBmad.jl/blob/ae9423d703b4e612ae66e9deb0546aa79f62a340/docs/src/track.md#L72-L73)<br>[`BeamTracking.jl/src/types.jl:15-18`](https://github.com/bmad-sim/BeamTracking.jl/blob/0bf3585fdc9c5b41859b23b92854376b933a4fe6/src/types.jl#L15-L18)<br>[`BeamTracking.jl/src/kernels/drift_kick.jl:66-67`](https://github.com/bmad-sim/BeamTracking.jl/blob/0bf3585fdc9c5b41859b23b92854376b933a4fe6/src/kernels/drift_kick.jl#L66-L67)<br>[`BeamTracking.jl/test/RungeKuttaTracking_test.jl:331-337`](https://github.com/bmad-sim/BeamTracking.jl/blob/0bf3585fdc9c5b41859b23b92854376b933a4fe6/test/RungeKuttaTracking_test.jl#L331-L337) |

Rectangular aperture tracking overwrites an already-lost particle's loss-side code when its frozen coordinates lie outside a later aperture. A particle at x = y = 0.2 m first lost at an `x2_limit = 0.1` m aperture has state 4 (`STATE_LOST_POS_X`), but a later `y2_limit = 0.1` m aperture changes it to state 6 (`STATE_LOST_POS_Y`). The state should remain 4: the drift kernel keeps an already-set loss state, and BeamTracking's Runge–Kutta tests assert that a lost particle keeps its state. Source inspection shows the same missing live-particle check in the elliptical aperture kernel. Loss-side diagnostics are silently corrupted; surviving particles are unaffected.

Run `julia LAT-11.jl`.

??? example "Reproducer: LAT-11.jl"

    [Download LAT-11.jl](LAT-11.jl)

    ```julia
    # LAT-11: Aperture kernels overwrite the state of particles that are not alive
    #
    # Rectangular aperture tracking overwrites an already-lost particle's loss-side code when its frozen
    # coordinates lie outside a later aperture. A particle at x = y = 0.2 m first lost at an `x2_limit =
    # 0.1` m aperture has state 4 (`STATE_LOST_POS_X`), but a later `y2_limit = 0.1` m aperture changes
    # it to state 6 (`STATE_LOST_POS_Y`). The state should remain 4: the drift kernel keeps an
    # already-set loss state, and BeamTracking's Runge–Kutta tests assert that a lost particle keeps its
    # state. Source inspection shows the same missing live-particle check in the elliptical aperture
    # kernel. Loss-side diagnostics are silently corrupted; surviving particles are unaffected.
    #
    # The script tracks two particles at x = y = 0.2 m through a rectangular x aperture (`x2_limit =
    # 0.1` m), a 1 m drift and a rectangular y aperture (`y2_limit = 0.1` m), recording the states. The
    # first particle must keep `STATE_LOST_POS_X`. The second starts as `STATE_PREBORN` and shows that
    # preborn particles are also given loss codes; no contract for preborn states is documented, so that
    # case is only printed.
    using SciBmad
    const BT = SciBmad.BeamTracking
    sp = Species("electron")
    first_ap = Marker(x1_limit=-1.0, x2_limit=0.1, y1_limit=-1.0, y2_limit=1.0, aperture_shape=ApertureShape.Rectangular)
    second_ap = Marker(x1_limit=-1.0, x2_limit=1.0, y1_limit=-1.0, y2_limit=0.1, aperture_shape=ApertureShape.Rectangular)
    bl = Beamline([first_ap, Drift(L=1.0), second_ap], species_ref=sp, E_ref=1e9)
    b = Bunch(v=[0.2 0.0 0.2 0.0 0.0 0.0; 0.2 0.0 0.2 0.0 0.0 0.0], species=sp, p_over_q_ref=bl.p_over_q_ref)
    b.state[2] = BT.STATE_PREBORN
    v0 = copy(b.v)
    # The aperture kernels assign loss codes without testing STATE_ALIVE, unlike e.g. the exact drift kernel.
    track!(b, bl.line[1]); s1 = copy(b.state)
    track!(b, bl.line[2]); track!(b, bl.line[3]); s3 = copy(b.state)
    println("particle 1: state after first aperture = ", s1[1], ", after second = ", s3[1], ", coordinates frozen = ", b.v[1, :] == v0[1, :])
    println("particle 2 (STATE_PREBORN = 0x00 at start): state after first aperture = ", s1[2], ", after second = ", s3[2])
    println("expected: particle 1 keeps STATE_LOST_POS_X = 0x04 (lost first at x2_limit, never reached the second aperture)")
    println("(particle 2 shows that a preborn particle is also given a loss code; no contract for preborn states is documented)")
    println(s3[1] != BT.STATE_LOST_POS_X ? "DEFECT REPRODUCED" : "NOT REPRODUCED")

    ```

## Provenance

| Role | Model | Effort | Date | Code examined |
|---|---|---|---|---|
| Found | `gpt-6-sol` | xhigh | 2026-10-22 | BeamTracking 0.8.7 (`65e343f9afda`), SciBmad 0.6.0 release packages, Julia 1.13.2, CPU |
| Confirmed | `gpt-6-astra` | xhigh | 2026-10-22 | BeamTracking 0.8.7 (`65e343f9afda`), SciBmad 0.6.0 release packages, Julia 1.13.2, CPU |
| Confirmed | `claude-opus-5-5` | high | 2026-10-22 | BeamTracking 0.8.7 (`65e343f9afda`), SciBmad 0.6.0 release packages, Julia 1.13.2, CPU |

