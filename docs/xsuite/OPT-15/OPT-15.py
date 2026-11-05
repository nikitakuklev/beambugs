"""OPT-15: Closest tune approach is inflated across an integer boundary

Twiss inflates `c_minus`, the closest-tune-approach coefficient, when two nearby eigentunes straddle
an integer. An analytic coupled-oscillator calculation and an independent eigenfrequency scan show
that shifting both tunes together must preserve their minimum separation. For eigentunes 0.9920 and
1.006, mixing angle 0.07 rad and modal beta function 2.7 m, `c_minus` is 0.1376 instead of 0.001954,
too large by a factor 70.43; shifting both tunes by ±0.09 restores the correct value. The error
appears in the 4D and 6D periodic calculations with positive line length and pure difference
coupling, including the complex coupling magnitudes; tracking maps, eigentunes and projected beta
functions remain correct.

The script builds rings of 24 or 48 rotated `LineSegmentMap` elements (modal beta 2.7 m, 0.8 GeV/c
protons) with pure difference coupling at mixing angles θ = 0.07, 0.31 and 0.53 rad, shifts both
tunes by −0.09, 0 and +0.09, and runs 4D and 6D Twiss. The exact closest tune approach, the
eigentune gap times |sin 2θ|, is confirmed by a 141-point eigenvalue scan and compared with
`c_minus`; tracked maps, tunes and projected beta functions are checked as well. Cases that do not
straddle an integer are the controls.
"""

import json

from _prebuilt import CTX, identity, kernels, np, xt

identity()


def rotation(tune, beta):
    c, s = np.cos(2 * np.pi * tune), np.sin(2 * np.pi * tune)
    return np.array([[c, beta * s], [-s / beta, c]])


results = []
for center, gap, theta in ((0.999, 0.014, 0.07), (1.004, 0.038, 0.31), (2.002, 0.022, 0.53)):
    mix = np.array([[np.cos(theta), np.sin(theta)], [-np.sin(theta), np.cos(theta)]])
    h = mix.T @ np.diag([center - gap / 2, center + gap / 2]) @ mix
    k = h[0, 1]
    detunings = np.linspace(-0.07, 0.07, 141)
    gaps = [
        np.diff(np.linalg.eigvalsh([[center + d / 2, k], [k, center - d / 2]]))[0]
        for d in detunings
    ]
    # For H=[[a,k],[k,b]], the eigentune gap is sqrt((a-b)^2+4*k^2).
    # Its minimum at fixed k is 2*abs(k), invariant under common tune shifts.
    expected = float(2 * abs(k))
    assert abs(min(gaps) - expected) < 2e-15
    for shift in (-0.09, 0.0, 0.09):
        q1, q2 = center - gap / 2 + shift, center + gap / 2 + shift
        for nseg in (24, 48):
            line = xt.Line(
                elements=[
                    xt.LineSegmentMap(
                        length=2.0,
                        qx=q1 / nseg,
                        qy=q2 / nseg,
                        betx=2.7,
                        bety=2.7,
                        qs=0.073 / nseg,
                        bets=11.0,
                        rot_s_rad=theta,
                    )
                    for _ in range(nseg)
                ]
            )
            line.particle_ref = xt.Particles(_context=CTX, p0c=0.8e9)
            line.build_tracker(_context=CTX)
            # Track all transverse Cartesian basis directions with both signs.
            rays = np.vstack([np.eye(4), -np.eye(4)]) * 1e-5
            p = xt.Particles(
                _context=CTX, p0c=0.8e9, x=rays[:, 0], px=rays[:, 1], y=rays[:, 2], py=rays[:, 3]
            )
            line.track(p)
            assert np.all(p.state == 1)
            diag = np.zeros((4, 4))
            diag[:2, :2] = rotation(q1, 2.7)
            diag[2:, 2:] = rotation(q2, 2.7)
            m = np.kron(mix.T, np.eye(2)) @ diag @ np.kron(mix, np.eye(2))
            actual_rays = np.array([p.x, p.px, p.y, p.py]).T
            track_error = float(np.max(np.abs(actual_rays - rays @ m.T)))
            assert track_error < 1e-17
            for method in ("4d", "6d"):
                for el in line.elements:
                    el.qs = 0.0 if method == "4d" else 0.073 / nseg
                tw = line.twiss(method=method, chrom=False)
                assert max(abs(tw.qx - q1), abs(tw.qy - q2)) < 1e-12
                beta_error = max(
                    abs(tw.betx[0] - 2.7 * np.cos(theta) ** 2),
                    abs(tw.betx2[0] - 2.7 * np.sin(theta) ** 2),
                )
                assert beta_error < 1e-11
                straddles = int(np.floor(q1)) != int(np.floor(q2))
                ratio = tw.c_minus / expected
                assert (
                    np.max(np.abs(np.abs(tw.c_minus_re + 1j * tw.c_minus_im) - tw.c_minus)) < 1e-12
                )
                assert np.max(np.abs(np.abs(tw.c_minus_local) - tw.c_minus)) < 1e-12
                if straddles:
                    # c_minus uses |frac(qx) - frac(qy)|, which turns a gap across an integer
                    # into 1 - gap and so inflates c_minus by (1 - gap)/gap.
                    assert abs(ratio - (1 - gap) / gap) < 2e-9
                else:
                    assert abs(tw.c_minus - expected) < 1e-12
                rec = dict(
                    center=center,
                    gap=gap,
                    theta=theta,
                    shift=shift,
                    nseg=nseg,
                    method=method,
                    qx=tw.qx,
                    qy=tw.qy,
                    expected=expected,
                    actual=tw.c_minus,
                    ratio=ratio,
                    eigenscan_min=float(min(gaps)),
                    track_error=track_error,
                    beta_error=float(beta_error),
                    straddles=straddles,
                )
                results.append(rec)
                print("COUPLING", json.dumps(rec))
                if nseg == 24 and method == "6d":
                    print(
                        f"TUNES q1={q1:g} q2={q2:g} expected={expected:.17g} "
                        f"actual={tw.c_minus:.17g} ratio={ratio:.12g}"
                    )
kernels()
print("DEFECT REPRODUCED")
