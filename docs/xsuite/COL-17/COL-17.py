"""COL-17: All-lost tracking calls freeze wake history

`WakeTracker.track` returns early when no particle is alive, before advancing the stored wake-source
history, so a turn in which the whole bunch is lost does not age the history. A turn without sources
must still age it, as calls with live particles outside the slicing window or with zero weight do,
and direct convolution gives the expected delays and expiry. With a three-turn history in a 20 m
ring and a 1.4 GeV/c proton source of weight 5, a witness one empty turn later receives `px` = 4.024
× 10⁻⁹ instead of 2.441 × 10⁻⁹, and two empty turns later 4.024 × 10⁻⁹ instead of 0, after the
source should have expired. Users who keep calling a wake tracker on fully lost bunches are
affected.

The script uses four slices over [−1, 1] m and a causal transverse wake W(d) = 10¹⁹·exp(d/40) for d
< −5 m and zero otherwise, so no kick arises within the same turn. A source at ζ = 0.75 m is
followed by zero, one or two explicitly numbered empty turns and then a witness at ζ = −0.75 m,
whose kick is compared with direct convolution. The empty turns consist of lost particles, live
particles outside the slicing window, or live particles of zero weight; the last two are controls.
It requires the installed prebuilt kernels (compilation is forbidden).
"""

from typing import ClassVar

import xfields as xf
from _prebuilt import CTX, identity, kernels, np, xt
from scipy.constants import e

identity()


# Delayed causal monopole transverse wake: no self-wake within this bunch.
class DelayedWake:
    source_exponents = (0, 0)
    test_exponents = (0, 0)
    source_moments: ClassVar[list] = ["num_particles"]
    _source_moment_exponents: ClassVar[list] = [1]
    plane = "x"
    kick = "px"

    def function_vs_zeta(self, zeta, beta0, dzeta):
        return np.where(zeta < -5, 1e19 * np.exp(zeta / 40), 0)


def make():
    return xf.WakeTracker(
        components=[DelayedWake()],
        num_turns=3,
        circumference=20,
        zeta_range=(-1, 1),
        num_slices=4,
        _context=CTX,
    )


def bunch(zeta, turn, state=1, weight=1):
    return xt.Particles(
        p0c=1.4e9, zeta=zeta, at_turn=turn, state=state, weight=weight, _context=CTX
    )


def populations(w):
    return [
        float(np.sum(w.moments_data.get_source_moment_profile("num_particles", i, 0)[1]))
        for i in range(3)
    ]


for gaps in (0, 1, 2):
    observations = {}
    for empty in ("lost", "outside", "zero_weight"):
        wake = make()
        wake.track(bunch(0.75, 0, weight=5))
        for t in range(1, gaps + 1):
            wake.track(
                bunch(
                    4 if empty == "outside" else 0.75,
                    t,
                    state=0 if empty == "lost" else 1,
                    weight=0 if empty == "zero_weight" else 1,
                )
            )
        before = populations(wake)
        test = bunch(-0.75, gaps + 1)
        wake.track(test)
        got = float(test.px[0])
        expected = (
            (5e19 * np.exp((-1.5 - (gaps + 1) * 20) / 40) * e / (1.4e9 * float(test.beta0[0])))
            if gaps + 1 < 3
            else 0.0
        )
        observations[empty] = got
        print(
            "gap",
            gaps,
            empty,
            "history_before_witness",
            before,
            "kick",
            f"{got:.17g}",
            "expected",
            f"{expected:.17g}",
        )
        if empty != "lost" or gaps == 0:
            np.testing.assert_allclose(got, expected, rtol=6e-15, atol=1e-27)
        else:
            assert abs(got - expected) > 1e-11
    np.testing.assert_allclose(
        observations["outside"], observations["zero_weight"], rtol=0, atol=1e-27
    )

kernels()
print("DEFECT REPRODUCED")
