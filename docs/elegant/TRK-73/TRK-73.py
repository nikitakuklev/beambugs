"""TRK-73: CCBEND ignores DX_DY_SIGN, so the documented legacy DX and DY signs at negative angle cannot be restored

For a negative-angle CCBEND, DX_DY_SIGN=−1 does not restore the documented earlier signs of DX and DY: the parameter is accepted but never read during tracking. With default fitting, a 0.6000 m, −0.04000 rad bend with quadrupole gradient K₁=0.2000 m⁻² and DY=1.000 mm sends an on-axis ray to y=−35.35 µm and y′=−118.5 µrad instead of +35.35 µm and +118.5 µrad. Reversing the supplied offset establishes the legacy result independently. Serial and three-rank tracking agree across all five alignment methods. The default positive setting retains the consistent displacement convention.

The script tracks three fixed rays through one 0.6 m, ±0.04 rad CCBEND with K1 = 0.2 m^-2 and a 1 mm DX or DY, for MALIGN_METHOD 0 to 4 with fitting and edge kicks off, and once with the element's default fitting and edges. The documented legacy convention reverses DX and DY for ANGLE < 0, so DX_DY_SIGN = -1 must reproduce the default setting with the offset negated; for MALIGN_METHOD 4 the expectation is also computed by the shared rigid-face Lorentz integration with the magnet displaced by the negated offset in laboratory axes. Positive-angle and zero-offset runs are controls, and --mpi repeats the fixed-fit cases in three-rank Pelegant.
"""

import _elegant as native
import numpy as np
from _ccbend import ANGLE, DEFAULT, K1, RAYS, Magnet, close, error, track
from _elegant import verdict

OFFSET = 1e-3  # DX or DY, m
SIGNS = (1, 0, -1)  # DX_DY_SIGN values; 1 is the default
ORACLE_TOL = 1e-15  # native method-4 transport against the independent integration
DEFECT_MIN = 1e-9  # a discrepancy this large is not rounding
AXES = (("DX", 0), ("DY", 1))  # parameter and index into the laboratory offset vector


def run(mode, angle, axis, offset, sign, **params):
    return track(RAYS, mode=mode, ANGLE=angle, **{axis: offset, "DX_DY_SIGN": sign}, **params)[0]


def main():
    modes = native.modes()
    failures = []
    legacy_runs = {}
    for mode in modes:
        for method in range(5):
            for axis, index in AXES:
                fixture = dict(MALIGN_METHOD=method)
                # Legacy convention (dictionary and release note): for ANGLE < 0 the sign of DX and
                # DY is reversed, so F_{-1}(D) = F_{+1}(-D); for ANGLE > 0 F_{-1}(D) = F_{+1}(D).
                negative = {s: run(mode, -ANGLE, axis, OFFSET, s, **fixture) for s in SIGNS}
                expected = run(mode, -ANGLE, axis, -OFFSET, 1, **fixture)
                legacy = error(negative[-1], expected)
                switch = max(error(negative[s], negative[1]) for s in (0, -1))
                positive = close(
                    run(mode, ANGLE, axis, OFFSET, -1, **fixture),
                    run(mode, ANGLE, axis, OFFSET, 1, **fixture),
                    0.0, f"{mode} method {method} {axis} positive-angle control",
                )  # fmt: skip
                failures.append(legacy > DEFECT_MIN)
                legacy_runs[mode, method, axis] = negative[-1]
                print(
                    f"{mode} MALIGN_METHOD={method} {axis}=1e-3 ANGLE=-0.04 DX_DY_SIGN=-1: max error "
                    f"vs DX_DY_SIGN=+1 with {axis}=-1e-3={legacy!r} expected=0 (legacy sign "
                    f"reversal); DX_DY_SIGN=0,-1 vs +1={switch:.4g}; control ANGLE=+0.04 "
                    f"DX_DY_SIGN=-1 vs +1={positive:.4g} expected=0"
                )
                if method == 4 and mode == "serial":
                    # Independent expectation: the magnet displaced rigidly in laboratory axes.
                    d = np.zeros(3)
                    d[index] = OFFSET
                    modern = close(
                        negative[1], Magnet(angle=-ANGLE, k1=K1, d=d).forward(RAYS), ORACLE_TOL,
                        f"method 4 {axis} DX_DY_SIGN=+1 against the integration",
                    )  # fmt: skip
                    oracle = error(negative[-1], Magnet(angle=-ANGLE, k1=K1, d=-d).forward(RAYS))
                    failures.append(oracle > DEFECT_MIN)
                    print(
                        f"serial MALIGN_METHOD=4 {axis}=1e-3 ANGLE=-0.04 DX_DY_SIGN=-1: max error vs "
                        f"laboratory integration with {axis}=-1e-3={oracle!r} expected=0; control "
                        f"DX_DY_SIGN=+1 vs integration with {axis}=+1e-3={modern:.4g}"
                    )
        zero = close(
            run(mode, -ANGLE, "DX", 0.0, -1, MALIGN_METHOD=4),
            run(mode, -ANGLE, "DX", 0.0, 1, MALIGN_METHOD=4),
            0.0, f"{mode} zero-offset control",
        )  # fmt: skip
        print(f"control {mode} ANGLE=-0.04 zero offset: DX_DY_SIGN=-1 vs +1={zero:.4g} expected=0")
    if "mpi" in modes:
        spread = max(
            error(legacy_runs["mpi", m, a], legacy_runs["serial", m, a])
            for m in range(5)
            for a, _ in AXES
        )
        print(
            f"control three-rank vs serial DX_DY_SIGN=-1 outputs, all methods: max difference={spread:.4g}"
        )

    # Default fitting, edge model, reference correction, alignment method and integration order.
    defaults = dict.fromkeys(
        ("OPTIMIZE_FSE", "OPTIMIZE_DX", "REFERENCE_CORRECTION", "FRINGEMODEL", "EDGE_ORDER"),
        DEFAULT,
    )
    defaults["INTEGRATION_ORDER"] = DEFAULT
    for axis, index in AXES:
        observed = run("serial", -ANGLE, axis, OFFSET, -1, **defaults)
        same = error(observed, run("serial", -ANGLE, axis, OFFSET, 1, **defaults))
        expected = run("serial", -ANGLE, axis, -OFFSET, 1, **defaults)
        failures.append(error(observed, expected) > DEFECT_MIN)
        position, slope = 2 * index, 2 * index + 1  # x, x' or y, y' of the on-axis ray
        print(
            f"default fit {axis}=1e-3 ANGLE=-0.04 DX_DY_SIGN=-1 on-axis ray: "
            f"{'xy'[index]}={observed[0, position]:.10e} {'xy'[index]}'={observed[0, slope]:.10e} "
            f"expected {expected[0, position]:.10e} {expected[0, slope]:.10e} "
            f"(DX_DY_SIGN=+1 with {axis}=-1e-3); DX_DY_SIGN=-1 vs +1={same:.4g}"
        )
    verdict(all(failures))


if __name__ == "__main__":
    main()
