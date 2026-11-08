---
title: "COL-12: BetFreqDep_SC1D_AccNode cannot be constructed, and its interpolator fails outside the β range"
hide:
  - toc
---

# COL-12 — `BetFreqDep_SC1D_AccNode` cannot be constructed, and its interpolator fails outside the β range

| | |
|---|---|
| Code | [PyORBIT](../index.md) · Collective effects |
| Priority | P4 |
| Failure | Crash or API hazard |
| Evidence | Reproduced, independently rechecked |
| Last reproduced | 2026-11-07 on PyORBIT commit [`28df660`](https://github.com/PyORBIT-Collaboration/PyORBIT3/commit/28df66018aeb), committed 2026-10-14 |
| Source | [sc1DNode.py#L179](https://github.com/PyORBIT-Collaboration/PyORBIT3/blob/28df66018aebe4e2b780b9747f4a2d7b8430a5c0/py/orbit/space_charge/sc1d/sc1DNode.py#L179-L179)<br>[sc1DNode.py#L202](https://github.com/PyORBIT-Collaboration/PyORBIT3/blob/28df66018aebe4e2b780b9747f4a2d7b8430a5c0/py/orbit/space_charge/sc1d/sc1DNode.py#L202-L202)<br>[sc1DNode.py#L227](https://github.com/PyORBIT-Collaboration/PyORBIT3/blob/28df66018aebe4e2b780b9747f4a2d7b8430a5c0/py/orbit/space_charge/sc1d/sc1DNode.py#L227-L227)<br>[sc1DNode.py#L104](https://github.com/PyORBIT-Collaboration/PyORBIT3/blob/28df66018aebe4e2b780b9747f4a2d7b8430a5c0/py/orbit/space_charge/sc1d/sc1DNode.py#L104-L104)<br>[sc1DNode.py#L266-L302](https://github.com/PyORBIT-Collaboration/PyORBIT3/blob/28df66018aebe4e2b780b9747f4a2d7b8430a5c0/py/orbit/space_charge/sc1d/sc1DNode.py#L266-L302)<br>[sc1DNode.py#L288](https://github.com/PyORBIT-Collaboration/PyORBIT3/blob/28df66018aebe4e2b780b9747f4a2d7b8430a5c0/py/orbit/space_charge/sc1d/sc1DNode.py#L288-L288)<br>[sc1DNode.py#L298-L300](https://github.com/PyORBIT-Collaboration/PyORBIT3/blob/28df66018aebe4e2b780b9747f4a2d7b8430a5c0/py/orbit/space_charge/sc1d/sc1DNode.py#L298-L300)<br>[ImpedanceNode.bilinterp](https://github.com/PyORBIT-Collaboration/PyORBIT3/blob/28df66018aebe4e2b780b9747f4a2d7b8430a5c0/py/orbit/impedances/ImpedanceNode.py#L492-L520)<br>[test_longscbunch.py#L285-L293](https://github.com/PyORBIT-Collaboration/PyORBIT3/blob/28df66018aebe4e2b780b9747f4a2d7b8430a5c0/tests/examples/SpaceCharge/sc1D/test_longscbunch.py#L285-L293) |

`BetFreqDep_SC1D_AccNode`, the β- and frequency-dependent longitudinal space-charge node, builds its impedance list with `range(self.nbins / 2 - 1)`, which raises `TypeError` in Python 3 for every bin count, so the node cannot be created. Its module's `bilinterp` also fails whenever β lies outside the table: below the range it raises `UnboundLocalError`, and above it indexes the table with a list and raises `TypeError`. The impedance package's copy of `bilinterp` clamps β to the table ends and returns the endpoint value for the same inputs, and `FreqDep_SC1D_AccNode` constructs normally.

Run `python COL-12.py`.

??? example "Reproducer: COL-12.py"

    [Download COL-12.py](COL-12.py)

    ```python
    """COL-12: `BetFreqDep_SC1D_AccNode` cannot be constructed, and its interpolator fails outside the β range

    `BetFreqDep_SC1D_AccNode`, the β- and frequency-dependent longitudinal space-charge node, builds its impedance list with `range(self.nbins / 2 - 1)`, which raises `TypeError` in Python 3 for every bin count, so the node cannot be created. Its module's `bilinterp` also fails whenever β lies outside the table: below the range it raises `UnboundLocalError`, and above it indexes the table with a list and raises `TypeError`. The impedance package's copy of `bilinterp` clamps β to the table ends and returns the endpoint value for the same inputs, and `FreqDep_SC1D_AccNode` constructs normally.

    The script constructs `BetFreqDep_SC1D_AccNode` with 64 and 128 bins and a two-point β and frequency table, and calls the module's `bilinterp` below, inside and above the β range. The controls are `FreqDep_SC1D_AccNode` with the same arguments and the impedance package's `bilinterp`, whose clamped bilinear values are derived beside the calls.
    """

    import os

    for _v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
        os.environ.setdefault(_v, "1")

    from orbit.core.bunch import Bunch
    from orbit.impedances.ImpedanceNode import bilinterp as bilinterp_imp
    from orbit.space_charge.sc1d import BetFreqDep_SC1D_AccNode, FreqDep_SC1D_AccNode
    from orbit.space_charge.sc1d.sc1DNode import bilinterp as bilinterp_sc1d

    bunch = Bunch()
    bunch.getSyncParticle().kinEnergy(1.0)
    betas, freqs = [0.5, 0.99], [0.0, 1e9]
    z_bf = [
        [complex(1, 0), complex(3, 0)],
        [complex(2, 0), complex(4, 0)],
    ]  # z_bf[beta index][freq index]


    def attempt(fn):
        try:
            return f"{fn()!r}"
        except Exception as e:  # noqa: BLE001 - report the exception type and message
            return f"{type(e).__name__}: {e}"


    bad, ctl_ok = False, True
    for nbins in (64, 128):
        obs = attempt(
            lambda nbins=nbins: (
                BetFreqDep_SC1D_AccNode(
                    3.0, 248.0, 1, 1, nbins, bunch, {"betas": betas, "freqs": freqs, "z_imp": z_bf}
                )
                and "constructed"
            )
        )
        ctl = attempt(
            lambda nbins=nbins: (
                FreqDep_SC1D_AccNode(3.0, 248.0, 1, 1, nbins, bunch, {"freqs": freqs, "z_imp": z_bf[0]})
                and "constructed"
            )
        )
        print(
            f"nbins = {nbins}: BetFreqDep_SC1D_AccNode observed {obs}; FreqDep_SC1D_AccNode (control) {ctl}; "
            f"expected 'constructed'"
        )
        bad |= obs != "'constructed'"
        ctl_ok &= ctl == "'constructed'"

    for beta, expected in ((0.3, complex(2, 0)), (0.745, complex(2.5, 0)), (0.995, complex(3, 0))):
        # Bilinear interpolation at f = 5e8 Hz (mid-table) with beta clamped to [0.5, 0.99]:
        # beta <= 0.5 -> (1 + 3)/2 = 2; beta = 0.745 (midpoint) -> (1 + 3 + 2 + 4)/4 = 2.5; beta >= 0.99 -> (2 + 4)/2 = 3.
        # bilinterp_sc1d annotates fxy as real although it interpolates complex impedances
        o = attempt(lambda beta=beta: bilinterp_sc1d(beta, 5e8, 1, 1, betas, freqs, z_bf))  # ty: ignore[invalid-argument-type]
        c = attempt(lambda beta=beta: bilinterp_imp(beta, 5e8, 1, 1, betas, freqs, z_bf))
        print(
            f"sc1d bilinterp(beta = {beta}): observed {o}; impedance-package bilinterp (control) {c}; expected {expected!r}"
        )
        bad |= o != repr(expected)
        ctl_ok &= c == repr(expected)
    print("DEFECT REPRODUCED" if bad and ctl_ok else "NOT REPRODUCED")

    ```

## Provenance

| Role | Model | Effort | Date | Code examined |
|---|---|---|---|---|
| Found | <span class="bb-nowrap">`claude-opus-5-5`</span> | high | <span class="bb-nowrap">2026-11-07</span> | main (`28df66018aeb`), source, serial, CPU, double |
| Confirmed | <span class="bb-nowrap">`claude-sonnet-5-5`</span> | high | <span class="bb-nowrap">2026-11-07</span> | main (`28df66018aeb`), source, serial, CPU, double |

