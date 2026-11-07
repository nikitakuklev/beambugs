---
title: "TRK-64: Zero-length TAPERAPR with unequal endpoint widths removes interior rays"
hide:
  - toc
---

# TRK-64 — Zero-length TAPERAPR with unequal endpoint widths removes interior rays

| | |
|---|---|
| Code | [Elegant](../index.md) · Tracking |
| Priority | P3 |
| Failure | Wrong physics |
| Evidence | Reproduced, independently rechecked |
| Last reproduced | 2026-11-06 on Elegant commit [`0355dd6`](https://github.com/rtsoliday/elegant/commit/0355dd6aca12), committed 2026-11-03 |
| Source | [src/limit_amplitudes.c:1816–1822](https://github.com/rtsoliday/elegant/blob/0355dd6aca12c861971fbf8f3ae78a0ded991734/src/limit_amplitudes.c#L1816-L1822)<br>[src/limit_amplitudes.c:1835–1895](https://github.com/rtsoliday/elegant/blob/0355dd6aca12c861971fbf8f3ae78a0ded991734/src/limit_amplitudes.c#L1835-L1895)<br>[src/track_data.c:3941–3949](https://github.com/rtsoliday/elegant/blob/0355dd6aca12c861971fbf8f3ae78a0ded991734/src/track_data.c#L3941-L3949) |

TAPERAPR with zero or omitted length rejects particles strictly inside both openings when either active transverse plane has unequal positive endpoint half-widths. An aperture tapering from 14.00 to 9.000 mm removes the centred ray and rays at ±6.000 mm, although all remain inside both endpoints. Direct endpoint inequalities establish survival without choosing a collapsed-taper convention. Both taper directions, both planes, reflection and displaced/rolled placement reproduce; positive-length and equal-width controls preserve the correct survivors.

Run `python TRK-64.py` with [_slicing.py](../helpers/_slicing.py) and [_elegant.py](../helpers/_elegant.py) in the same directory.

??? example "Reproducer: TRK-64.py"

    [Download TRK-64.py](TRK-64.py)

    ```python
    """TRK-64: Zero-length TAPERAPR with unequal endpoint widths removes interior rays

    TAPERAPR with zero or omitted length rejects particles strictly inside both openings when either active transverse plane has unequal positive endpoint half-widths. An aperture tapering from 14.00 to 9.000 mm removes the centred ray and rays at ±6.000 mm, although all remain inside both endpoints. Direct endpoint inequalities establish survival without choosing a collapsed-taper convention. Both taper directions, both planes, reflection and displaced/rolled placement reproduce; positive-length and equal-width controls preserve the correct survivors.

    Classify five fixed rays with affine endpoint inequalities in both transverse planes. Compare zero/omitted and positive lengths, equal and unequal openings, reflection, disabled-plane and placed-aperture controls using exact particle IDs.
    """

    from _slicing import close

    # isort: split
    import _elegant as native
    import numpy as np


    def track(lattice, rays):
        beam = dict(zip(("x", "xp", "y", "yp"), rays[:, :4].T, strict=True))
        beam.update(p=np.full(len(rays), 3200.0), particleID=np.arange(1, len(rays) + 1))
        result = native.run(
            {
                "line.lte": lattice,
                "run.ele": (
                    "&run_setup lattice=line.lte,use_beamline=BL,p_central=3200,output=out.sdds,losses=loss.sdds,random_number_seed=104729 &end\n"
                    "&run_control n_steps=1 &end\n&sdds_beam input=beam.sdds &end\n&track &end\n"
                ),
            },
            beams={"beam.sdds": beam},
            outputs=("out.sdds", "loss.sdds"),
        )
        out, lost = (native.particles(result[name]) for name in ("out.sdds", "loss.sdds"))
        close(sorted([*out["particleID"], *lost["particleID"]]), beam["particleID"], 0, "ID accounting")
        return out["particleID"].astype(int).tolist()


    def main():
        rays = np.zeros((5, 6))
        rays[:, :4] = [
            [0, 0, 0, 0],
            [0.006, 0.0002, -0.004, -0.0001],
            [-0.006, -0.0002, 0.004, 0.0001],
            [0.018, 0, 0, 0],
            [0, 0, 0.018, 0],
        ]
        faults = []
        for plane in ("X", "Y"):
            for a, b in ((0.014, 0.009), (0.009, 0.014), (0.009, 0.009)):
                for length in (None, 0, 1e-10, 0.31):
                    x0, x1, y0, y1 = (a, b, 0.011, 0.011) if plane == "X" else (0.011, 0.011, a, b)
                    prefix = "" if length is None else f"L={length},"
                    for reflected in (False, True):
                        lattice = f"A:TAPERAPR,{prefix}XSTART={x0},XEND={x1},YSTART={y0},YEND={y1}\nU:LINE=(A)\nBL:LINE=({'-U' if reflected else 'U'})\n"
                        ell = 0 if length is None else length
                        ends = [(x0, y0), (x1, y1)]
                        if reflected:
                            ends.reverse()
                        # Affine ray/wall differences attain their extrema at endpoints.
                        good = np.ones(len(rays), dtype=bool)
                        for s, (wx, wy) in zip((0, ell), ends, strict=True):
                            good &= abs(rays[:, 0] + s * rays[:, 1]) < wx
                            good &= abs(rays[:, 2] + s * rays[:, 3]) < wy
                        expected = (np.flatnonzero(good) + 1).tolist()
                        observed = track(lattice, rays)
                        if observed != expected:
                            if ell != 0 or a == b or observed:
                                raise RuntimeError(
                                    f"Unexpected survivor IDs: {observed}, expected {expected}"
                                )
                            faults.append(True)
                        elif ell == 0 and a != b:
                            faults.append(False)
                        print(
                            f"{plane} {a:g}->{b:g} L={length} reflected={reflected}: IDs={observed}, expected={expected} (strict affine endpoint inequalities)"
                        )
        ids = track("A:TAPERAPR,XSTART=.014,XEND=0,YSTART=.011,YEND=.011\nBL:LINE=(A)\n", rays)
        close(ids, [1, 2, 3, 4], 0, "disabled horizontal plane")
        print(f"disabled horizontal plane: IDs={ids}, expected=[1, 2, 3, 4] (vertical inequalities)")
        local = rays[:3].copy()
        roll, dx, dy = 0.23, 0.0011, -0.0007
        placed = local.copy()
        for ix, iy in ((0, 2), (1, 3)):
            placed[:, ix] = np.cos(roll) * local[:, ix] - np.sin(roll) * local[:, iy]
            placed[:, iy] = np.sin(roll) * local[:, ix] + np.cos(roll) * local[:, iy]
        placed[:, 0] += dx
        placed[:, 2] += dy
        ids = track(
            f"A:TAPERAPR,XSTART=.014,XEND=.009,YSTART=.011,YEND=.011,DX={dx},DY={dy},TILT={roll}\nBL:LINE=(A)\n",
            placed,
        )
        if ids not in ([], [1, 2, 3]):
            raise RuntimeError(f"Unexpected placed-aperture IDs: {ids}")
        faults.append(ids == [])
        print(f"DX/DY/TILT placement: IDs={ids}, expected=[1, 2, 3] (inverse local-frame inequalities)")
        native.verdict(any(faults))


    if __name__ == "__main__":
        main()

    ```

## Provenance

| Role | Model | Effort | Date | Code examined |
|---|---|---|---|---|
| Found | `gpt-6-astra` | xhigh | 2026-11-06 | master (`0355dd6aca12`), source with SDDS a3f730a14c49, CPU, double, serial, one thread |
| Confirmed | `gpt-6.1-sol` | xhigh | 2026-11-06 | master (`0355dd6aca12`), source with SDDS a3f730a14c49, CPU, double, serial, one thread |

