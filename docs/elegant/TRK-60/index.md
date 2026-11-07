---
title: "TRK-60: Zero-length TAPERAPC rejects every finite-coordinate particle"
hide:
  - toc
---

# TRK-60 — Zero-length TAPERAPC rejects every finite-coordinate particle

| | |
|---|---|
| Code | [Elegant](../index.md) · Tracking |
| Priority | P3 |
| Severity | Wrong physics |
| Evidence | Reproduced, independently rechecked |
| Last reproduced | 2026-11-06 on Elegant commit [`0355dd6`](https://github.com/rtsoliday/elegant/commit/0355dd6aca12), committed 2026-11-03 |
| Tags | <span class="bb-tag">partly-source-only</span> [?](../../methodology.md#tags) |
| Source | [src/limit_amplitudes.c:1642–1651](https://github.com/rtsoliday/elegant/blob/0355dd6aca12c861971fbf8f3ae78a0ded991734/src/limit_amplitudes.c#L1642-L1651)<br>[src/limit_amplitudes.c:1656–1674](https://github.com/rtsoliday/elegant/blob/0355dd6aca12c861971fbf8f3ae78a0ded991734/src/limit_amplitudes.c#L1656-L1674)<br>[src/limit_amplitudes.c:1675–1694](https://github.com/rtsoliday/elegant/blob/0355dd6aca12c861971fbf8f3ae78a0ded991734/src/limit_amplitudes.c#L1675-L1694)<br>[doc/elegant.tex:12784–12785](https://github.com/rtsoliday/elegant/blob/0355dd6aca12c861971fbf8f3ae78a0ded991734/doc/elegant.tex#L12784-L12785)<br>[src/track_data.c:3912–3918](https://github.com/rtsoliday/elegant/blob/0355dd6aca12c861971fbf8f3ae78a0ded991734/src/track_data.c#L3912-L3918)<br>[src/do_tracking.c:1583–1605](https://github.com/rtsoliday/elegant/blob/0355dd6aca12c861971fbf8f3ae78a0ded991734/src/do_tracking.c#L1583-L1605)<br>[doc/elegant.tex:12384–12385](https://github.com/rtsoliday/elegant/blob/0355dd6aca12c861971fbf8f3ae78a0ded991734/doc/elegant.tex#L12384-L12385) |

TAPERAPC with L=0, including an omitted length, rejects every particle because its entrance radius squared remains zero. A circular opening of radius 10.00 mm removes both its centred particle and a particle 5.000 mm from the centre, although both should survive; all losses are placed at the entrance. The radius inequality, an equivalent ECOL and a circular TAPERAPE establish the expected acceptance. Giving TAPERAPC a positive length restores these entrance classifications. The failure occurs with equal or unequal positive endpoint radii, with or without transverse offsets, and with either STICKY setting.

Run `python TRK-60.py` with [_loss_geometry.py](../helpers/_loss_geometry.py) and [_elegant.py](../helpers/_elegant.py) in the same directory.

??? example "Reproducer: TRK-60.py"

    [Download TRK-60.py](TRK-60.py)

    ```python
    """TRK-60: Zero-length TAPERAPC rejects every finite-coordinate particle

    TAPERAPC with L=0, including an omitted length, rejects every particle because its entrance radius squared remains zero. A circular opening of radius 10.00 mm removes both its centred particle and a particle 5.000 mm from the centre, although both should survive; all losses are placed at the entrance. The radius inequality, an equivalent ECOL and a circular TAPERAPE establish the expected acceptance. Giving TAPERAPC a positive length restores these entrance classifications. The failure occurs with equal or unequal positive endpoint radii, with or without transverse offsets, and with either STICKY setting.

    Track three fixed points with distances 0, 5 and 15 mm from the aperture centre and compare acceptance with the radius inequality. Check omitted, zero and positive lengths, equal and unequal endpoint radii, offsets, STICKY and equivalent thin ECOL/TAPERAPE controls.
    """

    from _loss_geometry import close, ids, shown, track

    # isort: split
    from _elegant import verdict


    def rays(dx, dy):
        return [[dx, 0, dy, 0], [dx + 0.004, 0, dy + 0.003, 0], [dx + 0.015, 0, dy, 0]]


    def main():
        defect = False
        for length in (None, 0, 1e-9, 0.2):
            for dx, dy in ((0, 0), (0.001, -0.002)):
                for rstart, rend in ((0.01, 0.01), (0.012, 0.008)):
                    ltext = "" if length is None else f"L={length},"
                    lattice = (
                        f"A:TAPERAPC,{ltext}RSTART={rstart},REND={rend},DX={dx},DY={dy}\nBL:LINE=(A)\n"
                    )
                    lost, out, _ = track(lattice, rays(dx, dy), p0=2000, global_coords=False)
                    # r²=0,25e-6,225e-6 m². Both endpoint radii bracket the first
                    # two points and exclude the third, even when the taper collapses.
                    if length is None or length == 0:
                        actual = out["particleID"].astype(int).tolist()
                        if actual == [1, 2]:
                            mismatch = False
                        else:
                            close(actual, [], 0)
                            mismatch = True
                        ids(lost, out, [1, 2, 3] if mismatch else [3], [] if mismatch else [1, 2])
                        defect |= mismatch
                    else:
                        ids(lost, out, [3], [1, 2])
                    close(lost["s"], [0] * len(lost["s"]), 0)
                    print(
                        f"TAPERAPC/L={'omitted' if length is None else length}/offset=({dx:g},{dy:g})/r=({rstart:g},{rend:g}): "
                        f"survivors={out['particleID'].astype(int).tolist()} expected=[1, 2] "
                        f"(distances 0,5 mm inside both radii; 15 mm outside); loss s=0"
                    )
        for kind, params in (
            ("ECOL", "X_MAX=.01,Y_MAX=.01"),
            ("TAPERAPE", "ASTART=.01,AEND=.01,BSTART=.01,BEND=.01"),
        ):
            lost, out, _ = track(
                f"A:{kind},L=0,{params},DX=.001,DY=-.002\nBL:LINE=(A)\n",
                rays(0.001, -0.002),
                p0=2000,
                global_coords=False,
            )
            ids(lost, out, [3], [1, 2])
            close(lost["s"], [0], 0)
            print(f"control/{kind}/L=0: survivors=[1, 2] expected=[1, 2] (10 mm radius inequality)")
        for sticky in (0, 1):
            lost, out, _ = track(
                f"A:TAPERAPC,L=0,RSTART=.01,REND=.01,STICKY={sticky}\nBL:LINE=(A)\n",
                [[0, 0, 0, 0]],
                p0=2000,
                origin=(0.32, -0.17),
            )
            mismatch = len(out["particleID"]) == 0
            ids(lost, out, [1] if mismatch else [], [] if mismatch else [1])
            defect |= mismatch
            if mismatch:
                close(lost["s"], [0], 0)
                close([lost["X"][0], lost["Z"][0]], [0.32, -0.17], 1e-15)
            print(
                f"TAPERAPC/sticky={sticky}/global=1: survivors={out['particleID'].astype(int).tolist()} "
                f"expected=[1] (r=0<10 mm); lost X,Z={shown([lost['X'][0], lost['Z'][0]]) if mismatch else 'none'} m"
            )
        verdict(defect)


    if __name__ == "__main__":
        main()

    ```

## Provenance

| Role | Model | Effort | Date | Code examined |
|---|---|---|---|---|
| Found | `gpt-6.1-sol` | xhigh | 2026-11-06 | master (`0355dd6aca12`), source with SDDS a3f730a14c49, CPU, double, serial, one thread |
| Confirmed | `gpt-6-astra` | xhigh | 2026-11-06 | master (`0355dd6aca12`), source with SDDS a3f730a14c49, CPU, double, serial, one thread |

