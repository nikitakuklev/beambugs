---
title: "IO-15: Expanded-Hamiltonian CSBEND aperture losses write uninitialized global coordinates"
hide:
  - toc
---

# IO-15 — Expanded-Hamiltonian CSBEND aperture losses write uninitialized global coordinates

| | |
|---|---|
| Code | [Elegant](../index.md) · Lattices, particles and I/O |
| Priority | P4 |
| Severity | Silent wrong result |
| Evidence | Reproduced, independently rechecked |
| Last reproduced | 2026-11-06 on Elegant master (`0355dd6`, committed 2026-11-03) |
| Tags | <span class="bb-tag">partly-source-only</span> [?](../../methodology.md#tags) |
| Source | [src/csbend.c:902–902](https://github.com/rtsoliday/elegant/blob/0355dd6aca12c861971fbf8f3ae78a0ded991734/src/csbend.c#L902-L902)<br>[src/csbend.c:2148–2170](https://github.com/rtsoliday/elegant/blob/0355dd6aca12c861971fbf8f3ae78a0ded991734/src/csbend.c#L2148-L2170)<br>[src/csbend.c:2247–2250](https://github.com/rtsoliday/elegant/blob/0355dd6aca12c861971fbf8f3ae78a0ded991734/src/csbend.c#L2247-L2250)<br>[src/csbend.c:1504–1533](https://github.com/rtsoliday/elegant/blob/0355dd6aca12c861971fbf8f3ae78a0ded991734/src/csbend.c#L1504-L1533)<br>[src/obstructionData.c:311–361](https://github.com/rtsoliday/elegant/blob/0355dd6aca12c861971fbf8f3ae78a0ded991734/src/obstructionData.c#L311-L361)<br>[src/track_data.c:1977–1977](https://github.com/rtsoliday/elegant/blob/0355dd6aca12c861971fbf8f3ae78a0ded991734/src/track_data.c#L1977-L1977)<br>[doc/elegant.tex:7206–7207](https://github.com/rtsoliday/elegant/blob/0355dd6aca12c861971fbf8f3ae78a0ded991734/doc/elegant.tex#L7206-L7207)<br>[doc/elegant.tex:12288–12290](https://github.com/rtsoliday/elegant/blob/0355dd6aca12c861971fbf8f3ae78a0ded991734/doc/elegant.tex#L12288-L12290)<br>[src/sdds_support.c:222–234](https://github.com/rtsoliday/elegant/blob/0355dd6aca12c861971fbf8f3ae78a0ded991734/src/sdds_support.c#L222-L234) |

With EXPAND_HAMILTONIAN=1 and global loss coordinates enabled, CSBEND aperture rejection returns without initializing X, Z or thetaX, the global horizontal position, longitudinal position and direction. The outer tracker copies unwritten values into the loss file. In a 1.000 m, 0.1500 rad bend, a loss at s=0.1189 m produces arbitrary data instead of X=0.3447 m, Z=−0.05314 m and thetaX=0.1332 rad. Geometric projection of the reported local loss state establishes the expected values, and Valgrind detects uninitialized bytes in the output. Disabling global coordinates leaves physical loss records unchanged; genuine obstruction-return paths initialize these fields.

Run `python IO-15.py` with [_loss_geometry.py](../helpers/_loss_geometry.py) and [_elegant.py](../helpers/_elegant.py) in the same directory.

??? example "Reproducer: IO-15.py"

    [Download IO-15.py](IO-15.py)

    ```python
    """IO-15: Expanded-Hamiltonian CSBEND aperture losses write uninitialized global coordinates

    With EXPAND_HAMILTONIAN=1 and global loss coordinates enabled, CSBEND aperture rejection returns without initializing X, Z or thetaX, the global horizontal position, longitudinal position and direction. The outer tracker copies unwritten values into the loss file. In a 1.000 m, 0.1500 rad bend, a loss at s=0.1189 m produces arbitrary data instead of X=0.3447 m, Z=−0.05314 m and thetaX=0.1332 rad. Geometric projection of the reported local loss state establishes the expected values, and Valgrind detects uninitialized bytes in the output. Disabling global coordinates leaves physical loss records unchanged; genuine obstruction-return paths initialize these fields.

    Setup: `valgrind` on `PATH`; the script runs serial Elegant under Memcheck.

    Run positive-angle expanded and exact CSBEND aperture losses under Valgrind, projecting the reported ordinary loss state into the floor frame. Check that disabling global output preserves ordinary records and removes undefined-byte reports, and that genuine obstruction returns initialize their fields. The verdict uses Memcheck rather than an arbitrary unwritten byte pattern.
    """

    from _loss_geometry import close, ids, obstruction, projection, same_physics, shown, track

    # isort: split
    from _elegant import verdict

    RAYS = [[0.008, 0.02, 0, 0], [0, 0, 0, 0]]


    def bend(expanded):
        return (
            f"B:CSBEND,L=1,ANGLE=.15,EXPAND_HAMILTONIAN={expanded},N_SLICES=16,"
            "INTEGRATION_ORDER=6,EDGE1_EFFECTS=0,EDGE2_EFFECTS=0,SYNCH_RAD=0,ISR=0\n"
        )


    def main():
        defect = False
        for expanded in (1, 0):
            lattice = "A:MAXAMP,X_MAX=.01,Y_MAX=.02\n" + bend(expanded) + "BL:LINE=(A,B)\n"
            on = track(lattice, RAYS, heading=0.13, origin=(0.32, -0.17), p0=2000, memcheck=True)
            off = track(
                lattice,
                RAYS,
                heading=0.13,
                origin=(0.32, -0.17),
                p0=2000,
                memcheck=True,
                global_coords=False,
            )
            lost, out, errors = on
            ids(lost, out)
            ids(*off[:2])
            same_physics(on[:2], off[:2])
            if off[2] or (not expanded and errors):
                raise RuntimeError("A Memcheck control has undefined output")
            expected = projection(lost, heading=0.13, origin=(0.32, -0.17))
            actual = [lost[k][0] for k in ("X", "Z", "thetaX")]
            if expanded:
                close(lost["s"][0], 0.11889376357232741, 2e-15)
                if errors:
                    defect = True
                    observed = "uninitialized bytes in dump_lost_particles"
                else:
                    close(actual, expected)
                    observed = shown(actual)
            else:
                # The independent sector-angle error has its own entry.
                close(actual[:2], expected[:2])
                observed = shown(actual[:2]) + " (positions)"
            print(
                f"CSBEND/expanded={expanded}: s={lost['s'][0]:.16g} m; global={observed}; "
                f"expected X,Z,thetaX={shown(expected)} (local state + sector floor projection); "
                f"Memcheck on={errors} off={off[2]} expected=0/0; physical switch=bit-identical"
            )
        for expanded in (0, 1):
            lost, out, errors = track(
                bend(expanded) + "BL:LINE=(B)\n",
                [[0, 0, 0, 0]],
                heading=0.13,
                origin=(0.32, -0.17),
                p0=2000,
                memcheck=True,
                files={"obstruction.sdds": obstruction(zmin=0.2, zmax=0.3)},
                commands="&obstruction_data input=obstruction.sdds &end\n",
            )
            ids(lost, out, (1,), ())
            if errors:
                raise RuntimeError("Obstruction-return control has undefined output")
            expected = projection(lost, heading=0.13, origin=(0.32, -0.17))
            actual = [lost[k][0] for k in ("X", "Z")]
            close(actual, expected[:2])
            print(
                f"CSBEND/expanded={expanded}/obstruction: X,Z={shown(actual)} expected={shown(expected[:2])} m "
                f"(sector floor projection); Memcheck={errors} expected=0, IDs=pass"
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

