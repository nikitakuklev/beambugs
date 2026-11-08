---
title: "TRK-52: A skew multipole loses its whole-element tilt in native consumers"
hide:
  - toc
---

# TRK-52 — A skew multipole loses its whole-element tilt in native consumers

| | |
|---|---|
| Code | [Bmad](../index.md) · Tracking, RF, fields and spin |
| Priority | P2 |
| Failure | Wrong physics |
| Evidence | Reproduced, independently rechecked |
| Last reproduced | 2026-11-05 on Bmad release [20261031-0](https://github.com/bmad-sim/bmad-ecosystem/tree/20261031-0) (`c4c629f`) |
| Tags | <span class="bb-tag">partly-source-only</span> [?](../../methodology.md#tags) |
| Source | [multipole_ele_to_kt contract](https://github.com/bmad-sim/bmad-ecosystem/blob/c4c629fc9071d3a56cb1ba5bc714baba65e9c5d3/bmad/code/multipole_ele_to_kt.f90#L4-L16)<br>[skew-coefficient branch](https://github.com/bmad-sim/bmad-ecosystem/blob/c4c629fc9071d3a56cb1ba5bc714baba65e9c5d3/bmad/code/multipole_ele_to_kt.f90#L57-L73)<br>[Bmad Manual, “Straight Line Element Orientation” (s:straight.orient)](https://github.com/bmad-sim/bmad-ecosystem/blob/c4c629fc9071d3a56cb1ba5bc714baba65e9c5d3/bmad/doc/attributes.tex#L608-L609)<br>[“Magnetostatic Multipole Fields” (s:mag.field)](https://github.com/bmad-sim/bmad-ecosystem/blob/c4c629fc9071d3a56cb1ba5bc714baba65e9c5d3/bmad/doc/electromagnetic-fields.tex#L96-L109) |

A multipole with any nonzero skew coefficient loses its whole-element tilt when converted to strengths and angles for native tracking. With normal strength K₁L = 0.5 m⁻¹, skew strength K₁SL = 0.2 m⁻¹ and tilt = 0.17 rad, a particle at (x, y) = (1, 2) mm receives normalized transverse momenta (pₓ, pᵧ) = (−0.1000, 1.200) × 10⁻³ instead of (−0.4945, 1.098) × 10⁻³. The same omission affects its matrix, reference-bend geometry, displays and exports that consume those angles. PTC, Taylor tracking and an equivalent strength-and-angle element reproduce the rotated field.

Run `python TRK-52.py` with [_bmad.py](../helpers/_bmad.py) and [_tracking_contracts_materials.py](../helpers/_tracking_contracts_materials.py) in the same directory.

??? example "Reproducer: TRK-52.py"

    [Download TRK-52.py](TRK-52.py)

    ```python
    """TRK-52: A skew multipole loses its whole-element tilt in native consumers

    A multipole with any nonzero skew coefficient loses its whole-element tilt when converted to strengths and angles for native tracking. With normal strength K₁L = 0.5 m⁻¹, skew strength K₁SL = 0.2 m⁻¹ and tilt = 0.17 rad, a particle at (x, y) = (1, 2) mm receives normalized transverse momenta (pₓ, pᵧ) = (−0.1000, 1.200) × 10⁻³ instead of (−0.4945, 1.098) × 10⁻³. The same omission affects its matrix, reference-bend geometry, displays and exports that consume those angles. PTC, Taylor tracking and an equivalent strength-and-angle element reproduce the rotated field.

    The script checks native, PTC and Taylor kicks, native matrices and floor geometry against the
    complex multipole formula. Equivalent elements without skew coefficients serve as controls, and
    Tao displays and MAD8, Elegant and SLICKTRACK export angles provide additional consumer checks.
    """

    import cmath
    import math
    import re

    import numpy as np
    from _bmad import compare, run, verdict
    from _tracking_contracts_materials import lattice, orbit, pipe_values, tao

    k, s, tilt = 0.5, 0.2, 0.17
    strength = math.hypot(k, s)
    t1 = -math.atan2(s, k) / 2
    # -dpx+i*dpy = (K1L+i*K1SL)*exp(-2i*tilt)*(x+i*y).
    c = (k + 1j * s) * cmath.exp(-2j * tilt)
    g = c * complex(0.001, 0.002)
    kick = np.array([-g.real, g.imag])
    matrix = np.array([-c.real, c.imag, c.imag, c.real])
    start = "particle_start[x]=.001\nparticle_start[y]=.002"
    forms = {
        "S": f"multipole,k1l={k},k1sl={s},tilt={tilt}",
        "E": f"multipole,k1l={strength:.17g},t1={t1:.17g},tilt={tilt}",
    }
    defects = []
    for name, definition in forms.items():
        for method in ("bmad_standard", "symp_lie_ptc", "taylor"):
            lat = lattice(f"e: {definition},tracking_method={method}", start=start)
            cmd = "pipe ele:orbit 1"
            actual = orbit(tao(lat, [cmd])[cmd])[[1, 3]]
            bad = compare(f"{name} {method} kick, normalized momenta", actual, kick, atol=3e-15)
            if name == "S" and method == "bmad_standard":
                defects.append(bad)
            elif bad:
                raise AssertionError("Equivalent-field tracking control failed")
        ptc_lat = lattice(f"e: {definition}, mat6_calc_method=symp_lie_ptc", start=start)
        matcmd = "pipe ele:mat6 1 mat6"
        ptc_block = tao(ptc_lat, [matcmd])[matcmd]
        ptc_matrix = np.array([pipe_values(ptc_block, str(i)) for i in range(1, 7)])
        if compare(
            f"{name} PTC matrix control", ptc_matrix[[1, 1, 3, 3], [0, 2, 0, 2]], matrix, atol=2e-14
        ):
            raise AssertionError("PTC matrix control failed")
        lat = lattice(f"e: {definition}", start=start)
        cmds = ["pipe ele:mat6 1 mat6", "pipe ele:multipoles 1|model", "show ele 1"]
        blocks = tao(lat, cmds)
        m = np.array([pipe_values(blocks[cmds[0]], str(i)) for i in range(1, 7)])
        bad = compare(
            f"{name} native (M21,M23,M41,M43)", m[[1, 1, 3, 3], [0, 2, 0, 2]], matrix, atol=2e-14
        )
        if name == "S":
            defects.append(bad)
        elif bad:
            raise AssertionError("Equivalent-field matrix control failed")
        line = next(x for x in blocks[cmds[1]].splitlines() if x.startswith("1;"))
        equivalent = float(line.split(";")[4])
        bad = compare(f"{name} Tao equivalent tilt (rad)", equivalent, t1 + tilt, atol=2e-14)
        if name == "S":
            defects.append(bad)
        elif bad:
            raise AssertionError("Equivalent-field display control failed")
        match = re.search(r"T1\(equiv\) =\s*(\S+)", blocks[cmds[2]])
        if match is None:
            raise ValueError(blocks[cmds[2]])
        print(f"{name} show ele rounded tilt={match[1]}; expected={t1 + tilt:.8g}")
        bad = compare(
            f"{name} show ele tilt (rad), four-significant-digit output",
            float(match[1]),
            t1 + tilt,
            atol=5e-5,
        )
        if name == "S":
            defects.append(bad)
        elif bad:
            raise AssertionError("Equivalent-field rounded display control failed")
        for flag, suffix, key in [("-mad8", "mad8", "T1"), ("-elegant", "lte", "TILT")]:
            output = run(
                "bmad_to_mad_sad_elegant",
                [flag, "lat.bmad"],
                files={"lat.bmad": lat},
                outputs=[f"lat.{suffix}"],
            )
            if re.search(r"\[(?:ERROR|FATAL|ABORT)\b", output.log):
                raise RuntimeError(output.log)
            exported = output.outputs[f"lat.{suffix}"].decode()
            matches = re.findall(rf"\b{key}\s*=\s*([-+.\dEe]+)", exported, re.IGNORECASE)
            if len(matches) != 1:
                raise ValueError(exported)
            bad = compare(
                f"{name} {suffix} exported tilt (rad)", float(matches[0]), t1 + tilt, atol=5e-10
            )
            if name == "S":
                defects.append(bad)
            elif bad:
                raise AssertionError("Equivalent-field export control failed")

        output = run("bmad_to_slicktrack", ["lat.bmad"], files={"lat.bmad": lat}, outputs=["lat.slick"])
        if re.search(r"\[(?:ERROR|FATAL|ABORT)\b", output.log):
            raise RuntimeError(output.log)
        records = [
            line.split()
            for line in output.outputs["lat.slick"].decode().splitlines()
            if re.match(r"^\s+4 E\b", line)
        ]
        if len(records) != 1:
            raise ValueError(output.outputs["lat.slick"].decode())
        bad = compare(
            f"{name} SLICKTRACK exported tilt (rad)", float(records[0][3]), t1 + tilt, atol=5e-9
        )
        if name == "S":
            defects.append(bad)
        elif bad:
            raise AssertionError("SLICKTRACK equivalent-field export control failed")

    # The reference-bending dipole rotates its floor displacement and dispersion by tilt.
    for name, quad in [("G", "k1sl=.2"), ("G0", "k1l=0"), ("GE", "k1l=.2,t1=-pi/4")]:
        lat = lattice(
            f"e: multipole,k0l=.001,k0l_status=bends_reference,tilt=.17,{quad}\nd: drift,l=1",
            line="e,d",
        )
        cmds = ["pipe ele:floor 2", "pipe ele:mat6 1 mat6"]
        blocks = tao(lat, cmds)
        floor = pipe_values(blocks[cmds[0]], "Reference")[:2]
        m = np.array([pipe_values(blocks[cmds[1]], str(i)) for i in range(1, 7)])
        for label, observed, expected in [
            ("floor x,y (m)", floor, -math.sin(0.001) * np.array([math.cos(tilt), math.sin(tilt)])),
            ("M26,M46", m[[1, 3], 5], 0.001 * np.array([math.cos(tilt), math.sin(tilt)])),
        ]:
            bad = compare(f"{name} {label}", observed, expected, atol=3e-15)
            if name == "G":
                defects.append(bad)
            elif bad:
                raise AssertionError("Reference-bend geometry control failed")
    verdict(any(defects))

    ```

## Provenance

| Role | Model | Effort | Date | Code examined |
|---|---|---|---|---|
| Found<br><small class="bb-scope">native multipole tilt with skew coefficients</small> | <span class="bb-nowrap">`claude-opus-5-5`</span> | xhigh | <span class="bb-nowrap">2026-10-24</span> | 20261021-0 (`d00a8f17e55a`), source, production, CPU, double, serial |
| Confirmed<br><small class="bb-scope">native skew-multipole kicks and derivatives</small> | <span class="bb-nowrap">`gpt-6-astra`</span> | xhigh | <span class="bb-nowrap">2026-11-05</span> | 20261031-0 (`c4c629fc9071`), source, production, GNU Fortran 15.3.0, CPU, double, serial, OpenMP=1 |
| Confirmed<br><small class="bb-scope">Taylor/PTC, field and floor-geometry consumers</small> | <span class="bb-nowrap">`claude-opus-5-5`</span> | xhigh | <span class="bb-nowrap">2026-10-24</span> | 20261021-0 (`d00a8f17e55a`), source, production, CPU, double, serial |

