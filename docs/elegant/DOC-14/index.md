---
title: "DOC-14: CCBEND and RFTMEZ0 interchange the EPITCH and EYAW axis descriptions"
hide:
  - toc
---

# DOC-14 — CCBEND and RFTMEZ0 interchange the EPITCH and EYAW axis descriptions

| | |
|---|---|
| Code | [Elegant](../index.md) · Tracking |
| Priority | P5 |
| Severity | Documentation or example mismatch |
| Evidence | Reproduced, independently rechecked |
| Last reproduced | 2026-11-06 on Elegant master (`0355dd6`, committed 2026-11-03) |
| Tags | <span class="bb-tag">partly-source-only</span> [?](../../methodology.md#tags) |
| Source | [src/track_data.c:3822–3823](https://github.com/rtsoliday/elegant/blob/0355dd6aca12c861971fbf8f3ae78a0ded991734/src/track_data.c#L3822-L3823)<br>[src/track_data.c:816–817](https://github.com/rtsoliday/elegant/blob/0355dd6aca12c861971fbf8f3ae78a0ded991734/src/track_data.c#L816-L817)<br>[src/ccbend.c:493–495](https://github.com/rtsoliday/elegant/blob/0355dd6aca12c861971fbf8f3ae78a0ded991734/src/ccbend.c#L493-L495)<br>[src/ccbend.c:526–529](https://github.com/rtsoliday/elegant/blob/0355dd6aca12c861971fbf8f3ae78a0ded991734/src/ccbend.c#L526-L529)<br>[src/malign_mat.c:1421–1437](https://github.com/rtsoliday/elegant/blob/0355dd6aca12c861971fbf8f3ae78a0ded991734/src/malign_mat.c#L1421-L1437)<br>[src/motion.cc:595–596](https://github.com/rtsoliday/elegant/blob/0355dd6aca12c861971fbf8f3ae78a0ded991734/src/motion.cc#L595-L596)<br>[src/tilt_matrices.c:218–273](https://github.com/rtsoliday/elegant/blob/0355dd6aca12c861971fbf8f3ae78a0ded991734/src/tilt_matrices.c#L218-L273)<br>[src/track_data.c:1965–1966](https://github.com/rtsoliday/elegant/blob/0355dd6aca12c861971fbf8f3ae78a0ded991734/src/track_data.c#L1965-L1966)<br>[doc/elegant.tex:6642–6653](https://github.com/rtsoliday/elegant/blob/0355dd6aca12c861971fbf8f3ae78a0ded991734/doc/elegant.tex#L6642-L6653)<br>[doc/elegant.tex:10828–10834](https://github.com/rtsoliday/elegant/blob/0355dd6aca12c861971fbf8f3ae78a0ded991734/doc/elegant.tex#L10828-L10834) |

The CCBEND and RFTMEZ0 parameter descriptions interchange the axes of EPITCH and EYAW. They describe EPITCH as rotation about the vertical axis and EYAW as rotation about the horizontal axis, while the implemented rotations use horizontal-axis pitch and vertical-axis yaw. The reversed descriptions appear in the native parameter dictionary, which supplies the manual's element listing. Rotation matrices establish the axes, and independent positive-angle CCBEND transport agrees with them. The parameter storage and argument association remain correct; the defect is in the descriptions.

Run `python DOC-14.py` with [_ccbend.py](../helpers/_ccbend.py) and [_elegant.py](../helpers/_elegant.py) in the same directory.

??? example "Reproducer: DOC-14.py"

    [Download DOC-14.py](DOC-14.py)

    ```python
    """DOC-14: CCBEND and RFTMEZ0 interchange the EPITCH and EYAW axis descriptions

    The CCBEND and RFTMEZ0 parameter descriptions interchange the axes of EPITCH and EYAW. They describe EPITCH as rotation about the vertical axis and EYAW as rotation about the horizontal axis, while the implemented rotations use horizontal-axis pitch and vertical-axis yaw. The reversed descriptions appear in the native parameter dictionary, which supplies the manual's element listing. Rotation matrices establish the axes, and independent positive-angle CCBEND transport agrees with them. The parameter storage and argument association remain correct; the defect is in the descriptions.

    Generate the native SDDS parameter dictionary and compare CCBEND and RFTMEZ0 with correctly labelled CSBEND and KQUAD controls. Derive the axes from elementary rotation matrices, then compare positive-angle CCBEND transport with independently placed rigid magnets. RFTMEZ0's implemented axes are established by source comparison.
    """

    import re
    import tempfile
    from pathlib import Path

    from _ccbend import Magnet, close, track

    # isort: split
    import _elegant as native
    import numpy as np


    def dictionary():
        # ASCII dictionary pages have heterogeneous string defaults; inspect their raw rows.
        with tempfile.TemporaryDirectory(prefix="axis-dictionary-") as tmp:
            path = Path(tmp) / "dictionary.sdds"
            native.run(
                {"run.ele": f'&print_dictionary filename="{path}",SDDS_form=1 &end\n'}, outputs=()
            )
            result = {}
            element = ""
            for line in path.read_text().splitlines():
                header = re.fullmatch(r"!\*+ element type ([A-Z0-9]+):", line)
                if header:
                    element = header[1]
                if element not in ("CCBEND", "RFTMEZ0", "CSBEND", "KQUAD"):
                    continue
                match = re.fullmatch(r'"(EPITCH|EYAW|PITCH|YAW)" .*"([^"]*)"', line)
                wanted = ("PITCH", "YAW") if element == "KQUAD" else ("EPITCH", "EYAW")
                if match and match[1] in wanted:
                    key = element, match[1]
                    if key in result:
                        raise RuntimeError(f"Duplicate dictionary row: {key}")
                    result[key] = match[2]
            if len(result) != 8:
                raise RuntimeError(f"Expected eight axis descriptions: {result}")
            return result


    def main():
        faults = []
        for (element, name), description in dictionary().items():
            expected = "horizontal" if name.endswith("PITCH") else "vertical"
            opposite = "vertical" if expected == "horizontal" else "horizontal"
            axis = re.search(r"about (horizontal|vertical) axis", description)
            if axis is None:
                raise RuntimeError(f"Unexpected axis description: {description}")
            if element in ("CSBEND", "KQUAD") and axis[1] != expected:
                raise RuntimeError(f"Dictionary control failed: {element} {name}")
            faults.append(axis[1] == opposite)
            print(
                f"{element} {name}: described axis={axis[1]}, expected={expected} (rotation matrix); text={description!r}"
            )
        a = 0.001
        # Rx(a) ez=(0,-sin a,cos a); Ry(a) ez=(sin a,0,cos a).
        print(
            f"rotation a=0.001 rad: pitch ez={[0.0, -float(np.sin(a)), float(np.cos(a))]}; yaw ez={[float(np.sin(a)), 0.0, float(np.cos(a))]} (horizontal x and vertical y axes)"
        )
        rays = np.array([[0, 0, 0, 0, 0], [0.0007, 0.0003, -0.0004, 0.0002, 0]])
        for parameter, key in (("EPITCH", "pitch"), ("EYAW", "yaw")):
            out, _ = track(rays, K1=0, MALIGN_METHOD=4, **{parameter: a})
            expected = Magnet(k1=0, **{key: a}).forward(rays)
            residual = close(out, expected, 2e-15, "positive-bend laboratory rotation")
            print(
                f"CCBEND {parameter}=0.001: central y={out[0, 2]:.17g} m, expected={expected[0, 2]:.17g} m; max transverse residual={residual:.3e} (rigid-face Lorentz integration)"
            )
        native.verdict(any(faults))


    if __name__ == "__main__":
        main()

    ```

## Provenance

| Role | Model | Effort | Date | Code examined |
|---|---|---|---|---|
| Found | `claude-opus-5-5` | high | 2026-11-06 | master (`0355dd6aca12`), source inspection |
| Confirmed | `gpt-6.1-sol` | xhigh | 2026-11-06 | master (`0355dd6aca12`), source with SDDS a3f730a14c49, CPU, double, serial, one thread |

