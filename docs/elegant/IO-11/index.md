---
title: "IO-11: MAD8 export silently drops QUAD and KQUAD transverse offsets"
hide:
  - toc
---

# IO-11 — MAD8 export silently drops QUAD and KQUAD transverse offsets

| | |
|---|---|
| Code | [Elegant](../index.md) · Lattices, particles and I/O |
| Priority | P4 |
| Failure | Wrong physics |
| Evidence | Reproduced, independently rechecked |
| Last reproduced | 2026-11-06 on Elegant commit [`0355dd6`](https://github.com/rtsoliday/elegant/commit/0355dd6aca12), committed 2026-11-03 |
| Tags | <span class="bb-tag">partly-source-only</span> [?](../../methodology.md#tags) |
| Source | [src/convert_to_mad8.c:59–68](https://github.com/rtsoliday/elegant/blob/0355dd6aca12c861971fbf8f3ae78a0ded991734/src/convert_to_mad8.c#L59-L68)<br>[src/convert_to_mad8.c:157–204](https://github.com/rtsoliday/elegant/blob/0355dd6aca12c861971fbf8f3ae78a0ded991734/src/convert_to_mad8.c#L157-L204)<br>[src/convert_to_mad8.c:144–147](https://github.com/rtsoliday/elegant/blob/0355dd6aca12c861971fbf8f3ae78a0ded991734/src/convert_to_mad8.c#L144-L147)<br>[src/track_data.c:373–374](https://github.com/rtsoliday/elegant/blob/0355dd6aca12c861971fbf8f3ae78a0ded991734/src/track_data.c#L373-L374)<br>[src/track_data.c:1259–1260](https://github.com/rtsoliday/elegant/blob/0355dd6aca12c861971fbf8f3ae78a0ded991734/src/track_data.c#L1259-L1260)<br>[src/compute_matrices.c:1490–1493](https://github.com/rtsoliday/elegant/blob/0355dd6aca12c861971fbf8f3ae78a0ded991734/src/compute_matrices.c#L1490-L1493)<br>[src/compute_matrices.c:1667–1670](https://github.com/rtsoliday/elegant/blob/0355dd6aca12c861971fbf8f3ae78a0ded991734/src/compute_matrices.c#L1667-L1670)<br>[doc/malign_method.tex:1–22](https://github.com/rtsoliday/elegant/blob/0355dd6aca12c861971fbf8f3ae78a0ded991734/doc/malign_method.tex#L1-L22)<br>[doc/elegant.tex:9587–9623](https://github.com/rtsoliday/elegant/blob/0355dd6aca12c861971fbf8f3ae78a0ded991734/doc/elegant.tex#L9587-L9623)<br>[MAD8 manual:2398–2415](https://github.com/MethodicalAcceleratorDesign/MAD-X/blob/7998e738eca575c82a424138517c2a17115f3cdb/doc/mad8usrguide/maduser.tex#L2398-L2415)<br>[MAD8 manual:6023–6030](https://github.com/MethodicalAcceleratorDesign/MAD-X/blob/7998e738eca575c82a424138517c2a17115f3cdb/doc/mad8usrguide/maduser.tex#L6023-L6030)<br>[MAD8 manual:6032–6087](https://github.com/MethodicalAcceleratorDesign/MAD-X/blob/7998e738eca575c82a424138517c2a17115f3cdb/doc/mad8usrguide/maduser.tex#L6032-L6087)<br>[doc/elegant.tex:9642–9647](https://github.com/rtsoliday/elegant/blob/0355dd6aca12c861971fbf8f3ae78a0ded991734/doc/elegant.tex#L9642-L9647)<br>[src/convert_to_mad8.c:248–257](https://github.com/rtsoliday/elegant/blob/0355dd6aca12c861971fbf8f3ae78a0ded991734/src/convert_to_mad8.c#L248-L257) |

Exporting a QUAD or KQUAD with transverse offsets DX and DY using elegantto's MAD8 output silently produces an aligned quadrupole. MAD8 represents these offsets through EALIGN commands, but the exporter writes neither those commands nor a warning. For a 0.4200 m quadrupole with normalized gradient K₁ = 1.700 m⁻², DX = 0.8000 mm, and DY = −0.6000 mm, an initially axial particle exits with slopes 0.5431 and 0.4501 mrad; the exported model gives zero. Native tracking and an independent affine map agree, and the omission occurs for every alignment-method setting.

Run `python IO-11.py` with [_misalign.py](../helpers/_misalign.py), [_elegant.py](../helpers/_elegant.py) and [_diagnostics.py](../helpers/_diagnostics.py) in the same directory.

??? example "Reproducer: IO-11.py"

    [Download IO-11.py](IO-11.py)

    ```python
    """IO-11: MAD8 export silently drops QUAD and KQUAD transverse offsets

    Exporting a QUAD or KQUAD with transverse offsets DX and DY using elegantto's MAD8 output silently produces an aligned quadrupole. MAD8 represents these offsets through EALIGN commands, but the exporter writes neither those commands nor a warning. For a 0.4200 m quadrupole with normalized gradient K₁ = 1.700 m⁻², DX = 0.8000 mm, and DY = −0.6000 mm, an initially axial particle exits with slopes 0.5431 and 0.4501 mrad; the exported model gives zero. Native tracking and an independent affine map agree, and the omission occurs for every alignment-method setting.

    Setup: `elegantto` of the Elegant build on `PATH`; the script exports lattices with it.

    Track an initially axial electron through displaced QUAD and KQUAD elements for all five alignment methods and both displacement signs. Compare QUAD with the independent affine map (I−M)d, KQUAD with exact canonical Lorentz-force integration, and the MAD8 export with its own nominal map and EALIGN contract. Check zero-offset controls and literal EALIGN inclusion through -ender.
    """

    import re

    from _misalign import close, coefficient, exact_quad_offset, export, fault, lattice, quad_map, track

    # isort: split
    # The helper sets numerical thread limits before NumPy initializes.
    import numpy as np
    from _elegant import verdict


    def mad8_axis(text):
        commands = re.findall(r"^\s*EALIGN\b[^\n]*", text, re.MULTILINE | re.IGNORECASE)
        if not commands:
            if re.search(r"\b(?:DX|DY)\b", text, re.IGNORECASE):
                raise RuntimeError("MAD8 offsets must be represented with EALIGN")
            return track(text, np.zeros((1, 4)))[0]
        if len(commands) != 1 or not re.search(r"\bRANGE\s*=\s*Q\b", commands[0], re.IGNORECASE):
            raise RuntimeError("Expected one EALIGN command for Q")
        # MAD8's single unrolled magnet has the same local and laboratory axes;
        # alignment commands translate its linear map by (I-M)d.
        d = np.array([coefficient(commands[0], "DX"), 0, coefficient(commands[0], "DY"), 0])
        matrix = quad_map(coefficient(text, "K1"), coefficient(text, "L"), coefficient(text, "TILT"))
        return d - matrix @ d


    def vector(values):
        return "(" + ",".join(f"{v:.16g}" for v in values) + ")"


    def main():
        defect = False
        matrix = quad_map(1.7, 0.42, 0.0)
        for kind in ("QUAD", "KQUAD"):
            for method in range(5):
                for dx, dy in ((0.0, 0.0), (8e-4, -6e-4), (-8e-4, 6e-4)):
                    source = lattice(kind, 0.42, 1.7, 0.0, dx, dy, method)
                    native = track(source, np.zeros((1, 4)))[0]
                    d = np.array([dx, 0.0, dy, 0.0])
                    affine = d - matrix @ d
                    expected = exact_quad_offset(1.7, 0.42, dx, dy) if kind == "KQUAD" else affine
                    close(native, expected, 3e-16)
                    text = export(source, "mad8")
                    close(coefficient(text, "L"), 0.42, 0)
                    close(coefficient(text, "K1"), 1.7, 0)
                    close(coefficient(text, "TILT"), 0, 0)
                    target = mad8_axis(text)
                    defect |= fault(target, affine, np.zeros(4), 3e-16)
                    print(
                        f"{kind} method={method} DX={dx:g} DY={dy:g}: native={vector(native)} "
                        f"expected={vector(expected)}; MAD8={vector(target)} "
                        f"affine expected={vector(affine)}; diagnostics=none "
                        "(x,xprime,y,yprime; affine=(I-M)d, KQUAD uses exact canonical drifts)"
                    )
        source = lattice("QUAD", 0.42, 1.7, 0.0, 8e-4, -6e-4)
        commands = "USE,BL\nEALIGN,RANGE=Q,DX=0.0008,DY=-0.0006\n"
        text = export(source, "mad8", ender=commands)
        if not text.endswith(commands):
            raise RuntimeError("Literal -ender inclusion failed")
        manual = mad8_axis(text)
        d = np.array([8e-4, 0, -6e-4, 0])
        close(manual, d - matrix @ d, 3e-16)
        print(
            f"EALIGN -ender control: observed={vector(manual)} expected={vector(d - matrix @ d)}; "
            "literal inclusion=pass (MAD8 local translation gives (I-M)d)"
        )
        verdict(defect)


    if __name__ == "__main__":
        main()

    ```

## Provenance

| Role | Model | Effort | Date | Code examined |
|---|---|---|---|---|
| Found | <span class="bb-nowrap">`gpt-6-astra`</span> | xhigh | <span class="bb-nowrap">2026-11-06</span> | master (`0355dd6aca12`), source with SDDS a3f730a14c49, CPU, double, serial elegant and elegantto, one thread |
| Confirmed | <span class="bb-nowrap">`gpt-6.1-sol`</span> | xhigh | <span class="bb-nowrap">2026-11-06</span> | master (`0355dd6aca12`), source with SDDS a3f730a14c49, CPU, double, serial elegant and elegantto, one thread |

