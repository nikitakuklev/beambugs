---
title: "IO-16: Lattice export loses reflected-bend physical end selection"
hide:
  - toc
---

# IO-16 — Lattice export loses reflected-bend physical end selection

| | |
|---|---|
| Code | [Elegant](../index.md) · Lattices, particles and I/O |
| Priority | P4 |
| Failure | Wrong physics |
| Evidence | Reproduced, independently rechecked |
| Last reproduced | 2026-11-06 on Elegant commit [`0355dd6`](https://github.com/rtsoliday/elegant/commit/0355dd6aca12), committed 2026-11-03 |
| Tags | <span class="bb-tag">partly-source-only</span> [?](../../methodology.md#tags) |
| Source | [src/convert_to_mad8.c:55–83](https://github.com/rtsoliday/elegant/blob/0355dd6aca12c861971fbf8f3ae78a0ded991734/src/convert_to_mad8.c#L55-L83)<br>[src/convert_to_mad8.c:161–168](https://github.com/rtsoliday/elegant/blob/0355dd6aca12c861971fbf8f3ae78a0ded991734/src/convert_to_mad8.c#L161-L168)<br>[src/convert_to_mad8.c:258–274](https://github.com/rtsoliday/elegant/blob/0355dd6aca12c861971fbf8f3ae78a0ded991734/src/convert_to_mad8.c#L258-L274)<br>[src/convert_to_mad8.c:343–350](https://github.com/rtsoliday/elegant/blob/0355dd6aca12c861971fbf8f3ae78a0ded991734/src/convert_to_mad8.c#L343-L350)<br>[src/mad_parse.c:563–571](https://github.com/rtsoliday/elegant/blob/0355dd6aca12c861971fbf8f3ae78a0ded991734/src/mad_parse.c#L563-L571)<br>[src/mad_parse.c:1458–1479](https://github.com/rtsoliday/elegant/blob/0355dd6aca12c861971fbf8f3ae78a0ded991734/src/mad_parse.c#L1458-L1479)<br>[src/compute_matrices.c:1453–1468](https://github.com/rtsoliday/elegant/blob/0355dd6aca12c861971fbf8f3ae78a0ded991734/src/compute_matrices.c#L1453-L1468)<br>[src/mad_parse.c:287–294](https://github.com/rtsoliday/elegant/blob/0355dd6aca12c861971fbf8f3ae78a0ded991734/src/mad_parse.c#L287-L294)<br>[src/get_beamline.c:1409–1410](https://github.com/rtsoliday/elegant/blob/0355dd6aca12c861971fbf8f3ae78a0ded991734/src/get_beamline.c#L1409-L1410)<br>[doc/elegant.tex:9588–9589](https://github.com/rtsoliday/elegant/blob/0355dd6aca12c861971fbf8f3ae78a0ded991734/doc/elegant.tex#L9588-L9589)<br>[doc/elegant.tex:9614–9615](https://github.com/rtsoliday/elegant/blob/0355dd6aca12c861971fbf8f3ae78a0ded991734/doc/elegant.tex#L9614-L9615)<br>[doc/elegant.tex:9623–9623](https://github.com/rtsoliday/elegant/blob/0355dd6aca12c861971fbf8f3ae78a0ded991734/doc/elegant.tex#L9623-L9623)<br>[src/convert_to_mad8.c:144–148](https://github.com/rtsoliday/elegant/blob/0355dd6aca12c861971fbf8f3ae78a0ded991734/src/convert_to_mad8.c#L144-L148)<br>[doc/sben.tex:31–36](https://github.com/rtsoliday/elegant/blob/0355dd6aca12c861971fbf8f3ae78a0ded991734/doc/sben.tex#L31-L36) |

Exporting a reflected asymmetric bend with elegantto retains its original entrance and exit face order. A 0.8000 m, 0.1600 rad sector bend with gradient K₁ = 0.7000 m⁻² and face angles −0.02000 and 0.1300 rad exports horizontal position response R₁₁ = 0.7694 instead of 0.7917, a 2.812% deficit. Independent integration of the linear equations establishes the selected-end map. MAD8 export affects SBEN, RBEN, CSBEND and CSRCSBEND, including mixed orientations sharing one definition; SAD serialization also loses end selection. Native reflection and saving preserve it.

Run `python IO-16.py` with [_misalign.py](../helpers/_misalign.py), [_slicing.py](../helpers/_slicing.py), [_elegant.py](../helpers/_elegant.py) and [_diagnostics.py](../helpers/_diagnostics.py) in the same directory.

??? example "Reproducer: IO-16.py"

    [Download IO-16.py](IO-16.py)

    ```python
    """IO-16: Lattice export loses reflected-bend physical end selection

    Exporting a reflected asymmetric bend with elegantto retains its original entrance and exit face order. A 0.8000 m, 0.1600 rad sector bend with gradient K₁ = 0.7000 m⁻² and face angles −0.02000 and 0.1300 rad exports horizontal position response R₁₁ = 0.7694 instead of 0.7917, a 2.812% deficit. Independent integration of the linear equations establishes the selected-end map. MAD8 export affects SBEN, RBEN, CSBEND and CSRCSBEND, including mixed orientations sharing one definition; SAD serialization also loses end selection. Native reflection and saving preserve it.

    Setup: `elegantto` of the Elegant build on `PATH`; the script exports lattices with it.

    Export four bend types to MAD8 and compare their reloaded linear maps with independent ODE integration and ordered edge kicks. Check forward, reflected, double-reflected, mixed and symmetric cases, native saving, SAD serialization and an unsupported-element warning.
    """

    import re
    import subprocess
    import tempfile
    from pathlib import Path

    from _misalign import export
    from _slicing import close, discrepancy, track

    # isort: split
    import _elegant as native
    import numpy as np
    from scipy.integrate import solve_ivp


    def bend(e1, e2, length):
        h, k1 = 0.16 / length, 0.7
        a = np.zeros((4, 4))
        a[0, 1] = a[2, 3] = 1
        a[1, 0], a[3, 2] = -(h * h + k1), k1
        solution = solve_ivp(
            lambda _, m: (a @ m.reshape(4, 4)).ravel(),
            (0, length),
            np.eye(4).ravel(),
            method="DOP853",
            rtol=3e-14,
            atol=1e-15,
        )
        if not solution.success:
            raise RuntimeError(solution.message)

        def edge(e):
            m = np.eye(4)
            m[1, 0], m[3, 2] = h * np.tan(e), -h * np.tan(e)
            return m

        # Ordered entrance kick, body fundamental matrix, then exit kick.
        result = edge(e2) @ solution.y[:, -1].reshape(4, 4) @ edge(e1)
        kx = np.sqrt(h * h + k1)
        close(
            result[0, 0],
            np.cos(kx * length) + np.sin(kx * length) / kx * h * np.tan(e1),
            2e-15,
            "closed-form R11",
        )
        return result


    def matrix(lattice):
        return track("matrix", lattice, np.zeros((1, 6)), matrix=True, p0=3200)["R"][:4, :4]


    def saved_lattice(lattice):
        with tempfile.TemporaryDirectory(prefix="bend-save-") as tmp:
            path = Path(tmp) / "saved.lte"
            native.run(
                {
                    "line.lte": lattice,
                    "run.ele": (
                        "&run_setup lattice=line.lte,use_beamline=BL,p_central=3200,default_order=1 &end\n"
                        "&run_control n_steps=1 &end\n"
                        f'&save_lattice filename="{path}",output_seq=1 &end\n'
                        "&bunched_beam n_particles_per_bunch=1 &end\n&track &end\n"
                    ),
                },
                outputs=(),
            )
            return path.read_text()


    def main():
        faults = []
        for kind in ("SBEN", "RBEN", "CSBEND", "CSRCSBEND"):
            arc = 0.8 * 0.08 / np.sin(0.08) if kind == "RBEN" else 0.8
            for e1, e2 in ((-0.02, 0.13), (0.06, 0.06)):
                ein, eout = (e1 + 0.08, e2 + 0.08) if kind == "RBEN" else (e1, e2)
                for seq, order in (
                    ("U", [False]),
                    ("-U", [True]),
                    ("-V", [False]),
                    ("U,D,-U", [False, True]),
                    ("-U,D,U", [True, False]),
                ):
                    lattice = f"B:{kind},L=.8,ANGLE=.16,K1=.7,E1={e1},E2={e2},HGAP=0,FINT=0\nD:DRIF,L=.17\nU:LINE=(B)\nV:LINE=(-U)\nBL:LINE=({seq})\n"
                    target = export(lattice, "mad8")
                    observed = matrix(target)
                    expected = faulty = np.eye(4)
                    for i, reflected in enumerate(order):
                        if i:
                            d = np.eye(4)
                            d[0, 1] = d[2, 3] = 0.17
                            expected = d @ expected
                            faulty = d @ faulty
                        expected = (
                            bend(eout, ein, arc) if reflected else bend(ein, eout, arc)
                        ) @ expected
                        faulty = bend(ein, eout, arc) @ faulty
                    if e1 == e2 or not any(order):
                        close(observed, expected, 5e-14, "export control")
                    else:
                        faults.append(discrepancy(observed, expected, faulty, 5e-14))
                    if kind in ("SBEN", "RBEN"):
                        close(matrix(lattice), expected, 5e-14, "native reflection")
                        close(matrix(saved_lattice(lattice)), expected, 5e-14, "native saving")
                    error = float(np.max(abs(observed - expected)))
                    print(
                        f"MAD8 {kind} faces={e1:g},{e2:g} sequence={seq}: R11={observed[0, 0]:.17g}, expected={expected[0, 0]:.17g}; max 4x4 error={error:.17g} (independent body ODE/ordered face kicks); warning=False"
                    )
        sad = export("B:SBEN,L=.8,ANGLE=.16,K1=.7,E1=-.02,E2=.13\nU:LINE=(B)\nBL:LINE=(-U)\n", "sad")
        values = {k: float(v) for k, v in re.findall(r"\b(E[12])=([+\-0-9.eE]+)", sad)}
        if set(values) != {"E1", "E2"}:
            raise RuntimeError(f"Unexpected SAD face fields: {values}")
        actual = np.array([values["E1"], values["E2"]])
        faults.append(discrepancy(actual, [0.8125, -0.125], [-0.125, 0.8125], 1e-15))
        print(
            f"SAD reflected SBEN: face fractions={actual.tolist()}, expected=[0.8125, -0.125] (selected face angles/ANGLE)"
        )
        with tempfile.TemporaryDirectory(prefix="unsupported-export-") as tmp:
            work = Path(tmp)
            (work / "input.lte").write_text(
                "A:TAPERAPR,L=.1,XSTART=.014,XEND=.009\nU:LINE=(A)\nBL:LINE=(-U)\nUSE,BL\n"
            )
            result = subprocess.run(
                ["elegantto", "input.lte", "output.mad", "-mad8"],
                cwd=work,
                capture_output=True,
                text=True,
                check=True,
                timeout=290,
            )
            if "not implemented---translated to MARK" not in result.stdout + result.stderr:
                raise RuntimeError("Unsupported-aperture export warning is missing")
        print(
            "unsupported TAPERAPR: warning='not implemented---translated to MARK', expected explicit warning (converter contract)"
        )
        native.verdict(any(faults))


    if __name__ == "__main__":
        main()

    ```

## Provenance

| Role | Model | Effort | Date | Code examined |
|---|---|---|---|---|
| Found | `gpt-6-astra` | xhigh | 2026-11-06 | master (`0355dd6aca12`), source with SDDS a3f730a14c49, CPU, double, serial, one thread |
| Confirmed | `gpt-6.1-sol` | xhigh | 2026-11-06 | master (`0355dd6aca12`), source with SDDS a3f730a14c49, CPU, double, serial, one thread |

