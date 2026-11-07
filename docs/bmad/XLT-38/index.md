---
title: "XLT-38: SciBmad export reuses one definition for distinct elements whose generated names collide"
hide:
  - toc
---

# XLT-38 — SciBmad export reuses one definition for distinct elements whose generated names collide

| | |
|---|---|
| Code | [Bmad](../index.md) · Translators |
| Priority | P2 |
| Failure | Wrong physics |
| Evidence | Reproduced, independently rechecked |
| Last reproduced | 2026-11-06 on Bmad release [20261031-0](https://github.com/bmad-sim/bmad-ecosystem/tree/20261031-0) (`c4c629f`) |
| Tags | <span class="bb-tag">partly-source-only</span> [?](../../methodology.md#tags) |
| Source | [writer contract, bmad/output/write_lattice_scibmad_format.f90:4–13](https://github.com/bmad-sim/bmad-ecosystem/blob/c4c629fc9071d3a56cb1ba5bc714baba65e9c5d3/bmad/output/write_lattice_scibmad_format.f90#L4-L13)<br>[unique-name step, :166–169](https://github.com/bmad-sim/bmad-ecosystem/blob/c4c629fc9071d3a56cb1ba5bc714baba65e9c5d3/bmad/output/write_lattice_scibmad_format.f90#L166-L169)<br>[this_create_unique_ele_names, :1472–1560](https://github.com/bmad-sim/bmad-ecosystem/blob/c4c629fc9071d3a56cb1ba5bc714baba65e9c5d3/bmad/output/write_lattice_scibmad_format.f90#L1472-L1560)<br>[ele_struct%name, bmad/modules/bmad_struct.f90:1432–1433](https://github.com/bmad-sim/bmad-ecosystem/blob/c4c629fc9071d3a56cb1ba5bc714baba65e9c5d3/bmad/modules/bmad_struct.f90#L1432-L1433)<br>[scibmad_ele_name, :888–910](https://github.com/bmad-sim/bmad-ecosystem/blob/c4c629fc9071d3a56cb1ba5bc714baba65e9c5d3/bmad/output/write_lattice_scibmad_format.f90#L888-L910)<br>[definition loop, :231–259](https://github.com/bmad-sim/bmad-ecosystem/blob/c4c629fc9071d3a56cb1ba5bc714baba65e9c5d3/bmad/output/write_lattice_scibmad_format.f90#L231-L259)<br>[add_this_name_to_list, bmad/output/write_lattice_file_mod.f90:489–516](https://github.com/bmad-sim/bmad-ecosystem/blob/c4c629fc9071d3a56cb1ba5bc714baba65e9c5d3/bmad/output/write_lattice_file_mod.f90#L489-L516)<br>[branch lines, :451–467](https://github.com/bmad-sim/bmad-ecosystem/blob/c4c629fc9071d3a56cb1ba5bc714baba65e9c5d3/bmad/output/write_lattice_scibmad_format.f90#L451-L467)<br>[write_scibmad_element, :863–883](https://github.com/bmad-sim/bmad-ecosystem/blob/c4c629fc9071d3a56cb1ba5bc714baba65e9c5d3/bmad/output/write_lattice_scibmad_format.f90#L863-L883)<br>[element names, bmad/doc/lattice-file.tex:219–232](https://github.com/bmad-sim/bmad-ecosystem/blob/c4c629fc9071d3a56cb1ba5bc714baba65e9c5d3/bmad/doc/lattice-file.tex#L219-L232)<br>[per-occurrence attributes, bmad/doc/lattice-file.tex:374–379](https://github.com/bmad-sim/bmad-ecosystem/blob/c4c629fc9071d3a56cb1ba5bc714baba65e9c5d3/bmad/doc/lattice-file.tex#L374-L379)<br>[line tagging, bmad/doc/lines-and-lists.tex:328–364](https://github.com/bmad-sim/bmad-ecosystem/blob/c4c629fc9071d3a56cb1ba5bc714baba65e9c5d3/bmad/doc/lines-and-lists.tex#L328-L364)<br>[bmad_to_scibmad driver, util_programs/bmad_to_scibmad/bmad_to_scibmad.f90:38–58](https://github.com/bmad-sim/bmad-ecosystem/blob/c4c629fc9071d3a56cb1ba5bc714baba65e9c5d3/util_programs/bmad_to_scibmad/bmad_to_scibmad.f90#L38-L58)<br>[foreign-format dispatcher, bmad/output/write_lattice_foreign_format.f90:75–76](https://github.com/bmad-sim/bmad-ecosystem/blob/c4c629fc9071d3a56cb1ba5bc714baba65e9c5d3/bmad/output/write_lattice_foreign_format.f90#L75-L76)<br>[Tao `write scibmad`, tao/code/tao_write_cmd.f90:636–660](https://github.com/bmad-sim/bmad-ecosystem/blob/c4c629fc9071d3a56cb1ba5bc714baba65e9c5d3/tao/code/tao_write_cmd.f90#L636-L660) |

Exporting a Bmad lattice to SciBmad silently replaces an element whose exported name equals another element's. The writer renames differing same-name elements to name_n1, name_n2, …, rewrites '.' and '#' as '_' and '_s', renames the beginning and end markers and truncates names at 40 characters, without checking the results against existing names. It then writes one definition per name and reuses it at every colliding position. Compared with the parsed Bmad lattice, quadrupoles q.a (k₁ = 0.4 m⁻²) and q_a (k₁ = −0.7 m⁻²) export as two copies of the first, and a tagged line t@cell followed by a 3 m drift t_a shrinks from 4.250 m to 1.500 m. Renaming the colliding element restores the lattice.

Run `python XLT-38.py` with [_bmad.py](../helpers/_bmad.py) in the same directory.

??? example "Reproducer: XLT-38.py"

    [Download XLT-38.py](XLT-38.py)

    ```python
    """XLT-38: SciBmad export reuses one definition for distinct elements whose generated names collide

    Exporting a Bmad lattice to SciBmad silently replaces an element whose exported name equals another element's. The writer renames differing same-name elements to name_n1, name_n2, …, rewrites '.' and '#' as '_' and '_s', renames the beginning and end markers and truncates names at 40 characters, without checking the results against existing names. It then writes one definition per name and reuses it at every colliding position. Compared with the parsed Bmad lattice, quadrupoles q.a (k₁ = 0.4 m⁻²) and q_a (k₁ = −0.7 m⁻²) export as two copies of the first, and a tagged line t@cell followed by a 3 m drift t_a shrinks from 4.250 m to 1.500 m. Renaming the colliding element restores the lattice.

    The script links an inline client against the production build selected by BMAD_DIST that
    parses each deck, prints the parsed element sequence, and calls the public SciBmad writer.
    It re-reads the emitted file, resolves every branch reference to its single definition, and
    compares kind, length and k1 position by position with the parsed lattice. Renamed decks
    are passing controls, and the writer's error flag must stay false for a silent defect.
    """

    from __future__ import annotations

    import os

    for _name in (
        "OMP_NUM_THREADS",
        "MKL_NUM_THREADS",
        "OPENBLAS_NUM_THREADS",
        "NUMEXPR_NUM_THREADS",
        "NPROC",
        "PJRT_NPROC",
    ):
        os.environ[_name] = "1"

    import re

    from _bmad import compare, fortran, verdict

    PROGRAM = r"""program scibmad_names_client
    use bmad
    implicit none
    type(lat_struct), target :: lat
    type(ele_struct), pointer :: ele
    integer i
    logical err

    call bmad_parser('case.bmad', lat, err_flag=err)
    if (err) error stop 'parse'
    do i = 0, lat%n_ele_track
      ele => lat%ele(i)
      write(*, '(a,1x,a,1x,a,2(1x,es25.16))') 'ELE', trim(ele%name), trim(key_name(ele%key)), &
                                               ele%value(l$), ele%value(k1$)
    enddo
    err = .true.
    call write_lattice_scibmad_format('out.jl', lat, err)
    write(*, '(a,1x,l1)') 'ERR_FLAG', err
    write(*, '(a)') 'CLIENT COMPLETED'
    end program
    """

    HEADER = """no_digested
    parameter[particle] = proton
    parameter[p0c] = 1e9
    parameter[geometry] = open
    """

    LONG = "w" * 39  # 39 characters; Bmad names hold 40, so "_n1" and "_n2" both truncate to "_"


    def deck(*lines: str) -> str:
        return "\n".join(lines) + "\n"


    QUADS = ("q.a: quadrupole, l=0.5, k1=0.4", "d: drift, l=1")
    REPEATED = ("q: quadrupole, l=0.5, k1=0.4", "d: drift, l=1")
    CHANGE_SECOND_Q = ("use, route", "expand_lattice", "q##2[k1] = 0.6")

    # (label, collision expected, deck body after the header)
    CASES = (
        (
            "period",
            True,
            deck(
                *QUADS, "q_a: quadrupole, l=0.5, k1=-0.7", "route: line = (q.a, d, q_a)", "use, route"
            ),
        ),
        (
            "period control",
            False,
            deck(
                *QUADS, "q_b: quadrupole, l=0.5, k1=-0.7", "route: line = (q.a, d, q_b)", "use, route"
            ),
        ),
        (
            "period drift",
            True,
            deck(*QUADS, "q_a: drift, l=2", "route: line = (q.a, d, q_a)", "use, route"),
        ),
        (
            "tag",
            True,
            deck(
                "a: drift, l=0.25",
                "b: drift, l=1",
                "t_a: drift, l=3",
                "cell: line = (a, b)",
                "route: line = (t@cell, t_a)",
                "use, route",
            ),
        ),
        (
            "tag control",
            False,
            deck(
                "a: drift, l=0.25",
                "b: drift, l=1",
                "u_a: drift, l=3",
                "cell: line = (a, b)",
                "route: line = (t@cell, u_a)",
                "use, route",
            ),
        ),
        (
            "suffix",
            True,
            deck(
                *REPEATED,
                "q_n1: quadrupole, l=0.5, k1=-0.7",
                "route: line = (q, d, q_n1, d, q)",
                *CHANGE_SECOND_Q,
            ),
        ),
        (
            "suffix control",
            False,
            deck(
                *REPEATED,
                "qx: quadrupole, l=0.5, k1=-0.7",
                "route: line = (q, d, qx, d, q)",
                *CHANGE_SECOND_Q,
            ),
        ),
        (
            "hash",
            True,
            deck(
                "a#2: drift, l=0.25",
                "b: drift, l=1",
                "a_s2: drift, l=3",
                "route: line = (a#2, b, a_s2)",
                "use, route",
            ),
        ),
        (
            "end alias",
            True,
            deck("end_b1: drift, l=3", "b: drift, l=1", "route: line = (end_b1, b)", "use, route"),
        ),
        (
            "length limit",
            True,
            deck(
                f"{LONG}: drift, l=0.25",
                "b: drift, l=1",
                f"route: line = ({LONG}, b, {LONG})",
                "use, route",
                "expand_lattice",
                "3[l] = 3",
            ),
        ),
    )

    TYPES = {"Drift": "DRIFT", "Quadrupole": "QUADRUPOLE", "Marker": "MARKER"}


    def native_and_export(deck: str) -> tuple[list[tuple[str, str, float, float]], bool, str]:
        """Parse and export in one fresh process; native (name, key, L, k1) and the emitted file."""
        result = fortran(PROGRAM, files={"case.bmad": HEADER + deck}, outputs=["out.jl"])
        lines = result.log.splitlines()
        if lines.count("CLIENT COMPLETED") != 1:
            raise RuntimeError(f"Missing native completion marker:\n{result.log}")
        native = []
        flag = None
        for line in lines:
            words = line.split()
            if words[:1] == ["ELE"]:
                native.append((words[1], words[2], float(words[3]), float(words[4])))
            elif words[:1] == ["ERR_FLAG"]:
                flag = words[1] == "T"
        if flag is None or not native:
            raise RuntimeError(f"Incomplete native output:\n{result.log}")
        return native, flag, result.outputs["out.jl"].decode()


    def reread(text: str) -> list[tuple[str, str, float, float]]:
        """Resolve each Branch reference of the emitted file to its single @elements definition."""
        block = re.search(r"^@elements begin\n(.*?)^end\s+# @elements", text, re.DOTALL | re.MULTILINE)
        branches = re.findall(r"Branch\(\[(.*?)\]", text, re.DOTALL)
        if block is None or len(branches) != 1:
            raise RuntimeError(f"Unexpected SciBmad file layout:\n{text}")
        definitions: dict[str, tuple[str, dict[str, float]]] = {}
        for line in block[1].splitlines():
            match = re.fullmatch(r"\s*(\w+) = (\w+)\((.*)\)\s*", line)
            if match is None:
                raise RuntimeError(f"Unparsed definition line: {line!r}")
            name, kind, args = match.groups()
            if name in definitions:
                raise RuntimeError(f"Name defined twice: {name}")
            values = {k: float(v) for k, v in re.findall(r"(\w+) =\s+([-+0-9.Ee]+)(?=,|$)", args)}
            definitions[name] = (kind, values)
        loaded = []
        for name in (n.strip() for n in branches[0].split(",")):
            kind, values = definitions[name]
            loaded.append((name, TYPES[kind], values.get("L", 0.0), values.get("Kn1", 0.0)))
        return loaded


    def main() -> None:
        defect = False
        for label, collision, deck in CASES:
            native, flag, text = native_and_export(deck)
            loaded = reread(text)
            if len(loaded) != len(native):
                raise RuntimeError(
                    f"{label}: exported branch has {len(loaded)} of {len(native)} positions"
                )
            # The expected value is Bmad's own parsed lattice, position by position. SciBmad has
            # one Marker type for the beginning element, the end element and ordinary markers.
            want_kinds = [
                "MARKER" if k.upper() == "BEGINNING_ELE" else k.upper() for _, k, _, _ in native
            ]
            got_kinds = [k for _, k, _, _ in loaded]
            want = [v for _, _, length, k1 in native for v in (length, k1)]
            want.append(sum(length for _, _, length, _ in native))
            got = [v for _, _, length, k1 in loaded for v in (length, k1)]
            got.append(sum(length for _, _, length, _ in loaded))
            print(
                f"{label}: exported references {[n for n, *_ in loaded]}, err_flag={flag}; "
                f"kinds observed={got_kinds} expected={want_kinds}"
            )
            failed = compare(
                f"{label}: per-position (L/m, k1/m^-2) then total length/m (native parse)",
                got,
                want,
                atol=0,
            )
            failed |= got_kinds != want_kinds
            if collision:
                # A silent defect: a definition is lost while the writer reports success.
                defect |= failed and not flag
            elif failed or flag:
                raise RuntimeError(f"Control failed: {label}")
        verdict(defect)


    if __name__ == "__main__":
        main()

    ```

## Provenance

| Role | Model | Effort | Date | Code examined |
|---|---|---|---|---|
| Found | `claude-opus-5-5` | high | 2026-11-06 | 20261031-0 (`c4c629fc9071d3a56cb1ba5bc714baba65e9c5d3`), source, production, GNU Fortran 15.3.0, CPU, double, OpenMP one thread, MPI disabled |
| Confirmed | `gpt-6.1-sol` | xhigh | 2026-11-06 | 20261031-0 (`c4c629fc9071d3a56cb1ba5bc714baba65e9c5d3`), source, production, GNU Fortran 15.3.0, CPU, double, OpenMP one thread, MPI disabled |
| Confirmed<br><small>public-writer export re-read against the parsed lattice</small> | `claude-opus-5-5` | high | 2026-11-06 | 20261031-0 (`c4c629fc9071d3a56cb1ba5bc714baba65e9c5d3`), source, production, GNU Fortran 15.3.0, CPU, double, OpenMP one thread, MPI disabled |

