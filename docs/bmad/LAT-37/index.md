---
title: "LAT-37: After expand_lattice, element references longer than 40 characters are rejected or silently truncated"
hide:
  - toc
---

# LAT-37 — After expand_lattice, element references longer than 40 characters are rejected or silently truncated

| | |
|---|---|
| Code | [Bmad](../index.md) · Lattices, particles and I/O |
| Priority | P3 |
| Severity | Silent wrong result |
| Evidence | Reproduced, independently rechecked |
| Last reproduced | 2026-11-07 on Bmad release [20261031-0](https://github.com/bmad-sim/bmad-ecosystem/tree/20261031-0) (`c4c629f`) |
| Tags | <span class="bb-tag">partly-source-only</span> [?](../../methodology.md#tags) |
| Discussed in | [[gh:https://github.com/bmad-sim/bmad-ecosystem/issues/1081]] |
| Source | [40-character element names, bmad/doc/lattice-file.tex:223–231](https://github.com/bmad-sim/bmad-ecosystem/blob/c4c629fc9071d3a56cb1ba5bc714baba65e9c5d3/bmad/doc/lattice-file.tex#L223-L231)<br>[post-expansion class, branch and occurrence references, bmad/doc/lattice-file.tex:312–325](https://github.com/bmad-sim/bmad-ecosystem/blob/c4c629fc9071d3a56cb1ba5bc714baba65e9c5d3/bmad/doc/lattice-file.tex#L312-L325)<br>[per-occurrence attribute assignments, bmad/doc/lattice-file.tex:374–379](https://github.com/bmad-sim/bmad-ecosystem/blob/c4c629fc9071d3a56cb1ba5bc714baba65e9c5d3/bmad/doc/lattice-file.tex#L374-L379)<br>[class-qualified wildcard assignments, bmad/doc/lattice-file.tex:505–516](https://github.com/bmad-sim/bmad-ecosystem/blob/c4c629fc9071d3a56cb1ba5bc714baba65e9c5d3/bmad/doc/lattice-file.tex#L505-L516)<br>[numeric, branch and range assignments, bmad/doc/lattice-file.tex:539–545](https://github.com/bmad-sim/bmad-ecosystem/blob/c4c629fc9071d3a56cb1ba5bc714baba65e9c5d3/bmad/doc/lattice-file.tex#L539-L545)<br>[the word_1 and name declarations, bmad/parsing/bmad_parser2.f90:59–60](https://github.com/bmad-sim/bmad-ecosystem/blob/c4c629fc9071d3a56cb1ba5bc714baba65e9c5d3/bmad/parsing/bmad_parser2.f90#L59-L60)<br>[initial read and reconstruction, bmad/parsing/bmad_parser2.f90:185–202](https://github.com/bmad-sim/bmad-ecosystem/blob/c4c629fc9071d3a56cb1ba5bc714baba65e9c5d3/bmad/parsing/bmad_parser2.f90#L185-L202)<br>[get_next_word, bmad/parsing/bmad_parser_mod.f90:327–333](https://github.com/bmad-sim/bmad-ecosystem/blob/c4c629fc9071d3a56cb1ba5bc714baba65e9c5d3/bmad/parsing/bmad_parser_mod.f90#L327-L333)<br>[word_read/set_this_word, sim_utils/string/word_read.f90:217–228](https://github.com/bmad-sim/bmad-ecosystem/blob/c4c629fc9071d3a56cb1ba5bc714baba65e9c5d3/sim_utils/string/word_read.f90#L217-L228)<br>[the multi-element classification and locator, bmad/parsing/bmad_parser2.f90:523–541](https://github.com/bmad-sim/bmad-ecosystem/blob/c4c629fc9071d3a56cb1ba5bc714baba65e9c5d3/bmad/parsing/bmad_parser2.f90#L523-L541)<br>[the missing-element diagnostic gate, bmad/parsing/bmad_parser2.f90:574–586](https://github.com/bmad-sim/bmad-ecosystem/blob/c4c629fc9071d3a56cb1ba5bc714baba65e9c5d3/bmad/parsing/bmad_parser2.f90#L574-L586)<br>[an 80-character word_1, bmad/parsing/bmad_parser.f90:77–77](https://github.com/bmad-sim/bmad-ecosystem/blob/c4c629fc9071d3a56cb1ba5bc714baba65e9c5d3/bmad/parsing/bmad_parser.f90#L77-L77)<br>[bmad_parser, bmad/parsing/bmad_parser.f90:1380–1385](https://github.com/bmad-sim/bmad-ecosystem/blob/c4c629fc9071d3a56cb1ba5bc714baba65e9c5d3/bmad/parsing/bmad_parser.f90#L1380-L1385) |

After expand_lattice, an attribute redefinition whose element reference is longer than 40 characters fails, although element names may themselves be 40 characters long. With a 38-character name, `name##2[k1] = 0.2`, the documented form for the second element of that name, stops parsing with "WORD ARGUMENT LENGTH (40) TOO SHORT"; 37-character names and numeric element indices work. With a class prefix the reference is cut to 40 characters without a message: `quadrupole::` followed by a 29-character name changes nothing, or sets k₁ of the quadrupole named by its first 28 characters. The same statement before expand_lattice sets the named element.

Run `python LAT-37.py` with [_bmad.py](../helpers/_bmad.py) in the same directory.

??? example "Reproducer: LAT-37.py"

    [Download LAT-37.py](LAT-37.py)

    ```python
    """LAT-37: After expand_lattice, element references longer than 40 characters are rejected or silently truncated

    After expand_lattice, an attribute redefinition whose element reference is
    longer than 40 characters fails, although element names may themselves be 40 characters long.
    With a 38-character name, `name##2[k1] = 0.2`, the documented form for the second element of
    that name, stops parsing with "WORD ARGUMENT LENGTH (40) TOO SHORT"; 37-character names and
    numeric element indices work. With a class prefix the reference is cut to 40 characters
    without a message: `quadrupole::` followed by a 29-character name changes nothing, or sets k₁
    of the quadrupole named by its first 28 characters. The same statement before expand_lattice
    sets the named element.

    The script compiles an inline parse client using BMAD_DIST and the shared runner.
    Each fixed-input deck runs in a fresh process and compares the returned quadrupole strengths
    with the manual's name, occurrence and inclusive-range matching rules.
    Boundary, numeric-index, unprefixed and pre-expansion controls distinguish reference truncation
    from invalid names; exact diagnostics identify the parser's zero-status early stop.
    """

    from __future__ import annotations

    from dataclasses import dataclass
    from pathlib import Path

    import _bmad

    PROGRAM = r"""program parser_reference
    use bmad
    implicit none
    type(lat_struct), target :: lat
    integer i
    logical err
    character(20) mode
    call get_command_argument(1, mode)
    if (mode == 'noexit') global_com%exit_on_error = .false.
    call bmad_parser('lat.bmad', lat, err_flag=err)
    write(*, '(a,1x,l1)') 'ERR_FLAG', err
    do i = 1, lat%n_ele_track
      if (lat%ele(i)%key /= quadrupole$) cycle
      write(*, '(a,1x,es25.16)') 'QUAD', lat%ele(i)%value(k1$)
    enddo
    write(*, '(a)') 'CLIENT COMPLETED'
    end program
    """


    @dataclass(frozen=True)
    class Case:
        label: str
        names: tuple[str, ...]
        target: str
        expected: tuple[float, ...]
        expand: bool = True
        noexit: bool = False
        control: bool = False


    def name(length: int) -> str:
        return "Q" + "A" * (length - 1)


    def deck(case: Case) -> str:
        definitions = "".join(
            f"{word}: quadrupole, l=0.5, k1=0.1\n" for word in dict.fromkeys(case.names)
        )
        line = ", DD, ".join(case.names)
        return (
            "no_digested\nparameter[particle]=proton\nparameter[p0c]=1e9\n"
            "parameter[geometry]=open\n" + definitions
            + f"DD: drift, l=1\nLN: line=({line})\nuse, LN\n"
            + ("expand_lattice\n" if case.expand else "")
            + f"{case.target}[k1]=0.2\n"
        )


    def cases() -> list[Case]:
        # Three occurrences have tracking indices 1, 3, 5. Only ##2 or index 3 selects
        # the middle one; a plain name selects all three, independently of its length.
        second = (0.1, 0.2, 0.1)
        all_three = (0.2, 0.2, 0.2)
        result = [
            Case(f"name{n}/##2", (name(n),) * 3, f"{name(n)}##2", second, control=n == 37)
            for n in (37, 38, 39, 40)
        ]
        for n in (37, 38, 40):
            result.append(Case(f"name{n}/index3", (name(n),) * 3, "3", second, control=True))
            result.append(Case(f"name{n}/plain", (name(n),) * 3, name(n), all_three, control=True))
        result.append(Case("name38/##2/noexit", (name(38),) * 3, f"{name(38)}##2", second, noexit=True))
        # The branch prefix costs three characters: length 34 fits, length 35 does not.
        for n in (34, 35):
            result.append(Case(f"name{n}/branch/##2", (name(n),) * 3, f"0>>{name(n)}##2",
                               second, control=n == 34))
        for n in (25, 26, 40):
            result.append(Case(f"name{n}/class/##2", (name(n),) * 3,
                               f"quadrupole::{name(n)}##2", second, control=n == 25))
        for n in (28, 29, 40):
            result.append(Case(f"name{n}/class", (name(n),) * 3,
                               f"quadrupole::{name(n)}", all_three, control=n == 28))
        result.append(Case("name40/class/pre-expansion", (name(40),) * 3,
                           f"quadrupole::{name(40)}", all_three, expand=False, control=True))
        prefix = name(28)
        pair = (prefix, prefix + "X")
        result.extend([
            Case("wrong-element/class", pair, f"quadrupole::{pair[1]}", (0.1, 0.2)),
            Case("wrong-element/pre-expansion", pair, f"quadrupole::{pair[1]}",
                 (0.1, 0.2), expand=False, control=True),
            Case("wrong-element/plain", pair, pair[1], (0.1, 0.2), control=True),
            Case("prefix28/class/control", pair, f"quadrupole::{prefix}", (0.2, 0.1), control=True),
            Case("name29/class/wildcard", (pair[1],) * 3, f"quadrupole::{prefix}*", all_three),
        ])
        for n in (19, 20):
            first, last = name(19) + "B", name(n - 1) + "C"
            # Inclusive first:last must set every quadrupole, including the middle QM.
            result.append(Case(f"range{21+n}", (first, "QM", last), f"{first}:{last}",
                               all_three, control=n == 19))
        return result


    def check(case: Case, log: str) -> bool:
        returned = "CLIENT COMPLETED" in log.splitlines()
        flags = [line.split()[1] for line in log.splitlines() if line.startswith("ERR_FLAG ")]
        observed = tuple(float(line.split()[1]) for line in log.splitlines() if line.startswith("QUAD "))
        diagnostics = ("ERROR IN", "[ERROR", "[FATAL", "[WARNING")
        short_word = all(text in log for text in (
            "WORD ARGUMENT LENGTH (40) TOO SHORT", "BAD WORD", "ELEMENT NOT FOUND",
        ))
        malformed = "MALFORMED ELEMENT(S) NAME" in log
        if not returned:
            if not (short_word or malformed) or "BMAD_PARSER FINISHED. EXITING ON ERRORS" not in log:
                raise RuntimeError(f"{case.label}: unexpected early termination:\n{log}")
            detail = "word-too-short+bad-word+element-not-found" if short_word else "malformed-range"
            observation = f"returned=False, diagnostics={detail}"
            defect = True
        else:
            if len(flags) != 1 or len(observed) != len(case.expected):
                raise RuntimeError(f"{case.label}: incomplete client output:\n{log}")
            if flags[0] == "T":
                if not case.noexit or not short_word:
                    raise RuntimeError(f"{case.label}: unexpected parser error:\n{log}")
            elif flags[0] != "F" or any(marker in log for marker in diagnostics):
                raise RuntimeError(f"{case.label}: unexpected diagnostic:\n{log}")
            # Both the parser and Python read the same literal 0.1/0.2 as binary64:
            # exact equality tests selection without introducing a numerical tolerance.
            defect = flags[0] != "F" or observed != case.expected
            observation = f"returned=True, err_flag={flags[0]}, k1={list(observed)}"
        print(f"{case.label} (reference length {len(case.target)}): status=0, {observation}; "
              f"expected err_flag=F, k1={list(case.expected)} m^-2 (manual matching)")
        if case.control and defect:
            raise RuntimeError(f"{case.label}: required control failed")
        return defect


    def main() -> None:
        compile_argv, link_argv = _bmad._recipe(False)
        defects = 0
        with _bmad._scratch() as temporary:
            work = Path(temporary)
            (work / "probe.f90").write_text(PROGRAM)
            _bmad._execute(compile_argv, work, "", _bmad.TIMEOUT)
            _bmad._execute(link_argv, work, "", _bmad.TIMEOUT)
            suite = cases()
            for i, case in enumerate(suite):
                args = [str(work / "probe"), "noexit" if case.noexit else "default"]
                result = _bmad._case(args, work / f"case-{i}", {"lat.bmad": deck(case)}, (), "", _bmad.TIMEOUT)
                defects += check(case, result.log)
        print(f"cases={len(suite)}; discrepancies={defects}; controls={sum(c.control for c in suite)} pass")
        _bmad.verdict(defects > 0)


    if __name__ == "__main__":
        main()

    ```

## Provenance

| Role | Model | Effort | Date | Code examined |
|---|---|---|---|---|
| Found | `gpt-6.1-sol` | xhigh | 2026-11-06 | 20261031-0 (`c4c629fc9071d3a56cb1ba5bc714baba65e9c5d3`), source, production, GNU Fortran 15.3.0, CPU, double, OpenMP one thread, MPI disabled |
| Confirmed | `claude-opus-5-5` | high | 2026-11-07 | 20261031-0 (`c4c629fc9071d3a56cb1ba5bc714baba65e9c5d3`), source, production, GNU Fortran 15.3.0, CPU, double, OpenMP one thread, MPI disabled |

