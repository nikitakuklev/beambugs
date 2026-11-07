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
