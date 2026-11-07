---
title: "LAT-16: nonlinear-twiss.ipynb Python example has a syntax error"
hide:
  - toc
---

# LAT-16 — `nonlinear-twiss.ipynb` Python example has a syntax error

| | |
|---|---|
| Code | [SciBmad](../index.md) · Lattice and API |
| Priority | P4 |
| Failure | Documentation or example mismatch |
| Evidence | Reproduced, independently rechecked |
| Last reproduced | 2026-10-31 on SciBmad commit [`ae9423d`](https://github.com/bmad-sim/SciBmad.jl/commit/ae9423d703b4), committed 2026-10-28 |
| Source | [`SciBmad.jl/examples/python/nonlinear-twiss.ipynb:333`](https://github.com/bmad-sim/SciBmad.jl/blob/ae9423d703b4e612ae66e9deb0546aa79f62a340/examples/python/nonlinear-twiss.ipynb#L333) |

The Python example notebook `nonlinear-twiss.ipynb` contains `nparray(tw2.dbeta1 / tw2.beta1` without a closing parenthesis, in notebook cell 15 counting all cells from zero (code cell 8 counting only code cells). Parsing the notebook's code cells reports `SyntaxError: '(' was never closed`, and an independent count finds one opening and no closing parenthesis on the offending line. It is the only syntax error found by that parser check.

Run `julia LAT-16.jl`.

??? example "Reproducer: LAT-16.jl"

    [Download LAT-16.jl](LAT-16.jl)

    ```julia
    # LAT-16: `nonlinear-twiss.ipynb` Python example has a syntax error
    #
    # The Python example notebook `nonlinear-twiss.ipynb` contains `nparray(tw2.dbeta1 / tw2.beta1`
    # without a closing parenthesis, in notebook cell 15 counting all cells from zero (code cell 8
    # counting only code cells). Parsing the notebook's code cells reports `SyntaxError: '(' was never
    # closed`, and an independent count finds one opening and no closing parenthesis on the offending
    # line. It is the only syntax error found by that parser check.
    #
    # Setup: `SCIBMAD_SOURCE` set to a clone of bmad-sim/SciBmad.jl at the pinned commit `5f0fb83`, and
    # `python3` on `PATH`; the script reads an example notebook from the clone and parses its cells with
    # CPython.
    #
    # The script parses every code cell of the notebook with CPython's `ast` module, after dropping
    # IPython magic and shell lines, and counts the parentheses on the offending raw notebook line;
    # every code cell of a shipped example should parse.
    haskey(ENV, "SCIBMAD_SOURCE") || error("set SCIBMAD_SOURCE to a SciBmad.jl checkout (commit 5f0fb83 or the build under test)")
    nb = joinpath(ENV["SCIBMAD_SOURCE"], "examples", "python", "nonlinear-twiss.ipynb")
    isfile(nb) || error("notebook not found at $nb")
    py = raw"""
    import ast, json, sys
    nb = json.load(open(sys.argv[1]))
    bad = 0
    for i, c in enumerate(nb["cells"]):
        if c["cell_type"] != "code":
            continue
        src = "".join(c["source"])
        src = "\n".join(l for l in src.splitlines() if not l.lstrip().startswith(("%", "!")))
        try:
            ast.parse(src)
        except SyntaxError as e:
            bad += 1
            print(f"code cell index {i}: SyntaxError: {e.msg}; line: {src.splitlines()[e.lineno-1].strip()}")
    print(f"cells with syntax errors: {bad}")
    """
    out = read(`python3 -I -c $py $nb`, String)
    print(out)
    line = readlines(nb)[333]
    println("notebook file line 333: ", strip(line), "  '(' count = ", count(==('('), line), ", ')' count = ", count(==(')'), line))
    println("expected: every code cell of a shipped example parses")
    println(occursin("cells with syntax errors: 0", out) ? "NOT REPRODUCED" : "DEFECT REPRODUCED")

    ```

## Provenance

| Role | Model | Effort | Date | Code examined |
|---|---|---|---|---|
| Found | `gpt-6-astra` | xhigh | 2026-10-22 | SciBmad 0.6.0 (`62399f930847`), SciBmad 0.6.0 release packages, Julia 1.13.2, CPU |
| Confirmed | `gpt-6-sol` | xhigh | 2026-10-22 | SciBmad 0.6.0 (`62399f930847`), SciBmad 0.6.0 release packages, Julia 1.13.2, CPU |
| Confirmed | `claude-opus-5-5` | high | 2026-10-22 | SciBmad 0.6.0 (`62399f930847`), SciBmad 0.6.0 release packages, Julia 1.13.2, CPU |

