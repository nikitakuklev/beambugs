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
