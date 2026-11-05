"""Build the BBU probe with the pin's libbsim and the shared Bmad recipe."""

from pathlib import Path

from _bmad import Result, _case, _execute, _recipe, _scratch, build


def bbu_fortran(program: str, files: dict[str, str]) -> Result:
    compile_argv, link_argv = _recipe(False)
    library = build() / "production/lib/libbsim.a"
    if not library.is_file():
        raise RuntimeError(f"Missing BBU library: {library}")
    # Static libraries must precede the Bmad libraries that satisfy their references.
    link_argv.insert(link_argv.index("probe.o") + 1, str(library))
    with _scratch() as tmp:
        work = Path(tmp)
        (work / "probe.f90").write_text(program)
        _execute(compile_argv, work, "", 290)
        _execute(link_argv, work, "", 290)
        return _case([str(work / "probe")], work / "run", files, (), "", 290)
