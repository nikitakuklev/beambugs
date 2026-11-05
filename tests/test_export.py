"""Exercise shared-helper publication from a ledger through generated downloads."""

import json
from pathlib import Path

import yaml

from tools import export


def snapshot(root: Path) -> dict[str, bytes]:
    return {
        str(path.relative_to(root)): path.read_bytes()
        for path in root.rglob("*")
        if path.is_file()
    }


def test_shared_helpers(tmp_path, monkeypatch):
    monkeypatch.setattr(export, "ROOT", tmp_path)
    monkeypatch.setattr(export, "DOCS", tmp_path / "docs")
    codes = {}
    for code in ("alpha", "beta"):
        ledger = tmp_path / code
        repro = ledger / "repro"
        repro.mkdir(parents=True)
        (ledger / "filed.md").write_text("")
        (repro / "_shared.py").write_text(
            f'"""Shared {code} setup."""\nimport _nested\n'
        )
        (repro / "_nested.py").write_text('"""Transitive helper."""\nimport _shared\n')
        (repro / "_extra.py").write_text('"""An additional helper."""\n')
        (repro / "_unused.py").write_text('"""Unused helper."""\n')
        rows, entries = [], []
        for number in (1, 2):
            entry_id = f"TRK-{number:02}"
            rows.append(
                f"| {entry_id} | P2 | Example | Reproduced, independently rechecked"
                " | Wrong physics | |"
            )
            entries.append(
                f"## {entry_id} — Example\n\n**Summary.** Example summary.\n\n"
                "**Provenance.**\n```yaml\n"
                "found:\n  - model: first\n    date: 2026-01-01\n"
                "    code: {ref: release, commit: abc123}\n"
                "confirmed:\n  - model: second\n    date: 2026-01-02\n"
                "    code: {ref: release, commit: abc123}\n```\n"
            )
            (repro / f"{entry_id}.py").write_text(
                f'"""{entry_id}: Example\n\nExample summary.\n\nDemonstrate it."""\n'
                "import _shared\n" + ("import _extra\n" if number == 2 else "")
            )
        (ledger / "tracking.md").write_text("\n".join(rows + entries))
        codes[code] = {
            "name": code,
            "homepage": "https://example.org",
            "repository": f"example/{code}",
            "ledger": str(ledger),
            "areas": {"tracking.md": "Tracking"},
            "modules": [],
            "run": "Run the downloaded script with its helpers.",
        }
    (tmp_path / "codes.yaml").write_text(yaml.safe_dump(codes))

    export.main(list(codes))
    docs = tmp_path / "docs"
    records = json.loads((docs / "defects.json").read_text())
    assert len(records) == 4
    for record in records:
        code, entry_id = record["code"], record["id"]
        page = docs / record["page"]
        expected = [f"{entry_id}.py", "../helpers/_shared.py", "../helpers/_nested.py"]
        if entry_id == "TRK-02":
            expected.append("../helpers/_extra.py")
        assert set(record["reproducers"]) == set(expected)
        assert not list(page.glob("_*.py"))
        markdown = (page / "index.md").read_text()
        helper_page = (docs / code / "helpers/index.md").read_text()
        for name in expected:
            source = tmp_path / code / "repro" / Path(name).name
            assert (page / name).read_bytes() == source.read_bytes()
            assert f"]({name})" in markdown
            if name.startswith("../"):
                assert f"../helpers/index.md#{source.stem}" in markdown
                assert f"{{ #{source.stem} }}" in helper_page
                assert source.read_text().splitlines()[0] in helper_page
                assert source.read_text().splitlines()[0] not in markdown
        assert not (docs / code / "helpers/_unused.py").exists()
        assert len(list((docs / code).rglob("_shared.py"))) == 1

    before_check = snapshot(docs)
    export.main(["--check", *codes])
    assert snapshot(docs) == before_check

    # Regeneration must remove helpers no entry imports, preserving the other code.
    script = tmp_path / "alpha/repro/TRK-02.py"
    script.write_text(script.read_text().replace("import _extra\n", ""))
    beta_before = snapshot(docs / "beta")
    export.main(["alpha"])
    assert not (docs / "alpha/helpers/_extra.py").exists()
    assert "_extra.py" not in (docs / "alpha/helpers/index.md").read_text()
    assert snapshot(docs / "beta") == beta_before
