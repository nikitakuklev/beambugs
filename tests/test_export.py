"""Exercise shared-helper publication from a ledger through generated downloads."""

import json
from pathlib import Path

import yaml
from tools import export

RUNS = (
    "found:\n  - model: first\n    date: 2026-01-02\n"
    "    code: {{ref: release, commit: {commit}}}\n"
    "confirmed:\n  - model: second\n    date: 2026-01-02\n"
    "    code: {{ref: release, commit: {commit}}}\n"
)
VERIFIED = (
    "verified:\n  date: 2026-01-03\n"
    "  code: {{ref: main, commit: head01}}\n  result: {result}\n"
)


def snapshot(root: Path) -> dict[str, bytes]:
    return {
        str(path.relative_to(root)): path.read_bytes()
        for path in root.rglob("*")
        if path.is_file()
    }


def seed_commits(root: Path, commits: dict[str, dict]) -> None:
    """Cached upstream facts, so the exporter needs no GitHub lookups."""
    (root / "tools").mkdir(exist_ok=True)
    (root / "tools" / "commits.json").write_text(json.dumps(commits))


def commit(committed: str, tip_seen: str | None = None) -> dict:
    return {
        "sha": "0" * 40,
        "committed": committed,
        "tip_seen": tip_seen,
        "superseded": tip_seen is None,
    }


def config(code: str, ledger: Path) -> dict:
    return {
        "name": code,
        "homepage": "https://example.org",
        "repository": f"example/{code}",
        "branch": "main",
        "repositories": {},
        "ledger": str(ledger),
        "areas": {"tracking.md": "Tracking"},
        "modules": [],
        "run": "Run the downloaded script with its helpers.",
    }


def provenance(runs: str, verified: str = "") -> str:
    return f"**Provenance.**\n```yaml\n{runs}{verified}```\n"


def isolate(tmp_path: Path, monkeypatch, published: set[str] = frozenset()) -> None:
    """Export into tmp_path, with `published` as the IDs on the public site."""
    monkeypatch.setattr(export, "ROOT", tmp_path)
    monkeypatch.setattr(export, "DOCS", tmp_path / "docs")
    monkeypatch.setattr(export, "published_ids", lambda code: set(published))


def test_shared_helpers(tmp_path, monkeypatch):
    isolate(tmp_path, monkeypatch)
    seed_commits(
        tmp_path,
        {f"example/{code}": {"abc123": commit("2026-01-01")} for code in ("alpha", "beta")},
    )
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
                + provenance(RUNS.format(commit="abc123"))
            )
            (repro / f"{entry_id}.py").write_text(
                f'"""{entry_id}: Example\n\nExample summary.\n\nDemonstrate it."""\n'
                "import _shared\n" + ("import _extra\n" if number == 2 else "")
            )
        (ledger / "tracking.md").write_text("\n".join(rows + entries))
        codes[code] = config(code, ledger)
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


def test_current_run_gate(tmp_path, monkeypatch):
    """Publication needs a head run or a commit at most CURRENT_DAYS old on the run date."""
    isolate(tmp_path, monkeypatch)
    seed_commits(
        tmp_path,
        {
            "example/alpha": {
                "fresh1": commit("2025-12-26"),  # 7 days old on 2026-01-02
                "stale1": commit("2025-12-25"),
                "tip001": commit("2025-12-01", tip_seen="2026-01-05"),
                "head01": commit("2026-01-03"),
            }
        },
    )
    cases = {
        "TRK-01": provenance(RUNS.format(commit="fresh1")),
        "TRK-02": provenance(RUNS.format(commit="stale1")),
        "TRK-03": provenance(RUNS.format(commit="tip001")),
        "TRK-04": provenance(RUNS.format(commit="stale1"), VERIFIED.format(result="reproduced")),
        "TRK-05": provenance(RUNS.format(commit="stale1"), VERIFIED.format(result="not reproduced")),
    }
    ledger = tmp_path / "alpha"
    (ledger / "repro").mkdir(parents=True)
    (ledger / "filed.md").write_text("")
    rows, entries = [], []
    for entry_id, block in cases.items():
        rows.append(
            f"| {entry_id} | P2 | Example | Reproduced, independently rechecked | Wrong physics | |"
        )
        entries.append(
            f"## {entry_id} — Example\n\n**Summary.** Example summary.\n\n"
            f"**Justification.** Example.\n\n{block}"
        )
    (ledger / "tracking.md").write_text("\n".join(rows + entries))
    (tmp_path / "codes.yaml").write_text(yaml.safe_dump({"alpha": config("alpha", ledger)}))

    export.main(["alpha"])
    published = json.loads((tmp_path / "docs/defects.json").read_text())
    records = {r["id"]: r["last_reproduced"] for r in published}
    assert set(records) == {"TRK-01", "TRK-03", "TRK-04", "TRK-05"}
    assert records["TRK-01"]["code"]["committed"] == "2025-12-26"
    assert not records["TRK-01"]["head"]
    assert records["TRK-03"]["head"]
    assert records["TRK-04"]["date"] == "2026-01-03" and records["TRK-04"]["head"]
    assert records["TRK-05"]["code"]["commit"] == "stale1"
    assert records["TRK-05"]["head_not_reproduced"]["code"]["commit"] == "head01"


def test_published_entries_stay(tmp_path, monkeypatch):
    """An entry on the public site may not leave it, unless codes.yaml lists it as merged;
    a filed.md row the parser cannot attribute is an error, not a dropped `filed` link."""
    seed_commits(tmp_path, {"example/alpha": {"abc123": commit("2026-01-01"), "stale1": commit("2025-12-01")}})
    ledger = tmp_path / "alpha"
    (ledger / "repro").mkdir(parents=True)
    filed = ledger / "filed.md"
    rows, entries = [], []
    for entry_id, sha in (("TRK-01", "abc123"), ("TRK-02", "stale1")):
        rows.append(f"| {entry_id} | P2 | Example | Reproduced, independently rechecked | Wrong physics | |")
        entries.append(
            f"## {entry_id} — Example\n\n**Summary.** Example summary.\n\n"
            f"**Justification.** Example.\n\n{provenance(RUNS.format(commit=sha))}"
        )
    (ledger / "tracking.md").write_text("\n".join(rows + entries))
    codes = {"alpha": config("alpha", ledger)}
    (tmp_path / "codes.yaml").write_text(yaml.safe_dump(codes))

    def check(published: set[str]) -> str:
        isolate(tmp_path, monkeypatch, published)
        try:
            export.main(["--check", "alpha"])
        except export.LedgerError as err:
            return str(err)
        return ""

    filed.write_text("| — | example/alpha | https://github.com/example/alpha/issues/1 |\n")
    assert check({"TRK-01"}) == ""
    # TRK-02 fails the current-run gate; TRK-03 is gone from the ledger.
    err = check({"TRK-01", "TRK-02", "TRK-03"})
    assert "TRK-02 (no current run" in err and "TRK-03 (not in the ledger)" in err
    codes["alpha"]["merged"] = {"TRK-02": "TRK-01", "TRK-03": "TRK-01"}
    (tmp_path / "codes.yaml").write_text(yaml.safe_dump(codes))
    assert check({"TRK-01", "TRK-02", "TRK-03"}) == ""
    codes["alpha"]["merged"] = {"TRK-03": "TRK-02"}
    (tmp_path / "codes.yaml").write_text(yaml.safe_dump(codes))
    assert "point to unpublished ['TRK-02']" in check({"TRK-01"})

    filed.write_text("| TRK-01 | example/alpha | https://github.com/example/alpha/issues/1 |\n")
    assert "neither a linked ID nor —" in check(set())
