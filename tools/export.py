"""Generate docs/<code>/ from a private upstream-defect ledger.

Usage: python tools/export.py <code> [<code> ...]

The ledger location and area files come from codes.yaml. Everything under
docs/<code>/, docs/defects.json and docs/index.md is regenerated; do not edit
those files by hand. The ledger format is defined by the ledger's shared
procedure (defect_hunting.md): entries `## <ID> — <title>` with bold fields,
a YAML provenance block, an index table per area file and filed.md.
"""

from __future__ import annotations

import ast
import importlib.metadata
import json
import re
import shutil
import sys
from dataclasses import dataclass
from pathlib import Path

import jinja2
import yaml

ROOT = Path(__file__).resolve().parent.parent
DOCS = ROOT / "docs"
TEMPLATES = ROOT / "tools" / "templates"

PUBLISHED_LEVELS = {"Reproduced, independently rechecked", "Probe-pinned"}
GITHUB_REF = re.compile(r"https://github\.com/([\w.-]+/[\w.-]+)/(issues|pull)/(\d+)")
BLOB_LINK = re.compile(
    r"\[([^\]]+)\]\((https://github\.com/[\w.-]+/[\w.-]+/blob/[^)\s]+)\)"
)
ENTRY_HEADING = re.compile(r"^## ([A-Z]+-\d+) — (.+)$", re.MULTILINE)
FIELD = re.compile(r"^\*\*([A-Z][^*\n]*?)\.\*\*[ \t]*", re.MULTILINE)
YAML_BLOCK = re.compile(r"```yaml\n(.*?)\n```", re.DOTALL)
UPSTREAM_RELATIONS = {"report", "fix", "discussion"}
TAGS = {
    "experimental",
    "extreme-regime",
    "contract-unsettled",
    "known-limitation",
    "partly-source-only",
}


class LedgerError(Exception):
    pass


@dataclass
class Entry:
    code: str
    id: str
    title: str
    area: str
    priority: str
    level: str
    severity: str
    fields: dict[str, str]
    provenance: dict
    upstream: list[dict]
    tags: list[str]
    locations: list[dict]
    repro: list[Path]
    brief: str | None

    @property
    def models(self) -> set[str]:
        items = self.provenance.get("found", []) + self.provenance.get("confirmed", [])
        return {item["model"] for item in items if item.get("model")}

    def gate_failures(self) -> list[str]:
        failures = []
        if self.level not in PUBLISHED_LEVELS:
            failures.append(f"level {self.level!r}")
        if not self.repro and "Justification" not in self.fields:
            failures.append("no reproducer or Justification")
        if len(self.models) < 2:
            failures.append(f"models {sorted(self.models)}")
        if not self.fields.get("Summary"):
            failures.append("no Summary")
        return failures


def split_fields(body: str) -> dict[str, str]:
    marks = list(FIELD.finditer(body))
    fields = {}
    for i, m in enumerate(marks):
        end = marks[i + 1].start() if i + 1 < len(marks) else len(body)
        fields[m.group(1)] = body[m.end() : end].strip()
    return fields


def parse_index_table(text: str, path: Path) -> dict[str, list[str]]:
    rows = {}
    for line in text.splitlines():
        m = re.match(r"^\| *\[?([A-Z]+-\d+)\]?[^|]* *\|(.*)\|\s*$", line)
        if m:
            rows[m.group(1)] = [c.strip() for c in m.group(2).split("|")]
    if not rows:
        raise LedgerError(f"{path}: no index table rows")
    return rows


def parse_filed(path: Path) -> dict[str, list[tuple[str, str, int]]]:
    filed: dict[str, list[tuple[str, str, int]]] = {}
    for line in path.read_text().splitlines():
        m = re.match(r"^\| \[([A-Z]+-\d+)\]", line)
        if m:
            filed[m.group(1)] = [(r, k, int(n)) for r, k, n in GITHUB_REF.findall(line)]
    return filed


def link_record(url: str, relation: str) -> dict:
    m = GITHUB_REF.match(url)
    if not m:
        raise LedgerError(f"not a GitHub issue or pull request URL: {url}")
    repo, kind, number = m.groups()
    return {
        "repository": repo,
        "number": int(number),
        "kind": "pull" if kind == "pull" else "issue",
        "url": url,
        "relation": relation,
    }


def upstream_links(
    entry_id: str,
    index_cell: str,
    filed: list[tuple[str, str, int]],
    declared: list[dict],
) -> list[dict]:
    """Our filings from filed.md plus others' issues/PRs declared in the provenance block."""
    links = [
        link_record(f"https://github.com/{r}/{k}/{n}", "filed") for r, k, n in filed
    ]
    for item in declared:
        if item.get("relation") not in UPSTREAM_RELATIONS:
            raise LedgerError(
                f"{entry_id}: upstream relation {item.get('relation')!r} not in "
                f"{sorted(UPSTREAM_RELATIONS)}"
            )
        links.append(link_record(item["url"], item["relation"]))
    known = {(link["repository"], link["number"]) for link in links}
    for repo, _kind, number in GITHUB_REF.findall(index_cell):
        if (repo, int(number)) not in known:
            raise LedgerError(
                f"{entry_id}: index links {repo}#{number}, which is neither in "
                "filed.md nor in the provenance upstream list"
            )
    return links


def parse_tags(entry_id: str, field: str | None) -> list[str]:
    if not field:
        return []
    tags = [t.strip().strip("`") for t in field.split(",")]
    unknown = set(tags) - TAGS
    if unknown:
        raise LedgerError(
            f"{entry_id}: unknown tags {sorted(unknown)}; allowed {sorted(TAGS)}"
        )
    return tags


def parse_provenance(entry_id: str, field: str | None) -> dict:
    if field is None:
        return {}
    m = YAML_BLOCK.search(field)
    if not m:
        raise LedgerError(f"{entry_id}: Provenance field without a yaml block")
    data = yaml.safe_load(m.group(1))
    if not isinstance(data, dict):
        raise LedgerError(f"{entry_id}: Provenance block is not a mapping")
    for role in ("found", "confirmed"):
        if not isinstance(data.get(role, []), list):
            raise LedgerError(f"{entry_id}: provenance {role} must be a list")
    return data


def public_provenance(data: dict) -> dict:
    """Drop local session IDs, internal notes and evidence links."""

    def clean(item: dict) -> dict:
        return {k: v for k, v in item.items() if k not in ("session", "evidence")}

    out = {
        role: [clean(i) for i in data.get(role, [])] for role in ("found", "confirmed")
    }
    if "verified" in data:
        out["verified"] = clean(data["verified"])
    return json.loads(json.dumps(out, default=str))


def repro_files(ledger: Path, entry_id: str) -> list[Path]:
    repro = ledger / "repro"
    files = sorted(repro.glob(f"{entry_id}.py")) + sorted(
        repro.glob(f"{entry_id}-*.py")
    )
    if (repro / entry_id).is_dir():
        files.append(repro / entry_id)
    return files


def describe(entry_id: str, source: str, filename: str) -> str:
    """Script docstring as escaped Markdown paragraphs."""
    doc = ast.get_docstring(ast.parse(source, filename))
    if not doc:
        raise LedgerError(f"{filename}: reproducer has no docstring")
    paragraphs = [" ".join(p.split()) for p in doc.split("\n\n") if p.strip()]
    if paragraphs[0].startswith(f"{entry_id}:"):
        # The main script's first line restates the entry title; a secondary
        # script's first line names what it adds.
        if Path(filename).stem == entry_id:
            paragraphs = paragraphs[1:]
        else:
            first = paragraphs[0][len(entry_id) + 1 :].lstrip()
            paragraphs[0] = first[:1].upper() + first[1:]
    text = "\n\n".join(paragraphs)
    return re.sub(r"([\\`*_\[\]<>#|])", r"\\\1", text)


def requirements(own_modules: set[str], files: list[Path]) -> list[str]:
    """Distributions of the third-party modules the reproducers import, from this environment."""
    roots: set[str] = set()
    for path in files:
        sources = [path] if path.is_file() else sorted(path.rglob("*.py"))
        for src in sources:
            for node in ast.walk(ast.parse(src.read_text(), str(src))):
                if isinstance(node, ast.Import):
                    roots.update(a.name.split(".")[0] for a in node.names)
                elif (
                    isinstance(node, ast.ImportFrom) and node.module and node.level == 0
                ):
                    roots.add(node.module.split(".")[0])
    distributions = importlib.metadata.packages_distributions()
    needed = set()
    for module in roots - set(sys.stdlib_module_names) - own_modules:
        if module not in distributions:
            raise LedgerError(
                f"{files[0].name}: no installed distribution provides {module!r}"
            )
        needed.update(distributions[module])
    return sorted(needed)


def parse_ledger(code: str, config: dict) -> list[Entry]:
    ledger = Path(config["ledger"])
    filed = parse_filed(ledger / "filed.md")
    entries = []
    for filename, area in config["areas"].items():
        path = ledger / filename
        text = path.read_text()
        index = parse_index_table(text, path)
        headings = list(ENTRY_HEADING.finditer(text))
        if set(index) != {h.group(1) for h in headings}:
            raise LedgerError(
                f"{path}: index IDs {sorted(index)} differ from entry headings "
                f"{sorted(h.group(1) for h in headings)}"
            )
        for h in headings:
            entry_id, title = h.group(1), h.group(2).strip()
            nxt = re.compile(r"^## ", re.MULTILINE).search(text, h.end())
            body = text[h.end() : nxt.start() if nxt else len(text)]
            fields = split_fields(body)
            row = index[entry_id]
            if len(row) != 5:
                raise LedgerError(
                    f"{path}: index row {entry_id} has {len(row)} cells after the ID, expected 5"
                )
            priority, _finding, level, severity, on_development = row
            brief = None
            if fields.get("Brief public", "").lower().startswith("yes"):
                brief_path = ledger / "drafts" / entry_id / "brief.md"
                if not brief_path.is_file():
                    raise LedgerError(
                        f"{entry_id}: Brief public is yes but {brief_path} is missing"
                    )
                brief = brief_path.read_text()
            provenance = parse_provenance(entry_id, fields.get("Provenance"))
            entries.append(
                Entry(
                    code=code,
                    id=entry_id,
                    title=title,
                    area=area,
                    priority=priority,
                    level=level,
                    severity=severity,
                    fields=fields,
                    provenance=provenance,
                    upstream=upstream_links(
                        entry_id,
                        on_development,
                        filed.get(entry_id, []),
                        provenance.get("upstream", []),
                    ),
                    tags=parse_tags(entry_id, fields.get("Tags")),
                    locations=[
                        {"label": label, "url": url}
                        for label, url in BLOB_LINK.findall(
                            fields.get("Repository and location", "")
                        )
                    ],
                    repro=repro_files(ledger, entry_id),
                    brief=brief,
                )
            )
    return entries


def entry_record(entry: Entry, config: dict) -> dict:
    return {
        "code": entry.code,
        "id": entry.id,
        "title": entry.title,
        "area": entry.area,
        "priority": entry.priority,
        "severity": entry.severity,
        "level": entry.level,
        "tags": entry.tags,
        "summary": entry.fields["Summary"],
        "justification": entry.fields.get("Justification"),
        "locations": entry.locations,
        "upstream": entry.upstream,
        "provenance": public_provenance(entry.provenance),
        "models": sorted(entry.models),
        "reproducers": [p.name for p in entry.repro],
        "requirements": requirements(set(config["modules"]), entry.repro),
        "page": f"{entry.code}/{entry.id}/",
    }


def export(code: str, config: dict, env: jinja2.Environment) -> list[dict]:
    entries = parse_ledger(code, config)
    out = DOCS / code
    if out.exists():
        shutil.rmtree(out)
    out.mkdir(parents=True)
    records, excluded = [], []
    for entry in entries:
        failures = entry.gate_failures()
        if failures:
            excluded.append((entry.id, failures))
            continue
        record = entry_record(entry, config)
        page = out / entry.id
        page.mkdir()
        sources, descriptions = {}, {}
        for src in entry.repro:
            if src.is_dir():
                shutil.copytree(src, page / src.name)
            else:
                shutil.copy2(src, page / src.name)
                sources[src.name] = src.read_text()
                descriptions[src.name] = describe(entry.id, sources[src.name], str(src))
        source_env = config.get("source_env")
        notes = []
        extra = [r for r in record["requirements"] if r != "numpy"]
        if extra:
            notes.append(f"Requires {', '.join(extra)}.")
        if source_env and any(source_env in t for t in sources.values()):
            notes.append(
                f"Set `{source_env}` to a clone of the {config['name']} repository."
            )
        (page / "index.md").write_text(
            env.get_template("entry.md.j2").render(
                e=record,
                code=config,
                sources=sources,
                descriptions=descriptions,
                note=" ".join(notes),
                brief=entry.brief,
            )
        )
        records.append(record)
    (out / "index.md").write_text(
        env.get_template("code_index.md.j2").render(
            code=config,
            areas={
                a: [r for r in records if r["area"] == a]
                for a in config["areas"].values()
            },
        )
    )
    (out / "defects.json").write_text(
        json.dumps(records, indent=1, ensure_ascii=False) + "\n"
    )
    print(f"{code}: published {len(records)}, excluded {len(excluded)}")
    for entry_id, failures in excluded:
        print(f"  {entry_id}: {'; '.join(failures)}")
    return records


def main(argv: list[str]) -> None:
    if not argv:
        raise SystemExit(__doc__)
    codes = yaml.safe_load((ROOT / "codes.yaml").read_text())
    unknown = set(argv) - set(codes)
    if unknown:
        raise SystemExit(
            f"unknown codes {sorted(unknown)}; codes.yaml has {sorted(codes)}"
        )
    env = jinja2.Environment(
        loader=jinja2.FileSystemLoader(TEMPLATES),
        undefined=jinja2.StrictUndefined,
        keep_trailing_newline=True,
        trim_blocks=True,
        lstrip_blocks=True,
    )
    for code in argv:
        export(code, codes[code], env)
    published = {}
    for code in codes:
        path = DOCS / code / "defects.json"
        if path.is_file():
            published[code] = json.loads(path.read_text())
    (DOCS / "defects.json").write_text(
        json.dumps(
            [r for recs in published.values() for r in recs],
            indent=1,
            ensure_ascii=False,
        )
        + "\n"
    )
    (DOCS / "index.md").write_text(
        env.get_template("home.md.j2").render(codes=codes, published=published)
    )


if __name__ == "__main__":
    main(sys.argv[1:])
