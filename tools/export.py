"""Generate docs/<code>/ from a private upstream-defect ledger.

Usage: python tools/export.py [--check] <code> [<code> ...]

The ledger location and area files come from codes.yaml. Everything under
docs/<code>/, docs/defects.json and docs/index.md is regenerated; do not edit
those files by hand. --check parses the ledgers and reports the entries that
miss the publication gate without writing anything. The ledger format is
defined by the ledgers' shared procedure (defect_hunting.md): entries
`## <ID> — <title>` with bold fields, a YAML provenance block, an index table
per area file and filed.md.
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
BLOB_URL = re.compile(r"https://github\.com/[\w.-]+/[\w.-]+/blob/[^)\s]+")
BLOB_LINK = re.compile(rf"\[((?:`[^`]*`|[^\]`])+)\]\(({BLOB_URL.pattern})\)")
ENTRY_HEADING = re.compile(r"^## ([A-Z]+-\d+) — (.+)$", re.MULTILINE)
FIELD = re.compile(r"^\*\*([A-Z][^*\n]*?)\.\*\*[ \t]*", re.MULTILINE)
YAML_BLOCK = re.compile(r"```yaml\n(.*?)\n```", re.DOTALL)
UPSTREAM_RELATIONS = {"report", "fix", "discussion"}
# Reproducer languages: code-fence name and the line-comment marker that opens
# a non-Python script's description.
LANGUAGES = {".py": ("python", None), ".jl": ("julia", "#"), ".sh": ("bash", "#")}
TAGS = {
    "experimental",
    "extreme-regime",
    "contract-unsettled",
    "known-limitation",
    "partly-source-only",
}
# Severity vocabulary of defect_hunting.md, most severe first; the code pages list
# entries by priority (P0 first), then by this order.
SEVERITIES = (
    "Wrong physics",
    "Silent wrong result",
    "Crash or API hazard",
    "Documentation or example mismatch",
)
# A priority cell starts with P0..P5; a qualifier may follow ("P3 (spin P4)").
PRIORITY = re.compile(r"P([0-5])\b")


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
        elif not script_matches(self):
            failures.append("reproducer does not open with the title and Summary")
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


def source_links(entry_id: str, field: str) -> list[dict]:
    links = list(BLOB_LINK.finditer(field))
    captured = {link.span(2) for link in links}
    missing = [url.group() for url in BLOB_URL.finditer(field) if url.span() not in captured]
    if missing:
        raise LedgerError(f"{entry_id}: unparsed source permalinks: {missing}")
    return [{"label": link.group(1), "url": link.group(2)} for link in links]


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
    files = [
        p
        for pattern in (f"{entry_id}.*", f"{entry_id}-*.*")
        for p in sorted(repro.glob(pattern))
        if p.is_file()
    ]
    for p in files:
        if p.suffix not in LANGUAGES:
            raise LedgerError(f"{p}: unknown reproducer language {p.suffix!r}")
    if (repro / entry_id).is_dir():
        files.append(repro / entry_id)
    return files + helpers(repro, files)


def helpers(repro: Path, files: list[Path]) -> list[Path]:
    """Shared modules `repro/_<name>.py` that the Python reproducers import, transitively."""
    found: list[Path] = []
    queue = [f for f in files if f.suffix == ".py"]
    while queue:
        tree = ast.parse(queue.pop().read_text())
        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                names = [a.name for a in node.names]
            elif isinstance(node, ast.ImportFrom) and node.module and node.level == 0:
                names = [node.module]
            else:
                continue
            for name in names:
                path = repro / f"{name}.py"
                if name.startswith("_") and path.is_file() and path not in found:
                    found.append(path)
                    queue.append(path)
    return found


def leading_comment(source: str, marker: str) -> str:
    """Leading comment block after an optional shebang; '#' alone separates paragraphs."""
    lines = source.splitlines()
    if lines and lines[0].startswith("#!"):
        lines = lines[1:]
    doc = []
    for line in lines:
        if not line.startswith(marker):
            break
        doc.append(line[len(marker) :].removeprefix(" "))
    return "\n".join(doc)


def description(source: str, filename: str) -> str:
    """A reproducer's opening description (docstring or leading comment)."""
    marker = LANGUAGES[Path(filename).suffix][1]
    if marker is None:
        doc = ast.get_docstring(ast.parse(source, filename))
    else:
        doc = leading_comment(source, marker)
    if not doc:
        raise LedgerError(f"{filename}: reproducer has no description")
    return doc


def script_matches(entry: Entry) -> bool:
    """The main script opens with `<ID>: <title>`, then the entry's Summary verbatim."""
    main = next((p for p in entry.repro if p.is_file() and p.stem == entry.id), None)
    if main is None:
        return True
    paragraphs = [" ".join(p.split()) for p in description(main.read_text(), str(main)).split("\n\n")]
    return paragraphs[:2] == [f"{entry.id}: {entry.title}", " ".join(entry.fields["Summary"].split())]


def last_reproduced(provenance: dict, name: str) -> dict:
    """Most recent execution: the development-branch run if it reproduced, else the latest
    found/confirmed run. Only dated facts are stated: a branch head moves on."""

    def code(c: dict) -> dict:
        # A bare branch name ("development") is qualified with the code's name.
        ref = c["ref"] if " " in c["ref"] else f"{name} {c['ref']}"
        return {"ref": ref, "commit": str(c["commit"])[:12]}

    verified = provenance.get("verified")
    if verified and verified.get("result") not in ("reproduced", "not reproduced"):
        raise LedgerError(f"verified.result {verified.get('result')!r}")
    if verified and verified["result"] == "reproduced":
        return {
            "date": str(verified["date"]),
            "code": code(verified["code"]),
            "head": True,
            "head_not_reproduced": None,
        }
    runs = provenance.get("found", []) + provenance.get("confirmed", [])
    last = max(runs, key=lambda r: str(r["date"]))
    failed = verified and {"date": str(verified["date"]), "code": code(verified["code"])}
    return {
        "date": str(last["date"]),
        "code": code(last["code"]),
        "head": False,
        "head_not_reproduced": failed or None,
    }


def requirements(own_modules: set[str], files: list[Path]) -> list[str]:
    """Distributions of the third-party modules the reproducers import, from this environment."""
    roots: set[str] = set()
    for path in files:
        sources = [path] if path.is_file() else sorted(path.rglob("*.py"))
        for src in (s for s in sources if s.suffix == ".py"):
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
            if not PRIORITY.match(priority):
                raise LedgerError(f"{entry_id}: priority {priority!r} does not start with P0..P5")
            if severity not in SEVERITIES:
                raise LedgerError(
                    f"{entry_id}: severity {severity!r} is not one of {list(SEVERITIES)}"
                )
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
                    locations=source_links(
                        entry_id, fields.get("Repository and location", "")
                    ),
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
        "last_reproduced": last_reproduced(entry.provenance, config["name"]),
        "models": sorted(entry.models),
        "reproducers": [p.name for p in entry.repro],
        "requirements": requirements(
            set(config["modules"]) | {p.stem for p in entry.repro}, entry.repro
        ),
        "page": f"{entry.code}/{entry.id}/",
    }


def render_entry(
    entry: Entry, config: dict, env: jinja2.Environment
) -> tuple[dict, str]:
    """Public record and entry page; validates the reproducer descriptions."""
    record = entry_record(entry, config)
    sources, languages, inputs = {}, {}, []
    for src in entry.repro:
        if src.is_dir():
            inputs += [
                str(p.relative_to(src.parent))
                for p in sorted(src.rglob("*"))
                if p.is_file()
            ]
        else:
            sources[src.name] = src.read_text()
            description(sources[src.name], str(src))
            languages[src.name] = LANGUAGES[src.suffix][0]
    page = env.get_template("entry.md.j2").render(
        e=record,
        code=config,
        sources=sources,
        languages=languages,
        inputs=inputs,
        brief=entry.brief,
    )
    return record, page


def report(
    code: str, records: list[dict], excluded: list[tuple[str, list[str]]]
) -> None:
    print(f"{code}: published {len(records)}, excluded {len(excluded)}")
    for entry_id, failures in excluded:
        print(f"  {entry_id}: {'; '.join(failures)}")


def check(code: str, config: dict, env: jinja2.Environment) -> None:
    records, excluded = [], []
    for entry in parse_ledger(code, config):
        failures = entry.gate_failures()
        if failures:
            excluded.append((entry.id, failures))
        else:
            records.append(render_entry(entry, config, env)[0])
    report(code, records, excluded)


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
        record, text = render_entry(entry, config, env)
        page = out / entry.id
        page.mkdir()
        for src in entry.repro:
            if src.is_dir():
                shutil.copytree(src, page / src.name)
            else:
                shutil.copy2(src, page / src.name)
        (page / "index.md").write_text(text)
        records.append(record)
    # Default table order: priority (P0 first), then severity, then the ledger's
    # area and entry order (records are already in that order; sorted() is stable).
    by_impact = sorted(
        records,
        key=lambda r: (
            int(PRIORITY.match(r["priority"]).group(1)),
            SEVERITIES.index(r["severity"]),
        ),
    )
    (out / "index.md").write_text(
        env.get_template("code_index.md.j2").render(
            code=config,
            extras=sorted({r for rec in records for r in rec["requirements"]} - {"numpy"}),
            rows=by_impact,
        )
    )
    (out / "defects.json").write_text(
        json.dumps(records, indent=1, ensure_ascii=False) + "\n"
    )
    report(code, records, excluded)
    return records


def main(argv: list[str]) -> None:
    check_only = "--check" in argv
    argv = [a for a in argv if a != "--check"]
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
    if check_only:
        for code in argv:
            check(code, codes[code], env)
        return
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
