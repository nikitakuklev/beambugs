"""Generate docs/<code>/ from a private upstream-defect ledger.

Usage: python tools/export.py [--check] <code> [<code> ...]

The ledger location and area files come from codes.yaml. Everything under
docs/<code>/, docs/defects.json and docs/index.md is regenerated; do not edit
those files by hand. --check parses the ledgers and reports the entries that
miss the publication gate without writing under docs/. The ledger format is
defined by the ledgers' shared procedure (defect_hunting.md): entries
`## <ID> — <title>` with bold fields, a YAML provenance block, an index table
per area file and filed.md.

Whether a run is current needs upstream commit facts, which come from the
GitHub API (GITHUB_TOKEN raises the rate limit) and are cached in
tools/commits.json; --check also updates that cache.
"""

from __future__ import annotations

import ast
import datetime as dt
import functools
import importlib.metadata
import json
import os
import re
import shutil
import subprocess
import sys
import urllib.error
import urllib.request
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path
from zoneinfo import ZoneInfo

import jinja2
import yaml

ROOT = Path(__file__).resolve().parent.parent
DOCS = ROOT / "docs"
TEMPLATES = ROOT / "tools" / "templates"
# What the public site serves; a published entry may not disappear from it.
PUBLIC_REF = "origin/main"

PUBLISHED_LEVELS = {"Reproduced, independently rechecked", "Probe-pinned"}
# A run is current when it executed the development-branch head or a commit at most
# this many days old on the run date. Provenance dates are America/Chicago days.
CURRENT_DAYS = 7
LEDGER_TZ = ZoneInfo("America/Chicago")
# Provenance roles: the sessions that found and confirmed the claim (the two-model gate).
ROLES = ("found", "confirmed")
PROVENANCE_KEYS = {*ROLES, "verified", "upstream", "note", "evidence"}
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

    def gate_failures(self, reproduction: dict) -> list[str]:
        failures = []
        if not reproduction["current"]:
            r = reproduction["last_reproduced"]
            failures.append(
                f"no current run (latest {r['date']} on {r['code']['ref']}, "
                f"committed {r['code']['committed']})"
            )
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
    """Our filings per entry. A row opens with the linked entry ID, or with `—` for a
    filing whose defect has no published entry; any other table row is an error, since
    a skipped row would silently drop the entry's `filed` links."""
    filed: dict[str, list[tuple[str, str, int]]] = {}
    for line in path.read_text().splitlines():
        if not line.startswith("|") or line.startswith(("| ID |", "|---", "| — |")):
            continue
        m = re.match(r"^\| \[([A-Z]+-\d+)\]", line)
        if not m:
            raise LedgerError(f"{path}: row opens with neither a linked ID nor —: {line[:80]}")
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
    unknown = set(data) - PROVENANCE_KEYS
    if unknown:
        raise LedgerError(f"{entry_id}: unknown provenance keys {sorted(map(str, unknown))}")
    for role in ROLES:
        if not isinstance(data.get(role, []), list):
            raise LedgerError(f"{entry_id}: provenance {role} must be a list")
    return data


def public_provenance(data: dict) -> dict:
    """Drop local session IDs, internal notes and evidence links."""

    def clean(item: dict) -> dict:
        return {k: v for k, v in item.items() if k not in ("session", "evidence")}

    out: dict = {role: [clean(i) for i in data.get(role, [])] for role in ROLES}
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


def is_helper(path: Path) -> bool:
    """Only imported repro/_*.py modules are shared; entry scripts keep their ID names."""
    return path.name.startswith("_") and path.suffix == ".py"


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


class UpstreamCommits:
    """Commit dates and observed development-branch tips, from the GitHub API.

    Cached in tools/commits.json as {repository: {commit[:12]: {committed, tip_seen,
    superseded}}}. The facts do not change once observed: a commit's date is fixed, and
    a commit seen as the branch tip on day T ran the head code (or newer) on every day
    from its commit date to T. Only a commit not yet seen superseded is looked up again,
    for a run after the last day it was seen as the tip."""

    def __init__(self, path: Path):
        self.path = path
        self.data: dict = json.loads(path.read_text()) if path.is_file() else {}
        self.tips: dict[tuple[str, str], str] = {}
        self.tags: dict[str, dict[str, list[str]]] = {}

    def _get(self, url: str) -> dict:
        request = urllib.request.Request(
            url, headers={"Accept": "application/vnd.github+json"}
        )
        token = os.environ.get("GITHUB_TOKEN")
        if token:
            request.add_header("Authorization", f"Bearer {token}")
        try:
            with urllib.request.urlopen(request, timeout=30) as response:
                return json.load(response)
        except urllib.error.HTTPError as err:
            if err.code in (404, 422):
                raise LedgerError(f"GitHub has no {url}") from err
            hint = "" if token else " Set GITHUB_TOKEN (e.g. $(gh auth token))."
            raise RuntimeError(f"GitHub API {err.code} for {url}: {err.reason}.{hint}") from err

    def _record(self, repo: str, commit: str) -> dict:
        key = commit[:12]
        record = self.data.setdefault(repo, {}).get(key)
        if record is None:
            c = self._get(f"https://api.github.com/repos/{repo}/commits/{commit}")
            if not c["sha"].startswith(commit):
                raise LedgerError(f"{repo}: commit {commit} resolves to {c['sha']}")
            committed = dt.datetime.fromisoformat(c["commit"]["committer"]["date"])
            record = {
                "sha": c["sha"],
                "committed": str(committed.astimezone(LEDGER_TZ).date()),
                "tip_seen": None,
                "superseded": False,
            }
            self.data[repo][key] = record
        return record

    def committed(self, repo: str, commit: str) -> dt.date:
        return dt.date.fromisoformat(self._record(repo, commit)["committed"])

    def tip_on(self, repo: str, branch: str, commit: str, day: dt.date) -> bool:
        """True if the commit was the branch tip on `day`, as far as observations show."""
        record = self._record(repo, commit)
        if record["tip_seen"] and day <= dt.date.fromisoformat(record["tip_seen"]):
            return True
        if record["superseded"]:
            return False
        if (repo, branch) not in self.tips:
            b = self._get(f"https://api.github.com/repos/{repo}/branches/{branch}")
            self.tips[repo, branch] = b["commit"]["sha"]
        if self.tips[repo, branch] != record["sha"]:
            record["superseded"] = True
            return False
        record["tip_seen"] = str(dt.datetime.now(LEDGER_TZ).date())
        return day <= dt.date.fromisoformat(record["tip_seen"])

    def release(self, repo: str, commit: str) -> str | None:
        """The tag on the commit, if any (the first in sorted order when several are).
        Tags are read from GitHub on every run rather than cached, since a commit can
        be tagged after it is first looked up."""
        if repo not in self.tags:
            by_sha: dict[str, list[str]] = {}
            page = 1
            while batch := self._get(
                f"https://api.github.com/repos/{repo}/tags?per_page=100&page={page}"
            ):
                for tag in batch:
                    by_sha.setdefault(tag["commit"]["sha"], []).append(tag["name"])
                page += 1
            self.tags[repo] = by_sha
        names = self.tags[repo].get(self._record(repo, commit)["sha"])
        return min(names) if names else None

    def save(self) -> None:
        self.path.write_text(json.dumps(self.data, indent=1, sort_keys=True) + "\n")


def run_repository(config: dict, ref: str) -> str:
    """Repository a run executed: `ref` is prefixed with the package when the code
    spans several repositories (`xtrack 0.118.2`); a bare ref is the code's own."""
    if " " not in ref:
        return config["repository"]
    package = ref.split()[0]
    if package not in config["repositories"]:
        raise LedgerError(f"ref {ref!r}: package {package!r} not in codes.yaml repositories")
    return config["repositories"][package]


def reproduction(provenance: dict, config: dict, commits: UpstreamCommits) -> dict:
    """The entry's last reproduction and whether it is current.

    Last reproduced is the most recent current reproducing run (`verified` when it
    reproduced, or any found or confirmed run), else the most recent
    reproducing run. A run is current when it is the recorded head verification, its
    commit was at most CURRENT_DAYS old on the run date, or the commit was still the
    branch tip. A `verified` run that did not reproduce is reported beside it; that head
    run keeps a fixed defect's entry current."""

    def code(run: dict) -> dict:
        c = run["code"]
        repo = run_repository(config, str(c["ref"]))
        # A bare branch or release name ("development") is qualified with the code's name.
        ref = str(c["ref"]) if " " in str(c["ref"]) else f"{config['name']} {c['ref']}"
        return {
            "ref": ref,
            "repository": repo,
            "commit": str(c["commit"])[:12],
            "committed": str(commits.committed(repo, str(c["commit"]))),
            # Verified against the repository's tags, not taken from the ref label.
            "release": commits.release(repo, str(c["commit"])),
        }

    def head(run: dict) -> bool:
        if run is verified:
            return True
        c = run["code"]
        repo = run_repository(config, str(c["ref"]))
        day = dt.date.fromisoformat(str(run["date"]))
        return commits.tip_on(repo, config["branch"], str(c["commit"]), day)

    def current(run: dict) -> bool:
        c = run["code"]
        repo = run_repository(config, str(c["ref"]))
        age = dt.date.fromisoformat(str(run["date"])) - commits.committed(repo, str(c["commit"]))
        return run is verified or age.days <= CURRENT_DAYS or head(run)

    def summary(run: dict) -> dict:
        return {"date": str(run["date"]), "code": code(run), "head": head(run)}

    verified = provenance.get("verified")
    if verified and verified.get("result") not in ("reproduced", "not reproduced"):
        raise LedgerError(f"verified.result {verified.get('result')!r}")
    runs = [run for role in ROLES for run in provenance.get(role, [])]
    if verified and verified["result"] == "reproduced":
        runs.append(verified)
    if not runs:
        raise LedgerError("provenance has no run")
    # Most recent first; on the same day the head verification ranks first.
    runs.sort(key=lambda r: (str(r["date"]), r is verified), reverse=True)
    last = next((r for r in runs if current(r)), None)
    fixed = verified if verified and verified["result"] == "not reproduced" else None
    return {
        "last_reproduced": {
            **summary(last or runs[0]),
            "head_not_reproduced": fixed and summary(fixed),
        },
        "current": last is not None or fixed is not None,
    }


@functools.cache
def _distributions() -> Mapping[str, list[str]]:
    # Scanning every installed distribution takes seconds; the environment is fixed for one run.
    return importlib.metadata.packages_distributions()


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
    distributions = _distributions()
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


def entry_record(entry: Entry, config: dict, last: dict) -> dict:
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
        "last_reproduced": last,
        "models": sorted(entry.models),
        # Download paths are relative to the entry page, including transitive helpers.
        "reproducers": [
            f"../helpers/{p.name}" if is_helper(p) else p.name for p in entry.repro
        ],
        "requirements": requirements(
            set(config["modules"]) | {p.stem for p in entry.repro}, entry.repro
        ),
        "page": f"{entry.code}/{entry.id}/",
    }


def render_entry(
    entry: Entry, config: dict, env: jinja2.Environment, last: dict
) -> tuple[dict, str]:
    """Public record and entry page; validates the reproducer descriptions."""
    record = entry_record(entry, config, last)
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
        helper_names=[p.name for p in entry.repro if is_helper(p)],
        # The script named by the entry ID; extra scripts and Justification-only entries add no run line.
        main=next((p.name for p in entry.repro if p.is_file() and p.stem == entry.id), None),
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


def published_ids(code: str) -> set[str]:
    """Entry IDs on the public site: the entry directories of docs/<code>/ on PUBLIC_REF."""
    try:
        out = subprocess.run(
            ["git", "-C", str(ROOT), "ls-tree", "-d", "--name-only", PUBLIC_REF, f"docs/{code}/"],
            capture_output=True, text=True, check=True,
        ).stdout
    except subprocess.CalledProcessError as err:
        raise LedgerError(f"cannot list {PUBLIC_REF}:docs/{code}/: {err.stderr.strip()}") from err
    names = {Path(p).name for p in out.splitlines()}
    return {n for n in names if re.fullmatch(r"[A-Z]+-\d+", n)}


def gated(
    code: str, entries: list[Entry], config: dict, commits: UpstreamCommits
) -> tuple[list[tuple[Entry, dict]], list[tuple[str, list[str]]]]:
    """Entries that pass the gate, with their last reproduction, and the excluded ones.

    Publication is permanent: every entry on the public site must still pass, unless
    codes.yaml `merged` names the entry that absorbed it."""
    passed, excluded = [], []
    for entry in entries:
        try:
            r = reproduction(entry.provenance, config, commits)
        except LedgerError as err:
            raise LedgerError(f"{entry.id}: {err}") from err
        failures = entry.gate_failures(r)
        if failures:
            excluded.append((entry.id, failures))
        else:
            passed.append((entry, r["last_reproduced"]))
    kept = {entry.id for entry, _ in passed}
    merged = config.get("merged", {})
    if unkept := sorted(set(merged.values()) - kept):
        raise LedgerError(f"{code}: merged entries point to unpublished {unkept}")
    reasons = dict(excluded)
    if lost := sorted(published_ids(code) - kept - set(merged)):
        details = "; ".join(f"{i} ({'; '.join(reasons.get(i, ['not in the ledger']))})" for i in lost)
        raise LedgerError(
            f"{code}: published entries would leave the site: {details}. A published "
            "entry stays, as a fixed entry once fixed (defect_hunting.md, Publication)"
        )
    return passed, excluded


def check(code: str, config: dict, env: jinja2.Environment, commits: UpstreamCommits) -> None:
    passed, excluded = gated(code, parse_ledger(code, config), config, commits)
    records = [render_entry(entry, config, env, last)[0] for entry, last in passed]
    report(code, records, excluded)


def export(
    code: str, config: dict, env: jinja2.Environment, commits: UpstreamCommits
) -> list[dict]:
    passed, excluded = gated(code, parse_ledger(code, config), config, commits)
    out = DOCS / code
    if out.exists():
        shutil.rmtree(out)
    out.mkdir(parents=True)
    records = []
    shared: set[Path] = set()
    for entry, last in passed:
        record, text = render_entry(entry, config, env, last)
        page = out / entry.id
        page.mkdir()
        for src in entry.repro:
            if is_helper(src):
                shared.add(src)
            elif src.is_dir():
                shutil.copytree(src, page / src.name)
            else:
                shutil.copy2(src, page / src.name)
        (page / "index.md").write_text(text)
        records.append(record)
    if shared:
        helper_dir = out / "helpers"
        helper_dir.mkdir()
        for src in sorted(shared):
            shutil.copy2(src, helper_dir / src.name)
        (helper_dir / "index.md").write_text(
            env.get_template("helpers.md.j2").render(
                code=config,
                sources={src.name: src.read_text() for src in sorted(shared)},
            )
        )
    # Default table order: priority (P0 first), then severity, then the ledger's
    # area and entry order (records are already in that order; sorted() is stable).
    by_impact = sorted(
        records,
        key=lambda r: (
            int(r["priority"][1]),  # parse_ledger validates the P0..P5 prefix.
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
    commits = UpstreamCommits(ROOT / "tools" / "commits.json")
    try:
        for code in argv:
            (check if check_only else export)(code, codes[code], env, commits)
    finally:
        commits.save()
    if check_only:
        return
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
