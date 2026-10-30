"""MkDocs hook: render [[gh:<issue or PR URL>]] as a state icon and a link.

States are fetched at build time, so the site shows whether a linked issue or
pull request is open, closed or merged without the exporter tracking it. Uses
GITHUB_TOKEN when set; unauthenticated requests are limited to 60 per hour.
Any API failure stops the build; set BEAMBUGS_OFFLINE=1 to render plain links
without states (local previews only).
"""

from __future__ import annotations

import json
import os
import re
import urllib.error
import urllib.request

PLACEHOLDER = re.compile(
    r"\[\[gh:(https://github\.com/([\w.-]+/[\w.-]+)/(?:issues|pull)/(\d+)[^\]\s]*)\]\]"
)
# (icon shortcode, css class, tooltip) per GitHub state.
ICONS = {
    "issue-open": ("octicons-issue-opened-16", "bb-open", "Open issue"),
    "issue-completed": ("octicons-issue-closed-16", "bb-done", "Closed issue (completed)"),
    "issue-not_planned": ("octicons-skip-16", "bb-dropped", "Closed issue (not planned)"),
    "issue-closed": ("octicons-issue-closed-16", "bb-done", "Closed issue"),
    "pull-open": ("octicons-git-pull-request-16", "bb-open", "Open pull request"),
    "pull-draft": ("octicons-git-pull-request-draft-16", "bb-dropped", "Draft pull request"),
    "pull-merged": ("octicons-git-merge-16", "bb-done", "Merged pull request"),
    "pull-closed": ("octicons-git-pull-request-closed-16", "bb-closed", "Closed pull request"),
}
_cache: dict[tuple[str, int], str] = {}


def _get(url: str) -> dict:
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
        hint = "" if token else " Set GITHUB_TOKEN (e.g. $(gh auth token))."
        raise RuntimeError(f"GitHub API {err.code} for {url}: {err.reason}.{hint}") from err


def _state(repo: str, number: int) -> str:
    key = (repo, number)
    if key not in _cache:
        issue = _get(f"https://api.github.com/repos/{repo}/issues/{number}")
        if "pull_request" in issue:
            pull = _get(f"https://api.github.com/repos/{repo}/pulls/{number}")
            if pull.get("merged_at"):
                state = "pull-merged"
            elif pull["state"] == "open" and pull.get("draft"):
                state = "pull-draft"
            else:
                state = f"pull-{pull['state']}"
        elif issue["state"] == "open":
            state = "issue-open"
        else:
            state = f"issue-{issue.get('state_reason') or 'closed'}"
        if state not in ICONS:
            raise RuntimeError(f"unexpected GitHub state {state!r} for {repo}#{number}")
        _cache[key] = state
    return _cache[key]


def _render(match: re.Match) -> str:
    url, repo, number = match.group(1), match.group(2), int(match.group(3))
    link = f"[{repo.split('/')[1]}#{number}]({url})"
    if os.environ.get("BEAMBUGS_OFFLINE") == "1":
        return link
    icon, css, tooltip = ICONS[_state(repo, number)]
    return f':{icon}:{{ .{css} title="{tooltip}" }} {link}'


def on_page_markdown(markdown: str, **kwargs) -> str:
    return PLACEHOLDER.sub(_render, markdown)
