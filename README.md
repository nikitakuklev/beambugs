# beambugs

Evidence for defects in accelerator and beam-physics codes. Each entry has a
summary, a standalone reproducer, the models that found and confirmed it, and
links to any upstream issue or pull request. It is an evidence store, not an
issue tracker.

Site: https://nikitakuklev.github.io/beambugs/

## Layout

| Path | Content |
|---|---|
| `docs/<code>/<ID>/` | Entry page (`index.md`) and its reproducer scripts. Generated. |
| `docs/<code>/helpers/` | Shared `_*.py` helpers, published once per code, and a source-viewing page (`index.md`). Generated. |
| `docs/<code>/index.md`, `docs/<code>/defects.json` | Per-code table and records. Generated. |
| `docs/defects.json`, `docs/index.md` | All records and the home page. Generated. |
| `docs/methodology.md` | Criteria, levels, severity, priority. Hand-written. |
| `codes.yaml` | Exported codes: names, upstream repository, ledger location, area files. |
| `tools/export.py`, `tools/templates/` | Exporter from the private ledgers. |
| `tools/commits.json` | Upstream commit dates and observed branch tips that decide whether a run is current. Generated. |
| `hooks/upstream_state.py` | Build hook that shows the current state of linked issues and PRs. |

Entry pages link to the source and download of every helper their scripts need,
including helpers imported by other helpers. Save these files beside the
downloaded scripts. In `defects.json`, `reproducers` contains download paths
relative to the entry's `page`, with helpers at `../helpers/<name>.py`.

## Regenerate and build

```sh
export GITHUB_TOKEN=$(gh auth token)   # commit lookups not yet in tools/commits.json
python tools/export.py impactx          # needs the private ledger named in codes.yaml
python tools/export.py --check impactx  # parse and report the gate, write nothing under docs/
mkdocs serve                            # BEAMBUGS_OFFLINE=1 skips GitHub state lookups
```

Pushing to `main` builds and deploys the site with GitHub Actions.

## License

Reproducer scripts and tools: MIT ([LICENSE](LICENSE)). Text: CC BY 4.0
(https://creativecommons.org/licenses/by/4.0/).
