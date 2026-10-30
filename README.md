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
| `docs/<code>/index.md`, `docs/<code>/defects.json` | Per-code table and records. Generated. |
| `docs/defects.json`, `docs/index.md` | All records and the home page. Generated. |
| `docs/methodology.md` | Criteria, levels, severity, priority. Hand-written. |
| `codes.yaml` | Exported codes: names, upstream repository, ledger location, area files. |
| `tools/export.py`, `tools/templates/` | Exporter from the private ledgers. |
| `hooks/upstream_state.py` | Build hook that shows the current state of linked issues and PRs. |

## Regenerate and build

```sh
python tools/export.py impactx          # needs the private ledger named in codes.yaml
mkdocs serve                            # BEAMBUGS_OFFLINE=1 skips GitHub state lookups
```

Pushing to `main` builds and deploys the site with GitHub Actions.

## License

Reproducer scripts and tools: MIT ([LICENSE](LICENSE)). Text: CC BY 4.0
(https://creativecommons.org/licenses/by/4.0/).
