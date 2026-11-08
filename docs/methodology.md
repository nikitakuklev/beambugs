# Methodology

## How an entry is established

1. **Discovery.** An AI model examines one part of a code against its documentation, its source, analytic results or another code, and writes a runnable reproducer for each suspected defect.
2. **Confirmation.** A different model rechecks the defect by an independent method: an analytic result, finite differences, an independent numerical calculation, the documented behavior or another code. Rerunning the first model's reproducer does not count.
3. **Separation from conventions.** A discrepancy becomes an entry only once it is shown not to come from a coordinate or sign convention, a documented approximation or an unsupported input. Comparisons with another code first match coordinates, signs, charge, reference normalization, slicing and splitting.

A person directed every investigation and chose what to examine; the models did the analysis and wrote the reproducers.
Pull requests filed from this evidence were reviewed by hand before filing.

## Publication criteria

An entry is published when all of these hold:

- its evidence level is **Reproduced, independently rechecked** or **Probe-pinned**;
- it has a standalone reproducer, or a stated reason why none is possible and what establishes the defect instead;
- at least two different models, identified by exact model ID, found and confirmed it;
- it has a current run.

A run is **current** when it ran on the head of the code's development branch, or on a commit at most seven days old on the day of the run. Upstream code changes faster than every entry can be rerun; the week bounds how much upstream change a reproduction can miss. For a code made of several packages, a head run uses the code's head with the newest version of each package that the code accepts.

A published entry stays on the site. When upstream fixes the defect, a run on the development head that no longer reproduces it marks the entry as fixed; the *Show fixed* switch above each code's table hides fixed entries.

*Last reproduced* gives the date of the most recent current run that reproduced the defect, and the code it ran: the release when the commit carries an upstream tag, otherwise the commit.

## Evidence levels

Every published entry has a reproducer and was confirmed by a second model. The level states what else supports the claim:

<div class="bb-defs" markdown>

| Level | Meaning |
|---|---|
| **Reproduced, independently rechecked** | A separate check agrees that the reproducer's result is wrong: the source read against its documented behavior, an analytic result, or an independent numerical calculation. |
| **Probe-pinned** | A test in the probe suite kept for the code asserts the correct behavior and is marked as an expected failure. The probe suite, which is not published here, checks the code's documented behavior and is rerun on each new upstream version; when the defect is fixed, the test starts passing and the suite reports it. |

</div>

The level applies to the entry's main claim. Reproductions use serial CPU builds in double precision unless the entry says otherwise; they do not establish GPU, MPI or single-precision behavior.

## Failure

The kind of failure; `severity` in `defects.json`.

<div class="bb-defs" markdown>

| Failure | Meaning |
|---|---|
| Wrong physics | The code computes a physically wrong result for a supported input. |
| Silent wrong result | A wrong number with no warning or error. |
| Crash or API hazard | An exception, NaN or API behavior that breaks a valid use. |
| Documentation or example mismatch | Labels, units or examples disagree with what the code computes. |

</div>

## Priority

Priority ranks the impact on users: how widely the affected element, mode or input path is used, whether the error is silent, and how large it is in realistic regimes.
A failure that raises an error ranks below a silent one of similar reach.
How widely something is used is a judgment; no usage data backs it.

<div class="bb-defs" markdown>

| Priority | Meaning |
|---|---|
| **P0** | Silently wrong physics on a widely used element or default path |
| **P1** | Silent, large error in a commonly used element, mode or import path |
| **P2** | Silently wrong physics in a less-used feature or option, or wrong with a warning |
| **P3** | Niche element or regime, or a NaN at valid input |
| **P4** | Experimental features, unrealistically weak strengths, helpers outside tracking, display-only errors, or configuration mistakes that raise an error |
| **P5** | Labels and units only, or an unphysical regime |

</div>

## Tags

Tags qualify an entry's scope or status; an untagged entry holds as stated.

<div class="bb-defs" markdown>

| Tag | Meaning |
|---|---|
| `experimental` | Affects a feature upstream documents as experimental, such as spin tracking. |
| `extreme-regime` | Appears only at parameters far outside typical use. |
| `contract-unsettled` | Whether it is a defect depends on intended behavior that upstream has not settled or documents ambiguously. |
| `known-limitation` | Upstream source or documentation acknowledges the behavior. |
| `partly-source-only` | Part of the stated scope is established by reading the source, not by running it. |

</div>

## Upstream links

The *Issue* and *PR* columns link upstream issues and pull requests that report or fix the defect, each with an icon for its current state: open, closed or merged.
:material-alpha-b-circle-outline:{ .bb-filed } marks those filed from this evidence.
*Discussed in*, on entry pages, lists issues or pull requests that raise the defect or its cause without reporting or fixing it.

## Provenance

Each entry lists the sessions that found and confirmed it: the exact model ID, the reasoning effort where the session recorded one, the date, and the code release, commit and build the session examined.
A note under a role limits that session to the part of the claim it names.

## Reproducers

Each reproducer builds its case inline, prints observed and expected values with the derivation of the expected value, and ends with `DEFECT REPRODUCED` or `NOT REPRODUCED`.
A nonzero exit status means the run itself failed and gives no verdict.
Each script opens with the entry's title and summary, then a short description of how it demonstrates the defect.
Installation and run instructions are on each code's page. A script that needs more, such as a source checkout named by an environment variable, MPI or Valgrind, says so in a paragraph starting `Setup:` after the summary, and stops with an error when a required variable is unset.
