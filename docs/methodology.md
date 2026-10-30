# Methodology

## How an entry is established

1. **Discovery.** An AI model examines one subsystem of a code against its documentation, source contracts, analytic results or another code, and writes a runnable reproducer for each candidate.
2. **Confirmation.** A different model rechecks the candidate with an independent method: an analytic invariant, finite differences, an independent numerical integration, the documented contract or another code. Rerunning the discovery reproducer does not count.
3. **Head verification.** The reproducer is run on a build of the code's current upstream head. The entry's *Last reproduced* row records the date, commit and build.

Before an observation becomes an entry, it is separated from coordinate conventions, documented model approximations and unsupported inputs.
Coordinates, signs, charge and reference normalization, slicing and splitting are resolved before comparing with another code.
Candidates that did not survive these checks are not published.

A human directed every investigation and chose what to examine; the models did the analysis and wrote the reproducers.
Pull requests filed from this evidence are reviewed by hand before filing.

## Publication criteria

An entry is published when all of these hold:

- its evidence level is **Reproduced, independently rechecked** or **Probe-pinned**;
- it has a standalone reproducer, or a stated reason why none is possible and what establishes the defect instead;
- the models that found and confirmed it include at least two distinct models, identified by exact model ID.

## Evidence levels

| Level | Meaning |
|---|---|
| **Probe-pinned** | A named check in a probe suite pins the claim with a strict expected failure. |
| **Reproduced, independently rechecked** | A runnable reproducer demonstrates the claim, and a separate source/contract, analytic or numerical check supports it. |

A level applies to the principal claim of an entry.
A CPU run does not establish GPU, MPI or single-precision behavior; unless an entry says otherwise, reproductions use serial CPU builds in double precision.

## Severity

| Severity | Meaning |
|---|---|
| Wrong physics | The model computes a physically wrong result for a supported input. |
| Silent wrong result | A wrong number with no warning or error. |
| Crash or API hazard | An exception, NaN or API behavior that breaks a valid use. |
| Documentation or example mismatch | Labels, units or examples disagree with what the code computes. |

## Priority

Priority ranks the physics impact on users: how widely the element, mode or input path is used, whether the error is silent, and its magnitude in realistic regimes.
A loud failure ranks below a silent error of similar reach.
Popularity is a judgment of typical use; no usage data backs it.

| Priority | Meaning |
|---|---|
| **P0** | Silently wrong physics on a widely used element or default path |
| **P1** | Silent, large error in a commonly used element, mode or import path |
| **P2** | Silently wrong physics in a less-used feature or option, or wrong with a warning |
| **P3** | Niche element or regime, or a loud NaN at valid input |
| **P4** | Experimental features, unrealistically weak strengths, helpers outside tracking, display-only or loud configuration errors |
| **P5** | Labels and units only, or an unphysical regime |

## Tags

Tags mark entries whose scope or status needs a qualifier.
An untagged entry is established as stated.

| Tag | Meaning |
|---|---|
| `experimental` | Affects a feature upstream documents as experimental, such as spin tracking. |
| `extreme-regime` | Appears only at parameters far outside typical use. |
| `contract-unsettled` | Whether it is a defect depends on intended behavior that upstream has not settled or documents ambiguously. |
| `known-limitation` | Upstream source or documentation acknowledges the behavior. |
| `partly-source-only` | Part of the stated scope is established by source reading, not execution. |

## Upstream links

*Issue* and *PR* list reports and fixes, with their current state: open, closed or merged.
*Filed from this evidence* marks filings made from the entry.
*Discussed in* lists issues or pull requests that raise the defect or its cause without reporting or fixing it.

## Provenance

Each entry lists the sessions that found and confirmed it: the exact model ID, the reasoning effort where the session recorded one, the date, and the code release, commit and build the session examined.
A row with a scope established only that part of the claim.

## Reproducers

Each reproducer builds its case inline, prints observed and expected values with the expected value's derivation, and ends with `DEFECT REPRODUCED` or `NOT REPRODUCED`.
A nonzero exit status means the run failed, not a verdict.
Reproducers that read the code's sources or git history take the path of a clone from an environment variable named on the entry page.
