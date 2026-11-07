---
title: "IO-07: Pelegant WATCH converts every species' kinetic energy with the electron mass"
hide:
  - toc
---

# IO-07 — Pelegant WATCH converts every species' kinetic energy with the electron mass

| | |
|---|---|
| Code | [Elegant](../index.md) · Lattices, particles and I/O |
| Priority | P4 |
| Severity | Silent wrong result |
| Evidence | Reproduced, independently rechecked |
| Last reproduced | 2026-11-06 on Elegant master (`0355dd6`, committed 2026-11-03) |
| Tags | <span class="bb-tag">partly-source-only</span> [?](../../methodology.md#tags) |
| Source | [sdds_support.c:1312–1323](https://github.com/rtsoliday/elegant/blob/0355dd6aca12c861971fbf8f3ae78a0ded991734/src/sdds_support.c#L1312-L1323)<br>[1326–1352](https://github.com/rtsoliday/elegant/blob/0355dd6aca12c861971fbf8f3ae78a0ded991734/src/sdds_support.c#L1326-L1352)<br>[495](https://github.com/rtsoliday/elegant/blob/0355dd6aca12c861971fbf8f3ae78a0ded991734/src/sdds_support.c#L495)<br>[521](https://github.com/rtsoliday/elegant/blob/0355dd6aca12c861971fbf8f3ae78a0ded991734/src/sdds_support.c#L521)<br>[elegant.tex:1497–1517](https://github.com/rtsoliday/elegant/blob/0355dd6aca12c861971fbf8f3ae78a0ded991734/doc/elegant.tex#L1497-L1517)<br>[elegant.c:3242–3247](https://github.com/rtsoliday/elegant/blob/0355dd6aca12c861971fbf8f3ae78a0ded991734/src/elegant.c#L3242-L3247)<br>[3285–3297](https://github.com/rtsoliday/elegant/blob/0355dd6aca12c861971fbf8f3ae78a0ded991734/src/elegant.c#L3285-L3297)<br>[elegant.tex:7422–7431](https://github.com/rtsoliday/elegant/blob/0355dd6aca12c861971fbf8f3ae78a0ded991734/doc/elegant.tex#L7422-L7431)<br>[constants.h:28–35](https://github.com/rtsoliday/SDDS/blob/a3f730a14c49ddd4f4d6347b42d60d34aa7cadc6/include/constants.h#L28-L35) |

In Pelegant, WATCH centroid and parameter output converts every species' mean kinetic energy using the electron rest energy. For a proton distribution with dimensionless momenta 0.75, 1.5, 2.75, and 5.5 in equal proportions, it reports 0.9670 MeV instead of 1776 MeV, an underestimate by a factor of 1836. The expected mean follows from K=(γ−1)mc², where γ is the Lorentz factor and m the proton mass. ID selection retains the error. Serial proton output, named electron and positron controls, momenta, and drift times are correct.

Run `python IO-07.py` with [_elegant.py](../helpers/_elegant.py), [_watch_energy.py](../helpers/_watch_energy.py) and [_diagnostics.py](../helpers/_diagnostics.py) in the same directory.

??? example "Reproducer: IO-07.py"

    [Download IO-07.py](IO-07.py)

    ```python
    """IO-07: Pelegant WATCH converts every species' kinetic energy with the electron mass

    In Pelegant, WATCH centroid and parameter output converts every species' mean kinetic energy using the electron rest energy. For a proton distribution with dimensionless momenta 0.75, 1.5, 2.75, and 5.5 in equal proportions, it reports 0.9670 MeV instead of 1776 MeV, an underestimate by a factor of 1836. The expected mean follows from K=(γ−1)mc², where γ is the Lorentz factor and m the proton mass. ID selection retains the error. Serial proton output, named electron and positron controls, momenta, and drift times are correct.

    Setup: always runs 3-rank Pelegant (Open MPI `mpirun` on `PATH`) beside serial Elegant, because the defect occurs only under MPI.

    Twenty fixed particles cross a 0.75 m exact drift before centroid, parameter and ID-selected WATCH outputs. A 70-digit relativistic oracle uses the selected species mass, with electron, positron and serial controls and independent momentum and flight-time checks.
    """

    from _elegant import modes, verdict
    from _watch_energy import ME, MMU, MP, check


    def main():
        defects = []
        for mode in ("serial", *modes(mpi_only=True)):
            for label, species, mass, tabulated in (
                ("electron", "electron", ME, True),
                ("positron", "positron", ME, True),
                ("proton", "proton", MP, False),
                ("muon", "muon", MMU, False),
                ("custom100_q1", "custom,mass_ratio=100,charge_ratio=-1", 100 * ME, False),
            ):
                defects.extend(check(mode, label, species, mass, tabulated=tabulated))
        verdict(all(defects))


    if __name__ == "__main__":
        main()

    ```

## Provenance

| Role | Model | Effort | Date | Code examined |
|---|---|---|---|---|
| Found | `gpt-6-astra` | xhigh | 2026-11-06 | master (`0355dd6aca12`), source with SDDS a3f730a14c49, CPU, double, 3-rank MPI with serial controls, one thread |
| Confirmed | `gpt-6.1-sol` | xhigh | 2026-11-06 | master (`0355dd6aca12`), source with SDDS a3f730a14c49, CPU, double, 3-rank MPI with serial controls, one thread |

