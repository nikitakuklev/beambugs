---
title: "IO-06: HISTOGRAM writes whole-beam charge for a selected particle-ID subset"
hide:
  - toc
---

# IO-06 — HISTOGRAM writes whole-beam charge for a selected particle-ID subset

| | |
|---|---|
| Code | [Elegant](../index.md) · Lattices, particles and I/O |
| Priority | P4 |
| Severity | Silent wrong result |
| Evidence | Reproduced, independently rechecked |
| Last reproduced | 2026-11-06 on Elegant master (`0355dd6`, committed 2026-11-03) |
| Tags | <span class="bb-tag">partly-source-only</span> [?](../../methodology.md#tags) |
| Source | [do_tracking.c:1894–1900](https://github.com/rtsoliday/elegant/blob/0355dd6aca12c861971fbf8f3ae78a0ded991734/src/do_tracking.c#L1894-L1900)<br>[sdds_support.c:1744–1750](https://github.com/rtsoliday/elegant/blob/0355dd6aca12c861971fbf8f3ae78a0ded991734/src/sdds_support.c#L1744-L1750)<br>[1831–1842](https://github.com/rtsoliday/elegant/blob/0355dd6aca12c861971fbf8f3ae78a0ded991734/src/sdds_support.c#L1831-L1842)<br>[1868–1872](https://github.com/rtsoliday/elegant/blob/0355dd6aca12c861971fbf8f3ae78a0ded991734/src/sdds_support.c#L1868-L1872)<br>[schema, 176–177](https://github.com/rtsoliday/elegant/blob/0355dd6aca12c861971fbf8f3ae78a0ded991734/src/sdds_support.c#L176-L177)<br>[installation, 774–777](https://github.com/rtsoliday/elegant/blob/0355dd6aca12c861971fbf8f3ae78a0ded991734/src/sdds_support.c#L774-L777)<br>[histogram.tex:46–57](https://github.com/rtsoliday/elegant/blob/0355dd6aca12c861971fbf8f3ae78a0ded991734/doc/histogram.tex#L46-L57)<br>[track_data.c:2663–2665](https://github.com/rtsoliday/elegant/blob/0355dd6aca12c861971fbf8f3ae78a0ded991734/src/track_data.c#L2663-L2665)<br>[compute_matrices.c:2201–2204](https://github.com/rtsoliday/elegant/blob/0355dd6aca12c861971fbf8f3ae78a0ded991734/src/compute_matrices.c#L2201-L2204)<br>[track_data.c:2621–2622](https://github.com/rtsoliday/elegant/blob/0355dd6aca12c861971fbf8f3ae78a0ded991734/src/track_data.c#L2621-L2622)<br>[selection, sdds_support.c:935–938](https://github.com/rtsoliday/elegant/blob/0355dd6aca12c861971fbf8f3ae78a0ded991734/src/sdds_support.c#L935-L938)<br>[charge, 999–1001](https://github.com/rtsoliday/elegant/blob/0355dd6aca12c861971fbf8f3ae78a0ded991734/src/sdds_support.c#L999-L1001) |

When HISTOGRAM selects a bunch or another particle-ID subset, its Charge parameter retains the whole live-beam charge while its particle count and bins describe the selected population. For a 0.5000 nC beam containing 0.1250 nC and 0.3750 nC bunches, both bunch files report 0.5000 nC. Scaling their normalized distributions by this charge overestimates the bunch charges by factors of four and 1.333. The mismatch occurs in serial and parallel output, including count and sparse modes; whole-beam histograms and selected WATCH charge output are correct.

Run `python IO-06.py` with [_diagnostics.py](../helpers/_diagnostics.py) and [_elegant.py](../helpers/_elegant.py) in the same directory.

??? example "Reproducer: IO-06.py"

    [Download IO-06.py](IO-06.py)

    ```python
    """IO-06: HISTOGRAM writes whole-beam charge for a selected particle-ID subset

    When HISTOGRAM selects a bunch or another particle-ID subset, its Charge parameter retains the whole live-beam charge while its particle count and bins describe the selected population. For a 0.5000 nC beam containing 0.1250 nC and 0.3750 nC bunches, both bunch files report 0.5000 nC. Scaling their normalized distributions by this charge overestimates the bunch charges by factors of four and 1.333. The mismatch occurs in serial and parallel output, including count and sparse modes; whole-beam histograms and selected WATCH charge output are correct.

    Twenty fixed particles form two bunches with a common macroparticle charge and known IDs, then cross an aperture with and without losses. The script compares whole-beam, bunch, partial-bunch, raw-count and sparse histograms with direct selected-ID charge sums, and checks normalized-bin integrals and a selected WATCH control. Use --mpi to add three-rank Pelegant.
    """

    import math

    import numpy as np
    from _diagnostics import conversion_defect, run, scalar
    from _elegant import modes, particles, verdict

    QMACRO = 2.5e-11
    RTOL = 5e-15
    XS = np.array([-0.002, -0.001, 0.0, 0.001, 0.002] * 4)
    IDS = np.array([*range(1, 6), *range(33, 48)])
    PROPS = "BINS=5,X_DATA=1,Y_DATA=0,LONGIT_DATA=0,BIN_SIZE_FACTOR=1.25"


    def case(mode, loss):
        lattice = f"""Q: CHARGE,PER_PARTICLE={QMACRO}
    A: RCOL,X_MAX={0.0015 if loss else 0.003},Y_MAX=1
    H: HISTOGRAM,FILENAME="all.sdds",{PROPS}
    I: HISTOGRAM,FILENAME="id.sdds",START_PID=1,END_PID=32,{PROPS}
    J: HISTOGRAM,FILENAME="slice.sdds",START_PID=2,END_PID=4,{PROPS}
    B: HISTOGRAM,FILENAME="bunch-%ld.sdds",BUNCH_SERIES=1,{PROPS}
    R: HISTOGRAM,FILENAME="raw.sdds",START_PID=1,END_PID=32,NORMALIZE=0,{PROPS}
    S: HISTOGRAM,FILENAME="sparse.sdds",START_PID=1,END_PID=32,SPARSE=1,{PROPS}
    F: HISTOGRAM,FILENAME="fullrange.sdds",START_PID=1,END_PID=64,{PROPS}
    W: WATCH,FILENAME="watch.sdds",MODE="coordinates",START_PID=1,END_PID=32
    BL: LINE=(Q,A,H,I,J,2*B,R,S,F,W)
    """
        commands = """&run_setup lattice=lattice.lte,use_beamline=BL,p_central=2,output=out.sdds,random_number_seed=24681 &end
    &run_control n_passes=1 &end
    &sdds_beam input=beam.sdds,use_bunched_mode=1 &end
    &track &end
    """
        retained = np.abs(XS) <= (0.0015 if loss else 0.003)
        bunch1 = retained & (IDS <= 32)
        selections = (
            ("all.sdds", retained, False), ("fullrange.sdds", retained, False),
            ("id.sdds", bunch1, True), ("slice.sdds", retained & (IDS >= 2) & (IDS <= 4), True),
            ("bunch-1.sdds", bunch1, True), ("bunch-2.sdds", retained & (IDS > 32), True),
            ("raw.sdds", bunch1, True), ("sparse.sdds", bunch1, True), ("watch.sdds", bunch1, False),
        )
        result = run(
            {"lattice.lte": lattice, "run.ele": commands}, mode,
            {"beam.sdds": {"x": XS, "p": np.full(20, 2.0), "particleID": IDS,
                           "t": np.where(IDS <= 32, 0.0, 1e-8), "IDSlotsPerBunch": 32}},
            ("out.sdds", *(name for name, _, _ in selections)),
        )
        if not np.array_equal(particles(result["out.sdds"])["particleID"], IDS[retained]):
            raise RuntimeError("Retained particle IDs differ from the aperture prediction")
        defects = []
        for name, selected, affected in selections:
            data = result[name]
            count = int(np.count_nonzero(selected))
            # CHARGE.PER_PARTICLE is a common charge magnitude; sum it over selected live IDs.
            expected = math.fsum(QMACRO for _ in range(count))
            charge = scalar(data, "Charge")
            if scalar(data, "Particles") != count:
                raise RuntimeError("Selected-particle-count control failed")
            schema = next(p for p in data.parameters if p.name == "Charge")
            if schema.nm["units"] != "C":
                raise RuntimeError("The Charge schema changed")
            if affected:
                faulty = QMACRO * np.count_nonzero(retained)
                defects.append(conversion_defect(charge, expected, faulty, rtol=RTOL))
            elif not math.isclose(charge, expected, rel_tol=RTOL, abs_tol=0):
                raise RuntimeError("Whole-beam or selected WATCH charge control failed")
            bins = "WATCH control"
            if name != "watch.sdds":
                freq = np.asarray(data[0, "xFrequency"])
                if not np.all(np.isfinite(freq)):
                    raise RuntimeError("Nonfinite histogram bins")
                units = next(c for c in data.columns if c.name == "xFrequency").nm["units"]
                if name == "raw.sdds":
                    expected_bins = [1, 0, 1, 0, 1] if loss else [1] * 5
                    if units != "Particles/Bin" or not np.array_equal(freq, expected_bins):
                        raise RuntimeError("Raw-bin control failed")
                    bins = f"counts={freq.astype(int).tolist()} exact"
                else:
                    # The native binning interval expands by 1+1e-12 to include edge particles.
                    # Derive the effective width from the known fixture, not stored centers.
                    width = float(np.ptp(XS[selected])) * 1.25 * (1 + 1e-12) / 5
                    integral = float(np.sum(freq)) * width
                    if units != "1/m" or not math.isclose(integral, 1, rel_tol=RTOL, abs_tol=0):
                        raise RuntimeError("Normalized-bin integral control failed")
                    bins = f"density integral={integral:.15g}; integrated charge={charge * integral:.15g} C"
            print(f"{mode}/loss={int(loss)}/{name}: Particles={count}; Charge={charge:.15g} C "
                  f"expected={expected:.15g} C (selected IDs * q_macro); {bins}")
        return defects


    def main():
        defects = []
        for mode in modes():
            for loss in (False, True):
                defects.extend(case(mode, loss))
        verdict(all(defects))


    if __name__ == "__main__":
        main()

    ```

## Provenance

| Role | Model | Effort | Date | Code examined |
|---|---|---|---|---|
| Found | `gpt-6-astra` | xhigh | 2026-11-06 | master (`0355dd6aca12`), source with SDDS a3f730a14c49, CPU, double, serial and 3-rank MPI, one thread |
| Confirmed<br><small>arbitrary selected particle-ID subsets, count and sparse modes, aperture losses</small> | `gpt-6.1-sol` | xhigh | 2026-11-06 | master (`0355dd6aca12`), source with SDDS a3f730a14c49, CPU, double, serial and 3-rank MPI, one thread |

