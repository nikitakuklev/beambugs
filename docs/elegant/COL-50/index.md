---
title: "COL-50: CSRCSBEND BIN_ONCE wake output corrupts retained histograms and accesses past their ends"
hide:
  - toc
---

# COL-50 — CSRCSBEND BIN_ONCE wake output corrupts retained histograms and accesses past their ends

| | |
|---|---|
| Code | [Elegant](../index.md) · Collective effects |
| Priority | P3 |
| Severity | Silent wrong result |
| Evidence | Reproduced, independently rechecked |
| Last reproduced | 2026-11-05 on Elegant commit [`0355dd6`](https://github.com/rtsoliday/elegant/commit/0355dd6aca12), committed 2026-11-03 |
| Source | [`track_through_csbendCSR`, csbend.c:3686–3695](https://github.com/rtsoliday/elegant/blob/0355dd6aca12c861971fbf8f3ae78a0ded991734/src/csbend.c#L3686-L3695)<br>[`track_through_csbendCSR`, csbend.c:3712–3716](https://github.com/rtsoliday/elegant/blob/0355dd6aca12c861971fbf8f3ae78a0ded991734/src/csbend.c#L3712-L3716)<br>[`track_through_csbendCSR`, csbend.c:2973–2979](https://github.com/rtsoliday/elegant/blob/0355dd6aca12c861971fbf8f3ae78a0ded991734/src/csbend.c#L2973-L2979)<br>[`csrcsbend_param`, track_data.c:2058–2069](https://github.com/rtsoliday/elegant/blob/0355dd6aca12c861971fbf8f3ae78a0ded991734/src/track_data.c#L2058-L2069)<br>[`track_through_csbendCSR`, csbend.c:3442–3449](https://github.com/rtsoliday/elegant/blob/0355dd6aca12c861971fbf8f3ae78a0ded991734/src/csbend.c#L3442-L3449)<br>[`track_through_csbendCSR`, csbend.c:3613–3617](https://github.com/rtsoliday/elegant/blob/0355dd6aca12c861971fbf8f3ae78a0ded991734/src/csbend.c#L3613-L3617) |

With `BIN_ONCE=1` and a wake `OUTPUT_FILE`, CSRCSBEND leaves its retained histogram scaled after writing each page and attempts to restore elements one past both histogram arrays. Later output pages should retain the initial density and derivative, but for a 100.0 pC bunch represented by 4096 particles, the second page integrates to 7.319×10⁻⁴ pC and the fourth to 3.921×10⁻¹⁴ pC. The derivative grows by a factor of 5.321×10⁷ per additional page instead of remaining unchanged, and Valgrind confirms invalid reads and writes. The tested tracking kicks are unchanged by wake output.

Run `python COL-50.py` with [_csr.py](../helpers/_csr.py) and [_elegant.py](../helpers/_elegant.py) in the same directory.

??? example "Reproducer: COL-50.py"

    [Download COL-50.py](COL-50.py)

    ```python
    """COL-50: CSRCSBEND BIN_ONCE wake output corrupts retained histograms and accesses past their ends

    With `BIN_ONCE=1` and a wake `OUTPUT_FILE`, CSRCSBEND leaves its retained histogram scaled after writing each page and attempts to restore elements one past both histogram arrays. Later output pages should retain the initial density and derivative, but for a 100.0 pC bunch represented by 4096 particles, the second page integrates to 7.319×10⁻⁴ pC and the fourth to 3.921×10⁻¹⁴ pC. The derivative grows by a factor of 5.321×10⁷ per additional page instead of remaining unchanged, and Valgrind confirms invalid reads and writes. The tested tracking kicks are unchanged by wake output.

    Setup: `valgrind` on `PATH`; the script runs some cases of serial Elegant under Memcheck.

    The script tracks deterministic Gaussian quantiles through a four-slice bend and compares all-page, last-only, interval-two and disabled wake output, with a rebinned control. It integrates the reported current, checks the repeated scaling against the macroparticle charge and bin width, and compares final coordinates. It also runs the retained and rebinned cases under Valgrind.
    """

    import re

    import numpy as np
    from _csr import C, page, track, valgrind
    from _elegant import particles, verdict
    from scipy.special import ndtri

    N, CHARGE, P0 = 4096, 100e-12, 5000.0
    RTOL = 3e-14
    BEAM = {
        "t": 1e-12 * ndtri((np.arange(N) + 0.5) / N),
        "p": np.full(N, P0),
        "particleID": np.arange(1, N + 1),
    }


    def elements(once: int, last: int = 0, output: bool = True, interval: int = 1) -> str:
        wake = (
            f',OUTPUT_FILE="wake.sdds",OUTPUT_LAST_WAKE_ONLY={last},OUTPUT_INTERVAL={interval}'
            if output
            else ""
        )
        return (
            f"Q: CHARGE,TOTAL={CHARGE:.17g}\n"
            "B: CSRCSBEND,L=0.8,ANGLE=0.08,N_SLICES=4,BINS=64,"
            f"BIN_ONCE={once},STEADY_STATE=1,SG_HALFWIDTH=1,SG_ORDER=1,"
            f"SYNCH_RAD=0,ISR=0{wake}\n"
        )


    def memory_check(label: str, lattice: str) -> tuple[int, int]:
        status, log = valgrind(lattice, "Q,B", BEAM, P0)
        if status:
            raise RuntimeError(f"Valgrind/Elegant exited {status}:\n{log[-4000:]}")
        match = re.search(r"ERROR SUMMARY: (\d+) errors from (\d+) contexts", log)
        if match is None:
            raise RuntimeError("Valgrind did not produce its error summary")
        errors, contexts = map(int, match.groups())
        if errors and label == "BIN_ONCE=1":
            # Require the reported accesses to be precisely beyond the two 64-double arrays.
            # Function names and allocation size are stable if a later build moves line numbers.
            accesses = re.findall(
                r"Invalid (?:read|write) of size 8\n[^\n]*track_through_csbendCSR[^\n]*"
                r"(?:(?!Invalid |ERROR SUMMARY:)[\s\S])*?0 bytes after a block of size 512 alloc'd",
                log,
            )
            if len(accesses) != 4 or (errors, contexts) != (16, 4):
                raise RuntimeError(f"Unexpected memory errors:\n{log}")
        print(f"serial Valgrind {label}: errors={errors} contexts={contexts}; expected=0 errors")
        return errors, contexts


    def main() -> None:
        results = {}
        for name, once, last, output, interval in (
            ("all", 1, 0, True, 1),
            ("last", 1, 1, True, 1),
            ("off", 1, 0, False, 1),
            ("rebin", 0, 0, True, 1),
            ("interval2", 1, 0, True, 2),
        ):
            result = track(
                elements(once, last, output, interval),
                "Q,B",
                BEAM,
                P0,
                outputs=("out.sdds", "wake.sdds") if output else ("out.sdds",),
            )
            final = particles(result["out.sdds"])
            if not np.array_equal(final["particleID"], BEAM["particleID"]):
                raise RuntimeError("Unexpected particle loss")
            values: dict = {"final": np.column_stack([final[k] for k in ("x", "xp", "y", "yp", "t", "p")])}
            if output:
                wake = result["wake.sdds"]
                pages = [page(wake, i) for i in range(wake.n_pages)]
                dct = pages[0][1]["BinSize"]
                density = np.array([columns["LinearDensity"] for columns, _ in pages])
                derivative = np.array([columns["LinearDensityDeriv"] for columns, _ in pages])
                # LinearDensity is current (C/s); each bin spans dt = dct/c, so sum(I dt) = Q.
                charges = np.sum(density, axis=1) * dct / C
                values.update(density=density, derivative=derivative, dct=dct, charge=charges)
                print(
                    f"serial {name}: integrated_charge_C={charges.tolist()}; expected={CHARGE:.16g} per page"
                )
            results[name] = values

        all_pages, last = results["all"], results["last"]
        if (
            len(all_pages["charge"]) != 4
            or len(last["charge"]) != 1
            or len(results["interval2"]["charge"]) != 2
        ):
            raise RuntimeError("Unexpected number of wake output pages")
        qfactor = CHARGE / N * C
        dfactor = CHARGE / N * C**2 / all_pages["dct"]
        print(
            f"serial extra scaling per page: density={qfactor:.16g} derivative={dfactor:.16g}; expected=1 for both"
        )
        np.testing.assert_allclose(all_pages["charge"][0], CHARGE, rtol=RTOL, atol=0)
        # Rebinning evolves the distribution and filtering introduces a small integration residual.
        # It must conserve charge to 3e-8 relative, as opposed to losing five orders per output.
        np.testing.assert_allclose(results["rebin"]["charge"], CHARGE, rtol=3e-8, atol=0)
        corrupted = []
        for key, factor in (("density", qfactor), ("derivative", dfactor)):
            data = all_pages[key]
            reference = np.broadcast_to(data[0], data.shape)
            changed = not np.allclose(data, reference, rtol=RTOL, atol=0)
            corrupted.append(changed)
            # A missing restore leaves physical-unit values in internal arrays. Each subsequent
            # output multiplies them again by q_macro*c (density) or q_macro*c^2/dct (derivative).
            expected = data[0][None, :] * (factor if changed else 1.0) ** np.arange(4)[:, None]
            np.testing.assert_allclose(data, expected, rtol=RTOL, atol=0)
            np.testing.assert_allclose(last[key][0], data[0], rtol=RTOL, atol=0)
            interval_expected = data[0][None, :] * (factor if changed else 1.0) ** np.arange(2)[:, None]
            np.testing.assert_allclose(results["interval2"][key], interval_expected, rtol=RTOL, atol=0)
            print(
                f"serial {key}: retained pages unchanged={not changed}; expected=True; last-only matches first page=True"
            )
        print(
            f"serial peak derivative_C_per_s2={np.max(abs(all_pages['derivative']), axis=1).tolist()}; expected={np.max(abs(all_pages['derivative'][0])):.16g} per page"
        )
        for name in ("last", "off", "interval2"):
            difference = np.max(abs(results[name]["final"] - all_pages["final"]))
            print(
                f"serial final coordinates all vs {name}: max_abs_difference={difference:.16g}; expected=0"
            )
            if not np.array_equal(results[name]["final"], all_pages["final"]):
                raise RuntimeError("Wake output changed tracking; the unchanged-kick control failed")
        errors, _ = memory_check("BIN_ONCE=1", elements(1))
        control_errors, _ = memory_check("BIN_ONCE=0", elements(0))
        if control_errors:
            raise RuntimeError("Rebinning control has memory errors")
        verdict(any(corrupted) or errors > 0)


    if __name__ == "__main__":
        main()

    ```

## Provenance

| Role | Model | Effort | Date | Code examined |
|---|---|---|---|---|
| Found<br><small>missing histogram restoration loop; source only</small> | `gpt-6-astra` | xhigh | 2026-11-04 | master (`0355dd6aca12`), source with SDDS a3f730a14c49, CPU, serial |
| Confirmed | `gpt-6.1-sol` | xhigh | 2026-11-05 | master (`0355dd6aca12`), source with SDDS a3f730a14c49, CPU, serial |

