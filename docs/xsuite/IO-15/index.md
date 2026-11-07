---
title: "IO-15: Repeated element occurrences fail MAD-X sequence export"
hide:
  - toc
---

# IO-15 — Repeated element occurrences fail MAD-X sequence export

| | |
|---|---|
| Code | [Xsuite](../index.md) · Lattices, particles and I/O |
| Priority | P4 |
| Failure | Crash or API hazard |
| Evidence | Reproduced, independently rechecked |
| Last reproduced | 2026-11-07 on xtrack release [v0.118.6](https://github.com/xsuite/xtrack/tree/v0.118.6) (`f09e0fb`) |
| Discussed in | [[gh:https://github.com/xsuite/xsuite/issues/481]] |
| Source | [to_madx_sequence, mad_writer.py:720–745](https://github.com/xsuite/xtrack/blob/ad5e808cdc5b9a881eac2f41e32a0b7d11cc7028/xtrack/mad_writer.py#L720-L745)<br>[public export contract, line.py:877–894](https://github.com/xsuite/xtrack/blob/ad5e808cdc5b9a881eac2f41e32a0b7d11cc7028/xtrack/line.py#L877-L894) |

Exporting a line with repeated references to an ordinary element raises KeyError instead of producing its MAD-X sequence. The writer stores positions under occurrence-specific names but retrieves them using the shared element name. Two 0.4000 m drifts separated by 0.7000 m should occupy centres 0.2000 and 1.300 m, as an independently written MAD-X sequence confirms. Distinct native identifiers export successfully; repeated markers and multipoles also fail.

Run `python IO-15.py` with [_madx_io.py](../helpers/_madx_io.py), [_prebuilt.py](../helpers/_prebuilt.py) and [_common.py](../helpers/_common.py) in the same directory.

??? example "Reproducer: IO-15.py"

    [Download IO-15.py](IO-15.py)

    ```python
    """IO-15: Repeated element occurrences fail MAD-X sequence export

    Exporting a line with repeated references to an ordinary element raises KeyError instead of producing its MAD-X sequence. The writer stores positions under occurrence-specific names but retrieves them using the shared element name. Two 0.4000 m drifts separated by 0.7000 m should occupy centres 0.2000 and 1.300 m, as an independently written MAD-X sequence confirms. Distinct native identifiers export successfully; repeated markers and multipoles also fail.

    Export repeated drifts, markers and multipoles and compare with independently specified occurrence positions. Parse a handwritten sequence and a uniquely named native control in MAD-X, using the same ordered physical bodies.
    """

    from _madx_io import consume
    from _prebuilt import identity, np, xt


    def main():
        identity()
        print("EXECUTION Python serialization and MAD-X parsing; no tracker or tracking kernel")
        failures = 0
        for kind in ("Drift", "Marker", "Multipole"):
            element = (
                xt.Drift(length=0.4)
                if kind == "Drift"
                else xt.Marker()
                if kind == "Marker"
                else xt.Multipole(knl=[0.001, 0.12, -0.3])
            )
            line = xt.Line(
                elements={"r": element, "gap": xt.Drift(length=0.7)}, element_names=["r", "gap", "r"]
            )
            length = 0.4 if kind == "Drift" else 0.0
            expected_s = [0.0, length, length + 0.7, 2 * length + 0.7]
            table = line.get_table()
            np.testing.assert_allclose(table.s, expected_s, rtol=0, atol=3e-16)
            try:
                source = line.to_madx_sequence("testseq")
            except KeyError as exc:
                failures += 1
                print(
                    f"REPEATED {kind}: observed=KeyError({exc!s}) expected=sequence "
                    f"entry_positions={expected_s} derivation=ordered physical lengths"
                )
            else:
                result = consume(source)
                ok = result["status"] == "accepted"
                failures += not ok
                print(f"REPEATED {kind}: observed={result} expected=accepted pass={ok}")
        oracle = "r:drift,l=0.4; testseq:sequence,l=1.5; r,at=0.2; r,at=1.3; endsequence;"
        result = consume(oracle)
        assert result["status"] == "accepted"
        rows = [e for e in result["elements"] if e[0] == "r"]
        np.testing.assert_allclose([e[1:3] for e in rows], [[0.2, 0.4], [1.3, 0.4]], rtol=0, atol=0)
        assert result["length"] == 1.5
        print(
            f"HANDWRITTEN_CONTROL: observed={rows} expected_centres=[0.2,1.3] "
            "expected_length=1.5 derivation=sum of physical lengths"
        )
        line = xt.Line(
            elements={
                "ra": xt.Drift(length=0.4),
                "gap": xt.Drift(length=0.7),
                "rb": xt.Drift(length=0.4),
            }
        )
        result = consume(line.to_madx_sequence("testseq"))
        assert result["status"] == "accepted"
        np.testing.assert_allclose(
            [e[1:3] for e in result["elements"]],
            [[0.2, 0.4], [0.75, 0.7], [1.3, 0.4]],
            rtol=0,
            atol=3e-16,
        )
        print(f"UNIQUE_CONTROL: observed={result} expected=three bodies at centres 0.2,0.75,1.3")
        print("DEFECT REPRODUCED" if failures else "NOT REPRODUCED")


    if __name__ == "__main__":
        main()

    ```

## Provenance

| Role | Model | Effort | Date | Code examined |
|---|---|---|---|---|
| Found | `gpt-6-astra` | xhigh | 2026-11-06 | xtrack 0.118.2 (`ad5e808cdc5b`), installed packages, prebuilt kernels (compilation forbidden), CPU, double, serial |
| Confirmed | `gpt-6.1-sol` | xhigh | 2026-11-06 | xtrack 0.118.2 (`ad5e808cdc5b`), installed packages, prebuilt kernels (compilation forbidden), CPU, double, serial |

