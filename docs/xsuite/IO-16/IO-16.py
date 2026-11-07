"""IO-16: Distinct native names collide after MAD-X escaping

MAD-X sequence export can map distinct native elements to the same target name and return text that MAD-X refuses to parse. Colon and slash replacements make q:one, q/one and q__one collide; case-only native differences also collide in MAD-X. Two quadrupoles of lengths 0.2000 and 0.4000 m become duplicate declarations despite having different gradients. Independently parsing a uniquely named control preserves both bodies and their expected positions.

Export quadrupoles with distinct native names that collide after escaping or case folding, then parse the text in MAD-X. Compare a uniquely named control with independently specified lengths, gradients and centre positions.
"""

from _madx_io import consume
from _prebuilt import identity, np, xt


def main():
    identity()
    print("EXECUTION Python serialization and MAD-X parsing; no tracker or tracking kernel")
    failures = 0
    for first, second in (
        ("q:one", "q__one"),
        ("q/one", "q__one"),
        ("q:one", "q/one"),
        ("Qone", "qone"),
        ("qa", "qb"),
    ):
        line = xt.Line(
            elements={
                first: xt.Quadrupole(length=0.2, k1=0.7),
                "gap": xt.Drift(length=0.3),
                second: xt.Quadrupole(length=0.4, k1=-0.2),
            }
        )
        assert len(set(line.element_names)) == 3
        source = line.to_madx_sequence("testseq")
        result = consume(source)
        ok = result["status"] == "accepted"
        if ok:
            np.testing.assert_allclose(
                [e[1:] for e in result["elements"]],
                [[0.1, 0.2, 0.7], [0.35, 0.3, 0.0], [0.7, 0.4, -0.2]],
                rtol=0,
                atol=2e-16,
            )
            assert abs(result["length"] - 0.9) < 2e-16
        if first == "qa":
            assert ok
        failures += not ok
        definitions = [s for s in source.splitlines() if ": quadrupole" in s]
        print(
            f"NAMES {first},{second}: declarations={definitions} observed={result} "
            "expected=two distinct quadrupoles at 0.1,0.7m with lengths 0.2,0.4m "
            f"derivation=independent physical body positions pass={ok}"
        )
    print("DEFECT REPRODUCED" if failures else "NOT REPRODUCED")


if __name__ == "__main__":
    main()
