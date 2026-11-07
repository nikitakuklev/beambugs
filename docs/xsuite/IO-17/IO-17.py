"""IO-17: Reverse MAD-X loading crashes on string LINE components

Loading MAD-X input with reverse_lines can crash when any imported LINE has string composer components. The reverse loader treats every component as a placement object, including components in lines not selected for reversal. A valid 0.7300 m drift–quadrupole–drift LINE raises AttributeError, while the equivalent SEQUENCE reverses successfully with the expected order and fields. Forward loading succeeds for both representations.

Independently validate LINE and SEQUENCE inputs in MAD-X, then load both forward and reversed in Xtrack. Check nested LINE input and a reverse SEQUENCE with a separate unselected LINE, comparing order, total length and quadrupole field parity.
"""

from _madx_io import consume
from _prebuilt import identity, np, xt


def main():
    identity()
    print("EXECUTION native Python import and MAD-X parsing; no tracker or tracking kernel")
    definitions = "a:drift,l=0.17; b:quadrupole,l=0.25,k1=0.4; c:drift,l=0.31;"
    sequence = "ring:sequence,l=0.73,refer=entry; a,at=0; b,at=0.17; c,at=0.42; endsequence;"
    cases = {
        "LINE": definitions + "ring:line=(a,b,c);",
        "SEQUENCE": definitions + sequence,
        "SEQUENCE with unused LINE": definitions + "extra:line=(a);" + sequence,
        "nested LINE": definitions + "inner:line=(a,b); ring:line=(inner,c);",
    }
    failures = 0
    for label, source in cases.items():
        target = consume(source, "ring")
        assert target["status"] == "accepted"
        assert abs(target["length"] - 0.73) < 2e-16
        for reverse in (False, True):
            try:
                env = xt.load(
                    string=source, format="madx", reverse_lines=["ring"] if reverse else None
                )
            except AttributeError as exc:
                if not reverse:
                    raise
                failures += 1
                print(
                    f"IMPORT {label} reverse={reverse}: observed=AttributeError({exc}) "
                    "expected=accepted reversed 0.73m lattice derivation=same physical objects"
                )
                continue
            line = env.lines["ring"]
            names = list(line.element_names)
            strength = [float(line[n].k1) for n in names if isinstance(line[n], xt.Quadrupole)]
            np.testing.assert_allclose(strength, [-0.4 if reverse else 0.4], rtol=0, atol=0)
            assert abs(line.get_length() - 0.73) < 2e-16
            # Reversal adds a line-name suffix to each environment identifier.
            order = [n.split("/")[0] for n in names]
            assert order == (["c", "b", "a"] if reverse else ["a", "b", "c"])
            print(
                f"IMPORT {label} reverse={reverse}: observed=accepted length={line.get_length():.17g} "
                f"names={names} k1={strength} expected_length=0.73 "
                "derivation=reverse order and source-defined normal-quadrupole parity"
            )
    print("DEFECT REPRODUCED" if failures else "NOT REPRODUCED")


if __name__ == "__main__":
    main()
