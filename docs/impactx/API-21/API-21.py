"""API-21: MAD-X nominal thin dipoles import as straight-reference steering kicks

Importing a MAD-X MULTIPOLE with a nonzero dipole coefficient KNL[0], which MAD-X defines as a zero-length reference bend, turns the bend into a steering kick. For a 17.00 mrad bend, the design particle receives normalized horizontal momentum pₓ = −0.01700 instead of zero, and the reference heading does not turn. At reference speed β₀ = 0.9428, the dispersive coefficient R₂₆ = ∂pₓ/∂pt and the path-length coefficient R₅₁ = ∂t/∂x (t the arrival-time distance, pt = −ΔE/(P₀c)) are zero instead of −0.01803 and +0.01803. The MAD-X manual and native MAD-X TRACK, TWISS and SURVEY establish the expected result. An imported HKICKER remains a correct straight-reference kick, and native ThinDipole bends the reference correctly.

Setup: run pinned to one CPU (`taskset -c <n>`); the shared `_slicing` helper refuses to run on more, and child processes inherit that CPU.

The script compares both signs of a nominal normal dipole at all model floors with native MAD-X TRACK, TWISS and SURVEY and with the R₂₆/R₅₁ coefficients of the analytic thin-bend Hamiltonian. Ordinary-kicker, zero-strength and native ThinDipole cases control kick and reference-bend semantics. Temporary decks and MAD-X scratch files live beside the shared helper and are removed on exit.
"""

from _slicing import environment, run


def main():
    environment()
    failures = 0
    cases = [(a, "multipole", m) for a in (0.017, -0.017) for m in ("linear", "paraxial", "exact")]
    cases += [
        (0.017, "kicker", "linear"),
        (0.0, "multipole", "linear"),
        (0.017, "native", "linear"),
    ]
    for angle, kind, model in cases:
        definition = (
            f"b:hkicker,kick={-angle};"
            if kind == "kicker"
            else f"b:multipole,knl={{{angle}}},lrad=0;"
        )
        result = run(
            dict(
                kind="thin",
                angle=angle,
                model=model,
                native_bend=kind == "native",
                deck="beam,particle=electron,gamma=3;"
                + definition
                + "cell:line=(b);use,period=cell;",
            )
        )
        # pt=-DeltaE/(P0*c), so delta=sqrt(1-2*pt/beta+pt^2)-1.
        # G=-theta*x*delta gives R26=-theta/beta and R51=+theta/beta.
        target = angle / result["beta"] if kind != "kicker" else 0
        expected_design = -angle if kind == "kicker" else 0
        expected_heading = -angle if kind != "kicker" else 0
        matrix = result["matrix"]
        native = result["native"]
        assert abs(native["matrix"][1][5] + target) < 2e-15
        assert abs(native["matrix"][4][0] - target) < 2e-15
        assert abs(native["rays"][0][1] - expected_design) < 2e-15
        assert abs(native["survey_theta"] - expected_heading) < 2e-15
        residual = max(
            abs(matrix[1][5] + target),
            abs(matrix[4][0] - target),
            abs(result["rays"][0][1] - expected_design),
            abs(result["heading"] - expected_heading),
        )
        if kind in ("kicker", "native") or angle == 0:
            assert residual < 2e-15
        else:
            failures += residual > 2e-15
        print(
            "THIN",
            kind,
            angle,
            model,
            "design_px",
            result["rays"][0][1],
            "expected_design_px",
            expected_design,
            "R26_R51",
            matrix[1][5],
            matrix[4][0],
            "expected",
            -target,
            target,
            "heading",
            result["heading"],
            "expected_heading",
            expected_heading,
            "oracle=canonical_generator/native_MAD-X_TRACK_TWISS_SURVEY",
        )
    print("DEFECT REPRODUCED" if failures else "NOT REPRODUCED")


if __name__ == "__main__":
    main()
