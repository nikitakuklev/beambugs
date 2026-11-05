"""TAO-17: Phase residual recentering depends on array names rather than data types

Renaming an array of phase constraints changes Tao’s optimization objective even when the data types, targets, weights, and reference elements stay identical. A single phase.a datum in an array named phase.a has zero merit, while the same datum in match.a has merit 0.2197. The manual describes removal of the arbitrary phase origin by data type and reference element, without this array-name requirement. An explicit BEGINNING reference gives the same nonzero merit for the named array; unequal weights also expose the arithmetic mean used for recentering.

The script compares identical phase data under two array names, an explicit-reference control, and an unequal-weight control. It derives the drift phase analytically and computes the expected residuals and merit contributions.
"""

import numpy as np
from _bmad import compare, verdict
from _tao_contracts_normal_form import LATTICE, record, scalar, tao


def merit(name, reference, targets, weights):
    init = f"""&tao_params
 global%opt_with_ref = F
 global%opt_with_base = F
/
&tao_d2_data
 d2_data%name = '{name}'
 n_d1_data = 1
/
&tao_d1_data
 ix_d1_data = 1
 d1_data%name = 'a'
 default_data_type = 'phase.a'
 default_merit_type = 'target'
"""
    for i, (target, weight) in enumerate(zip(targets, weights, strict=True), 1):
        init += f" datum({i})%ele_name = 'E'\n datum({i})%ele_ref_name = '{reference}'\n datum({i})%meas = {target}\n datum({i})%weight = {weight}\n"
    datum_files = [f"datum-{i}.txt" for i in range(1, len(targets) + 1)]
    result = tao(
        LATTICE,
        [
            *[
                f"pipe -write {filename} data {name}.a[{i}]"
                for i, filename in enumerate(datum_files, 1)
            ],
            "pipe -write merit.txt merit",
        ],
        files={"phase.init": init + "/\n"},
        options=["-lat", "lat.bmad", "-init", "phase.init"],
        outputs=["merit.txt", *datum_files],
    )
    for filename in datum_files:
        text = result.outputs[filename].decode()
        for flag in ("good_model", "useit_opt"):
            assert record(text, flag)[3] == "T", text
        assert record(text, "data_type")[3] == "phase.a", text
        assert record(text, "ele_ref_name")[3] == reference, text
        assert not compare(
            "active phase datum (rad), analytic drift", scalar(text, "model_value"), phi, atol=3e-15
        )
    return float(result.outputs["merit.txt"])


# Drift phase from the Courant-Snyder matrix: atan2(L,beta0-alpha0*L).
phi = np.arctan2(1.7, 3 - 0.2 * 1.7)
named = merit("phase", "", [0.1], [1])
renamed = merit("match", "", [0.1], [1])
reference = merit("phase", "BEGINNING", [0.1], [1])
weighted = merit("phase", "", [0.1, 0.3], [1, 9])
assert not compare("named single phase, free origin", named, 0, atol=1e-15)
assert not compare("explicit BEGINNING reference", reference, (phi - 0.1) ** 2, atol=3e-15)
# The manual specifies an average phase; the implementation uses the arithmetic mean.
residuals = phi - np.array([0.1, 0.3])
residuals -= residuals.mean()
assert not compare(
    "unequal weights, arithmetic recentering control",
    weighted,
    np.dot([1, 9], residuals**2),
    atol=3e-15,
)
verdict(
    compare(
        "renamed single phase, zero residual from the free-origin contract", renamed, 0, atol=3e-15
    )
)
