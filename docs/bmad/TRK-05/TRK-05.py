"""TRK-05: AC cylindrical magnetic modes contain an extra radial azimuthal factor

AC cylindrical magnetic modes with azimuthal order above one return an extra radial magnetic field.
For an order-two, 80 MHz mode at (11, −7, 190) mm, the horizontal field is −1.491 µT instead of
372.4 µT from the curl of the documented electric field. Cartesian finite differences give a
magnetic divergence of −0.06803 T/m where Maxwell’s equations require zero. Electric-coefficient
modes and order-one magnetic modes pass the same checks.

The script constructs 28 parsed maps and evaluates 4,116 samples around 84 centers for orders 1, 2
and 4, both Fourier signs, two frequencies and both coefficient families. An entire Bessel series
supplies an independent analytic curl, while Cartesian stencils check divergence and Faraday’s law.
"""

import os

# Set numerical pool sizes before NumPy/SciPy are imported.
for _thread_variable in (
    "OMP_NUM_THREADS",
    "OPENBLAS_NUM_THREADS",
    "MKL_NUM_THREADS",
    "NUMEXPR_NUM_THREADS",
):
    os.environ[_thread_variable] = "1"


import numpy as np
from _bmad import compare, verdict
from _s3_19_cylindrical import PROGRAM, modes_and_samples, oracle
from _s3_19_fields_photons_ptc import control, execute

modes, samples, groups, files = modes_and_samples()
output = execute(PROGRAM, [], files)
fields = {}
for line in output.splitlines():
    if not line.startswith("SAMPLE "):
        continue
    tokens = line.split()
    ident, ie = map(int, tokens[1:3])
    values = np.array(list(map(float, tokens[3:])))
    assert ident not in fields and samples[ident][0] == ie
    control(values[:4], samples[ident][1], 0)
    if not np.isfinite(values).all():
        raise RuntimeError(f"Nonfinite sample {ident}")
    fields[ident] = values[4:]
assert len(fields) == len(samples)
defect = False
max_control = max_div = max_faraday = 0.0
for group in groups:
    mode = modes[group["ie"] - 1]
    e, b, extra = oracle(mode, group["pos"])
    actual = fields[group["center"]]
    control(actual[:3], e, 2e-10 + 3e-13 * max(abs(e)))
    is_control = mode["family"] == "e" or mode["m"] == 1
    error = float(max(abs(actual[3:] - b)))
    if is_control:
        max_control = max(max_control, control(actual[3:], b, 3e-15))
    else:
        defect |= error > 3e-15
    compare(
        f"mode={group['ie']} family={mode['family']} m={mode['m']} "
        f"f={mode['frequency']:g} Hz Fourier-index={mode['index']} point={group['point']}, "
        "B (T), curl(E)/(i omega)",
        actual[3:],
        b,
        atol=3e-15,
    )
    # The 4-point stencils differentiate the library E and B in Cartesian coordinates.
    for stencils in group["stencils"]:
        d = np.array(
            [
                (np.array([1, -8, 8, -1]) / (12 * h)) @ np.array([fields[i] for i in ids])
                for h, ids in stencils
            ]
        )
        curl_e = np.array([d[1, 2] - d[2, 1], d[2, 0] - d[0, 2], d[0, 1] - d[1, 0]])
        divergence = float(np.trace(d[:3, 3:]))
        scale = max(float(max(abs(curl_e))), float(max(abs(d[3, 3:]))))
        faraday = float(max(abs(curl_e + d[3, 3:]))) / scale
        if is_control:
            assert abs(divergence) < 2e-9 and faraday < 2e-9
        max_div = max(max_div, abs(divergence))
        max_faraday = max(max_faraday, faraday)
        if (
            mode["family"] == "b"
            and mode["m"] == 2
            and mode["index"] == 1
            and mode["frequency"] == 8e7
            and mode["coefficient"].real > 0
            and group["point"] == 0
        ):
            print(
                f"h={stencils[0][0]:g} m: div B={divergence:.16g} T/m expected=0; Faraday x={(curl_e + d[3, 3:])[0]:.16g} V/m² expected=0"
            )
print(
    f"{len(modes)} maps, {len(groups)} centers, {len(fields)} samples; electric/m=1 control max B error={max_control:.3g} T"
)
print(
    f"Max |div B|={max_div:.16g} T/m; max relative Faraday residual={max_faraday:.16g}; expected=0"
)
verdict(defect)
