"""LAT-10: openPMD beam reading converts units with the inverse unitSI ratio

Reading an openPMD beam file in units other than Bmad's metres, eV/c and seconds applies the
inverse unit conversion, silently changing the physical coordinates. A stored value multiplied
by its unitSI gives its SI value, so conversion requires the file's unitSI divided by the Bmad
unit's SI size; the reader uses the reciprocal. Positions stored in millimetres give x = 1000 m
instead of 1 mm, SI momenta give a reference momentum of 2.856×10⁻⁴⁸ eV/c instead of 10 MeV/c,
and nanosecond times give 4.100×10⁹ s instead of 4.100 ns. Files in Bmad's own units read
correctly.

The script writes, with h5py, an openPMD BeamPhysics file of one electron in Bmad's units and three
variants that store the same physical values with positions in mm, momenta in kg m/s or times in
ns, as other openPMD writers do. It reads each with the production program
`beam_file_translate_format` and compares the coordinates Bmad prints with the physical values. The
file in Bmad's own units is the control.
"""

import io

import h5py
import numpy as np
from _bmad import compare, run, verdict

EV_C = 1.602176634e-19 / 299792458.0  # SI size of 1 eV/c, kg m/s

# One electron: x = 1 mm, y = -2 mm, z = 0, px = 10 keV/c, py = -20 keV/c, pz = 10 MeV/c,
# reference momentum 10 MeV/c, time 4.1 ns.
POSITION = {"x": 1e-3, "y": -2e-3, "z": 0.0}  # m
MOMENTUM = {"x": 1e4, "y": -2e4, "z": 1e7}  # eV/c
P0C = 1e7  # eV/c
TIME = 4.1e-9  # s


def component(group, name, value, unit_si, dimension, constant=False):
    """Store value (in SI) as value/unit_si with that unitSI, the openPMD convention."""
    stored = value / unit_si
    if constant:
        obj = group.create_group(name)
        obj.attrs["value"] = np.float64(stored)
        obj.attrs["shape"] = np.array([1], dtype=np.uint64)
    else:
        obj = group.create_dataset(name, data=np.array([stored], dtype=np.float64))
    obj.attrs["unitSI"] = np.float64(unit_si)
    obj.attrs["unitDimension"] = np.array(dimension, dtype=np.float64)


def beam_file(length_unit, momentum_unit, time_unit):
    """openPMD BeamPhysics file of the electron, with the given unitSI per quantity."""
    buffer = io.BytesIO()
    with h5py.File(buffer, "w") as file:
        for key, value in {
            "openPMD": "2.0.0",
            "openPMDextension": "BeamPhysics;SpeciesType",
            "basePath": "/data/%T/",
            "particlesPath": "particles/",
        }.items():
            file.attrs[key] = np.bytes_(value)
        group = file.create_group("data/0/particles/electron")
        group.attrs["numParticles"] = np.int32(1)
        group.attrs["speciesType"] = np.bytes_("electron")
        length, momentum, time = (
            [1, 0, 0, 0, 0, 0, 0],
            [1, 1, -1, 0, 0, 0, 0],
            [0, 0, 1, 0, 0, 0, 0],
        )
        for axis, x in POSITION.items():
            component(group, f"position/{axis}", x, length_unit, length)
        for axis, p in MOMENTUM.items():
            component(group, f"momentum/{axis}", p * EV_C, momentum_unit, momentum)
        component(group, "totalMomentumOffset", P0C * EV_C, momentum_unit, momentum, True)
        component(group, "time", TIME, time_unit, time)
    return buffer.getvalue()


def validate_file(data):
    """Check each record against its SI value independently of the file-writing routine."""
    with h5py.File(io.BytesIO(data), "r") as file:
        group = file["data/0/particles/electron"]
        physical = {
            **{f"position/{axis}": x for axis, x in POSITION.items()},
            **{f"momentum/{axis}": p * EV_C for axis, p in MOMENTUM.items()},
            "totalMomentumOffset": P0C * EV_C,
            "time": TIME,
        }
        for name, expected in physical.items():
            obj = group[name]
            stored = obj[()] if isinstance(obj, h5py.Dataset) else obj.attrs["value"]
            np.testing.assert_allclose(stored * obj.attrs["unitSI"], expected, atol=0, rtol=3e-15)


def read_back(data):
    """Bmad's coordinates for the electron: x, px, y, py, z, pz, p0c, time."""
    validate_file(data)
    result = run(
        "beam_file_translate_format",
        ["beam.h5", "ascii"],
        files={"beam.h5": data},
        outputs=["beam.dat"],
    )
    if "ERROR" in result.log or "FATAL" in result.log:
        raise RuntimeError(f"Beam translation reported an error:\n{result.log}")
    rows = [line.split() for line in result.outputs["beam.dat"].decode().splitlines()]
    rows = [row for row in rows if row and not row[0].startswith("#")]
    if len(rows) != 1 or rows[0][13] != "Alive":
        raise RuntimeError(f"Expected one live particle:\n{result.log}")
    return np.array([float(value) for value in rows[0][1:8] + rows[0][9:10]])


# Bmad coordinates of the electron: px, py are momenta over p0c; pz = |p|/p0c - 1, written
# without cancellation; z = -beta c t, the z Bmad assigns to a particle read with time t at s = 0.
p = np.array(list(MOMENTUM.values()))
total = np.sqrt(p @ p)
beta = total / np.hypot(total, 0.51099895069e6)  # electron mass, eV
PZ = (p @ p - P0C**2) / (P0C * (total + P0C))
EXPECTED = np.array(
    [
        POSITION["x"],
        p[0] / P0C,
        POSITION["y"],
        p[1] / P0C,
        -beta * 299792458.0 * TIME,
        PZ,
        P0C,
        TIME,
    ]
)
OTHER = [0, 1, 2, 3, 4, 6, 7]
RTOL = 1e-13  # one unit conversion and the z/beta reconstruction round far below this
PZ_ATOL = 4e-16  # pz is the difference of two numbers near 1


def check(label, observed):
    """True when the read coordinates differ from the physical values."""
    other = compare(
        f"{label}: x, px, y, py, z, p0c, time, expected the physical values",
        observed[OTHER],
        EXPECTED[OTHER],
        atol=0,
        rtol=RTOL,
    )
    pz = compare(f"{label}: pz, expected {float(PZ)!r}", observed[5], PZ, atol=PZ_ATOL)
    return other or pz


print("units: x, y, z in m; px, py, pz relative to p0c; p0c in eV/c; time in s")
if check("Bmad units (m, eV/c, s)", read_back(beam_file(1.0, EV_C, 1.0))):
    raise RuntimeError("The file in Bmad's own units does not read back; comparison not valid.")
defect = False
for label, units, unaffected in (
    ("positions in mm (unitSI 1e-3)", (1e-3, EV_C, 1.0), [1, 3, 4, 5, 6, 7]),
    ("momenta in kg m/s (unitSI 1)", (1.0, 1.0, 1.0), [0, 1, 2, 3, 5, 7]),
    ("time in ns (unitSI 1e-9)", (1.0, EV_C, 1e-9), [0, 1, 2, 3, 5, 6]),
):
    observed = read_back(beam_file(*units))
    # A unit conversion of one quantity must preserve the other coordinates. pz loses
    # about 1e-16 to subtraction, while the other controls use a relative tolerance.
    for index in unaffected:
        np.testing.assert_allclose(
            observed[index], EXPECTED[index], atol=PZ_ATOL if index == 5 else 0, rtol=RTOL
        )
    defect |= check(label, observed)
verdict(defect)
