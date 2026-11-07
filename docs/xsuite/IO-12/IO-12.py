"""IO-12: XYShift stores NaN for omitted zero offsets

Constructing the deprecated XYShift element with either transverse offset omitted stores NaN instead of its documented zero default. A 2.500 mm horizontal shift turns a finite vertical coordinate into NaN; subsequent line tracking marks the particle lost. Saving and reloading an explicitly valid element also introduces NaN whenever serialization omits a zero offset. Direct coordinate subtraction and round-trip preservation establish the expected values. Supplying both offsets works before serialization, and Translation preserves omitted coordinates.

The script applies the documented coordinate subtraction to fixed finite particles and checks omitted arguments, explicit offsets, survey and backtracking. It reloads valid zero-offset elements through element dictionaries, line dictionaries and an in-memory JSON file, with copy and Translation controls. It uses serial prebuilt kernels with compilation forbidden.
"""

import io
import warnings

from _prebuilt import CTX, identity, kernels, np, xt

XY = np.array([0.0012, -0.0034])
DX, DY = 0.0025, -0.0018
matches = []
np.set_printoptions(precision=17, linewidth=10000)


def record(label, observed, expected, derivation, *, control=False):
    observed, expected = np.asarray(observed), np.asarray(expected)
    passed = bool(np.all(np.isfinite(observed)) and np.array_equal(observed, expected))
    if control:
        np.testing.assert_array_equal(observed, expected)
    print(
        f"{label}: observed={observed} expected={expected} derivation={derivation} match={passed}"
    )
    matches.append(passed)


def particle():
    return xt.Particles(_context=CTX, p0c=2.3e9, x=XY[0], y=XY[1])


def main():
    identity()
    # The deprecation warning is documented; it does not replace zero defaults.
    warnings.filterwarnings("ignore", message="XYShift is deprecated", category=FutureWarning)
    for name, kwargs, offsets in (
        ("none", {}, (0.0, 0.0)),
        ("dx_only", {"dx": DX}, (DX, 0.0)),
        ("dy_only", {"dy": DY}, (0.0, DY)),
        ("explicit_zero_dx", {"dx": 0.0}, (0.0, 0.0)),
        ("both_zero", {"dx": 0.0, "dy": 0.0}, (0.0, 0.0)),
        ("explicit_horizontal", {"dx": DX, "dy": 0.0}, (DX, 0.0)),
        ("both_nonzero", {"dx": DX, "dy": DY}, (DX, DY)),
    ):
        element = xt.XYShift(_context=CTX, **kwargs)
        explicit = "dx" in kwargs and "dy" in kwargs
        record(
            name + "/fields",
            [getattr(element, field) for field in ("dx", "dy")],
            offsets,
            "omitted displacement=0",
            control=explicit,
        )
        forward = particle()
        element.track(forward)
        record(
            name + "/forward",
            [forward.x[0], forward.y[0]],
            XY - offsets,
            "(x,y)_out=(x,y)_in-(dx,dy)",
            control=explicit,
        )
        assert forward.state[0] == 1
        survey = xt.Line(elements=[element]).survey()
        record(
            name + "/survey",
            survey.XYZ[-1],
            [*offsets, 0.0],
            "reference origin advances by (dx,dy,0)",
            control=explicit,
        )
        line = xt.Line(elements=[element, xt.Drift(_context=CTX, length=0.2)])
        line.build_tracker(_context=CTX)
        backward = particle()
        line.track(backward, ele_start=0, num_elements=1, backtrack=True)
        record(
            name + "/backtrack",
            [backward.x[0], backward.y[0]],
            XY + offsets,
            "inverse translation adds (dx,dy)",
            control=explicit,
        )
        tracked = particle()
        line.track(tracked)
        record(
            name + "/line state",
            tracked.state,
            [1],
            "finite translated coordinates survive a 0.2 m drift",
            control=explicit,
        )
        if explicit:
            # Serialization legitimately elides scalar zeros. Reconstruction must
            # restore the documented defaults, preserving this finite map.
            saved = element.to_dict()
            print(f"{name}/serialized: {saved}")
            saved_line = xt.Line(elements=[element], element_names=["shift"])
            buffer = io.StringIO()
            saved_line.to_json(buffer)
            buffer.seek(0)
            for path, restored in (
                ("element_dict", xt.XYShift.from_dict(saved)),
                ("line_dict", xt.Line.from_dict(saved_line.to_dict())["shift"]),
                ("line_json", xt.Line.from_json(buffer)["shift"]),
                ("copy", element.copy()),
            ):
                record(
                    name + "/" + path,
                    [restored.dx, restored.dy],
                    offsets,
                    "round-trip preservation of explicitly valid offsets",
                    control=path == "copy" or 0.0 not in offsets,
                )
    for name, kwargs, offsets in (
        ("none", {}, (0.0, 0.0)),
        ("x", {"shift_x": DX}, (DX, 0.0)),
        ("y", {"shift_y": DY}, (0.0, DY)),
    ):
        element = xt.Translation(_context=CTX, **kwargs)
        tracked = particle()
        element.track(tracked)
        record(
            "Translation/" + name,
            [tracked.x[0], tracked.y[0]],
            XY - offsets,
            "documented replacement subtracts finite zero-default offsets",
            control=True,
        )
    kernels()
    print(
        f"COMPARISONS total={len(matches)} pass={sum(matches)} fail={len(matches) - sum(matches)}"
    )
    print("NOT REPRODUCED" if all(matches) else "DEFECT REPRODUCED")


if __name__ == "__main__":
    main()
