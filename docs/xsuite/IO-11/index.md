---
title: "IO-11: Survey assigns fictitious physical exits to thin elements"
hide:
  - toc
---

# IO-11 — Survey assigns fictitious physical exits to thin elements

| | |
|---|---|
| Code | [Xsuite](../index.md) · Lattices, particles and I/O |
| Priority | P4 |
| Failure | Silent wrong result |
| Evidence | Reproduced, independently rechecked |
| Last reproduced | 2026-11-07 on xtrack release [v0.118.6](https://github.com/xsuite/xtrack/tree/v0.118.6) (`f09e0fb`) |
| Source | [`get_misaligned_element_survey`, misalignment_survey.py:368–405](https://github.com/xsuite/xtrack/blob/ad5e808cdc5b9a881eac2f41e32a0b7d11cc7028/xtrack/survey/misalignment_survey.py#L368-L405)<br>[thin-slice declaration, slice_elements_thin.py:18–23](https://github.com/xsuite/xtrack/blob/ad5e808cdc5b9a881eac2f41e32a0b7d11cc7028/xtrack/beam_elements/slice_elements_thin.py#L18-L23)<br>[tracking wrapper, track_local_particle_with_transformations.h:99–107](https://github.com/xsuite/xtrack/blob/ad5e808cdc5b9a881eac2f41e32a0b7d11cc7028/xtrack/headers/track_local_particle_with_transformations.h#L99-L107)<br>[nominal survey, survey.py:439–441](https://github.com/xsuite/xtrack/blob/ad5e808cdc5b9a881eac2f41e32a0b7d11cc7028/xtrack/survey/survey.py#L439-L441)<br>[length and thickness contract, multipole.py:42–46](https://github.com/xsuite/xtrack/blob/ad5e808cdc5b9a881eac2f41e32a0b7d11cc7028/xtrack/beam_elements/multipole.py#L42-L46)<br>[actual entrance and exit frames including misalignments, survey.rst:38–43](https://github.com/xsuite/xsuite/blob/e0913e0a18028b2fce06991aebed6150876124b3/docs/survey.rst#L38-L43)<br>[rotation matrices, xtrack.tex:3735–3760](https://github.com/xsuite/xsuite/blob/e0913e0a18028b2fce06991aebed6150876124b3/docs/physics_manual/xtrack.tex#L3735-L3760) |

Surveying thin elements with yaw or pitch reports a fictitious separation between their physical entrance and exit. The survey uses a thin slice's integrated-strength weight as a geometric length and also mishandles complete thin Multipoles carrying a nonzero length field. A 0.8000 m quadrupole with 1.700 mrad yaw and four equal kicks acquires a 0.3400 mm horizontal exit error per kick. Independent rigid geometry requires coincident endpoints; thick elements and thick slices give the correct positions.

Run `python IO-11.py` with [_prebuilt.py](../helpers/_prebuilt.py) and [_common.py](../helpers/_common.py) in the same directory.

??? example "Reproducer: IO-11.py"

    [Download IO-11.py](IO-11.py)

    ```python
    """IO-11: Survey assigns fictitious physical exits to thin elements

    Surveying thin elements with yaw or pitch reports a fictitious separation between their physical entrance and exit. The survey uses a thin slice's integrated-strength weight as a geometric length and also mishandles complete thin Multipoles carrying a nonzero length field. A 0.8000 m quadrupole with 1.700 mrad yaw and four equal kicks acquires a 0.3400 mm horizontal exit error per kick. Independent rigid geometry requires coincident endpoints; thick elements and thick slices give the correct positions.

    The script compares physical survey endpoints with a straight rigid axis derived from the documented rotation matrices. It checks four magnetic and two RF families, thick controls, standalone thin Multipoles and reversed survey endpoints. Zero-strength native kick maps independently check that the supported thin elements have no geometric advance, using serial prebuilt kernels with compilation forbidden.
    """

    import math

    from _prebuilt import CTX, identity, kernels, np, xt

    ATOL = 2e-15
    LENGTH = 0.8
    SHIFT = np.array([0.0011, -0.0007, 0.0002])
    FAMILIES = ("Quadrupole", "Sextupole", "Octupole", "Multipole", "Cavity", "CrabCavity")
    matches = []
    np.set_printoptions(precision=17, linewidth=10000)


    def close(actual, expected):
        np.testing.assert_allclose(actual, expected, rtol=0, atol=ATOL)


    def record(label, observed, expected, derivation, *, control=False):
        observed, expected = np.asarray(observed), np.asarray(expected)
        passed = bool(
            np.all(np.isfinite(observed)) and np.allclose(observed, expected, rtol=0, atol=ATOL)
        )
        if control:
            close(observed, expected)
        print(
            f"{label}: observed={observed} expected={expected} derivation={derivation} match={passed}"
        )
        matches.append(passed)
        return passed


    def rotations(yaw, pitch, roll=0.0):
        # Manual (x,y,s) convention: positive yaw sends s toward x; pitch sends s toward y.
        cy, sy = math.cos(yaw), math.sin(yaw)
        cp, sp = math.cos(pitch), math.sin(pitch)
        cr, sr = math.cos(roll), math.sin(roll)
        return (
            np.array([[cy, 0, sy], [0, 1, 0], [-sy, 0, cy]])
            @ np.array([[1, 0, 0], [0, cp, sp], [0, -sp, cp]])
            @ np.array([[cr, -sr, 0], [sr, cr, 0], [0, 0, 1]])
        )


    def survey_case(kind, yaw, pitch, *, mode="thin", scheme="Uniform", anchor=0.4, roll=0.0):
        strengths = {
            "Quadrupole": {"k1": 0.7},
            "Sextupole": {"k2": 2.0},
            "Octupole": {"k3": 3.0},
            "Multipole": {"knl": [0.0, 0.56], "isthick": True},
            "Cavity": {"voltage": 2e5, "frequency": 4e8},
            "CrabCavity": {"crab_voltage": 2e5, "frequency": 4e8},
        }
        element = getattr(xt, kind)(
            _context=CTX,
            length=LENGTH,
            rot_y_rad=yaw,
            rot_x_rad=pitch,
            rot_s_rad=roll,
            rot_shift_anchor=anchor,
            shift_x=SHIFT[0],
            shift_y=SHIFT[1],
            shift_s=SHIFT[2],
            **strengths[kind],
        )
        line = xt.Line(elements=[element], element_names=["mag"])
        if mode != "whole":
            slicing = (
                xt.slicing.Custom([0.08, 0.33, 0.69], mode=mode)
                if scheme == "Custom"
                else getattr(xt, scheme)(4, mode=mode)
            )
            line.slice_thick_elements(slicing_strategies=[xt.Strategy(slicing=slicing)])
        survey = line.survey(include_element_frames=True)
        nominal = line.survey()
        for col in ("X", "Y", "Z", "s", "theta", "phi", "psi"):
            np.testing.assert_array_equal(survey[col], nominal[col])
        rotation = rotations(yaw, pitch, roll)
        axis = rotation[:, 2]
        for i, name in enumerate(line.element_names):
            node = line[name]
            inherits = getattr(node, "rot_and_shift_from_parent", False)
            if not (inherits or getattr(node, "allow_rot_and_shift", False)):
                close(survey.XYZ_elem_start[i], survey.XYZ_ref_start[i])
                continue
            offset = node.slice_offset if inherits else 0.0
            weight = node.weight if inherits else 1.0
            # A point at parent coordinate t lies at d+a*ez+(t-a)*R*ez.
            # A strength weight changes the geometric endpoint only for a thick node.
            entrance = SHIFT + np.array([0.0, 0.0, anchor]) + (offset - anchor) * axis
            length = LENGTH * weight if node.isthick else 0.0
            expected = entrance + length * axis
            close(survey.XYZ_elem_start[i], entrance)
            close(survey.E_elem_start[i], rotation)
            close(survey.E_elem_end[i], rotation)
            control = node.isthick or weight == 0 or (yaw == 0 and pitch == 0)
            passed = record(
                f"{kind}/{mode}/{scheme}/{name} yaw={yaw:g} pitch={pitch:g} anchor={anchor:g}",
                survey.XYZ_elem_end[i],
                expected,
                "P(t)=d+a*ez+(t-a)*R*ez; thin exit=entrance",
                control=control,
            )
            if not passed:
                close(survey.XYZ_elem_end[i] - expected, LENGTH * weight * (axis - [0, 0, 1]))
            close(survey.get_frame(name, which="elem_end").XYZ, survey.XYZ_elem_end[i])
        return line, survey


    def main():
        identity()
        for kind in FAMILIES:
            survey_case(kind, 0.0017, 0.0)
            survey_case(kind, 0.0, -0.0023)
            for mode in ("thick", "whole"):
                survey_case(kind, 0.0017, -0.0023, mode=mode)
        for scheme in ("Teapot", "Custom"):
            for anchor in (0.0, 0.4, 0.8):
                survey_case("Quadrupole", -0.0017, 0.0023, scheme=scheme, anchor=anchor, roll=0.19)
        survey_case("Quadrupole", 0.0, 0.0, roll=0.19)
        _, survey = survey_case("Quadrupole", 0.0017, 0.0)
        reversed_survey = survey.reverse()
        for name in ("mag..0", "mag..1", "mag..2", "mag..3"):
            i = list(reversed_survey.name).index(name)
            record(
                f"reversed/{name}",
                reversed_survey.XYZ_elem_start[i],
                reversed_survey.XYZ_elem_end[i],
                "thin endpoint coincidence",
            )
        for thick in (False, True):
            element = xt.Multipole(
                _context=CTX, length=LENGTH, isthick=thick, knl=[0, 0.56], rot_y_rad=0.0017
            )
            survey = xt.Line(elements=[element]).survey(include_element_frames=True)
            expected = LENGTH * rotations(0.0017, 0.0)[:, 2] if thick else np.zeros(3)
            record(
                f"standalone Multipole isthick={thick}",
                survey.XYZ_elem_end[0],
                expected,
                "thick exit=L*R*ez; thin exit=0",
                control=thick,
            )
        for kind in FAMILIES:
            parent = getattr(xt, kind)(
                _context=CTX, length=LENGTH, rot_y_rad=0.0017, rot_x_rad=-0.0023, rot_shift_anchor=0.4
            )
            node = getattr(xt, "ThinSlice" + kind)(
                _buffer=parent._buffer, _parent=parent, weight=0.25, slice_offset=0.16
            )
            particle = xt.Particles(
                _context=CTX, p0c=2.3e9, x=0.001, y=-0.002, px=0.0003, py=-0.0002, zeta=0.0007
            )
            attrs = ("x", "px", "y", "py", "zeta", "delta", "s")
            before = np.array([getattr(particle, key)[0] for key in attrs])
            node.track(particle)
            assert particle.state[0] == 1
            record(
                f"zero-strength native {kind}",
                [getattr(particle, key)[0] for key in attrs],
                before,
                "zero-strength thin map=identity including s",
                control=True,
            )
        kernels()
        print(
            f"COMPARISONS total={len(matches)} pass={sum(matches)} fail={len(matches) - sum(matches)}"
        )
        print("NOT REPRODUCED" if all(matches) else "DEFECT REPRODUCED")


    if __name__ == "__main__":
        main()

    ```

## Provenance

| Role | Model | Effort | Date | Code examined |
|---|---|---|---|---|
| Found | <span class="bb-nowrap">`gpt-6-astra`</span> | xhigh | <span class="bb-nowrap">2026-11-06</span> | xtrack 0.118.2 (`ad5e808cdc5b`), installed packages, prebuilt kernels (compilation forbidden), CPU, double, serial |
| Confirmed | <span class="bb-nowrap">`gpt-6.1-sol`</span> | xhigh | <span class="bb-nowrap">2026-11-06</span> | xtrack 0.118.2 (`ad5e808cdc5b`), installed packages, prebuilt kernels (compilation forbidden), CPU, double, serial |

