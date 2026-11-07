---
title: "TRK-43: FOREST and Bmad rest masses give different drift timing"
hide:
  - toc
---

# TRK-43 — FOREST and Bmad rest masses give different drift timing

| | |
|---|---|
| Code | [Bmad](../index.md) · Tracking, RF, fields and spin |
| Priority | P3 |
| Failure | Wrong physics |
| Evidence | Probe-pinned |
| Last reproduced | 2026-11-05 on Bmad release [20261031-0](https://github.com/bmad-sim/bmad-ecosystem/tree/20261031-0) (`c4c629f`) |
| Tags | <span class="bb-tag">partly-source-only</span> [?](../../methodology.md#tags) |
| Source | [FOREST rest masses](https://github.com/bmad-sim/bmad-ecosystem/blob/c4c629fc9071d3a56cb1ba5bc714baba65e9c5d3/forest/code/a_scratch_size.f90#L97-L103)<br>[physical_constants](https://github.com/bmad-sim/bmad-ecosystem/blob/c4c629fc9071d3a56cb1ba5bc714baba65e9c5d3/sim_utils/interfaces/physical_constants.f90#L10-L65) |

Changing an electron drift from native tracking to PTC changes its longitudinal coordinate because the two libraries use different electron rest masses. At reference momentum 1 MeV/c, relative momentum offset δ = 0.2 and length 1.7 m, the exit longitudinal coordinate differs by 8.450 × 10⁻¹⁰ m. The exact finite-speed drift formula agrees with each result when supplied with that library’s mass. The proton rest masses also differ in source.

Run `python TRK-43.py` with [_bmad.py](../helpers/_bmad.py) and [_tracking_contracts_materials.py](../helpers/_tracking_contracts_materials.py) in the same directory.

??? example "Reproducer: TRK-43.py"

    [Download TRK-43.py](TRK-43.py)

    ```python
    """TRK-43: FOREST and Bmad rest masses give different drift timing

    Changing an electron drift from native tracking to PTC changes its longitudinal coordinate because the two libraries use different electron rest masses. At reference momentum 1 MeV/c, relative momentum offset δ = 0.2 and length 1.7 m, the exit longitudinal coordinate differs by 8.450 × 10⁻¹⁰ m. The exact finite-speed drift formula agrees with each result when supplied with that library’s mass. The proton rest masses also differ in source.

    The script tracks the same fixed electron with native and PTC methods and evaluates the exact
    drift using each rest mass. It checks the transverse coordinates and the native result as
    controls; BMAD_SOURCE supplies the proton mass declarations.
    """

    import numpy as np
    from _bmad import compare, source, verdict
    from _tracking_contracts_materials import lattice, orbit, tao

    initial = np.array([0.001, 0.012, -0.002, -0.006, 0.03, 0.2])
    start = "\n".join(
        f"particle_start[{k}]={v}"
        for k, v in zip(("x", "px", "y", "py", "z", "pz"), initial, strict=True)
    )


    def exact(mass):
        # z=-beta*c*(t-t0): drift adds L*beta/beta0 minus the slanted path length.
        p0, length = 1e6, 1.7
        beta = 1.2 * p0 / np.hypot(1.2 * p0, mass)
        beta0 = p0 / np.hypot(p0, mass)
        ps = np.sqrt(1.2**2 - initial[1] ** 2 - initial[3] ** 2)
        expected = initial.copy()
        expected[[0, 2]] += length * initial[[1, 3]] / ps
        expected[4] += length * (beta / beta0 - 1.2 / ps)
        return expected


    command = "pipe ele:orbit 1"
    results = {}
    for method in ("bmad_standard", "symp_lie_ptc"):
        lat = lattice(f"e: drift, l=1.7, tracking_method={method}", start=start, p0c=1e6)
        results[method] = orbit(tao(lat, [command])[command])
    expected = exact(510998.95069)
    if compare(
        "native control, exact drift (m and normalized momenta)",
        results["bmad_standard"],
        expected,
        atol=3e-15,
        rtol=5e-13,
    ):
        raise AssertionError("Native drift control failed")
    if compare(
        "PTC transverse control",
        results["symp_lie_ptc"][[0, 1, 2, 3, 5]],
        expected[[0, 1, 2, 3, 5]],
        atol=3e-15,
        rtol=5e-13,
    ):
        raise AssertionError("PTC transverse control failed")
    defect = compare(
        "PTC z (m), Bmad-mass exact drift",
        results["symp_lie_ptc"][4],
        expected[4],
        atol=3e-15,
        rtol=5e-13,
    )
    if defect and compare(
        "PTC z (m), FOREST-mass explanation",
        results["symp_lie_ptc"][4],
        exact(510998.9461)[4],
        atol=3e-15,
    ):
        raise AssertionError("FOREST mass no longer explains the result")
    for path, needle in [
        ("forest/code/a_scratch_size.f90", "pmap="),
        ("sim_utils/interfaces/physical_constants.f90", ":: m_proton"),
    ]:
        lines = [
            s.strip()
            for s in (source() / path).read_text(encoding="latin-1").splitlines()
            if needle in s
        ]
        if len(lines) != 1:
            raise ValueError(f"Ambiguous proton declaration in {path}")
        print("proton source:", lines[0])
    verdict(defect)

    ```

## Provenance

| Role | Model | Effort | Date | Code examined |
|---|---|---|---|---|
| Found | `gpt-6-astra` | xhigh | 2026-10-20 | 20261020-0 (`19b8afb46a40`), source, production, CPU, double, serial |
| Confirmed | `gpt-6-astra` | xhigh | 2026-10-21 | 20261020-0 (`19b8afb46a40`), source, production, CPU, double, serial |
| Confirmed | `gpt-6.1-sol` | xhigh | 2026-11-05 | 20261031-0 (`c4c629fc9071`), source, production, GNU Fortran 15.3.0, CPU, double, serial, OpenMP=1 |

