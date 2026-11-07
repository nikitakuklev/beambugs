---
title: "IO-13: Empty particle selections raise a zero-capacity exception"
hide:
  - toc
---

# IO-13 — Empty particle selections raise a zero-capacity exception

| | |
|---|---|
| Code | [Xsuite](../index.md) · Lattices, particles and I/O |
| Priority | P4 |
| Severity | Crash or API hazard |
| Evidence | Reproduced, independently rechecked |
| Last reproduced | 2026-11-07 on xtrack main (`f09e0fb`, head build) |
| Tags | <span class="bb-tag">partly-source-only</span> [?](../../methodology.md#tags) |
| Source | [Particles.filter subset contract, particles.py:819–833](https://github.com/xsuite/xtrack/blob/ad5e808cdc5b9a881eac2f41e32a0b7d11cc7028/xtrack/particles/particles.py#L819-L833)<br>[CPU staging and allocation, lines 835–853](https://github.com/xsuite/xtrack/blob/ad5e808cdc5b9a881eac2f41e32a0b7d11cc7028/xtrack/particles/particles.py#L835-L853)<br>[constructor capacity policy, lines 321–338](https://github.com/xsuite/xtrack/blob/ad5e808cdc5b9a881eac2f41e32a0b7d11cc7028/xtrack/particles/particles.py#L321-L338)<br>[particlesmanip.rst:243–248](https://github.com/xsuite/xsuite/blob/e0913e0a18028b2fce06991aebed6150876124b3/docs/particlesmanip.rst#L243-L248) |

`Particles.filter` raises `ValueError` whenever a valid Boolean mask selects no records, including a live selection after total beam loss. It allocates the result with an explicit capacity of zero, which its constructor rejects, although direct empty construction succeeds with capacity, live count, lost count and total charge all zero. Nonempty selections, including selecting all three lost particles, succeed; the failure occurs in the selection helper and leaves the original particle counts and charge intact.

Run `python IO-13.py` with [_prebuilt.py](../helpers/_prebuilt.py) and [_common.py](../helpers/_common.py) in the same directory.

??? example "Reproducer: IO-13.py"

    [Download IO-13.py](IO-13.py)

    ```python
    """IO-13: Empty particle selections raise a zero-capacity exception

    `Particles.filter` raises `ValueError` whenever a valid Boolean mask selects no records, including a live selection after total beam loss. It allocates the result with an explicit capacity of zero, which its constructor rejects, although direct empty construction succeeds with capacity, live count, lost count and total charge all zero. Nonempty selections, including selecting all three lost particles, succeed; the failure occurs in the selection helper and leaves the original particle counts and charge intact.

    A Python record-subset calculation supplies exact selected identities, weights and live charge. A native aperture loses all three particles before filtering the live records, with direct-empty, copy and nonempty controls in serial and one-thread OpenMP contexts. It uses installed prebuilt kernels with compilation forbidden.
    """

    from _prebuilt import CTX, identity, kernels, np, xt

    # Import after the helper sets numerical thread and compilation limits.
    # isort: split
    import xobjects as xo

    matches = []


    def exact(label, observed, expected, derivation):
        np.testing.assert_array_equal(observed, expected)
        print(f"{label}: observed={observed} expected={expected} derivation={derivation} match=True")


    def subset(label, particles, mask):
        # A plain record-selection oracle does not use Particles.filter or its counts.
        rows = [(int(pid), float(w), int(state), float(charge))
                for pid, w, state, charge, keep in zip(particles.particle_id, particles.weight,
                    particles.state, particles.charge, mask) if keep]
        expected_ids = sorted(row[0] for row in rows)
        expected_weight = sum(row[1] for row in rows)
        expected_charge = sum(row[1] * row[3] for row in rows if row[2] > 0)
        before = [particles._num_active_particles, particles._num_lost_particles]
        try:
            selected = particles.filter(mask)
        except ValueError as exc:
            if str(exc) != "Explicitly provided `_capacity` has to be greater than zero.":
                raise
            assert not rows, (label, rows)
            np.testing.assert_array_equal(
                [particles._num_active_particles, particles._num_lost_particles], before)
            matches.append(False)
            print(f"{label}: observed={exc!r} expected=empty Particles "
                  f"records=0 weight=0 live_charge/e=0 derivation=zero true mask entries match=False")
            return
        observed_ids = sorted(int(pid) for pid in selected.particle_id)
        live = selected.state > 0
        observed = [selected._capacity, selected._num_active_particles, selected._num_lost_particles,
                    float(np.sum(selected.weight)), float(np.sum(selected.weight[live] * selected.charge[live]))]
        expected = [len(rows), sum(row[2] > 0 for row in rows), sum(row[2] <= 0 for row in rows),
                    expected_weight, expected_charge]
        passed = observed_ids == expected_ids and observed == expected
        if rows:
            assert passed, (label, observed_ids, expected_ids, observed, expected)
        matches.append(passed)
        print(f"{label}: observed_ids={observed_ids} expected_ids={expected_ids} "
              f"observed_metadata={observed} expected_metadata={expected} "
              f"derivation=record subset and exact selected-weight/charge sums match={passed}")


    def main():
        identity()
        for context in (CTX, xo.ContextCpu(omp_num_threads=1)):
            label = "serial" if context is CTX else "OpenMP-one-thread"
            print(f"CONTEXT helper={label}")
            particles = xt.Particles(_context=context, p0c=7e12, q0=2.0,
                                     x=[0.002, -0.004, 0.006], particle_id=[7, 19, 23],
                                     weight=[2.0, 3.0, 5.0])
            subset(label + " all true", particles, np.ones(3, dtype=bool))
            subset(label + " one selected", particles, np.array([False, True, False]))
            subset(label + " all false live", particles, np.zeros(3, dtype=bool))
            subset(label + " coordinate cut", particles, particles.x > 1.0)
            # Dimensions are metres: the native aperture loses every initial particle.
            if context is CTX:
                xt.LimitRect(_context=context, min_x=-0.001, max_x=0.001,
                             min_y=-0.001, max_y=0.001).track(particles)
                exact("native aperture lost states", particles.state, [0, 0, 0],
                      "all initial |x| exceed the 0.001 m half aperture")
            else:
                particles.state[:] = 0
            particles.reorganize()
            exact(label + " all-lost cached counts",
                  [particles._num_active_particles, particles._num_lost_particles], [0, 3],
                  "zero positive states and three lost records")
            live = particles.state > 0
            exact(label + " live charge/e", np.sum(particles.weight[live] * particles.charge[live]),
                  0.0, "empty live-record charge sum")
            subset(label + " live after total loss", particles, live)
            subset(label + " lost after total loss", particles, particles.state <= 0)
            empty = xt.Particles(_context=context, p0c=7e12, q0=2.0, x=np.array([], dtype=float))
            exact(label + " direct empty metadata",
                  [empty._capacity, empty._num_active_particles, empty._num_lost_particles,
                   float(np.sum(empty.weight)), float(np.sum(empty.charge * empty.weight))],
                  [0, 0, 0, 0, 0], "empty arrays have zero rows, number weight and charge")
            exact(label + " empty copy capacity", empty.copy()._capacity, 0,
                  "copy preserves an empty collection")
            subset(label + " empty source", empty, np.array([], dtype=bool))
            padded = xt.Particles(_context=context, _capacity=8, x=[0.1, 0.2])
            subset(label + " spare capacity", padded, np.zeros(8, dtype=bool))
        kernels()
        print(f"SELECTIONS total={len(matches)} pass={sum(matches)} fail={len(matches) - sum(matches)}")
        print("NOT REPRODUCED" if all(matches) else "DEFECT REPRODUCED")


    if __name__ == "__main__":
        main()

    ```

## Provenance

| Role | Model | Effort | Date | Code examined |
|---|---|---|---|---|
| Found | `gpt-6-astra` | xhigh | 2026-11-06 | xtrack 0.118.2 (`ad5e808cdc5b`), installed packages, Python selection helper, kernel compilation forbidden, CPU, double, serial |
| Confirmed | `gpt-6.1-sol` | xhigh | 2026-11-06 | xtrack 0.118.2 (`ad5e808cdc5b`), installed packages, prebuilt aperture kernel (compilation forbidden), CPU, double, serial; selection helper also OpenMP with one thread |

