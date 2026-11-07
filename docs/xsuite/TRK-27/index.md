---
title: "TRK-27: Slicing helper names overwrite existing physical elements"
hide:
  - toc
---

# TRK-27 — Slicing helper names overwrite existing physical elements

| | |
|---|---|
| Code | [Xsuite](../index.md) · Tracking |
| Priority | P2 |
| Severity | Wrong physics |
| Evidence | Reproduced, independently rechecked |
| Last reproduced | 2026-11-07 on xtrack main (`f09e0fb`, head build) |
| Tags | <span class="bb-tag">partly-source-only</span> [?](../../methodology.md#tags) |
| Source | [Slicer helper allocation, slicing.py:293–306](https://github.com/xsuite/xtrack/blob/ad5e808cdc5b9a881eac2f41e32a0b7d11cc7028/xtrack/slicing.py#L293-L306)<br>[entry-map names](https://github.com/xsuite/xtrack/blob/ad5e808cdc5b9a881eac2f41e32a0b7d11cc7028/xtrack/slicing.py#L373-L376)<br>[drift/body names](https://github.com/xsuite/xtrack/blob/ad5e808cdc5b9a881eac2f41e32a0b7d11cc7028/xtrack/slicing.py#L415-L431)<br>[slice_thick_elements contract](https://github.com/xsuite/xtrack/blob/ad5e808cdc5b9a881eac2f41e32a0b7d11cc7028/xtrack/line.py#L2366-L2407)<br>[cut_at_s](https://github.com/xsuite/xtrack/blob/ad5e808cdc5b9a881eac2f41e32a0b7d11cc7028/xtrack/line.py#L3884-L3952)<br>[documents thin/thick Uniform and Teapot slicing and positional cuts](https://github.com/xsuite/xsuite/blob/e0913e0a18028b2fce06991aebed6150876124b3/docs/environment.rst#L818-L832) |

Slicing a thick element named h silently replaces any existing element named h_entry or h_exit, which it generates for boundary markers, or h_aper..0, h_aper..1, … when h has an associated aperture; elements used by other lines in the same environment are replaced too. A zero-gradient 0.6000 m quadrupole h followed by a 1.300 m drift named h_entry becomes a 0.6000 m line, because the drift turns into a zero-length marker. At px = 0.002000 the exit position is 2.600 mm short of the analytic 1.900 m drift. Thin and thick slicing and cuts at specified positions share the failure; generated slice-body and entry-map names avoid existing names.

Run `python TRK-27.py` with [_prebuilt.py](../helpers/_prebuilt.py) and [_common.py](../helpers/_common.py) in the same directory.

??? example "Reproducer: TRK-27.py"

    [Download TRK-27.py](TRK-27.py)

    ```python
    """TRK-27: Slicing helper names overwrite existing physical elements

    Slicing a thick element named h silently replaces any existing element named h_entry or h_exit, which it generates for boundary markers, or h_aper..0, h_aper..1, … when h has an associated aperture; elements used by other lines in the same environment are replaced too. A zero-gradient 0.6000 m quadrupole h followed by a 1.300 m drift named h_entry becomes a 0.6000 m line, because the drift turns into a zero-length marker. At px = 0.002000 the exit position is 2.600 mm short of the analytic 1.900 m drift. Thin and thick slicing and cuts at specified positions share the failure; generated slice-body and entry-map names avoid existing names.

    Slice a zero-gradient quadrupole beside a drift and compare with the analytic summed-length drift map. Exercise boundary and aperture collisions, safe names, cuts, and a second line sharing the environment.
    """

    from _prebuilt import CTX, identity, kernels, np, xt


    def make(name):
        host = xt.Quadrupole(
            _context=CTX, length=0.6, k1=0, edge_entry_active=False, edge_exit_active=False
        )
        elements = {"h": host, name: xt.Drift(_context=CTX, length=1.3)}
        if name == "h_aper..0":
            elements["ap"] = xt.LimitRect(_context=CTX, min_x=-0.1, max_x=0.1, min_y=-0.1, max_y=0.1)
            host.name_associated_aperture = "ap"
        return xt.Line(elements=elements, element_names=["h", name])


    def check(label, line, control=False):
        length = line.get_length()
        line.skip_end_turn_actions = True
        line.build_tracker(_context=CTX)
        p = xt.Particles(_context=CTX, p0c=2e9, x=0.001, px=0.002, y=-0.002, py=-0.001)
        line.track(p)
        observed = np.array([getattr(p, n)[0] for n in ("x", "px", "y", "py")])
        # Expanded drift at delta=0: x=x0+L*px, y=y0+L*py.
        expected = np.array([0.001 + 1.9 * 0.002, 0.002, -0.002 - 1.9 * 0.001, -0.001])
        ok = abs(length - 1.9) < 5e-16 and np.allclose(observed, expected, atol=2e-16, rtol=0)
        assert p.state[0] > 0
        if control:
            assert ok
        print(
            f"DRIFT {label}: length={length:.17g} expected_length=1.9 observed={observed} "
            f"expected={expected} max_error={np.max(abs(observed - expected)):.17g} "
            f"derivation=analytic 1.9m expanded drift pass={ok}"
        )
        return not ok


    def main():
        identity()
        failures = 0
        for mode in ("thin", "thick"):
            for scheme in (xt.Teapot, xt.Uniform):
                for name in ("h_entry", "h_exit", "h_aper..0", "h..0", "h..entry_map", "safe"):
                    line = make(name)
                    line.slice_thick_elements(
                        [xt.Strategy(scheme(3, mode=mode), element_type=xt.Quadrupole)]
                    )
                    failures += check(
                        f"{scheme.__name__} {mode} {name}",
                        line,
                        control=name in ("h..0", "h..entry_map", "safe"),
                    )
        line = make("h_entry")
        failures += check("unsliced", line, control=True)
        line = make("h_entry")
        line.cut_at_s([0.3])
        failures += check("cut_at_s", line)
        env = xt.Environment()
        env.new("h", xt.Quadrupole, length=0.6)
        env.new("h_entry", xt.Drift, length=1.3)
        main_line = env.new_line(components=["h"])
        other = env.new_line(components=["h_entry"])
        assert other.get_length() == 1.3
        main_line.slice_thick_elements([xt.Strategy(xt.Teapot(2))])
        ok = other.get_length() == 1.3
        failures += not ok
        print(
            f"SHARED_ENVIRONMENT: observed_length={other.get_length()} expected=1.3 "
            f"derivation=untouched line retains its physical drift pass={ok}"
        )
        kernels()
        print(f"FAILED_PRESERVATION_CASES {failures}")
        print("DEFECT REPRODUCED" if failures else "NOT REPRODUCED")


    if __name__ == "__main__":
        main()

    ```

## Provenance

| Role | Model | Effort | Date | Code examined |
|---|---|---|---|---|
| Found | `gpt-6-astra` | xhigh | 2026-11-06 | xtrack 0.118.2 (`ad5e808cdc5b`), installed packages, prebuilt kernels (compilation forbidden), CPU, double, serial |
| Confirmed | `gpt-6.1-sol` | xhigh | 2026-11-06 | xtrack 0.118.2 (`ad5e808cdc5b`), installed packages, prebuilt kernels (compilation forbidden), CPU, double, serial |
| Confirmed | `claude-opus-5-5` | high | 2026-11-07 | xtrack 0.118.2 (`ad5e808cdc5b`), installed packages, prebuilt kernels (compilation forbidden), CPU, double, serial |

