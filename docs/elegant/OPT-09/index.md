---
title: "OPT-09: Quartic path term enters cubic momentum compaction"
hide:
  - toc
---

# OPT-09 — Quartic path term enters cubic momentum compaction

| | |
|---|---|
| Code | [Elegant](../index.md) · Optics |
| Priority | P2 |
| Severity | Wrong physics |
| Evidence | Reproduced, independently rechecked |
| Last reproduced | 2026-11-04 on Elegant commit [`0355dd6`](https://github.com/rtsoliday/elegant/commit/0355dd6aca12), committed 2026-11-03 |
| Source | [`compute_twiss_parameters`, `alpha3` accumulation, twiss.cc:2400–2404](https://github.com/rtsoliday/elegant/blob/0355dd6aca12c861971fbf8f3ae78a0ded991734/src/twiss.cc#L2400-L2404)<br>[`x = x(0) + η δ + η2 δ² + η3 δ³`, twiss.cc:253–259](https://github.com/rtsoliday/elegant/blob/0355dd6aca12c861971fbf8f3ae78a0ded991734/src/twiss.cc#L253-L259)<br>[`alphac3` is the third-order momentum compaction, doc/elegant.tex:8254–8255](https://github.com/rtsoliday/elegant/blob/0355dd6aca12c861971fbf8f3ae78a0ded991734/doc/elegant.tex#L8254-L8255)<br>[`alphac3` as an optimization symbol, doc/elegant.tex:6184–6185](https://github.com/rtsoliday/elegant/blob/0355dd6aca12c861971fbf8f3ae78a0ded991734/doc/elegant.tex#L6184-L6185) |

Elegant's `twiss_output` reports the third-order momentum compaction `alphac3`, the δ³ coefficient of the relative path length with δ the relative momentum offset, but adds a product of two second-order dispersions that belongs to δ⁴. For a prescribed stable 10 m cell with second-order term T166 = 0.2 m (closed orbit x = 0.1 m·δ²) and path term T511 = 0.3 m⁻¹ (path length ∝ x²), the path is 10 m + 0.003 m·δ⁴, so `alphac3` must be 0, but Elegant writes 3.000×10⁻⁴. Tracking 12 particles on the closed orbit reproduces the quartic path, and removing either term gives zero.

Run `python OPT-09.py` with [_elegant.py](../helpers/_elegant.py) in the same directory.

??? example "Reproducer: OPT-09.py"

    [Download OPT-09.py](OPT-09.py)

    ```python
    """OPT-09: Quartic path term enters cubic momentum compaction

    Elegant's `twiss_output` reports the third-order momentum compaction `alphac3`, the δ³ coefficient
    of the relative path length with δ the relative momentum offset, but adds a product of two
    second-order dispersions that belongs to δ⁴. For a prescribed stable 10 m cell with second-order
    term T166 = 0.2 m (closed orbit x = 0.1 m·δ²) and path term T511 = 0.3 m⁻¹ (path length ∝ x²), the
    path is 10 m + 0.003 m·δ⁴, so `alphac3` must be 0, but Elegant writes 3.000×10⁻⁴. Tracking 12
    particles on the closed orbit reproduces the quartic path, and removing either term gives zero.

    The script prescribes a stable 10 m cell as an explicit matrix with second-order dispersion and a
    quadratic path term, in serial Elegant (and with `--mpi` in 3-rank Pelegant). It compares the
    `alphac3` written by `twiss_output` with the analytic value zero, after checking that 12 particles
    on the exact closed orbit acquire the prescribed quartic path with no cubic term when tracked
    through the same cell. Two controls remove either the second-order dispersion or the path term,
    which leaves no quartic path and must give `alphac3` = 0.
    """

    import sys

    import numpy as np
    from _elegant import modes, particles, run, verdict

    C = 299792458.0  # m/s, c_mks of SDDS
    LENGTH, P0 = 10.0, 1000.0  # cell length (m), reference momentum (m c)
    TUNES = (0.17, 0.23)  # betatron tunes; the cell is a rotation by 2 pi nu with beta = 1 m, alpha = 0
    DELTAS = np.array([-0.3, -0.25, -0.2, -0.15, -0.1, -0.05, 0.05, 0.1, 0.15, 0.2, 0.25, 0.3])
    TOL = 1e-12


    def lattice(t166, t511):
        """EMATRIX cell: x' rotation, x and delta^2 term T166, and path s += T511 x^2."""
        mu_x, mu_y = (2 * np.pi * nu for nu in TUNES)
        entries = {
            "R11": np.cos(mu_x), "R12": np.sin(mu_x), "R21": -np.sin(mu_x), "R22": np.cos(mu_x),
            "R33": np.cos(mu_y), "R34": np.sin(mu_y), "R43": -np.sin(mu_y), "R44": np.cos(mu_y),
            "R55": 1.0, "R66": 1.0, "T166": t166, "T511": t511,
        }  # fmt: skip
        text = ",".join(f"{k}={float(v)!r}" for k, v in entries.items())
        return f"E: EMATRIX,L={LENGTH!r},C5={LENGTH!r},{text}\nBL: LINE=(E)\n"


    def commands(track):
        """Run deck: twiss_output on the cell, then either the closed-orbit beam or a dummy bunch."""
        beam = (
            "&sdds_beam input=beam.sdds &end\n"
            if track
            else "&bunched_beam n_particles_per_bunch=12 &end\n"
        )
        return (
            f"&run_setup lattice=lattice.lte, use_beamline=BL, p_central={P0!r}, default_order=3,\n"
            " output=out.sdds &end\n"
            "&run_control n_steps=1, n_passes=1 &end\n"
            "&twiss_output filename=twiss.sdds, concat_order=3, matched=1 &end\n"
            f"{beam}&track &end\n"
        )


    def closed_orbit(t166):
        """Closed orbit z = eta2 delta^2 (x, x') of the cell, from z = R z + (T166, 0) delta^2.

        The first-order dispersion is zero (R16 = R26 = 0) and the x update has no other nonlinear
        term, so the orbit is exactly quadratic in delta: eta2 = (1 - R)^-1 (T166, 0).
        """
        mu = 2 * np.pi * TUNES[0]
        rotation = np.array([[np.cos(mu), np.sin(mu)], [-np.sin(mu), np.cos(mu)]])
        return np.linalg.solve(np.eye(2) - rotation, [t166, 0.0])


    def run_case(mode, t166, t511, track):
        beam = None
        if track:
            eta2 = closed_orbit(t166)
            beam = {
                "beam.sdds": {
                    "x": eta2[0] * DELTAS**2,
                    "xp": eta2[1] * DELTAS**2,
                    "p": P0 * (1 + DELTAS),
                    "particleID": np.arange(1, len(DELTAS) + 1),
                }
            }
        out = run(
            {"lattice.lte": lattice(t166, t511), "run.ele": commands(track)},
            mode,
            beams=beam,
            outputs=("twiss.sdds", "out.sdds"),
        )
        return out["twiss.sdds"], particles(out["out.sdds"])


    def fmt(values):
        return "[" + ", ".join(f"{v:.10g}" for v in np.ravel(values)) + "]"


    def require(condition, message):
        """Exit nonzero (an invalid comparison, not a verdict) when an oracle gate fails."""
        if not condition:
            sys.exit(f"{message}: the comparison is not valid.")


    def compare(label, observed, expected, tol=TOL):
        """Print one case; return True when observed and expected disagree."""
        observed, expected = np.asarray(observed, float), np.asarray(expected, float)
        if not (np.isfinite(observed).all() and np.isfinite(expected).all()):
            sys.exit(f"{label}: nonfinite value, the comparison is not valid.")
        bad = not np.allclose(observed, expected, rtol=0, atol=tol)
        print(f"{label}: observed={fmt(observed)} expected={fmt(expected)}")
        return bad


    defect = False
    for mode in modes():
        # Controls: without second-order dispersion or without the path term there is no quartic
        # path, and the cubic coefficient is zero.
        for name, t166, t511 in (("T166=0", 0.0, 0.3), ("T511=0", 0.2, 0.0)):
            twiss, _ = run_case(mode, t166, t511, track=False)
            if compare(f"{mode} control {name}, alphac3", twiss["alphac3"][0], 0.0):
                sys.exit("A control has nonzero alphac3: the comparison is not valid.")

        twiss, out = run_case(mode, 0.2, 0.3, track=True)
        eta2 = closed_orbit(0.2)
        print(
            f"{mode} second-order dispersion: observed={float(twiss['etax2'][0])!r} expected={float(eta2[0])!r}"
        )
        # Oracle gates: the tracked particles must realize the prescribed map on the closed orbit.
        require(
            np.array_equal(out["particleID"], np.arange(1, len(DELTAS) + 1)),
            "the output particle IDs are incomplete",
        )
        require(
            all(np.isfinite(out[name]).all() for name in ("x", "xp", "p", "t")),
            "the tracked output is nonfinite",
        )
        require(
            np.allclose(out["p"], P0 * (1 + DELTAS), rtol=TOL, atol=0),
            "the tracked momentum differs from p = P0 (1 + delta)",
        )
        require(abs(float(twiss["etax2"][0]) - eta2[0]) <= TOL, "etax2 differs from the closed orbit")
        require(
            np.allclose(out["x"], eta2[0] * DELTAS**2, rtol=0, atol=TOL)
            and np.allclose(out["xp"], eta2[1] * DELTAS**2, rtol=0, atol=TOL),
            "the tracked closed orbit is not retained",
        )
        # Tracked path through the cell: s = beta c t with the output momentum, and the exact map gives
        # s = C5 + T511 x^2 with x = eta2 delta^2, i.e. 10 m + 0.003 m delta^4.
        p = out["p"]
        path = p / np.sqrt(1 + p**2) * C * out["t"]
        exact = LENGTH + 0.3 * (eta2[0] * DELTAS**2) ** 2
        path_error = np.max(np.abs(path - exact))
        print(f"{mode} tracked path, max |s - (10 + 0.3 x^2)|={path_error:.3g} m")
        require(path_error <= TOL, "the tracked path differs from the prescribed map")
        # Fit (s - C)/C = a1 delta + ... + a5 delta^5 to the tracked path. The path is
        # 10 + T511 eta2^2 delta^4, so the cubic coefficient a3 is zero and the quartic one is
        # T511 eta2^2 / C = 0.0003; both cross-check the tracking before alphac3 is compared.
        fit = np.polynomial.polynomial.polyfit(DELTAS, (path - LENGTH) / LENGTH, 5)
        expected_a4 = 0.3 * eta2[0] ** 2 / LENGTH
        print(f"{mode} fitted a3 and a4: observed={fmt(fit[3:5])} expected={fmt([0.0, expected_a4])}")
        require(np.isfinite(fit).all(), "the fitted path coefficients are nonfinite")
        require(
            abs(fit[3]) <= TOL and abs(fit[4] - expected_a4) <= TOL,
            "the tracked path coefficients differ from the prescribed map",
        )
        # The exact cubic coefficient is zero (derivation above), so alphac3 is compared with zero.
        defect |= compare(f"{mode} alphac3", twiss["alphac3"][0], 0.0)
    print(f"comparison: absolute tolerance {TOL:g}")
    verdict(defect)

    ```

## Provenance

| Role | Model | Effort | Date | Code examined |
|---|---|---|---|---|
| Found | `gpt-6.1-sol` | xhigh | 2026-10-15 | 2026.4.0 (`d177ff5347c7`), source reading, not executed for this claim |
| Confirmed | `gpt-6-astra` | xhigh | 2026-10-15 | 2026.4.0 (`d177ff5347c7`), source with SDDS 3fa650ee9a1c, CPU, serial |
| Confirmed | `gpt-6.1-sol` | xhigh | 2026-10-15 | 2026.4.0 (`d177ff5347c7`), source with SDDS 3fa650ee9a1c, CPU, serial |
| Confirmed | `gpt-6-astra` | high | 2026-10-25 | master (`75a2e1a1a58b`), source with SDDS a60b9bbed7cc, CPU, serial and 3-rank Pelegant |
| Confirmed | `gpt-6.1-sol` | high | 2026-10-25 | master (`75a2e1a1a58b`), source with SDDS a60b9bbed7cc, CPU, serial and 3-rank Pelegant |

