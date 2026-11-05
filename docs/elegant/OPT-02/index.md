---
title: "OPT-02: Cubic compaction omits momentum–transverse quadratic terms"
hide:
  - toc
---

# OPT-02 — Cubic compaction omits momentum–transverse quadratic terms

| | |
|---|---|
| Code | [Elegant](../index.md) · Optics |
| Priority | P2 |
| Severity | Wrong physics |
| Evidence | Probe-pinned |
| Last reproduced | 2026-11-04 on Elegant master (`0355dd6`) |
| Source | [`compute_twiss_parameters`, `alpha3` accumulation, src/twiss.cc:2398–2411](https://github.com/rtsoliday/elegant/blob/0355dd6aca12c861971fbf8f3ae78a0ded991734/src/twiss.cc#L2398-L2411)<br>[`alpha2` accumulation, src/twiss.cc:2384–2386](https://github.com/rtsoliday/elegant/blob/0355dd6aca12c861971fbf8f3ae78a0ded991734/src/twiss.cc#L2384-L2386)<br>[`alphac3` defined as the third-order momentum compaction, manual, doc/elegant.tex:8254–8258](https://github.com/rtsoliday/elegant/blob/0355dd6aca12c861971fbf8f3ae78a0ded991734/doc/elegant.tex#L8254-L8258) |

Elegant's `twiss_output` computes the third-order momentum compaction αc3 of a periodic cell but leaves out the path-length terms in which the momentum offset δ multiplies a transverse coordinate that is itself driven by δ². Take a 10 m cell with betatron tunes 0.17 and 0.23, a second-order term T166 = 0.2 m (horizontal shift ∝ δ²) and a path term T561 = 0.3 (path length ∝ xδ). The closed orbit is x = 0.1 m δ², so the path length is 10 m + 0.03 m δ³ and αc3 = 0.003. Elegant writes the correct second-order dispersion 0.1 m but αc3 = 0.

Download these shared helpers into the same directory as the reproducer scripts:

- [Download _elegant.py](../helpers/_elegant.py) · [View source](../helpers/index.md#_elegant)

??? example "Reproducer: OPT-02.py"

    [Download OPT-02.py](OPT-02.py)

    ```python
    """OPT-02: Cubic compaction omits momentum–transverse quadratic terms

    Elegant's `twiss_output` computes the third-order momentum compaction αc3 of a periodic cell but
    leaves out the path-length terms in which the momentum offset δ multiplies a transverse coordinate
    that is itself driven by δ². Take a 10 m cell with betatron tunes 0.17 and 0.23, a second-order term
    T166 = 0.2 m (horizontal shift ∝ δ²) and a path term T561 = 0.3 (path length ∝ xδ). The closed orbit
    is x = 0.1 m δ², so the path length is 10 m + 0.03 m δ³ and αc3 = 0.003. Elegant writes the correct
    second-order dispersion 0.1 m but αc3 = 0.

    The script writes a 10 m periodic EMATRIX cell with stable horizontal and vertical rotations
    (tunes 0.17 and 0.23), `T166 = 0.2` and `T561 = 0.3`, and runs `twiss_output` at third order in
    serial Elegant (and with `--mpi` in 3-rank Pelegant). It compares `etax2`, `alphac`, `alphac2` and
    `alphac3` with the exact closed orbit x(δ) = η₂δ², where (I - R)η₂ = (T166, T266). A control cell
    with first-order dispersion and `T511` instead of `T561` produces the same kind of path term from
    two transverse coordinates, which Elegant includes in `alphac2`.
    """

    import sys

    import numpy as np
    from _elegant import modes, run, verdict

    LENGTH = 10.0  # m, both C5 and L of the cell
    TUNES = (0.17, 0.23)  # horizontal and vertical betatron tunes of the cell, cycles
    RTOL, ATOL = 3e-13, 1e-15


    def rotation(tune):
        c, s = np.cos(2 * np.pi * tune), np.sin(2 * np.pi * tune)
        return np.array([[c, s], [-s, c]])


    R = np.eye(6)
    R[:2, :2], R[2:4, 2:4] = rotation(TUNES[0]), rotation(TUNES[1])


    def lattice(extra):
        elements = ",".join(f"R{i + 1}{j + 1}={float(R[i, j])!r}" for i in range(6) for j in range(6))
        return f"E: EMATRIX,L={LENGTH!r},C5={LENGTH!r},{elements}{extra}\nBL: LINE=(E)\n"


    COMMANDS = (
        "&run_setup lattice=lattice.lte, use_beamline=BL, p_central=1000, default_order=3 &end\n"
        "&run_control n_steps=1, n_passes=1 &end\n"
        "&twiss_output filename=twiss.sdds, matched=1 &end\n"
        "&bunched_beam n_particles_per_bunch=12 &end\n"
        "&track &end\n"
    )


    def twiss(mode, extra):
        out = run({"lattice.lte": lattice(extra), "run.ele": COMMANDS}, mode, outputs=("twiss.sdds",))
        return out["twiss.sdds"]


    def fmt(values):
        return "[" + ", ".join(f"{v:.16g}" for v in np.atleast_1d(values)) + "]"


    def compare(label, observed, expected, controls=0):
        """Print one case; return True when the entries after the first `controls` disagree.

        All values must be finite and the first `controls` entries must agree, otherwise the run is
        invalid and exits nonzero before any verdict.
        """
        print(f"{label}: observed={fmt(observed)} expected={fmt(expected)}")
        if not (np.all(np.isfinite(observed)) and np.all(np.isfinite(expected))):
            sys.exit(f"{label}: nonfinite value: invalid run")
        if not np.allclose(observed[:controls], expected[:controls], rtol=RTOL, atol=ATOL):
            sys.exit(f"{label}: the first {controls} entries disagree with the oracle: invalid run")
        return not np.allclose(observed[controls:], expected[controls:], rtol=RTOL, atol=ATOL)


    def orbit_coefficient(source):
        """Closed-orbit coefficient c of x = c*delta^n: (I - R) c = source, horizontal component."""
        return np.linalg.solve(np.eye(2) - R[:2, :2], source)[0]


    defect = False
    print("columns: etax (m), etax2 (m), alphac, alphac2, alphac3")
    for mode in modes():
        # Defect case: the quadratic terms are T166 (x shift 0.2 delta^2) and T561 (path 0.3 x delta).
        # The closed orbit is x = eta2 delta^2 with (I - R) eta2 = (0.2, 0), so eta2 = 0.1 m, and the
        # path per turn is C5 + 0.3 x delta = 10 m + 0.3*eta2*delta^3, hence alphac3 = 0.3*eta2/L.
        # No first- or second-order path term exists, so alphac = alphac2 = 0.
        t = twiss(mode, ",T166=0.2,T561=0.3")
        eta2 = orbit_coefficient([0.2, 0.0])
        observed = [t[0, "etax"][-1], t["etax2"][0], t["alphac"][0], t["alphac2"][0], t["alphac3"][0]]
        expected = [0.0, eta2, 0.0, 0.0, 0.3 * eta2 / LENGTH]
        # etax, etax2, alphac and alphac2 are mandatory controls; only alphac3 carries the defect flag.
        defect |= compare(f"{mode} T166=0.2, T561=0.3", observed, expected, controls=4)
        # Control: R16 = 0.2 gives first-order dispersion eta1 = (I - R)^-1 (0.2, 0); the path term
        # T511 x^2 = 0.3 eta1^2 delta^2 gives alphac2 = 0.3*eta1^2/L, a transverse-transverse
        # product that Elegant does include.
        t = twiss(mode, ",R16=0.2,T511=0.3")
        eta1 = orbit_coefficient([0.2, 0.0])
        observed = [t[0, "etax"][-1], t["etax2"][0], t["alphac"][0], t["alphac2"][0], t["alphac3"][0]]
        expected = [eta1, 0.0, 0.0, 0.3 * eta1**2 / LENGTH, 0.0]
        if compare(f"{mode} control, R16=0.2, T511=0.3", observed, expected):
            sys.exit("The T511 control disagrees with the oracle: the comparison is not valid.")
    print(f"comparison: rtol={RTOL:g}, atol={ATOL:g}")
    verdict(defect)

    ```

## Provenance

| Role | Model | Effort | Date | Code examined |
|---|---|---|---|---|
| Found | `gpt-6.1-sol` | xhigh | 2026-10-15 | 2026.4.0 (`d177ff5347c7`), source with SDDS 3fa650ee9a1c, CPU, serial |
| Confirmed | `gpt-6-astra` | xhigh | 2026-10-15 | 2026.4.0 (`d177ff5347c7`), source with SDDS 3fa650ee9a1c, CPU, serial |
| Confirmed | `gpt-6.1-sol` | xhigh | 2026-10-15 | 2026.4.0 (`d177ff5347c7`), source with SDDS 3fa650ee9a1c, CPU, serial |
| Confirmed | `gpt-6-astra` | high | 2026-10-25 | master (`75a2e1a1a58b`), source with SDDS a60b9bbed7cc, CPU, serial and 3-rank Pelegant |
| Confirmed | `gpt-6.1-sol` | high | 2026-10-25 | master (`75a2e1a1a58b`), source with SDDS a60b9bbed7cc, CPU, serial and 3-rank Pelegant |

