---
title: "SCT-18: IBSCATTER uses the electron's classical radius and mass for every species"
hide:
  - toc
---

# SCT-18 — IBSCATTER uses the electron's classical radius and mass for every species

| | |
|---|---|
| Code | [Elegant](../index.md) · Scattering and pressure |
| Priority | P3 |
| Severity | Wrong physics |
| Evidence | Reproduced, independently rechecked |
| Last reproduced | 2026-11-04 on Elegant master (`7514db3`) |
| Tags | <span class="bb-tag">partly-source-only</span> [?](../../methodology.md#tags) |
| Source | [`IBSRate` prefactor, zibs.c:110–113](https://github.com/rtsoliday/elegant/blob/0355dd6aca12c861971fbf8f3ae78a0ded991734/src/zibs.c#L110-L113)<br>[`coulombLog`, zibs.c:250–295](https://github.com/rtsoliday/elegant/blob/0355dd6aca12c861971fbf8f3ae78a0ded991734/src/zibs.c#L250-L295)<br>[rate calls, ibs_tracking.c:362–405](https://github.com/rtsoliday/elegant/blob/0355dd6aca12c861971fbf8f3ae78a0ded991734/src/ibs_tracking.c#L362-L405)<br>[update, ibs_tracking.c:465–490](https://github.com/rtsoliday/elegant/blob/0355dd6aca12c861971fbf8f3ae78a0ded991734/src/ibs_tracking.c#L465-L490)<br>[`particleRadius`, elegant.c:3308](https://github.com/rtsoliday/elegant/blob/0355dd6aca12c861971fbf8f3ae78a0ded991734/src/elegant.c#L3308)<br>[change_particle, elegant.tex:1497–1514](https://github.com/rtsoliday/elegant/blob/0355dd6aca12c861971fbf8f3ae78a0ded991734/doc/elegant.tex#L1497-L1514)<br>[IBSCATTER, ibscatter.tex:1–4](https://github.com/rtsoliday/elegant/blob/0355dd6aca12c861971fbf8f3ae78a0ded991734/doc/ibscatter.tex#L1-L4)<br>[`ibs_iterations`, elegant.tex:5617–5624](https://github.com/rtsoliday/elegant/blob/0355dd6aca12c861971fbf8f3ae78a0ded991734/doc/elegant.tex#L5617-L5624)<br>[moments.c:1592–1594](https://github.com/rtsoliday/elegant/blob/0355dd6aca12c861971fbf8f3ae78a0ded991734/src/moments.c#L1592-L1594) |

After change_particle selects a muon, a proton or a custom species, IBSCATTER computes its intrabeam-scattering growth rates with the electron's classical radius and mass, and so reports and applies the electron rates of the same beam. At fixed velocity, geometric phase space and particle count, the Bjorken–Mtingwa rate scales as r_c² ln Λ, with r_c the classical radius and ln Λ the Coulomb logarithm; for protons this is 4.106×10⁻⁷ of the electron rate. For a 1 nC beam at βγ = 30, the element gives protons a longitudinal rate of 8.067 s⁻¹, 2.436×10⁶ times the correct 3.312×10⁻⁶ s⁻¹. The IBS computed by moments_output is a separate calculation that uses the species' classical radius.

Download these shared helpers into the same directory as the reproducer scripts:

- [Download _elegant.py](../helpers/_elegant.py) · [View source](../helpers/index.md#_elegant)

??? example "Reproducer: SCT-18.py"

    [Download SCT-18.py](SCT-18.py)

    ```python
    """SCT-18: IBSCATTER uses the electron's classical radius and mass for every species

    After change_particle selects a muon, a proton or a custom species, IBSCATTER computes its
    intrabeam-scattering growth rates with the electron's classical radius and mass, and so reports and
    applies the electron rates of the same beam. At fixed velocity, geometric phase space and particle
    count, the Bjorken–Mtingwa rate scales as r_c² ln Λ, with r_c the classical radius and ln Λ the
    Coulomb logarithm; for protons this is 4.106×10⁻⁷ of the electron rate. For a 1 nC beam at βγ = 30,
    the element gives protons a longitudinal rate of 8.067 s⁻¹, 2.436×10⁶ times the correct
    3.312×10⁻⁶ s⁻¹. The IBS computed by moments_output is a separate calculation that uses the species'
    classical radius.

    The script tracks one fixed 4096-particle beam at βγ = 30 through a 0.5 m drift line ending in an
    IBSCATTER, as electron, positron, muon and proton, first with the updates switched off to read the
    growth-rate parameters, then with a smooth, amplified longitudinal update to measure the applied
    growth of the momentum spread. The expected species rates follow from the electron's by mass
    scaling at fixed velocity, geometric phase space and particle count, with the Coulomb logarithm
    from the element's own cutoff prescription. The positron run is the control, with a ratio of one.
    """

    import numpy as np
    from _elegant import particles, run, verdict

    C = 299792458.0  # SDDS constants.h
    ME_KG, ME_MEV = 9.1093897e-31, 0.51099906  # SDDS constants.h
    # Species masses of elegant.c (static mass[] table) over the electron mass.
    MASS_RATIO = {
        "electron": 1.0,
        "positron": 1.0,
        "muon": 1.88353109e-28 / ME_KG,
        "proton": 1.6726485e-27 / ME_KG,
    }
    P0, COUNT = 30.0, 4096  # beta*gamma (p in units of m c of the species) and macroparticles
    BETA_X, BETA_Y, EX, EY = 3.0, 4.0, 2e-6, 1e-6  # m, m, m, m (geometric emittances)
    SIGMA_DELTA, SIGMA_Z = 0.002, 0.02  # 1, m


    def beam():
        """Matched beam with exactly the requested second moments (orthonormalized normal samples)."""
        s = np.random.default_rng(7).normal(size=(COUNT, 6))
        s -= s.mean(axis=0)
        s = np.linalg.qr(s)[0] * np.sqrt(COUNT)
        return {
            "x": np.sqrt(EX * BETA_X) * s[:, 0],
            "xp": np.sqrt(EX / BETA_X) * s[:, 1],
            "y": np.sqrt(EY * BETA_Y) * s[:, 2],
            "yp": np.sqrt(EY / BETA_Y) * s[:, 3],
            "t": SIGMA_Z / C * s[:, 4],
            "p": P0 * (1 + SIGMA_DELTA * s[:, 5]),
        }


    # Rates only: the element writes its rates and leaves the beam unchanged.
    RATES = 'FILENAME="ibs.sdds",DO_X=0,DO_Y=0,DO_Z=0'
    # Applied growth: smooth (deterministic) longitudinal update only, amplified by FACTOR so that the
    # electron's relative variance growth of the momentum deviation is 2.7e-4.
    GROWTH = "DO_X=0,DO_Y=0,DO_Z=1,SMOOTH=1,FACTOR=1e4"


    def track(species, ibs_options, outputs):
        lattice = (
            "Q: CHARGE,TOTAL=1e-9\n"
            "D: DRIF,L=0.25\n"
            f"IB: IBSCATTER,ISRING=0,BUNCHED_BEAM_MODE=0,{ibs_options}\n"
            "L: LINE=(Q,D,D,IB)\n"
        )
        commands = (
            f'&change_particle name="{species}" &end\n'
            f'&run_setup lattice="line.lte",use_beamline=L,p_central={P0!r},output=out.sdds,default_order=2 &end\n'
            "&run_control &end\n"
            f"&twiss_output matched=0,beta_x={BETA_X!r},beta_y={BETA_Y!r} &end\n"
            '&sdds_beam input="beam.sdds" &end\n'
            "&track &end\n"
        )
        return run(
            {"line.lte": lattice, "run.ele": commands}, "serial", {"beam.sdds": beam()}, outputs=outputs
        )


    def rates(species):
        sdds = track(species, RATES, ("ibs.sdds",))["ibs.sdds"]
        par = {p.name: float(p.data[0]) for p in sdds.parameters}
        col = {c.name: np.asarray(c.data[0]) for c in sdds.columns if c.name != "ElementName"}
        return par, col


    def electron_coulomb_log(par, col):
        """ln(rmax/rmin) per element by the cutoff prescription of coulombLog() in zibs.c, for electrons.

        For a species of mass ratio M at the same beta*gamma and geometric beam, the transverse
        temperature scales as M, so both minimum impact parameters (classical 1.44e-7/T and quantum
        hbar c/(2 sqrt(2 m c^2 T_perp))) scale as 1/M; the maximum is min(sigma_x, Debye length), and the
        Debye length grows as sqrt(M). When sigma_x is the maximum for electrons, ln(Lambda_M) =
        ln(Lambda_e) + ln(M).
        """
        gamma = np.hypot(P0, 1.0)
        e_mev = np.hypot(gamma, 1.0) * ME_MEV  # zibs.c:258 as written
        transverse_ev = 0.5e6 * (gamma * e_mev - ME_MEV) * (par["emitx0"] / col["betax"])
        temp_ev = 2 * transverse_ev
        sx = 100 * np.sqrt(par["emitx0"] * col["betax"])  # cm
        sy = 100 * np.sqrt(par["emity0"] * col["betay"])
        density = par["Particles"] / (8 * np.pi**1.5 * sx * sy * 100 * par["sigmaz0"])  # cm^-3
        debye = 743.4 * np.sqrt(temp_ev / density)
        if not np.all(debye > sx):
            raise SystemExit(
                "the Debye length limits rmax; the 1/M scaling of ln(Lambda) does not hold"
            )
        rmin = np.maximum(
            1.44e-7 / temp_ev, 1.9732858e-11 / (2 * np.sqrt(2e-6 * transverse_ev * ME_MEV))
        )
        return np.log(sx / rmin)


    def average(col, values):
        """The element's growth-rate parameters are the trapezoid average of the local rates over s."""
        return np.trapezoid(values, col["s"]) / (col["s"][-1] - col["s"][0])


    def delta_growth(species):
        """Relative growth of the variance of the momentum deviation through the line."""
        p = particles(track(species, GROWTH, ("out.sdds",))["out.sdds"])["p"]
        return np.var(p / P0 - 1) / np.var(beam()["p"] / P0 - 1) - 1


    LOCAL = {"xGrowthRate": "dIBSRatex", "yGrowthRate": "dIBSRatey", "zGrowthRate": "dIBSRatez"}
    e_par, e_col = rates("electron")
    log_e = electron_coulomb_log(e_par, e_col)
    for name, local in LOCAL.items():
        if not np.isclose(average(e_col, e_col[local]), e_par[name], rtol=1e-12, atol=0):
            raise SystemExit(f"electron {name} is not the trapezoid average of {local}")
    print(f"electron: (x,y,z)GrowthRate={[e_par[n] for n in LOCAL]} /s; "
          f"Coulomb log {log_e.min():.6f} to {log_e.max():.6f}")  # fmt: skip


    def expected_rates(species):
        """Species growth-rate parameters from the electron's local rates by mass scaling.

        Bjorken-Mtingwa: rate = N r_c^2 c lnLambda/(8 pi beta^3 gamma^4 ex ey sigma_delta sigma_z) times
        an integral of the optics and emittances only. At equal |q|, beta*gamma, geometric beam and N,
        only r_c^2 ~ 1/M^2 and lnLambda change: local rate ratio = (1 + ln M/lnLambda_e)/M^2.
        """
        m = MASS_RATIO[species]
        factor = (1 + np.log(m) / log_e) / m**2
        return np.array([average(e_col, e_col[local] * factor) for local in LOCAL.values()])


    defect = True
    for species in ("positron", "muon", "proton"):
        par, _ = rates(species)
        observed, expected = np.array([par[n] for n in LOCAL]), expected_rates(species)
        print(f"{species}: (x,y,z)GrowthRate observed={observed.tolist()} expected={expected.tolist()}")
        print(f"  z rate over electron: observed={observed[2] / e_par['zGrowthRate']:.10g} "
              f"expected={expected[2] / e_par['zGrowthRate']:.10g}")  # fmt: skip
        match = np.allclose(observed, expected, rtol=1e-6, atol=0)
        if species == "positron" and not match:
            raise SystemExit("control: positron rates differ from electron rates")
        if species != "positron":
            defect &= not match

    # The rates set the applied growth (ibs_tracking.c:465): with the electron's growth g_e, the correct
    # proton growth is g_e times the proton/electron longitudinal rate ratio, to first order in g_e.
    g_e, g_p = delta_growth("electron"), delta_growth("proton")
    z_ratio = expected_rates("proton")[2] / e_par["zGrowthRate"]
    print(f"applied growth of var(delta): electron={g_e:.10g}; proton observed={g_p:.10g} "
          f"expected={g_e * z_ratio:.4g}")  # fmt: skip
    defect &= bool(np.isclose(g_p, g_e, rtol=1e-12, atol=0)) and g_e > 1e-4
    print("comparison: rates rtol=1e-6; applied growth equal to the electron's at rtol=1e-12")
    verdict(defect)

    ```

## Provenance

| Role | Model | Effort | Date | Code examined |
|---|---|---|---|---|
| Found | `claude-opus-5-5` | high | 2026-11-04 | master (`7514db36f764`), source with SDDS a3f730a14c49, CPU, serial |
| Confirmed (IBSCATTER rates through zibs.c; the claim that moments_output shares them was refuted) | `gpt-6.1-sol` | xhigh | 2026-11-04 | master (`7514db36f764`), source with SDDS a3f730a14c49, CPU, serial |

