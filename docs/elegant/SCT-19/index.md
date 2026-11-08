---
title: "SCT-19: moments_output IBS Coulomb logarithm mixes electron-unit momenta with the species mass"
hide:
  - toc
---

# SCT-19 — moments_output IBS Coulomb logarithm mixes electron-unit momenta with the species mass

| | |
|---|---|
| Code | [Elegant](../index.md) · Scattering and pressure |
| Priority | P3 |
| Failure | Wrong physics |
| Evidence | Reproduced, independently rechecked |
| Last reproduced | 2026-11-04 on Elegant commit [`7514db3`](https://github.com/rtsoliday/elegant/commit/7514db36f764), committed 2026-11-04 |
| Tags | <span class="bb-tag">partly-source-only</span> [?](../../methodology.md#tags) |
| Source | [moments.c:1496–1503](https://github.com/rtsoliday/elegant/blob/0355dd6aca12c861971fbf8f3ae78a0ded991734/src/moments.c#L1496-L1503)<br>[moments.c:1531–1539](https://github.com/rtsoliday/elegant/blob/0355dd6aca12c861971fbf8f3ae78a0ded991734/src/moments.c#L1531-L1539)<br>[Coulomb log, moments.c:1555–1582](https://github.com/rtsoliday/elegant/blob/0355dd6aca12c861971fbf8f3ae78a0ded991734/src/moments.c#L1555-L1582)<br>[moments.c:1592–1594](https://github.com/rtsoliday/elegant/blob/0355dd6aca12c861971fbf8f3ae78a0ded991734/src/moments.c#L1592-L1594)<br>[elegant.c:3284–3308](https://github.com/rtsoliday/elegant/blob/0355dd6aca12c861971fbf8f3ae78a0ded991734/src/elegant.c#L3284-L3308)<br>[change_particle, elegant.tex:1497–1517](https://github.com/rtsoliday/elegant/blob/0355dd6aca12c861971fbf8f3ae78a0ded991734/doc/elegant.tex#L1497-L1517)<br>[`ibs_iterations`, elegant.tex:5617–5624](https://github.com/rtsoliday/elegant/blob/0355dd6aca12c861971fbf8f3ae78a0ded991734/doc/elegant.tex#L5617-L5624) |

After change_particle selects a muon or a custom heavy species, moments_output with ibs_iterations forms the Coulomb logarithm ln Λ of its intrabeam-scattering (IBS) diffusion from momentum spreads in electron units but a classical minimum impact parameter with the species mass M, which inflates that cutoff by (M/mₑ)². Where it exceeds the other cutoff, ln Λ and the IBS diffusion, proportional to it, are too small. The correct value follows from Kubo and Oide's cutoffs in physical units on the same beam moments. In a radiating test ring at normalized momentum βγ = 1200 with 1 pC, ln Λ is 12.76–18.63% low for muons and 86.64–95.11% low for antiprotons, a custom species. Electrons and positrons are not affected.

Run `python SCT-19.py` with [_elegant.py](../helpers/_elegant.py) in the same directory.

??? example "Reproducer: SCT-19.py"

    [Download SCT-19.py](SCT-19.py)

    ```python
    """SCT-19: moments_output IBS Coulomb logarithm mixes electron-unit momenta with the species mass

    After change_particle selects a muon or a custom heavy species, moments_output with ibs_iterations
    forms the Coulomb logarithm ln Λ of its intrabeam-scattering (IBS) diffusion from momentum spreads
    in electron units but a classical minimum impact parameter with the species mass M, which inflates
    that cutoff by (M/mₑ)². Where it exceeds the other cutoff, ln Λ and the IBS diffusion, proportional
    to it, are too small. The correct value follows from Kubo and Oide's cutoffs in physical units on
    the same beam moments. In a radiating test ring at normalized momentum βγ = 1200 with 1 pC, ln Λ is
    12.76–18.63% low for muons and 86.64–95.11% low for antiprotons, a custom species. Electrons and
    positrons are not affected.

    The script runs moments_output with two IBS iterations on a radiating, coupled ten-cell FODO ring at
    βγ = 1200 with a 1 pC bunch, once each for the electron, the muon and an antiproton (the manual's
    custom species of the proton mass and the electron's charge). From the first, radiation-only beam
    moments of the output it evaluates Kubo and Oide's minimum and maximum impact parameters in physical
    units for each species and compares the resulting Coulomb logarithm with Elegant's at every element
    with length. The electron is the control and must agree to 10⁻¹²; the mixed-unit formula is also
    evaluated to show it reproduces Elegant's heavy-species values.
    """

    import numpy as np
    from _elegant import run, verdict

    # SDDS constants.h and the static species table of elegant.c
    C, E_CHARGE, ME = 2.99792458e8, 1.60217733e-19, 9.1093897e-31  # m/s, C, kg
    RE = 2.81794092e-15  # m, electron classical radius re_mks
    EPS0 = 1e7 / (4 * np.pi * C**2)  # F/m, epsilon_o
    SPECIES = {  # label: (change_particle arguments, mass in kg)
        "electron": ("name=electron", ME),
        "muon": ("name=muon", 1.88353109e-28),
        "antiproton": (f"name=custom, mass_ratio={1.6726485e-27 / ME!r}, charge_ratio=1", 1.6726485e-27),
    }  # fmt: skip
    P0, CHARGE = 1200.0, 1e-12  # beta*gamma, bunch charge in C

    # QF is rolled by 20 mrad to couple the planes, so the equilibrium vertical emittance is finite.
    LATTICE = (
        f"C: CHARGE,TOTAL={CHARGE!r}\n"
        "QF: KQUAD,L=0.25,K1=1.1,TILT=0.02\n"
        "QD: KQUAD,L=0.25,K1=-1.1\n"
        "D: DRIF,L=0.6\n"
        f"B: CSBEND,L=1.2,ANGLE={np.pi / 10!r},SYNCH_RAD=1,ISR=1,N_SLICES=20\n"
        "RF: RFCA,VOLT=1e6,FREQ=500e6,PHASE=180\n"
        "CELL: LINE=(QF,D,B,D,QD,D,B,D)\n"
        "RING: LINE=(C,10*CELL,RF)\n"
    )


    def moments(particle):
        """First output page of moments_output (radiation-only moments and the Coulomb log computed
        from them for the first IBS update) and the largest radiation damping time.

        With ibs_iterations=2 the first update uses the full charge (moments.c: iteration >=
        ibs_iterations-3), and ibs_output_iterations=1 writes that page."""
        commands = (
            f"&change_particle {particle} &end\n"
            f"&run_setup lattice=ring.lte, use_beamline=RING, p_central={P0!r}, default_order=2 &end\n"
            "&run_control &end\n"
            '&twiss_output filename="ring.twi", matched=1, radiation_integrals=1 &end\n'
            "&rf_setup harmonic=100, total_voltage=1e6 &end\n"
            '&moments_output filename="ring.mom", ibs_iterations=2, ibs_output_iterations=1 &end\n'
            "&bunched_beam &end\n"
            "&track &end\n"
        )
        out = run({"ring.lte": LATTICE, "run.ele": commands}, outputs=("ring.mom", "ring.twi"))
        mom, twi = out["ring.mom"], out["ring.twi"]
        col = {c.name: np.asarray(mom[0, c.name]) for c in mom.columns}
        par = {p.name: p.data[0] for p in twi.parameters}
        return col, max(float(par[name]) for name in ("taux", "tauy", "taudelta"))


    def sigma_matrix(col, row):
        """Lab-frame second moments in (x, x', y, y', s, delta) at the end of the element in `row`."""
        s = np.zeros((6, 6))
        for i in range(6):
            for j in range(i, 6):
                s[i, j] = s[j, i] = (
                    col[f"s{i + 1}"][row] ** 2 if i == j else col[f"s{i + 1}{j + 1}"][row]
                )
        return s


    def eigenvalues(cxx):
        """Eigenvalues of the 3x3 spatial covariance, ascending, each to full relative precision.

        The boosted longitudinal size exceeds the vertical by ~10^5, so the small eigenvalues come from
        the inverse matrix and the middle one from the determinant."""
        largest = np.linalg.eigvalsh(cxx)[-1]
        smallest = 1 / np.linalg.eigvalsh(np.linalg.inv(cxx))[-1]
        return np.array([smallest, np.linalg.det(cxx) / (smallest * largest), largest])


    def coulomb_log(sigma, beta_gamma, tau_lab, mass, radius, bmin2_scale=1.0):
        """Kubo-Oide Coulomb log ln(bmax/bmin) of one element, in physical units.

        Lab moments are converted to momenta with the species' reference momentum p0 = M c beta*gamma
        and boosted to the beam frame (z -> gamma z, p_z -> p0 delta/gamma). The conditional momentum
        covariance at fixed position (Schur complement) has trace U; the rms velocity is sqrt(U)/M.
        bmax = min(spatial rms sizes, n^(-1/3)); bmin1 = 1/sqrt(pi n v tau) with tau the largest
        damping time in the beam frame; bmin2 = sqrt(2) r_c (M c)^2 / U = sqrt(2) r_c c^2/v^2, the
        classical cutoff with the factor sqrt(2) of moments.c. bmin2_scale = (M/me)^2 gives Elegant's
        mixed-unit variant, which uses electron-unit momenta (U (me/M)^2) with the species mass.
        """
        gamma = np.hypot(beta_gamma, 1.0)
        p0 = mass * C * beta_gamma
        t = np.diag([1, p0, 1, p0, gamma, p0 / gamma])
        cov = t @ sigma @ t
        xx, pp = [0, 2, 4], [1, 3, 5]
        cxx, cpp, cpx = cov[np.ix_(xx, xx)], cov[np.ix_(pp, pp)], cov[np.ix_(pp, xx)]
        trace = np.trace(cpp - cpx @ np.linalg.solve(cxx, cpx.T))
        eig = eigenvalues(cxx)
        density = CHARGE / E_CHARGE / (8 * np.sqrt(np.pi**3 * np.prod(eig)))  # m^-3
        bmax = min(np.sqrt(eig).min(), density ** (-1 / 3))
        bmin1 = 1 / np.sqrt(np.pi * density * (np.sqrt(trace) / mass) * tau_lab / gamma)
        bmin2 = np.sqrt(2) * radius * (mass * C) ** 2 / trace * bmin2_scale
        return np.log(bmax / max(bmin1, bmin2)), bmin1, bmin2


    defect = True
    for label, (particle, mass) in SPECIES.items():
        col, tau = moments(particle)
        # Elegant's electron constant; the species radius q^2/(4 pi eps0 M c^2) otherwise (elegant.c).
        radius = RE if mass == ME else E_CHARGE**2 / (4 * np.pi * EPS0 * mass * C**2)
        rows = [i for i, kind in enumerate(col["ElementType"]) if kind in ("KQUAD", "CSBEND", "DRIF")]
        if not np.allclose(col["pCentral0"][rows], P0, rtol=1e-12, atol=0):
            raise SystemExit("reference momentum is not constant around the ring")
        observed = col["CoulombLog"][rows]
        cases = [coulomb_log(sigma_matrix(col, r), P0, tau, mass, radius) for r in rows]
        expected = np.array([c[0] for c in cases])
        mixed = np.array(
            [
                coulomb_log(sigma_matrix(col, r), P0, tau, mass, radius, (mass / ME) ** 2)[0]
                for r in rows
            ]
        )
        ratio = observed / expected
        _, bmin1, bmin2 = cases[0]
        print(f"{label}: QF#1 CoulombLog observed={observed[0]:.8f} expected={expected[0]:.8f} "
              f"mixed-unit prediction={mixed[0]:.8f}; bmin1={bmin1:.5e} m, physical bmin2={bmin2:.5e} m, "
              f"mixed bmin2={bmin2 * (mass / ME) ** 2:.5e} m")  # fmt: skip
        print(f"  observed/expected over {len(rows)} elements: {ratio.min():.6f} to {ratio.max():.6f}; "
              f"max |observed - mixed| = {np.abs(observed - mixed).max():.2e}")  # fmt: skip
        if label == "electron":
            if not np.allclose(observed, expected, rtol=1e-12, atol=0):
                raise SystemExit("control: electron Coulomb log differs from the independent value")
        else:
            defect &= bool(np.all(ratio < 0.99) and np.allclose(observed, mixed, rtol=1e-7, atol=0))
    print("comparison: electron rtol=1e-12; defect needs observed/expected < 0.99 at every element "
          "and observed equal to the mixed-unit prediction at rtol=1e-7")  # fmt: skip
    verdict(defect)

    ```

## Provenance

| Role | Model | Effort | Date | Code examined |
|---|---|---|---|---|
| Found | <span class="bb-nowrap">`claude-opus-5-5`</span> | high | <span class="bb-nowrap">2026-11-04</span> | master (`0355dd6aca12`), source reading, not executed for this claim |
| Confirmed | <span class="bb-nowrap">`gpt-6.1-sol`</span> | xhigh | <span class="bb-nowrap">2026-11-04</span> | master (`7514db36f764`), source with SDDS a3f730a14c49, CPU, serial |

