---
title: "COL-27: Averaged nonlinear SCMULT normalizes its kick with the unfiltered beam sizes"
hide:
  - toc
---

# COL-27 — Averaged nonlinear SCMULT normalizes its kick with the unfiltered beam sizes

| | |
|---|---|
| Code | [Elegant](../index.md) · Collective effects |
| Priority | P2 |
| Severity | Wrong physics |
| Evidence | Reproduced, independently rechecked |
| Last reproduced | 2026-11-04 on Elegant master (`0355dd6`, head build) |
| Tags | <span class="bb-tag">contract-unsettled</span> [?](../../methodology.md#tags) |
| Source | [filter, insertSCeffects.cc:316–323](https://github.com/rtsoliday/elegant/blob/0355dd6aca12c861971fbf8f3ae78a0ded991734/src/insertSCeffects.cc#L316-L323)<br>[nonlinear kick, insertSCeffects.cc:592–602](https://github.com/rtsoliday/elegant/blob/0355dd6aca12c861971fbf8f3ae78a0ded991734/src/insertSCeffects.cc#L592-L602)<br>[accumulation, insertSCeffects.cc:837–856](https://github.com/rtsoliday/elegant/blob/0355dd6aca12c861971fbf8f3ae78a0ded991734/src/insertSCeffects.cc#L837-L856)<br>[pass start, insertSCeffects.cc:706–713](https://github.com/rtsoliday/elegant/blob/0355dd6aca12c861971fbf8f3ae78a0ded991734/src/insertSCeffects.cc#L706-L713)<br>[averaging_factor, elegant.tex:4492–4503](https://github.com/rtsoliday/elegant/blob/0355dd6aca12c861971fbf8f3ae78a0ded991734/doc/elegant.tex#L4492-L4503)<br>[linear force, elegant.tex:4529–4538](https://github.com/rtsoliday/elegant/blob/0355dd6aca12c861971fbf8f3ae78a0ded991734/doc/elegant.tex#L4529-L4538) |

With nonlinear SCMULT and size-averaging factor f < 1, the filtered rms beam sizes shape the field, but the integrated strength that normalizes the kick is still accumulated from the unfiltered sizes, although the manual says the filtered sizes are used in calculating the kicks. An independent Gaussian-field integral with the filtered sizes in both places supplies the expected kick; at f = 1 it reproduces Elegant. For a 2.400 nC bunch at momentum 10.00 mₑc whose second-turn rms sizes, 1.605 mm and 0.9067 mm, filter to 1.526 mm and 0.8267 mm with f = 0.2500, a particle's horizontal and vertical slope kicks are 5.341% and 7.091% too small.

Run `python COL-27.py` with [_elegant.py](../helpers/_elegant.py) in the same directory.

??? example "Reproducer: COL-27.py"

    [Download COL-27.py](COL-27.py)

    ```python
    """COL-27: Averaged nonlinear SCMULT normalizes its kick with the unfiltered beam sizes

    With nonlinear SCMULT and size-averaging factor f < 1, the filtered rms beam sizes shape the field,
    but the integrated strength that normalizes the kick is still accumulated from the unfiltered
    sizes, although the manual says the filtered sizes are used in calculating the kicks. An
    independent Gaussian-field integral with the filtered sizes in both places supplies the expected
    kick; at f = 1 it reproduces Elegant. For a 2.400 nC bunch at momentum 10.00 mₑc whose second-turn
    rms sizes, 1.605 mm and 0.9067 mm, filter to 1.526 mm and 0.8267 mm with f = 0.2500, a particle's
    horizontal and vertical slope kicks are 5.341% and 7.091% too small.

    The script tracks a 243-particle bunch with exact second moments for two turns around a ring made
    of one symplectic 45-degree EMATRIX rotation (1 m, β = 1 m) with a nonlinear SCMULT after it, for
    f = 1 (control) and f = 0.25. The expected kick evaluates the elliptical Gaussian field by a real
    Poisson integral, without Elegant's complex error function, with the filtered sizes in both the
    field and the integrated strength; a second prediction, which keeps the strength from the
    unfiltered sizes, reproduces what Elegant does. At f = 1 the two coincide with the native kick.
    """

    import itertools
    import sys

    import numpy as np
    from _elegant import modes, particles, run, verdict
    from scipy.integrate import quad

    # SDDS constants.h, as Elegant uses them.
    C, QE, RE = 299792458.0, 1.60217733e-19, 2.81794092e-15
    P = 10.0  # electron momentum, units of m_e c
    GAMMA = np.hypot(1, P)
    BETA_C = C * P / GAMMA
    CHARGE = 2.4e-9  # C
    SX, SY, SZ = 1.5e-3, 0.8e-3, 0.025  # m, rms sizes of the initial bunch
    LENGTH = 1.0  # m, integrating length of the EMATRIX segment
    PHASE = np.pi / 4  # transverse phase advance of the segment, both planes, beta = 1 m
    FACTORS = (1.0, 0.25)
    PROBE = 61  # zero-based row of the reported particle (particleID 62)
    # Elegant's complex error function, which uses single-precision reciprocals, carries relative
    # errors up to 2.0e-7 in this bunch; the Poisson integral is accurate to ~1e-12.
    RTOL, ATOL = 5e-7, 1e-18

    # Every combination of -sqrt(1.5), 0, +sqrt(1.5) in (x, xp, y, yp, z): each coordinate has unit
    # variance and zero mean, and the planes are uncorrelated, so the scaled bunch has exactly the
    # rms sizes SX, SY, SZ and slope spreads SX/beta, SY/beta.
    UNIT = np.array(list(itertools.product([-np.sqrt(1.5), 0.0, np.sqrt(1.5)], repeat=5)))
    N = len(UNIT)


    def ematrix(name, length):
        """EMATRIX line: the same phase advance PHASE in both transverse planes at beta = 1 m."""
        c, s = float(np.cos(PHASE)), float(np.sin(PHASE))
        terms = [f"R{i}{i}=1" for i in (5, 6)]
        for k in (1, 3):
            terms += [
                f"R{k}{k}={c!r}",
                f"R{k}{k + 1}={s!r}",
                f"R{k + 1}{k}={-s!r}",
                f"R{k + 1}{k + 1}={c!r}",
            ]
        return f"{name}: EMATRIX,L={length!r},ORDER=1," + ",".join(terms) + "\n"


    def track(mode, factor):
        """Coordinates by particle ID at the segment entrance and before/after SCMULT, both passes."""
        lattice = (
            f"Q: CHARGE,TOTAL={CHARGE!r}\n"
            'WI: WATCH,FILENAME="entrance.sdds",MODE="coordinates"\n'
            + ematrix("R", LENGTH)
            + 'WP: WATCH,FILENAME="before.sdds",MODE="coordinates"\n'
            'WA: WATCH,FILENAME="after.sdds",MODE="coordinates"\n'
            "RING: LINE=(Q,WI,R,WP,WA)\n"
        )
        # insert_sceffects places the SCMULT after WP, between the two WATCH elements.
        commands = (
            "&global_settings echo_namelists=0 &end\n"
            '&insert_sceffects name="WP",skip=1,horizontal=1,vertical=1,nonlinear=1,'
            f"averaging_factor={factor!r} &end\n"
            f"&run_setup lattice=lattice.lte,use_beamline=RING,p_central={P!r},default_order=1 &end\n"
            "&twiss_output matched=1 &end\n"
            "&run_control n_steps=1,n_passes=2 &end\n"
            "&sdds_beam input=beam.sdds &end\n"
            "&track &end\n"
        )
        x, xp, y, yp, z = UNIT.T
        beam = {
            "x": SX * x,
            "xp": SX * xp,
            "y": SY * y,
            "yp": SY * yp,
            "t": SZ * z / BETA_C,
            "p": np.full(N, P),
            "particleID": np.arange(1, N + 1),
        }
        names = ("entrance.sdds", "before.sdds", "after.sdds")
        result = run(
            {"lattice.lte": lattice, "run.ele": commands}, mode, {"beam.sdds": beam}, outputs=names
        )
        pages = {}
        for name in names:
            if result[name].n_pages != 2:
                sys.exit(f"{mode}: {name} has {result[name].n_pages} pages for 2 passes")
            pages[name] = [particles(result[name], k) for k in (0, 1)]
            if any(not np.array_equal(p["particleID"], np.arange(1, N + 1)) for p in pages[name]):
                sys.exit(f"{mode}: particles lost")
        return pages


    def moments(page):
        """Centroids and rms sizes (x, y, s) as SCMULT computes them, s = beta c t."""
        coordinates = np.column_stack([page["x"], page["y"], BETA_C * page["t"]])
        return coordinates.mean(axis=0), coordinates.std(axis=0)


    def strength(entrance, exit_):
        """Integrated strength sum over the segment, trapezoid in beta/(sigma_i (sigma_x + sigma_y)).

        SCMULT accumulates L/2 [beta_a/(a_i (a_x + a_y)) + beta_b/(b_i (b_x + b_y))] per element,
        with a and b the rms sizes at the element's entrance and exit; beta = 1 m at both ends here,
        and the kick divides by the beta at the SCMULT, also 1 m.
        """
        a, b = entrance[:2], exit_[:2]
        return LENGTH / 2 * (1 / (a * a.sum()) + 1 / (b * b.sum()))


    def kicks(page, centroid, sigma, d):
        """Slope kicks (x', y') of the Gaussian bunch field for strength d and sizes sigma.

        The kick is K lambda(z) d sigma_i (sigma_x + sigma_y) G_i, with K = 2 N r_e/(beta^2 gamma^3),
        the Gaussian line density lambda = exp(-z^2/(2 sigma_z^2))/(sqrt(2 pi) sigma_z) and the
        elliptical Gaussian field from the real Poisson integral
          G_x = x/2 int_0^inf exp(-x^2/(2(sx^2+u)) - y^2/(2(sy^2+u)))/((sx^2+u)^(3/2) (sy^2+u)^(1/2)) du
        (planes exchanged for G_y), whose central gradient is 1/[sx (sx + sy)], so that near the axis
        the kick is the manual's linear kick K L lambda x/[sx (sx + sy)]. Rescaling u by max(sigma)^2
        keeps the quadrature well conditioned.
        """
        k = 2 * CHARGE / QE * RE / (P * P * GAMMA)
        sx, sy, sz = sigma
        scale = max(sx, sy) ** 2
        a, b = sx * sx / scale, sy * sy / scale
        out = np.empty((N, 2))
        for i in range(N):
            x, y = page["x"][i] - centroid[0], page["y"][i] - centroid[1]
            z = BETA_C * page["t"][i] - centroid[2]

            def integrand(v, plane, x=x, y=y):
                ax, by = a + v, b + v
                return np.exp(-0.5 * (x * x / (scale * ax) + y * y / (scale * by))) / (
                    np.sqrt(ax * by) * (ax if plane == 0 else by)
                )

            g = [
                coordinate
                / (2 * scale)
                * quad(integrand, 0, np.inf, args=(plane,), epsabs=2e-13, epsrel=2e-13)[0]
                for plane, coordinate in ((0, x), (1, y))
            ]
            density = np.exp(-0.5 * (z / sz) ** 2) / (np.sqrt(2 * np.pi) * sz)
            out[i] = k * density * d * np.array([sx * (sx + sy) * g[0], sy * (sx + sy) * g[1]])
        return out


    def fmt(values):
        return "[" + ", ".join(f"{v:.16g}" for v in np.ravel(values)) + "]"


    defect = False
    for mode in modes():
        for factor in FACTORS:
            pages = track(mode, factor)
            entrance, before, after = (
                pages[n][1] for n in ("entrance.sdds", "before.sdds", "after.sdds")
            )
            _, sigma_entrance = moments(entrance)
            _, sigma_last = moments(pages["before.sdds"][0])  # first-turn sizes, stored unfiltered
            centroid, sigma_raw = moments(before)
            # The first kick changes only slopes and acts after the segment, so the second-turn
            # entrance sizes equal the initial ones; the filter acts only at the SCMULT.
            if np.max(np.abs(sigma_entrance - moments(pages["entrance.sdds"][0])[1])) > 1e-18:
                sys.exit(f"{mode} f={factor}: the entrance sizes changed between passes")
            # The manual's filter with the preceding turn's effective size, sigma_hat_0 = sigma_0.
            sigma_hat = factor * sigma_raw + (1 - factor) * sigma_last
            observed = np.column_stack([after["xp"] - before["xp"], after["yp"] - before["yp"]])
            # Effective sizes throughout: the filtered sizes at the SCMULT enter the strength too.
            expected = kicks(before, centroid, sigma_hat, strength(sigma_entrance, sigma_hat))
            # Elegant: filtered sizes in the field and density, strength from the unfiltered sizes.
            mixture = kicks(before, centroid, sigma_hat, strength(sigma_entrance, sigma_raw))
            label = f"{mode} f={factor}"
            print(
                f"{label}: sizes at SCMULT (x, y) mm unfiltered={fmt(sigma_raw[:2] * 1e3)} "
                f"filtered={fmt(sigma_hat[:2] * 1e3)}"
            )
            print(
                f"{label}, second turn, particle {PROBE + 1}: observed={fmt(observed[PROBE])} "
                f"expected={fmt(expected[PROBE])} deficit={fmt(1 - observed[PROBE] / expected[PROBE])}"
            )
            print(
                f"{label}, second turn, particle {PROBE + 1}: unfiltered-strength prediction={fmt(mixture[PROBE])}"
            )
            if not np.allclose(observed, mixture, rtol=RTOL, atol=ATOL):
                sys.exit(f"{label}: the kicks follow neither prediction")
            matches = bool(np.allclose(observed, expected, rtol=RTOL, atol=ATOL))
            if factor == 1.0 and not matches:
                sys.exit(f"{label}: the unfiltered control disagrees with the field integral")
            print(
                f"{label}: maximum |observed - expected| over particles {np.max(np.abs(observed - expected)):.6g} rad"
            )
            defect |= not matches
    print(f"comparison: rtol={RTOL:g}, atol={ATOL:g}")
    verdict(defect)

    ```

## Provenance

| Role | Model | Effort | Date | Code examined |
|---|---|---|---|---|
| Found | `gpt-6-astra` | xhigh | 2026-11-04 | master (`0355dd6aca12`), source with SDDS a3f730a14c49, CPU, serial |
| Confirmed | `gpt-6.1-sol` | xhigh | 2026-11-04 | master (`0355dd6aca12`), source with SDDS a3f730a14c49, CPU, serial |

