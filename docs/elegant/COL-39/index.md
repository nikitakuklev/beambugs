---
title: "COL-39: Stupakov-model CSRDRIFT trapezoid integration omits the first interval behind each particle"
hide:
  - toc
---

# COL-39 — Stupakov-model CSRDRIFT trapezoid integration omits the first interval behind each particle

| | |
|---|---|
| Code | [Elegant](../index.md) · Collective effects |
| Priority | P2 |
| Severity | Wrong physics |
| Evidence | Reproduced, independently rechecked |
| Last reproduced | 2026-11-04 on Elegant master (`7514db3`) |
| Source | [Stupakov sum, csbend.c:5088–5112](https://github.com/rtsoliday/elegant/blob/0355dd6aca12c861971fbf8f3ae78a0ded991734/src/csbend.c#L5088-L5112)<br>[CSRDRIFT manual, csrdrift.tex:21–27](https://github.com/rtsoliday/elegant/blob/0355dd6aca12c861971fbf8f3ae78a0ded991734/doc/csrdrift.tex#L21-L27) |

CSRDRIFT with USE_STUPAKOV=1 and the upstream bend’s default TRAPAZOID_INTEGRATION=1 starts its trapezoid sum one bin behind each particle and omits the interval from zero to one bin. For a 35 μm rms Gaussian bunch after a 0.6 m, 0.3 rad bend with 400 bins, adaptive integration of the same formula gives a wake that the code misses by up to 7.085% of its peak 1 mm behind the bend, 5.306% at 10 mm and 2.348% at 100 mm. Restoring that interval reduces these errors to 0.1425%, 0.09961% and 0.05389%. The non-trapezoid path uses a right-endpoint sum instead.

Download these shared helpers into the same directory as the reproducer scripts:

- [Download _csr.py](../helpers/_csr.py) · [View source](../helpers/index.md#_csr)
- [Download _elegant.py](../helpers/_elegant.py) · [View source](../helpers/index.md#_elegant)

??? example "Reproducer: COL-39.py"

    [Download COL-39.py](COL-39.py)

    ```python
    """COL-39: Stupakov-model CSRDRIFT trapezoid integration omits the first interval behind each particle

    CSRDRIFT with USE_STUPAKOV=1 and the upstream bend’s default TRAPAZOID_INTEGRATION=1 starts its
    trapezoid sum one bin behind each particle and omits the interval from zero to one bin. For a 35
    μm rms Gaussian bunch after a 0.6 m, 0.3 rad bend with 400 bins, adaptive integration of the
    same formula gives a wake that the code misses by up to 7.085% of its peak 1 mm behind the bend,
    5.306% at 10 mm and 2.348% at 100 mm. Restoring that interval reduces these errors to 0.1425%,
    0.09961% and 0.05389%. The non-trapezoid path uses a right-endpoint sum instead.

    The script tracks 240001 electrons (Gaussian quantiles, σ_ct = 35 μm, p = 4000 mₑc, 10⁻¹⁶ C so
    the beam does not change) through a one-slice 0.6 m, 0.3 rad CSRCSBEND (400 bins,
    STEADY_STATE=1) and a CSRDRIFT with USE_STUPAKOV=1 and one kick at its middle, for drifts of
    2 mm, 20 mm and 200 mm, and reads the drift's exported wake. Expected is the Stupakov–Emma
    post-bend wake of the Gaussian bunch, integrated adaptively in the retarded angle. The script
    then adds back only the omitted first interval, integrated from the exported density
    derivative, which must remove almost all of the error.
    """

    import numpy as np
    from _csr import E_CHARGE, R_E, C, gaussian_beam, page, track
    from _elegant import verdict
    from scipy.integrate import quad
    from scipy.optimize import brentq

    P0, SIGMA, CHARGE, BINS = 4000.0, 35e-6, 1e-16, 400
    L, ANGLE = 0.6, 0.3
    RHO = L / ANGLE
    N = CHARGE / E_CHARGE
    BEND = (
        f"L={L},ANGLE={ANGLE},N_SLICES=1,INTEGRATION_ORDER=6,SG_HALFWIDTH=1,SGDERIV_HALFWIDTH=3,"
        f"SGDERIV_ORDER=3,BINS={BINS},STEADY_STATE=1"
    )
    BEAM = gaussian_beam(240001, SIGMA, P0)
    TOLERANCE = 0.005  # of the expected peak; with the first interval restored the error is < 0.15%


    def density(z):
        return np.exp(-0.5 * (z / SIGMA) ** 2) / (np.sqrt(2 * np.pi) * SIGMA)


    def density_prime(z):
        return -z / SIGMA**2 * density(z)


    def delay(phi: float, x: float) -> float:
        """Path difference u between a particle at x = d/rho behind the bend exit and the source
        point at retarded angle phi before the exit (Stupakov and Emma, LCLS-TN-01-12, eq. 18)."""
        return RHO * phi**3 * (phi + 4 * x) / (24 * (phi + x))


    def weight(phi: float, x: float) -> float:
        """(du/dphi)/(phi + 2x) = rho phi^2 (phi + 2x) / (8 (phi + x)^2): the kernel du/(phi + 2x) of
        the post-bend wake written in the retarded angle, regular at phi = 0."""
        return RHO * phi**2 * (phi + 2 * x) / (8 * (phi + x) ** 2)


    def expected_wake(z: np.ndarray, x: float, dz: float) -> np.ndarray:
        """Energy change over a drift step dz at distance x rho behind the bend exit (eqs. 19, 22):

            Delta gamma(z) = (4 N r_e dz/rho) [ -lambda(z + u_max)/(A + 2x)
                                                 + int_0^A lambda'(z + u(phi)) du/(phi + 2x) ],
        u_max = u(A); z grows toward the tail. The bunch is far shorter than the distance to the
        bend entrance, so the contribution of the field emitted before the bend (case C) vanishes.
        """
        k4 = 4 * N * R_E * dz / RHO
        u_max = delay(ANGLE, x)
        values = []
        for zz in z:
            integral = quad(
                lambda phi, zz=zz: density_prime(zz + delay(phi, x)) * weight(phi, x),
                0,
                ANGLE,
                epsabs=1e-3,
                epsrel=2e-10,
                limit=400,
            )[0]
            values.append(k4 * (integral - density(zz + u_max) / (ANGLE + 2 * x)))
        return np.array(values)


    def first_interval(lambda_prime: np.ndarray, h: float, x: float, dz: float) -> np.ndarray:
        """The omitted part: the same integral over 0 <= u <= h (phi from 0 to phi_h), with the
        exported density derivative interpolated linearly between bin i and bin i + 1."""
        k4 = 4 * N * R_E * dz / RHO
        phi_h = brentq(lambda f: delay(f, x) - h, 0, ANGLE, xtol=1e-16)
        out = np.zeros(len(lambda_prime))
        for i in range(len(lambda_prime) - 1):
            a, b = lambda_prime[i], lambda_prime[i + 1]
            out[i] = (
                k4
                * quad(
                    lambda f, a=a, b=b: (a + (b - a) * delay(f, x) / h) * weight(f, x),
                    0,
                    phi_h,
                    epsrel=1e-11,
                )[0]
            )
        return out


    defect = False
    for drift in (0.002, 0.02, 0.2):
        lattice = (
            f"Q: CHARGE,TOTAL={CHARGE!r}\nB: CSRCSBEND,{BEND}\n"
            f'D: CSRDRIFT,L={drift},USE_STUPAKOV=1,N_KICKS=1,STUPAKOV_OUTPUT="wake.sdds"'
        )
        wake, info = page(track(lattice, "Q,B,D", BEAM, P0, outputs=("wake.sdds",))["wake.sdds"])
        x = drift / 2 / RHO  # the single kick sits at the middle of the drift
        if not np.isclose(info["x"], x, rtol=1e-12):
            raise SystemExit("The kick is not at the middle of the drift.")
        expected = expected_wake(wake["s"], x, drift)
        peak = np.max(np.abs(expected))
        error = np.max(np.abs(wake["DeltaGamma"] - expected)) / peak
        h = wake["s"][1] - wake["s"][0]
        restored = wake["DeltaGamma"] + first_interval(
            wake["LinearDensityDeriv"] / (CHARGE * C**2), h, x, drift
        )
        error_restored = np.max(np.abs(restored - expected)) / peak
        print(
            f"serial kick {drift / 2:g} m behind the bend: peak observed={np.max(np.abs(wake['DeltaGamma'])):.10g} "
            f"expected={peak:.10g}; max |wake - expected| / peak={100 * error:.4g}%; "
            f"with the first interval added back {100 * error_restored:.4g}%"
        )
        if error_restored > TOLERANCE:
            raise SystemExit("Adding back the first interval does not reproduce the expected wake.")
        defect |= error > TOLERANCE
    print(f"tolerance {TOLERANCE:.1%} of the expected peak")
    verdict(defect)

    ```

## Provenance

| Role | Model | Effort | Date | Code examined |
|---|---|---|---|---|
| Found | `gemini-4-argon-medium` |  | 2026-11-04 | master (`7514db36f764`), source reading, not executed for this claim |
| Found (executed verification) | `claude-opus-5-5` | high | 2026-11-04 | master (`7514db36f764`), source with SDDS a3f730a14c49, CPU, serial |
| Confirmed | `gpt-6.1-sol` | xhigh | 2026-11-04 | master (`7514db36f764`), source with SDDS a3f730a14c49, CPU, serial |

