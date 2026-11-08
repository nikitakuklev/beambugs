---
title: "DIA-12: The KVDist3D docstring gives a 100% emittance of eight times the RMS value; the generator produces six"
hide:
  - toc
---

# DIA-12 — The `KVDist3D` docstring gives a 100% emittance of eight times the RMS value; the generator produces six

| | |
|---|---|
| Code | [PyORBIT](../index.md) · Diagnostics |
| Priority | P5 |
| Failure | Documentation or example mismatch |
| Evidence | Reproduced, independently rechecked |
| Last reproduced | 2026-11-07 on PyORBIT commit [`28df660`](https://github.com/PyORBIT-Collaboration/PyORBIT3/commit/28df66018aeb), committed 2026-10-14 |
| Source | [`distribution_generators.py#L182-L229`](https://github.com/PyORBIT-Collaboration/PyORBIT3/blob/28df66018aebe4e2b780b9747f4a2d7b8430a5c0/py/orbit/bunch_generators/distribution_generators.py#L182-L229)<br>[`distribution_generators.py#L236-L257`](https://github.com/PyORBIT-Collaboration/PyORBIT3/blob/28df66018aebe4e2b780b9747f4a2d7b8430a5c0/py/orbit/bunch_generators/distribution_generators.py#L236-L257) |

The `KVDist3D` docstring says "100% emittance is 8 times bigger for 3D KV distribution", but the constructor scales each plane's Twiss container by 6 and samples the unit 5-sphere, so the normalized amplitudes satisfy Hx + Hy + Hz = 6 exactly, with H = (u² + (αu + βu′)²)/(βε) and ε the RMS emittance, and no single-plane amplitude exceeds 6. The factor eight belongs to the 3D waterbag, whose docstring states it correctly. A full emittance taken from the docstring overestimates that of the generated KV beam by 33%. For 200000 particles the sum is 6.000000 for every particle and the largest single-plane H is 5.997.

Run `python DIA-12.py`.

??? example "Reproducer: DIA-12.py"

    [Download DIA-12.py](DIA-12.py)

    ```python
    """DIA-12: The `KVDist3D` docstring gives a 100% emittance of eight times the RMS value; the generator produces six

    The `KVDist3D` docstring says "100% emittance is 8 times bigger for 3D KV distribution", but the
    constructor scales each plane's Twiss container by 6 and samples the unit 5-sphere, so the
    normalized amplitudes satisfy Hx + Hy + Hz = 6 exactly, with H = (u² + (αu + βu′)²)/(βε) and ε the
    RMS emittance, and no single-plane amplitude exceeds 6. The factor eight belongs to the 3D waterbag,
    whose docstring states it correctly. A full emittance taken from the docstring overestimates that of
    the generated KV beam by 33%. For 200000 particles the sum is 6.000000 for every particle and the
    largest single-plane H is 5.997.

    The script draws 200000 particles from `KVDist3D` with α = (0.7, −0.4, 0.2), β = (5, 3, 8) m and
    RMS emittances (2, 3, 5)e-6 m (seed 11), evaluates each plane's normalized amplitude
    H = (u² + (αu + βu′)²)/(βε) with the input RMS emittance ε, and reports the RMS emittance ratio, the
    range of Hx + Hy + Hz and the largest single-plane H. The control does the same for
    `WaterBagDist3D`, whose docstring gives the factor eight.
    """

    import random

    import numpy as np
    from orbit.bunch_generators import KVDist3D, TwissContainer, WaterBagDist3D

    ALPHA = (0.7, -0.4, 0.2)
    BETA = (5.0, 3.0, 8.0)
    EPS = (2e-6, 3e-6, 5e-6)
    N = 200000


    def sample(cls):
        random.seed(11)
        gen = cls(*(TwissContainer(a, b, e) for a, b, e in zip(ALPHA, BETA, EPS)))
        c = np.array([gen.getCoordinates() for _ in range(N)])
        H, ratio = [], []
        for p in range(3):
            u, up = c[:, 2 * p], c[:, 2 * p + 1]
            H.append((u**2 + (ALPHA[p] * u + BETA[p] * up) ** 2) / (BETA[p] * EPS[p]))
            cov = np.cov(u, up, bias=True)
            ratio.append(np.sqrt(np.linalg.det(cov)) / EPS[p])
        return np.array(H), ratio


    # KV: points uniform on the unit 5-sphere have <u_k^2> = 1/6 per coordinate, so a container emittance
    # of 6 eps gives RMS eps and Hx + Hy + Hz = 6 for every particle; the 100% emittance of one plane is
    # max Hx * eps = 6 eps. Waterbag: the uniform 6-ball has <u_k^2> = 1/8, so its bound is 8 eps.
    ok = True
    for name, cls, factor in (
        ("KVDist3D", KVDist3D, 6.0),
        ("WaterBagDist3D (control)", WaterBagDist3D, 8.0),
    ):
        H, ratio = sample(cls)
        tot = H.sum(axis=0)
        print(
            f"{name}: RMS emittance / input {', '.join(f'{r:.4f}' for r in ratio)}; "
            f"Hx + Hy + Hz in [{tot.min():.6f}, {tot.max():.6f}]; largest single-plane H "
            f"{', '.join(f'{h.max():.4f}' for h in H)}; docstring factor 8"
        )
        ok &= factor - 1e-3 < tot.max() < factor + 1e-9
        if cls is KVDist3D:
            ok &= np.ptp(tot) < 1e-9  # every KV particle lies on the shell sum H = 6
    print("DEFECT REPRODUCED" if ok else "NOT REPRODUCED")

    ```

## Provenance

| Role | Model | Effort | Date | Code examined |
|---|---|---|---|---|
| Found | <span class="bb-nowrap">`claude-opus-5-5`</span> | high | <span class="bb-nowrap">2026-11-07</span> | main (`28df66018aeb`), source, serial, CPU, double |
| Confirmed | <span class="bb-nowrap">`claude-sonnet-5-5`</span> | high | <span class="bb-nowrap">2026-11-07</span> | main (`28df66018aeb`), source, serial, CPU, double |

