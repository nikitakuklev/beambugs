---
title: "COL-23: Coupled IBS diffusion is rotated with transposed eigenvectors"
hide:
  - toc
---

# COL-23 — Coupled IBS diffusion is rotated with transposed eigenvectors

| | |
|---|---|
| Code | [Bmad](../index.md) · Collective effects |
| Priority | P2 |
| Failure | Wrong physics |
| Evidence | Reproduced, independently rechecked |
| Last reproduced | 2026-11-05 on Bmad release [20261031-0](https://github.com/bmad-sim/bmad-ecosystem/tree/20261031-0) (`c4c629f`) |
| Source | [beam_envelope_ibs, bmad/multiparticle/envelope_mod.f90:865–933](https://github.com/bmad-sim/bmad-ecosystem/blob/c4c629fc9071d3a56cb1ba5bc714baba65e9c5d3/bmad/multiparticle/envelope_mod.f90#L865-L933) |

For a beam with coupled momentum covariance, the envelope IBS calculation rotates its diffusion with transposed eigenvectors. Rotating the input covariance must rotate the diffusion tensor by the same rotation. A three-axis rotation produces a relative tensor error of 1.530 and changes the first rest-frame diagonal from the expected +8.280 × 10⁻²⁰ m⁻¹ to −1.071 × 10⁻¹⁹ m⁻¹. The unrotated control agrees with independent Kubo quadrature.

Run `python COL-23.py` with [_bmad.py](../helpers/_bmad.py) in the same directory.

??? example "Reproducer: COL-23.py"

    [Download COL-23.py](COL-23.py)

    ```python
    """COL-23: Coupled IBS diffusion is rotated with transposed eigenvectors

    For a beam with coupled momentum covariance, the envelope IBS calculation rotates its diffusion with transposed eigenvectors. Rotating the input covariance must rotate the diffusion tensor by the same rotation. A three-axis rotation produces a relative tensor error of 1.530 and changes the first rest-frame diagonal from the expected +8.280 × 10⁻²⁰ m⁻¹ to −1.071 × 10⁻¹⁹ m⁻¹. The unrotated control agrees with independent Kubo quadrature.

    The script compiles a driver against BMAD_DIST and evaluates diagonal and rotated covariance matrices. It compares the rest-frame diffusion with rotational covariance and independent Kubo integrals; all inputs are deterministic.
    """

    import numpy as np
    from _bmad import compare, fortran, verdict
    from scipy.integrate import quad

    PROGRAM = """program probe_envelope
    use bmad
    use envelope_mod, only: beam_envelope_ibs, eigensys
    implicit none
    real(rp) sigma(6,6), cmat(6,6)
    integer i, j, nc
    open(11,file='envelope_cases.dat',status='old')
    read(11,*) nc
    do j=1,nc
      do i=1,6
        read(11,*) sigma(i,:)
      enddo
      call beam_envelope_ibs(sigma,cmat,.false.,0.01_rp,1000*m_electron,1.0e8_rp,electron$)
      do i=1,6
        write(*,'(6es25.16)') cmat(i,:)
      enddo
    enddo
    close(11)
    end program
    """
    gamma = 1000.0
    boost = np.diag([1, 1, 1, 1, 1 + gamma**2 / (1 + gamma), 1 / gamma])
    invboost = np.linalg.inv(boost)
    u = np.array([1.0, 4.0, 9.0]) * 1e-12
    a, b, c = 0.37, 0.61, 0.29
    qx = np.array([[1, 0, 0], [0, np.cos(a), -np.sin(a)], [0, np.sin(a), np.cos(a)]])
    qy = np.array([[np.cos(b), 0, np.sin(b)], [0, 1, 0], [-np.sin(b), 0, np.cos(b)]])
    qz = np.array([[np.cos(c), -np.sin(c), 0], [np.sin(c), np.cos(c), 0], [0, 0, 1]])
    q = qz @ qy @ qx
    sigmas = []
    for rot in [np.eye(3), q]:
        rest = np.diag([1e-6, u[0], 1e-6, u[1], 1e-6, u[2]])
        rest[np.ix_([1, 3, 5], [1, 3, 5])] = rot @ np.diag(u) @ rot.T
        sigmas.append(invboost @ rest @ invboost)
    payload = "2\n" + "".join(
        " ".join(f"{v:.17e}" for v in row) + "\n" for sigma in sigmas for row in sigma
    )
    out = fortran(PROGRAM, files={"envelope_cases.dat": payload}).log
    cmats = np.fromstring(out, sep=" ").reshape(2, 6, 6)
    restcs = [(boost @ v @ boost)[np.ix_([1, 3, 5], [1, 3, 5])] for v in cmats]
    pred = q @ restcs[0] @ q.T
    np.set_printoptions(precision=9)
    print("rest-frame diffusion, rotated beam (production):\n", restcs[1])
    print("expected by rotational covariance Q C0 Q^T:\n", pred)
    print("relative Frobenius error:", np.linalg.norm(restcs[1] - pred) / np.linalg.norm(pred))
    print("energy conservation: trace(C), expected 0 =", np.trace(restcs[1]))
    # Independent Kubo eigenvalue quadrature with normalized numpy eigenvectors.
    g = []
    for i in range(3):
        u1, u2, u3 = u[[i, (i + 1) % 3, (i + 2) % 3]]
        g.append(
            quad(
                lambda s, u1=u1, u2=u2, u3=u3: (
                    2
                    * u1
                    * np.sin(s) ** 2
                    * np.cos(s)
                    / np.sqrt(
                        (np.sin(s) ** 2 + u1 / u2 * np.cos(s) ** 2)
                        * (np.sin(s) ** 2 + u1 / u3 * np.cos(s) ** 2)
                    )
                ),
                0,
                np.pi / 2,
                epsabs=1e-26,
                epsrel=1e-12,
            )[0]
        )
    r = 2.8179403227e-15
    vol1 = 1e-9
    vol = (4 * np.pi) ** 1.5 * vol1
    bmax = min(1e-3, (vol / 1e8) ** (1 / 3))
    bmin = r / (np.sum(u) * gamma**2)
    ci = r * r * 1e8 * np.log(bmax / bmin) / (4 * np.pi * gamma**4 * vol1 * np.sqrt(np.prod(u)))
    dw = ci * (sum(g) - 3 * np.array(g))
    independent = q @ np.diag(dw) @ q.T
    print("independent numpy + quadrature:\n", independent)
    print(
        "diagonal control relative error:", np.linalg.norm(restcs[0] - np.diag(dw)) / np.linalg.norm(dw)
    )
    print(
        "rotated relative error:", np.linalg.norm(restcs[1] - independent) / np.linalg.norm(independent)
    )

    if compare(
        "unrotated diffusion control (m^-1), Kubo quadrature",
        restcs[0],
        np.diag(dw),
        atol=1e-30,
        rtol=2e-11,
    ):
        raise RuntimeError("The unrotated quadrature control failed")
    verdict(
        compare("rotated diffusion (m^-1), Q C0 Q^T", restcs[1], independent, atol=1e-30, rtol=2e-11)
    )

    ```

## Provenance

| Role | Model | Effort | Date | Code examined |
|---|---|---|---|---|
| Found | <span class="bb-nowrap">`gpt-6.1-sol`</span> | xhigh | <span class="bb-nowrap">2026-10-21</span> | 20261020-0 (`19b8afb46a40`), source, production, CPU, double, serial |
| Confirmed | <span class="bb-nowrap">`gpt-6-astra`</span> | xhigh | <span class="bb-nowrap">2026-10-21</span> | 20261020-0 (`19b8afb46a40`), source, production, CPU, double, serial |

