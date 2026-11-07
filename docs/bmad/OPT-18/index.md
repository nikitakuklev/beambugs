---
title: "OPT-18: The mode-flip manual example reverses forward conjugation"
hide:
  - toc
---

# OPT-18 — The mode-flip manual example reverses forward conjugation

| | |
|---|---|
| Code | [Bmad](../index.md) · Optics |
| Priority | P4 |
| Severity | Documentation or example mismatch |
| Evidence | Reproduced, independently rechecked |
| Last reproduced | 2026-11-05 on Bmad release [20261031-0](https://github.com/bmad-sim/bmad-ecosystem/tree/20261031-0) (`c4c629f`) |
| Source | [Bmad manual, “Coupling and Normal Modes”, mode-flip example `t0ef0`, bmad/doc/optics.tex:156–188](https://github.com/bmad-sim/bmad-ecosystem/blob/c4c629fc9071d3a56cb1ba5bc714baba65e9c5d3/bmad/doc/optics.tex#L156-L188)<br>[forward transport convention, bmad/doc/optics.tex:11–18](https://github.com/bmad-sim/bmad-ecosystem/blob/c4c629fc9071d3a56cb1ba5bc714baba65e9c5d3/bmad/doc/optics.tex#L11-L18) |

Following the mode-flip example in the optics manual transports the one-turn matrix with the conjugation reversed. If coordinates obey x₂ = M₁₂ x₁, the matrix at the second point must be T₂ = M₁₂ T₁ M₁₂⁻¹, as the earlier convention in the same chapter states. For an explicit symplectic example with unequal off-diagonal blocks, the displayed formula gives T₂₁₁ = 2.697 instead of −0.5236 and fails the identity T₂ M₁₂ = M₁₂ T₁.

Run `python OPT-18.py` with [_bmad.py](../helpers/_bmad.py) and [_dispersion_and_moments.py](../helpers/_dispersion_and_moments.py) in the same directory.

??? example "Reproducer: OPT-18.py"

    [Download OPT-18.py](OPT-18.py)

    ```python
    """OPT-18: The mode-flip manual example reverses forward conjugation

    Following the mode-flip example in the optics manual transports the one-turn matrix with the
    conjugation reversed. If coordinates obey x₂ = M₁₂ x₁, the matrix at the second point must be
    T₂ = M₁₂ T₁ M₁₂⁻¹, as the earlier convention in the same chapter states. For an explicit
    symplectic example with unequal off-diagonal blocks, the displayed formula gives T₂₁₁ = 2.697
    instead of −0.5236 and fails the identity T₂ M₁₂ = M₁₂ T₁.

    The script reads the manual formula through BMAD_SOURCE and builds an explicit symplectic mode-flip
    example. It evaluates both conjugations using the compiled Bmad matrix inversion routine, then
    checks the forward vector-map identity in NumPy.
    """

    import numpy as np
    from _bmad import compare, source, verdict
    from _dispersion_and_moments import execute, require

    manual = (source() / "bmad/doc/optics.tex").read_text()
    wrong = r"\bfT_2 = \bfM_{12}^{-1} \, \bfT_1 \, \bfM_{12}"
    correct = r"\bfT_2 = \bfM_{12} \, \bfT_1 \, \bfM_{12}^{-1}"
    if wrong in manual:
        reversed_order = True
    elif correct in manual:
        reversed_order = False
    else:
        raise RuntimeError("Mode-flip formula changed; inspect its convention")
    # Independent E and F are symplectic drifts, so M12=[[0,E],[F,0]] is symplectic.
    e = np.array([[1.0, 2.0], [0.0, 1.0]])
    f = np.array([[1.0, 3.0], [0.0, 1.0]])
    m = np.block([[np.zeros((2, 2)), e], [f, np.zeros((2, 2))]])
    t1 = np.zeros((4, 4))
    for j, mu in enumerate((0.4, 0.7)):
        t1[2 * j : 2 * j + 2, 2 * j : 2 * j + 2] = [[np.cos(mu), np.sin(mu)], [-np.sin(mu), np.cos(mu)]]
    j4 = np.kron(np.eye(2), [[0, 1], [-1, 0]])
    require(m.T @ j4 @ m, j4)
    text = "\n".join(" ".join(map(str, row)) for row in np.vstack([m, t1])) + "\n"
    # Bmad matrix inversion evaluates both expressions in the manual's own 4D convention.
    rows = execute(
        r"""
    program conjugation
    use bmad
    implicit none
    real(rp) m(4,4),mi(4,4),t1(4,4),forward(4,4),reverse(4,4)
    integer i
    character(8) tag
    open(10,file='input.dat',status='old')
    do i=1,4
    read(10,*) m(i,:)
    enddo
    do i=1,4
    read(10,*) t1(i,:)
    enddo
    close(10)
    call mat_inverse(m,mi)
    forward=matmul(matmul(m,t1),mi)
    reverse=matmul(matmul(mi,t1),m)
    do i=1,4
    write(tag,'(a,i0)') 'f',i
    write(*,'(a,a,4es25.17)') 'RESULT ',trim(tag),forward(i,:)
    write(tag,'(a,i0)') 'r',i
    write(*,'(a,a,4es25.17)') 'RESULT ',trim(tag),reverse(i,:)
    enddo
    end program
    """,
        {"input.dat": text},
    )
    forward = np.array([rows[f"f{i}"] for i in range(1, 5)])
    reverse = np.array([rows[f"r{i}"] for i in range(1, 5)])
    # x2=M12*x1 implies T2*M12=M12*T1; this checks the transported vector map directly.
    require(forward @ m, m @ t1)
    observed = reverse if reversed_order else forward
    verdict(
        compare(
            "mode-flip manual T2, forward vector-map identity",
            observed,
            forward,
            atol=1e-14,
            rtol=1e-14,
        )
    )

    ```

## Provenance

| Role | Model | Effort | Date | Code examined |
|---|---|---|---|---|
| Found<br><small>source check</small> | `gpt-6-astra` | xhigh | 2026-10-20 | 20261020-0 (`19b8afb46a40`) |
| Confirmed<br><small>source check</small> | `gpt-6.1-sol` | xhigh | 2026-10-20 | 20261020-0 (`19b8afb46a40`) |
| Confirmed<br><small>manual forward conjugation, exact rational symplectic counterexample</small> | `gpt-6.1-sol` | xhigh | 2026-11-05 | 20261031-0 (`c4c629fc9071`) |

