---
title: "COL-34: Envelope IBS argument comments call a damping time a rate"
hide:
  - toc
---

# COL-34 — Envelope IBS argument comments call a damping time a rate

| | |
|---|---|
| Code | [Bmad](../index.md) · Collective effects |
| Priority | P5 |
| Severity | Documentation or example mismatch |
| Evidence | Reproduced, independently rechecked |
| Last reproduced | 2026-11-05 on Bmad 20261031-0 (`c4c629f`) |
| Tags | <span class="bb-tag">partly-source-only</span> [?](../../methodology.md#tags) |
| Source | [ibs_matrix_C argument comments, bmad/multiparticle/envelope_mod.f90:720–728](https://github.com/bmad-sim/bmad-ecosystem/blob/c4c629fc9071d3a56cb1ba5bc714baba65e9c5d3/bmad/multiparticle/envelope_mod.f90#L720-L728)<br>[beam_envelope_ibs argument comments, bmad/multiparticle/envelope_mod.f90:758–766](https://github.com/bmad-sim/bmad-ecosystem/blob/c4c629fc9071d3a56cb1ba5bc714baba65e9c5d3/bmad/multiparticle/envelope_mod.f90#L758-L766)<br>[tail-cut impact parameter, bmad/multiparticle/envelope_mod.f90:909–918](https://github.com/bmad-sim/bmad-ecosystem/blob/c4c629fc9071d3a56cb1ba5bc714baba65e9c5d3/bmad/multiparticle/envelope_mod.f90#L909-L918)<br>[tau_a comment, bmad/multiparticle/ibs_mod.f90:10–12](https://github.com/bmad-sim/bmad-ecosystem/blob/c4c629fc9071d3a56cb1ba5bc714baba65e9c5d3/bmad/multiparticle/ibs_mod.f90#L10-L12)<br>[driver damping-time calculation, bsim/envelope_ibs/envelope_ibs.f90:210–216](https://github.com/bmad-sim/bmad-ecosystem/blob/c4c629fc9071d3a56cb1ba5bc714baba65e9c5d3/bsim/envelope_ibs/envelope_ibs.f90#L210-L216)<br>[longest-mode choice and call, bsim/envelope_ibs/envelope_ibs.f90:239–252](https://github.com/bmad-sim/bmad-ecosystem/blob/c4c629fc9071d3a56cb1ba5bc714baba65e9c5d3/bsim/envelope_ibs/envelope_ibs.f90#L239-L252) |

When callers follow the envelope IBS argument comments, they supply a damping rate where the tail-cut calculation requires a damping time in seconds. The implementation divides a ratio of volume to speed by that input to form a squared impact parameter, which requires time units. In a fixed diagonal beam covariance, passing 100 instead of a 0.01 s damping time increases the diffusion norm by a factor of 1.621. The envelope driver already passes the longest mode damping time; the argument comments should describe that convention.

Download these shared helpers into the same directory as the reproducer scripts:

- [Download _bmad.py](../helpers/_bmad.py) · [View source](../helpers/index.md#_bmad)
- [Download _collective_contracts.py](../helpers/_collective_contracts.py) · [View source](../helpers/index.md#_collective_contracts)

??? example "Reproducer: COL-34.py"

    [Download COL-34.py](COL-34.py)

    ```python
    """COL-34: Envelope IBS argument comments call a damping time a rate

    When callers follow the envelope IBS argument comments, they supply a damping rate where the tail-cut calculation requires a damping time in seconds. The implementation divides a ratio of volume to speed by that input to form a squared impact parameter, which requires time units. In a fixed diagonal beam covariance, passing 100 instead of a 0.01 s damping time increases the diffusion norm by a factor of 1.621. The envelope driver already passes the longest mode damping time; the argument comments should describe that convention.

    The script evaluates the envelope IBS kernel on the same diagonal covariance with a damping time
    and its reciprocal. It compares the finite diffusion outputs and reads the argument descriptions
    from BMAD_SOURCE. BMAD_DIST selects the production library.
    """

    import numpy as np
    from _bmad import fortran, verdict
    from _collective_contracts import rows, source_text

    r = rows(
        fortran("""program damping_time
    use envelope_mod
    implicit none
    real(rp) sigma(6,6),mat(6,6),tau(2)
    integer i,j
    sigma=0
    sigma(1,1)=1e-6_rp
    sigma(3,3)=1e-6_rp
    sigma(5,5)=1e-6_rp
    sigma(2,2)=1e-10_rp
    sigma(4,4)=4e-10_rp
    sigma(6,6)=9e-6_rp
    tau=[0.01_rp,100.0_rp]
    do j=1,2
     call beam_envelope_ibs(sigma,mat,.true.,tau(j),1000*m_electron,1e8_rp,electron$)
     write(*,'(a,i0,6es25.16)') 'RESULT tau',j,(mat(i,i),i=1,6)
    enddo
    end program
    """).log
    )
    if not np.isfinite([r["tau1"], r["tau2"]]).all() or np.linalg.norm(r["tau1"]) == 0:
        raise RuntimeError("Nonfinite or zero diffusion fixture")
    print(f"tau=0.01 s: diagonal diffusion={r['tau1'].tolist()} per m")
    print(f"tau=100 (inverse of 0.01): diagonal diffusion={r['tau2'].tolist()} per m")
    print(
        f"diffusion norm ratio, inverse-time input / time input={np.linalg.norm(r['tau2']) / np.linalg.norm(r['tau1']):.16g}"
    )
    text = source_text("bmad/multiparticle/envelope_mod.f90")
    wrong_header = "horizontal betatron damping rate" in text
    print(f"API headers label tau as damping rate={wrong_header}; tail-cut dimensions require seconds")
    # vol/N has units m^3 and ptrans*c has units m/s. Dividing their
    # ratio by tau in seconds gives bmin2^2 in m^2; a rate gives m^2*s^2.
    verdict(wrong_header and not np.allclose(r["tau1"], r["tau2"], rtol=1e-12, atol=0))

    ```

## Provenance

| Role | Model | Effort | Date | Code examined |
|---|---|---|---|---|
| Found (source/contract discrepancy) | `gemini-4-argon-high` |  | 2026-10-20 | 20261020-0 (`19b8afb46a40`) |
| Confirmed (source/contract discrepancy) | `gpt-6-astra` | xhigh | 2026-10-20 | 20261020-0 (`19b8afb46a40`) |
| Confirmed | `gpt-6.1-sol` | xhigh | 2026-11-05 | 20261031-0 (`c4c629fc9071`), source, production, GNU Fortran 15.3.0, CPU, double, serial, OpenMP=1 |

