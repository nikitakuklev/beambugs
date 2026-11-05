---
title: "COL-67: Initial IBS rates use equilibrium beam sizes instead of natural sizes"
hide:
  - toc
---

# COL-67 — Initial IBS rates use equilibrium beam sizes instead of natural sizes

| | |
|---|---|
| Code | [Bmad](../index.md) · Collective effects |
| Priority | P4 |
| Severity | Silent wrong result |
| Evidence | Reproduced, independently rechecked |
| Last reproduced | 2026-11-05 on Bmad 20261031-0 (`c4c629f`) |
| Source | [full-current equilibrium calculation, bsim/ibs_ring/ibs_ring.f90:238–244](https://github.com/bmad-sim/bmad-ecosystem/blob/c4c629fc9071d3a56cb1ba5bc714baba65e9c5d3/bsim/ibs_ring/ibs_ring.f90#L238-L244)<br>[natural-emittance output contract and rate calls, bsim/ibs_ring/ibs_ring.f90:277–292](https://github.com/bmad-sim/bmad-ecosystem/blob/c4c629fc9071d3a56cb1ba5bc714baba65e9c5d3/bsim/ibs_ring/ibs_ring.f90#L277-L292)<br>[zero-current inputs and saved mode, bsim/ibs_ring/ibs_ring.f90:181–214](https://github.com/bmad-sim/bmad-ecosystem/blob/c4c629fc9071d3a56cb1ba5bc714baba65e9c5d3/bsim/ibs_ring/ibs_ring.f90#L181-L214)<br>[ibs_equib_der mutates element beam sizes, bmad/multiparticle/ibs_mod.f90:296–339](https://github.com/bmad-sim/bmad-ecosystem/blob/c4c629fc9071d3a56cb1ba5bc714baba65e9c5d3/bmad/multiparticle/ibs_mod.f90#L296-L339) |

When ibs_ring uses its derivative method to find equilibrium, its initial growth-rate output uses the equilibrium beam sizes instead of the specified natural, zero-current values. In a 0.001 mA example, horizontal and vertical emittances exceed their requested natural values by 1.523% and 12.39%. At the same population, the first element's vertical growth rate is 2601 s⁻¹ instead of 3025 s⁻¹, 14.01% low. Evaluating the rate kernel with the requested natural beam sizes establishes the expected result.

Download these shared helpers into the same directory as the reproducer scripts:

- [Download _bmad.py](../helpers/_bmad.py) · [View source](../helpers/index.md#_bmad)
- [Download _collective_contracts.py](../helpers/_collective_contracts.py) · [View source](../helpers/index.md#_collective_contracts)

??? example "Reproducer: COL-67.py"

    [Download COL-67.py](COL-67.py)

    ```python
    """COL-67: Initial IBS rates use equilibrium beam sizes instead of natural sizes

    When ibs_ring uses its derivative method to find equilibrium, its initial growth-rate output
    uses the equilibrium beam sizes instead of the specified natural, zero-current values. In a
    0.001 mA example, horizontal and vertical emittances exceed their requested natural values by
    1.523% and 12.39%. At the same population, the first element's vertical growth rate is
    2601 s⁻¹ instead of 3025 s⁻¹, 14.01% low. Evaluating the rate kernel with the requested
    natural beam sizes establishes the expected result.

    The script runs the production driver and a diagnostic rebuild with read-only calculations
    before the initial-rate call. Direct Bjorken–Mtingwa calls compare the actual and natural beam
    sizes at identical population; both diagnostic rate files must equal production output.
    BMAD_DIST and BMAD_SOURCE select the matching build and source tree.
    """

    import io

    import numpy as np
    from _bmad import compare, fortran, run, verdict
    from _collective_contracts import rows, source_text

    FILES = {
        "ring.bmad": """parameter[particle]=electron
    parameter[p0c]=1e9
    qf:quad,l=0.5,k1=0.6,tilt=0.05
    qd:quad,l=0.5,k1=-0.5
    d:drift,l=0.3
    b:sbend,l=3,angle=pi/8,e1=0.07,e2=0.07
    rf:rfcavity,l=1.2,rf_frequency=250e6,voltage=1e6
    cell:line=(d,qf,d,b,d,qd,d,rf,b)
    ring:line=(8*cell)
    use,ring
    """,
        "run.in": """&parameters
    lat_file='ring.bmad'
    a_emit=1e-9
    b_emit=1e-11
    energy_spread=0.001
    fake_3HC=-1
    high_current=0.001
    low_current=0.001
    delta_current=0.001
    ibs_formula='bjmt'
    clog_to_use=1
    eqb_method='der'
    ratio=0
    initial_blow_up=3,3,3
    set_dispersion=.false.
    eta_set=0
    etap_set=0
    inductance=0
    granularity=-1
    /
    """,
    }
    text = source_text("bsim/ibs_ring/ibs_ring.f90")
    natural_contract = "!Growth rates at full current at natural emittances" in text
    declaration = "namelist /parameters/ &"
    call = "call ibs_rates1turn(lat,ibs_sim_params,rates,granularity)"
    if text.count(declaration) != 1 or text.count(call) != 1:
        raise RuntimeError("Expected unique driver declaration and initial-rate call")
    text = text.replace(
        declaration,
        "type(ele_struct) audit_ele\nreal(rp) audit_lambda,audit_log\ninteger audit_j\n" + declaration,
    )
    text = text.replace(
        call,
        """write(*,'(a,4es25.16)') 'RESULT emittances ',lat%ele(1)%a%emit,lat%ele(1)%b%emit,a_emit,b_emit
    write(*,'(a,es25.16)') 'RESULT population ',lat%param%n_part
    do audit_j=1,2
     audit_ele=lat%ele(1)
     if (audit_j==2) then
      audit_ele%a%emit=a_emit
      audit_ele%b%emit=b_emit
      audit_ele%z%sigma_p=energy_spread
      audit_ele%z%sigma=mode0%sig_z
     endif
     audit_lambda=lat%param%n_part/audit_ele%z%sigma
     call multi_coulomb_log(ibs_sim_params,audit_ele,audit_log,audit_lambda)
     call bjmt1(audit_ele,audit_log,rates,audit_lambda)
     write(*,'(a,i0,3es25.16)') 'RESULT rates',audit_j,rates%inv_Tp,rates%inv_Ta,rates%inv_Tb
    enddo
    """
        + call,
    )
    # The target sizes come directly from the zero-current input contract: the
    # transverse emittances and energy spread are specified in run.in, and mode0
    # stores the zero-current bunch length. Both reference calculations use the
    # same actual population, isolating beam-size state from current conversion.
    production = run("ibs_ring", ["run.in"], files=FILES, outputs=["ibs_rates0.out", "ibs_rates.out"])
    diagnostic = fortran(text, ["run.in"], files=FILES, outputs=list(production.outputs))
    for result in (production, diagnostic):
        if "Step          1 of          1 complete!" not in result.log:
            raise RuntimeError(result.log)
    for name in production.outputs:
        actual = np.loadtxt(io.BytesIO(diagnostic.outputs[name]))
        control = np.loadtxt(io.BytesIO(production.outputs[name]))
        if not np.isfinite(actual).all() or not np.array_equal(actual, control):
            raise RuntimeError(f"Diagnostic build changed production output: {name}")
    print("production/diagnostic control: both numeric rate files identical")
    data = rows(diagnostic.log)
    emittances, population = data["emittances"], data["population"]
    if emittances.shape != (4,) or population.shape != (1,) or population[0] <= 0:
        raise RuntimeError("Invalid diagnostic beam-state records")
    published = np.loadtxt(io.BytesIO(production.outputs["ibs_rates0.out"]))[0, 1:]
    # The production file rounds each rate to six decimal places.
    if compare(
        "first-element rate-file control (1/s), direct kernel at actual sizes",
        published,
        data["rates1"],
        atol=5.01e-7,
    ):
        raise RuntimeError("Direct-kernel control does not match the published rate row")
    size_error = compare(
        "initial-output emittances (m), requested zero-current inputs",
        emittances[:2],
        emittances[2:],
        atol=0,
        rtol=1e-12,
    )
    rate_error = compare(
        "initial-output rates (1/s), direct kernel at natural sizes and identical population",
        data["rates1"],
        data["rates2"],
        atol=0,
        rtol=1e-10,
    )
    print(f"population held fixed: {population[0]:.16g} particles")
    print(f"source specifies natural-emittance rates: {natural_contract}")
    verdict(natural_contract and size_error and rate_error)

    ```

## Provenance

| Role | Model | Effort | Date | Code examined |
|---|---|---|---|---|
| Found | `gpt-6-astra` | xhigh | 2026-11-05 | 20261031-0 (`c4c629fc9071`), source, production, GNU Fortran 15.3.0, CPU, double, serial, OpenMP=1; driver rebuilt with one diagnostic write |
| Confirmed | `gpt-6.1-sol` | xhigh | 2026-11-05 | 20261031-0 (`c4c629fc9071`), source, production, GNU Fortran 15.3.0, CPU, double, serial, OpenMP=1; driver rebuilt with read-only diagnostics and reference calculations |

