---
title: "COL-44: ibs_linac omits charge-squared scaling from ISR energy-spread growth"
hide:
  - toc
---

# COL-44 — ibs_linac omits charge-squared scaling from ISR energy-spread growth

| | |
|---|---|
| Code | [Bmad](../index.md) · Collective effects |
| Priority | P3 |
| Severity | Wrong physics |
| Evidence | Reproduced, independently rechecked |
| Last reproduced | 2026-11-05 on Bmad release [20261031-0](https://github.com/bmad-sim/bmad-ecosystem/tree/20261031-0) (`c4c629f`) |
| Source | [ibs_linac electron Cq constant](https://github.com/bmad-sim/bmad-ecosystem/blob/c4c629fc9071d3a56cb1ba5bc714baba65e9c5d3/bsim/ibs_linac/ibs_linac.f90#L38-L39)<br>[ISR variance increment](https://github.com/bmad-sim/bmad-ecosystem/blob/c4c629fc9071d3a56cb1ba5bc714baba65e9c5d3/bsim/ibs_linac/ibs_linac.f90#L187-L206)<br>[lin_i3_E7 construction](https://github.com/bmad-sim/bmad-ecosystem/blob/c4c629fc9071d3a56cb1ba5bc714baba65e9c5d3/bmad/code/radiation_integrals.f90#L570-L575)<br>[classical_radius](https://github.com/bmad-sim/bmad-ecosystem/blob/c4c629fc9071d3a56cb1ba5bc714baba65e9c5d3/bmad/code/classical_radius.f90#L14-L25) |

When ibs_linac computes incoherent synchrotron-radiation energy-spread growth for multiply
charged ions, it uses the electron radiation coefficient for every species. The coefficient
must scale with Q², where Q is the charge in elementary-charge units: the inverse-mass factors
in the quantum constant and classical radius cancel the squared rest energy. For carbon ions
with Q = 6, the calculated variance increment is 36 times too small. Unit-charge species retain
the correct charge scaling.

Run `python COL-44.py` with [_bmad.py](../helpers/_bmad.py) in the same directory.

??? example "Reproducer: COL-44.py"

    [Download COL-44.py](COL-44.py)

    ```python
    """COL-44: ibs_linac omits charge-squared scaling from ISR energy-spread growth

    When ibs_linac computes incoherent synchrotron-radiation energy-spread growth for multiply
    charged ions, it uses the electron radiation coefficient for every species. The coefficient
    must scale with Q², where Q is the charge in elementary-charge units: the inverse-mass factors
    in the quantum constant and classical radius cancel the squared rest energy. For carbon ions
    with Q = 6, the calculated variance increment is 36 times too small. Unit-charge species retain
    the correct charge scaling.

    The script runs the production ibs_linac executable for carbon charge states 1 and 6 at equal
    Lorentz factor and bend geometry. IBS and stochastic radiation tracking are disabled, so the
    reported energy spread isolates the ISR variance increment. The fixed random seed initializes
    the same bunch distribution, and the expected variance ratio is Q².
    """

    import re

    from _bmad import compare, run, verdict

    NAMELIST = """&ibs_linac_params
     lat_name='lat.bmad'
     use_beam=.false.
     ibs_on=.false.
     ISR_energy_spread_on=.true.
     radiation_fluctuations_on=.false.
     radiation_damping_on=.false.
     initial_slice_energy_spread_eV=0
     beam_init%n_particle=100
     beam_init%bunch_charge=1e-9
     beam_init%a_norm_emit=1e-6
     beam_init%b_norm_emit=1e-6
     beam_init%sig_z=1e-3
     beam_init%sig_pz=1e-4
    /
    """
    values = []
    for species in ("#12C+1", "#12C+6"):
        lattice = f"""parameter[particle]={species}
    parameter[geometry]=open
    parameter[e_tot]=1000*mass_of({species})
    beginning[beta_a]=10
    beginning[beta_b]=10
    ran_seed=12345
    b: sbend,l=1,g=0.1
    ln: line=(b)
    use,ln
    """
        log = run("ibs_linac", ["run.in"], files={"run.in": NAMELIST, "lat.bmad": lattice}).log
        if "END_DATA" not in log or "problem with" in log:
            raise RuntimeError(log)
        data = log.split("BEGIN_DATA")[1].split("END_DATA")[0]
        rows = [line.split() for line in data.splitlines() if re.match(r"^\s*\d+\.", line)]
        # Bmad appends an END element; it contributes no extra radiation integral.
        if len(rows) != 2 or any(len(row) != 9 for row in rows) or rows[0] != rows[1]:
            raise RuntimeError(log)
        spread = float(rows[0][5])
        if spread <= 0:
            raise RuntimeError("ISR variance control must be positive")
        values.append(spread**2)
        print(f"{species} ISR variance increment, eV^2: {spread**2:.12g}")
    # Equal gamma and geometry give equal I3*gamma^7. Cq*r0*(mc^2)^2 is
    # proportional to Q^2, so the expected variance ratio is exactly 36.
    verdict(
        compare(
            "C6/C1 ISR variance ratio; Cq r0 (mc^2)^2 scales as Q^2",
            values[1] / values[0],
            36.0,
            atol=0,
            rtol=3e-7,
        )
    )

    ```

## Provenance

| Role | Model | Effort | Date | Code examined |
|---|---|---|---|---|
| Found<br><small>source check</small> | `gpt-6.1-sol` | xhigh | 2026-10-22 | ibs-ion-classical-radius (`1bf1982abf8e`) |
| Confirmed<br><small>source check</small> | `claude-opus-5-5` | high | 2026-10-22 | 20261021-0 (`d00a8f17e55a`) |
| Confirmed<br><small>ISR variance charge scaling</small> | `gpt-6-astra` | xhigh | 2026-11-05 | 20261031-0 (`c4c629fc9071`), source, production, GNU Fortran 15.3.0, CPU, double, serial, OpenMP=1 |

