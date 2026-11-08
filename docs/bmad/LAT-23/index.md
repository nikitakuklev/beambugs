---
title: "LAT-23: Mixed Gaussian/ellipse renormalization ignores charge weights"
hide:
  - toc
---

# LAT-23 — Mixed Gaussian/ellipse renormalization ignores charge weights

| | |
|---|---|
| Code | [Bmad](../index.md) · Lattices, particles and I/O |
| Priority | P2 |
| Failure | Wrong physics |
| Evidence | Reproduced, independently rechecked |
| Last reproduced | 2026-11-05 on Bmad release [20261031-0](https://github.com/bmad-sim/bmad-ecosystem/tree/20261031-0) (`c4c629f`) |
| Source | [init_random_distribution count-weighted normalization, bmad/multiparticle/beam_utils.f90:600–663](https://github.com/bmad-sim/bmad-ecosystem/blob/c4c629fc9071d3a56cb1ba5bc714baba65e9c5d3/bmad/multiparticle/beam_utils.f90#L600-L663)<br>[centering and overlay, bmad/multiparticle/beam_utils.f90:711–725](https://github.com/bmad-sim/bmad-ecosystem/blob/c4c629fc9071d3a56cb1ba5bc714baba65e9c5d3/bmad/multiparticle/beam_utils.f90#L711-L725)<br>[combine_bunch_distributions coordinate-only overlay, bmad/multiparticle/beam_utils.f90:1116–1126](https://github.com/bmad-sim/bmad-ecosystem/blob/c4c629fc9071d3a56cb1ba5bc714baba65e9c5d3/bmad/multiparticle/beam_utils.f90#L1116-L1126)<br>[ellipse shell weights, bmad/multiparticle/beam_utils.f90:889–908](https://github.com/bmad-sim/bmad-ecosystem/blob/c4c629fc9071d3a56cb1ba5bc714baba65e9c5d3/bmad/multiparticle/beam_utils.f90#L889-L908)<br>[Bmad manual, Beam_Init_Struct Structure: mixed distributions, bmad/doc/beam-init.tex:135–147](https://github.com/bmad-sim/bmad-ecosystem/blob/c4c629fc9071d3a56cb1ba5bc714baba65e9c5d3/bmad/doc/beam-init.tex#L135-L147)<br>[Bmad manual, renormalization flags, bmad/doc/beam-init.tex:241–245](https://github.com/bmad-sim/bmad-ecosystem/blob/c4c629fc9071d3a56cb1ba5bc714baba65e9c5d3/bmad/doc/beam-init.tex#L241-L245)<br>[Bmad manual, Multiparticle Simulation: ellipse distribution, bmad/doc/multiparticle.tex:43–129](https://github.com/bmad-sim/bmad-ecosystem/blob/c4c629fc9071d3a56cb1ba5bc714baba65e9c5d3/bmad/doc/multiparticle.tex#L43-L129) |

Mixing Gaussian coordinates with an ellipse distribution leaves unequal shell charges attached to coordinates normalized with equal particle weights. With both renormalization flags enabled, the charge-weighted centroid and widths therefore miss the requested values. In a 40-particle example, the horizontal variance is 2.916 × 10⁻⁶ m² instead of 3.850 × 10⁻⁶ m², making the RMS width 12.97% too small. An equal-charge shell control recovers the requested moments, and the Gaussian coordinates stay identical as the shell cutoff changes.

Run `python LAT-23.py` with [_bmad.py](../helpers/_bmad.py) in the same directory.

??? example "Reproducer: LAT-23.py"

    [Download LAT-23.py](LAT-23.py)

    ```python
    """LAT-23: Mixed Gaussian/ellipse renormalization ignores charge weights

    Mixing Gaussian coordinates with an ellipse distribution leaves unequal shell charges attached to coordinates normalized with equal particle weights. With both renormalization flags enabled, the charge-weighted centroid and widths therefore miss the requested values. In a 40-particle example, the horizontal variance is 2.916 × 10⁻⁶ m² instead of 3.850 × 10⁻⁶ m², making the RMS width 12.97% too small. An equal-charge shell control recovers the requested moments, and the Gaussian coordinates stay identical as the shell cutoff changes.

    The script initializes the three retained longitudinal-ellipse cases with 40 particles and seed 731. It recomputes moments directly from particle charges and coordinates, checks the equal-charge and equal-count controls, and compares the unequal-charge cases with the requested Twiss covariance.
    """

    import numpy as np
    from _bmad import compare, fortran, verdict

    PROGRAM = """program verify
    use bmad
    use beam_utils
    use random_mod
    implicit none
    type(lat_struct) lat
    type(ele_struct) ele
    type(beam_init_struct) bi
    type(bunch_struct) bunch
    real(rp) q, cutoff, center(6)
    integer iu, nc, ic, id, i, plane, shells, seed, flags, nphi
    logical err

    call bmad_parser('base.bmad', lat)
    ele = lat%ele(0)
    open(newunit=iu, file='initialization.in', status='old')
    read(iu,*) nc
    do ic=1,nc
      read(iu,*) id, plane, shells, nphi, cutoff, seed, flags, q
      bi = beam_init_struct()
      bi%distribution_type = 'RAN_GAUSS'
      bi%ellipse(plane)%n_ellipse = shells
      bi%ellipse(plane)%part_per_ellipse = nphi
      bi%ellipse(plane)%sigma_cutoff = cutoff
      bi%distribution_type(plane) = 'ELLIPSE'
      bi%n_particle = shells*nphi
      bi%a_emit = 7.0e-7_rp
      bi%b_emit = 1.3e-6_rp
      bi%sig_z = 0.006_rp
      bi%sig_pz = 0.0008_rp
      bi%dpz_dz = -0.04_rp
      center = [0.0011_rp, -0.00012_rp, -0.0007_rp, 0.00008_rp, 0.002_rp, 0.0_rp]
      bi%center = center
      bi%bunch_charge = q
      bi%renorm_center = btest(flags,0)
      bi%renorm_sigma = btest(flags,1)
      bi%random_engine = 'pseudo'
      bi%random_gauss_converter = 'exact'
      call ran_seed_put(seed)
      call init_bunch_distribution(ele, lat%param, bi, 0, bunch, err)
      if (err) error stop 'initialization failed'
      write(*,'(a,4i8)') 'INIT ', id, size(bunch%particle), count(bunch%particle%state == alive$), flags
      do i=1,size(bunch%particle)
        write(*,'(a,2i8,7es26.17)') 'I ', id, i, bunch%particle(i)%charge, bunch%particle(i)%vec
      enddo
    enddo
    close(iu)
    end program
    """
    LATTICE = """parameter[particle] = electron
    parameter[p0c] = 8e8
    parameter[geometry] = open
    beginning[beta_a] = 5.5
    beginning[alpha_a] = -0.65
    beginning[beta_b] = 2.7
    beginning[alpha_b] = 0.25
    endmark: marker
    testline: line = (endmark)
    use, testline
    """
    cutoffs = [np.sqrt(8 * np.log(2)), 1.2, 4.6]
    lines = ["3"] + [f"{i} 3 2 20 {cutoff:.17g} 731 3 2.5e-10" for i, cutoff in enumerate(cutoffs, 1)]
    result = fortran(
        PROGRAM,
        files={"base.bmad": LATTICE, "initialization.in": "\n".join(lines) + "\n"},
    )
    center = np.array([0.0011, -0.00012, -0.0007, 0.00008, 0.002, 0])
    # In (x,px), Sigma=emit*[[beta,-alpha],[-alpha,(1+alpha^2)/beta]].
    target = 7e-7 * np.array([[5.5, 0.65], [0.65, (1 + 0.65**2) / 5.5]])
    bad = False
    previous = None
    for case, cutoff in enumerate(cutoffs, 1):
        metadata = [
            [int(v) for v in line.split()[1:]]
            for line in result.log.splitlines()
            if line.startswith("INIT ") and int(line.split()[1]) == case
        ]
        if metadata != [[case, 40, 40, 3]]:
            raise RuntimeError(
                f"Expected 40 live particles with both renormalization flags: {metadata}"
            )
        raw = [
            line.split()
            for line in result.log.splitlines()
            if line.startswith("I ") and int(line.split()[1]) == case
        ]
        data = np.array([[float(v) for v in row[3:]] for row in raw])
        np.testing.assert_array_equal([int(row[2]) for row in raw], np.arange(1, 41))
        assert data.shape == (40, 7) and np.isfinite(data).all()
        weights, xy = data[:, 0], data[:, 1:3]
        assert (weights > 0).all()
        np.testing.assert_allclose(weights.sum(), 2.5e-10, rtol=3e-15)
        weights = weights / weights.sum()
        mu = weights @ xy
        delta = xy - mu
        cov = (delta.T * weights) @ delta
        equal_mu = xy.mean(axis=0)
        equal_delta = xy - equal_mu
        np.testing.assert_allclose(equal_mu, center[:2], rtol=0, atol=4e-18)
        np.testing.assert_allclose(equal_delta.T @ equal_delta / 40, target, rtol=2e-13, atol=0)
        if previous is not None:
            np.testing.assert_array_equal(xy, previous)
        previous = xy
        # Two equal-count shells have outer charge fraction exp(-cutoff^2/8).
        tail = np.exp(-(cutoff**2) / 8)
        np.testing.assert_allclose(
            np.sort(weights) * 20, np.repeat(sorted([tail, 1 - tail]), 20), rtol=3e-14
        )
        mean_bad = compare(f"cutoff={cutoff:.16g} weighted center x (m)", mu[0], center[0], atol=4e-18)
        cov_bad = compare(
            f"cutoff={cutoff:.16g} weighted variance x (m^2), emit*beta",
            cov[0, 0],
            target[0, 0],
            atol=0,
            rtol=2e-13,
        )
        if case == 1 and (mean_bad or cov_bad):
            raise RuntimeError("Equal-charge control failed")
        print(
            f"cutoff={cutoff:.16g}: mean_x_error={mu[0] - center[0]:.16g} m; rms_relative_error={np.sqrt(cov[0, 0] / target[0, 0]) - 1:.16g}"
        )
        bad |= mean_bad or cov_bad
    verdict(bad)

    ```

## Provenance

| Role | Model | Effort | Date | Code examined |
|---|---|---|---|---|
| Found | <span class="bb-nowrap">`gpt-6-astra`</span> | xhigh | <span class="bb-nowrap">2026-10-29</span> | 20261020-0 (`19b8afb46a40`), source-matched 20261021-0 (d00a8f17e55a) libraries, production, CPU, double, serial |
| Confirmed<br><small class="bb-scope">mixed Gaussian/ellipse distributions with unequal shell weights</small> | <span class="bb-nowrap">`gpt-6.1-sol`</span> | xhigh | <span class="bb-nowrap">2026-10-29</span> | 20261020-0 (`19b8afb46a40`), source-matched 20261021-0 (d00a8f17e55a) libraries, production, CPU, double, serial |

