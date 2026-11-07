---
title: "LAT-06: Lazy seeding initializes the wrong random state"
hide:
  - toc
---

# LAT-06 — Lazy seeding initializes the wrong random state

| | |
|---|---|
| Code | [Bmad](../index.md) · Lattices, particles and I/O |
| Priority | P4 |
| Severity | Silent wrong result |
| Evidence | Reproduced, independently rechecked |
| Last reproduced | 2026-11-05 on Bmad release [20261031-0](https://github.com/bmad-sim/bmad-ecosystem/tree/20261031-0) (`c4c629f`) |
| Tags | <span class="bb-tag">partly-source-only</span> [?](../../methodology.md#tags) |
| Source | [`ran_uniform_scalar`, sim_utils/math/random_mod.f90:939–970](https://github.com/bmad-sim/bmad-ecosystem/blob/c4c629fc9071d3a56cb1ba5bc714baba65e9c5d3/sim_utils/math/random_mod.f90#L939-L970)<br>[Ziggurat lazy initialization, sim_utils/math/random_mod.f90:250–259](https://github.com/bmad-sim/bmad-ecosystem/blob/c4c629fc9071d3a56cb1ba5bc714baba65e9c5d3/sim_utils/math/random_mod.f90#L250-L259)<br>[exact-Gaussian lazy initialization and rejection loop, sim_utils/math/random_mod.f90:339–379](https://github.com/bmad-sim/bmad-ecosystem/blob/c4c629fc9071d3a56cb1ba5bc714baba65e9c5d3/sim_utils/math/random_mod.f90#L339-L379)<br>[`this_seed_put`, sim_utils/math/random_mod.f90:813–837](https://github.com/bmad-sim/bmad-ecosystem/blob/c4c629fc9071d3a56cb1ba5bc714baba65e9c5d3/sim_utils/math/random_mod.f90#L813-L837)<br>[`pointer_to_ran_state`, sim_utils/math/random_mod.f90:1129–1136](https://github.com/bmad-sim/bmad-ecosystem/blob/c4c629fc9071d3a56cb1ba5bc714baba65e9c5d3/sim_utils/math/random_mod.f90#L1129-L1136) |

Passing a fresh explicit random state to the random-number generator selects that state but seeds the global or thread state instead. With seed 12345, six uniform draws are all zero; subsequent scalar Ziggurat and fresh scalar exact-Gaussian calls fail to return. Lazy initialization must seed the selected state or reject it. Copies obtained through the documented seeding and state APIs produce matching scalar and vector streams.

Run `python LAT-06.py` with [_bmad.py](../helpers/_bmad.py) and [_s3_11_particle_io.py](../helpers/_s3_11_particle_io.py) in the same directory.

??? example "Reproducer: LAT-06.py"

    [Download LAT-06.py](LAT-06.py)

    ```python
    """LAT-06: Lazy seeding initializes the wrong random state

    Passing a fresh explicit random state to the random-number generator selects that state but seeds the global or thread state instead. With seed 12345, six uniform draws are all zero; subsequent scalar Ziggurat and fresh scalar exact-Gaussian calls fail to return. Lazy initialization must seed the selected state or reject it. Copies obtained through the documented seeding and state APIs produce matching scalar and vector streams.

    The script computes the seed-12345 uniform stream independently with integer recurrences and binary32 rounding, and checks initialized explicit-state scalar and vector controls against it. It then draws from a fresh explicit state. Each scalar Gaussian call runs in a separate process with a 290-second watchdog and a flushed marker immediately before the call. A timeout counts only after that marker and successful initialized controls.
    """

    import numpy as np
    from _bmad import compare, fortran, verdict
    from _s3_11_particle_io import execute, program


    def uniform_reference(seed, count):
        # Independent integer recurrence: 32-bit Marsaglia xorshift combined with
        # Park-Miller modulo 2^31-1. Seeding uses the constants in ran_seed_put.
        # The API's real(sp) multiplier and product round to binary32 before widening.
        ix = 777755555 ^ abs(seed)
        iy = (888889999 ^ abs(seed)) | 1
        modulus = 2**31 - 1
        am = np.nextafter(np.float32(1), np.float32(0)) / np.float32(modulus)
        values = []
        for _ in range(count):
            ix ^= (ix << 13) & 0xFFFFFFFF
            ix ^= ix >> 17
            ix ^= (ix << 5) & 0xFFFFFFFF
            iy = (16807 * iy) % modulus
            word = ((ix ^ iy) & modulus) | 1
            values.append(float(np.float32(am * np.float32(word))))
        return np.array(values)


    expected_uniform = uniform_reference(12345, 6)
    control = execute("""
    call ran_seed_put(12345)
    call ran_default_state(get_state=initial)
    rs = initial
    call ran_uniform(v, rs)
    call emit('uniform', v)
    rs = initial
    do i = 1, 6
      call ran_uniform(scalar(i), rs)
    enddo
    call emit('uniform_scalar', scalar)
    do j = 1, 2
      rs = initial
      if (j == 1) rs%gauss_converter = ziggurat$
      if (j == 2) rs%gauss_converter = exact_gaussian$
      call ran_gauss(v, rs)
      rs = initial
      if (j == 1) rs%gauss_converter = ziggurat$
      if (j == 2) rs%gauss_converter = exact_gaussian$
      do i = 1, 6
        call ran_gauss(scalar(i), rs)
      enddo
      write(label, '(a,i0)') 'gauss', j
      call emit(trim(label), v - scalar)
    enddo
    """)
    if not np.all((control["uniform"] > 0) & (control["uniform"] < 1)):
        raise RuntimeError("Initialized uniform stream invalid")
    if compare(
        "initialized uniform control; independent seed-12345 recurrence",
        control["uniform"],
        expected_uniform,
        atol=0,
    ):
        raise RuntimeError("Independent uniform control failed")
    if compare(
        "initialized uniform scalar/vector control",
        control["uniform_scalar"],
        control["uniform"],
        atol=0,
    ):
        raise RuntimeError("Uniform stream control failed")
    for j in (1, 2):
        if compare(
            f"initialized Gaussian {j} scalar/vector residual",
            control[f"gauss{j}"],
            np.zeros(6),
            atol=2e-15,
        ):
            raise RuntimeError("Gaussian stream control failed")
    fresh = execute("""
    rs%seed = 12345
    call ran_uniform(v, rs)
    call emit('uniform', v)
    call emit('am', [real(rs%am,rp)])
    """)
    defect = compare(
        "fresh explicit uniform; independent seed-12345 recurrence",
        fresh["uniform"],
        expected_uniform,
        atol=0,
    )
    print(
        f"fresh explicit state multiplier am={fresh['am'][0]:.17g}; initialized multiplier must be positive"
    )
    for converter in ("ziggurat", "exact_gaussian"):
        prefix = "call ran_uniform(v, rs)" if converter == "ziggurat" else ""
        body = f"""
    rs%seed = 12345
    rs%gauss_converter = {converter}$
    {prefix}
    print *, 'ENTER_GAUSSIAN {converter}'
    flush(6)
    call ran_gauss(x, rs)
    call emit('gaussian', [x])
    """
        try:
            log = fortran(program(body), timeout=290).log
        except TimeoutError as exc:
            if f"ENTER_GAUSSIAN {converter}" not in str(exc):
                raise
            print(f"{converter}: observed=timeout after 290 s expected=one finite draw")
            defect = True
        else:
            from _s3_11_particle_io import rows

            if "COMPLETED" not in log or "gaussian" not in rows(log):
                raise RuntimeError(log)
            print(f"{converter}: observed={rows(log)['gaussian'].tolist()} expected=one finite draw")
    verdict(defect)

    ```

## Provenance

| Role | Model | Effort | Date | Code examined |
|---|---|---|---|---|
| Found | `gpt-6.1-sol` | xhigh | 2026-10-21 | 20261020-0 (`19b8afb46a40`), source, production, CPU, double, serial |
| Confirmed | `gpt-6-astra` | xhigh | 2026-10-21 | 20261020-0 (`19b8afb46a40`), source, production, CPU, double, serial |

