---
title: "OPT-16: NAFF's Gram–Schmidt step is conjugated and unnormalized"
hide:
  - toc
---

# OPT-16 — NAFF's Gram–Schmidt step is conjugated and unnormalized

| | |
|---|---|
| Code | [SciBmad](../index.md) · Optics and maps |
| Priority | P2 |
| Failure | Silent wrong result |
| Evidence | Probe-pinned |
| Last reproduced | 2026-10-31 on FundamentalFrequencies release [v0.1.2](https://github.com/bmad-sim/FundamentalFrequencies.jl/tree/v0.1.2) (`f208dc0`) |
| Source | [`src/FundamentalFrequencies.jl:108-114`](https://github.com/bmad-sim/FundamentalFrequencies.jl/blob/f208dc0724b6c22dc85d9f018c6e8b50bce37591/src/FundamentalFrequencies.jl#L108-L114)<br>[`:111`](https://github.com/bmad-sim/FundamentalFrequencies.jl/blob/f208dc0724b6c22dc85d9f018c6e8b50bce37591/src/FundamentalFrequencies.jl#L111)<br>[`:113`](https://github.com/bmad-sim/FundamentalFrequencies.jl/blob/f208dc0724b6c22dc85d9f018c6e8b50bce37591/src/FundamentalFrequencies.jl#L113)<br>[`:26`](https://github.com/bmad-sim/FundamentalFrequencies.jl/blob/f208dc0724b6c22dc85d9f018c6e8b50bce37591/src/FundamentalFrequencies.jl#L26)<br>[`:60`](https://github.com/bmad-sim/FundamentalFrequencies.jl/blob/f208dc0724b6c22dc85d9f018c6e8b50bce37591/src/FundamentalFrequencies.jl#L60) |

FundamentalFrequencies' `naff` (Numerical Analysis of Fundamental Frequencies) uses a conjugated, unnormalized Gram–Schmidt projection, corrupting later spectral lines and amplitudes. Under the package's own inner product, removing an earlier basis signal Uⱼ from Uᵢ requires the coefficient ⟨Uᵢ,Uⱼ⟩/⟨Uⱼ,Uⱼ⟩, and a corrected projection restores orthogonality. For 513 samples over 512 intervals with frequencies 0.2134, 0.2179 and 0.3171 cycles per sample and amplitudes 1, 0.6 and 0.02, the third reported line rediscovers the first and has zero amplitude. Correcting the projection and amplitude normalization recovers the missing 0.3171 line with amplitude 0.02000. The first two frequency estimates are unchanged by that correction, and changing `tol` has no effect.

Run `julia OPT-16.jl`.

??? example "Reproducer: OPT-16.jl"

    [Download OPT-16.jl](OPT-16.jl)

    ```julia
    # OPT-16: NAFF's Gram–Schmidt step is conjugated and unnormalized
    #
    # FundamentalFrequencies' `naff` (Numerical Analysis of Fundamental Frequencies) uses a conjugated,
    # unnormalized Gram–Schmidt projection, corrupting later spectral lines and amplitudes. Under the
    # package's own inner product, removing an earlier basis signal Uⱼ from Uᵢ requires the coefficient
    # ⟨Uᵢ,Uⱼ⟩/⟨Uⱼ,Uⱼ⟩, and a corrected projection restores orthogonality. For 513 samples over 512
    # intervals with frequencies 0.2134, 0.2179 and 0.3171 cycles per sample and amplitudes 1, 0.6 and
    # 0.02, the third reported line rediscovers the first and has zero amplitude. Correcting the
    # projection and amplitude normalization recovers the missing 0.3171 line with amplitude 0.02000.
    # The first two frequency estimates are unchanged by that correction, and changing `tol` has no
    # effect.
    #
    # The script builds an exact three-tone complex signal of 513 samples, the first two tones 2.3/512
    # cycles per sample apart, and runs `naff` for three lines. It compares the result with the true
    # tones and with a copy of `naff` in which only the projection and amplitude lines are corrected. It
    # also evaluates ⟨U₂,U₁⟩ after the code's and after the correct projection at the returned
    # frequencies, and reruns `naff` with `tol = 0.1` to show that the keyword has no effect.
    using SciBmad
    const FF = SciBmad.FundamentalFrequencies
    const BS = SciBmad.BatchSolve
    # Copy of FF.naff (same window, FFT seed, Brent search, duplicate test) with the projection and
    # amplitude lines corrected.
    function naff_fixed(data, n_frequencies=1; window_order=1)
      n_particles = size(data, 1); turns = size(data, 2)-1
      chi = @. Float64(2^window_order * factorial(window_order)^2 / factorial(2*window_order) * (1 + cos(pi * (-turns:2:turns) / turns))^window_order)
      signal_res = complex(copy(data)); f_resolution = 1/turns
      U = zeros(ComplexF64, n_particles, turns+1, n_frequencies)
      frequencies = zeros(n_particles, n_frequencies); amplitudes = zeros(ComplexF64, n_particles, n_frequencies)
      good = fill(true, n_particles)
      inner_prod(f,g) = sum(@.(f * chi' * conj(g)), dims=2) ./ turns
      for i in 1:n_frequencies
        w = view(signal_res, :, 1:turns) .* view(chi, 1:turns)'
        y = FF.fft(w, 2); __, idx = findmax(abs.(y), dims=2); idx = map(x->x[2], idx)
        sh = @. ifelse(idx - 1 <= div(turns, 2), idx - 1, idx - 1 - turns); guess = sh .* f_resolution
        sol = BS.brent(guess; xa=(guess .- f_resolution/2), xb=(guess .+ f_resolution/2)) do x
          -abs.(inner_prod(signal_res, @.(exp(2*pi*im*(0:turns)'*x))))
        end
        cur = sol.u
        if i != 1
          dmin, __ = findmin(abs.(view(frequencies, :, 1:(i-1)) .- cur), dims=2); @. good = !(dmin < f_resolution)
        end
        frequencies[:, i] .= cur
        U[:,:,i] .= ifelse.(good, exp.(2 .* pi .* im .* (0:turns)' .* cur), 0)
        for j in 1:i-1
          U[:,:,i] .-= ifelse.(good, inner_prod(view(U,:,:,i), view(U,:,:,j)) ./ inner_prod(view(U,:,:,j), view(U,:,:,j)) .* view(U,:,:,j), 0)
        end
        amplitudes[:, i] .= ifelse.(good, inner_prod(signal_res, view(U,:,:,i)) ./ inner_prod(view(U,:,:,i), view(U,:,:,i)), 0)
        signal_res .-= ifelse.(good, view(amplitudes, :, i) .* view(U,:,:,i), 0)
      end
      return frequencies, amplitudes
    end
    N = 512; n = 0:N
    f = [0.2134, 0.2134 + 2.3 / N, 0.3171]; a = [1.0, 0.6cis(0.7), 0.02cis(0.3)]
    s = reshape(sum(a[k] .* cis.(2pi * f[k] .* n) for k in 1:3), 1, :)
    fb, ab = FF.naff(s, 3); fx, ax = naff_fixed(s, 3)
    println("true    f=", f, " |a|=", abs.(a))
    println("package f=", vec(fb), " |a|=", abs.(vec(ab)))
    println("patched f=", vec(fx), " |a|=", abs.(vec(ax)))
    # Package inner product ⟨f,g⟩ = Σ f χ conj(g)/N, linear in f. Removing U_j from U_i needs U_i −=
    # ⟨U_i,U_j⟩/⟨U_j,U_j⟩·U_j, and the residual's coefficient of U_i is ⟨r,U_i⟩/⟨U_i,U_i⟩. The package
    # subtracts ⟨U_j,U_i⟩·U_j and never divides by ⟨U_i,U_i⟩, so later basis vectors are not orthogonal
    # to earlier ones and each subtraction re-injects earlier lines.
    chi = @. 1 + cos(pi * (-N:2:N) / N)
    ip(u, v) = sum(u .* chi .* conj.(v)) / N
    e1 = cis.(2pi * fb[1] .* n); e2 = cis.(2pi * fb[2] .* n)
    Ucode = e2 .- ip(e1, e2) .* e1; Ufix = e2 .- ip(e2, e1) / ip(e1, e1) .* e1
    println("independent: <e2,e1>=", ip(e2, e1), "  <U2,U1> code projection=", ip(Ucode, e1), "  correct projection=", ip(Ufix, e1))
    ft, at = FF.naff(s, 3; tol = 0.1)
    println("independent: naff(...; tol=0.1) identical to default: ", ft == fb && at == ab)
    println(abs(fb[3] - f[3]) > 1e-6 ? "DEFECT REPRODUCED" : "NOT REPRODUCED")

    ```

## Provenance

| Role | Model | Effort | Date | Code examined |
|---|---|---|---|---|
| Found | <span class="bb-nowrap">`gpt-6-astra`</span> | xhigh | <span class="bb-nowrap">2026-10-22</span> | FundamentalFrequencies 0.1.2 (`f208dc0724b6`), release sources, Julia 1.13.2, CPU, Float64 |
| Found | <span class="bb-nowrap">`gpt-6-sol`</span> | xhigh | <span class="bb-nowrap">2026-10-22</span> | FundamentalFrequencies 0.1.2 (`f208dc0724b6`), release sources, Julia 1.13.2, CPU, Float64 |
| Confirmed | <span class="bb-nowrap">`claude-opus-5-5`</span> | high | <span class="bb-nowrap">2026-10-22</span> | FundamentalFrequencies 0.1.2 (`f208dc0724b6`), release sources, Julia 1.13.2, CPU, Float64 |

