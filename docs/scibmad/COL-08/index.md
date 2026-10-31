---
title: "COL-08: Cathode-image field loses Float64 accuracy at large distance/cell ratio"
hide:
  - toc
---

# COL-08 — Cathode-image field loses Float64 accuracy at large distance/cell ratio

| | |
|---|---|
| Code | [SciBmad](../index.md) · Collective effects and spin |
| Priority | P2 |
| Severity | Silent wrong result |
| Evidence | Probe-pinned |
| Last reproduced | 2026-10-31 on SpaceCharge main (`ae30463`) |
| Source | [`SpaceCharge.jl/src/green_functions.jl:35-38`](https://github.com/bmad-sim/SpaceCharge.jl/blob/ae30463305729aec2f04d4687ffd547b94750cb2/src/green_functions.jl#L35-L38)<br>[`41-67`](https://github.com/bmad-sim/SpaceCharge.jl/blob/ae30463305729aec2f04d4687ffd547b94750cb2/src/green_functions.jl#L41-L67)<br>[`103-112`](https://github.com/bmad-sim/SpaceCharge.jl/blob/ae30463305729aec2f04d4687ffd547b94750cb2/src/green_functions.jl#L103-L112)<br>[`SpaceCharge.jl/src/solvers/free_space.jl:49-56`](https://github.com/bmad-sim/SpaceCharge.jl/blob/ae30463305729aec2f04d4687ffd547b94750cb2/src/solvers/free_space.jl#L49-L56)<br>[`green_functions.jl:144-158`](https://github.com/bmad-sim/SpaceCharge.jl/blob/ae30463305729aec2f04d4687ffd547b94750cb2/src/green_functions.jl#L144-L158)<br>[`green_functions.jl:169-175`](https://github.com/bmad-sim/SpaceCharge.jl/blob/ae30463305729aec2f04d4687ffd547b94750cb2/src/green_functions.jl#L169-L175)<br>[`green_functions.jl:54`](https://github.com/bmad-sim/SpaceCharge.jl/blob/ae30463305729aec2f04d4687ffd547b94750cb2/src/green_functions.jl#L54)<br>[`green_functions.jl:61-66`](https://github.com/bmad-sim/SpaceCharge.jl/blob/ae30463305729aec2f04d4687ffd547b94750cb2/src/green_functions.jl#L61-L66)<br>[`free_space.jl:53-56`](https://github.com/bmad-sim/SpaceCharge.jl/blob/ae30463305729aec2f04d4687ffd547b94750cb2/src/solvers/free_space.jl#L53-L56) |

`solve!(at_cathode=true)` loses Float64 accuracy when the cathode image is far away compared with the mesh-cell size, producing spurious transverse fields. An on-axis source has zero transverse field by symmetry, and numerical integration over the charged cell and high-precision kernel evaluation independently give the expected field. With 1.000 nC at a mesh node 1.000 cm from the cathode, 1.000 µm cells and consistent manual bounds on a 32³ mesh, the horizontal field is 133.9 V/m instead of zero, beside an expected axial field of −2.247 × 10⁴ V/m. Errors are negligible for sources within about 100 cell widths of the cathode, but fine grids with distant images suffer cancellation and FFT contamination from unused kernel planes; the shipped example reaches this regime.

??? example "Reproducer: COL-08.jl"

    [Download COL-08.jl](COL-08.jl)

    ```julia
    # COL-08: Cathode-image field loses Float64 accuracy at large distance/cell ratio
    #
    # `solve!(at_cathode=true)` loses Float64 accuracy when the cathode image is far away compared with
    # the mesh-cell size, producing spurious transverse fields. An on-axis source has zero transverse
    # field by symmetry, and numerical integration over the charged cell and high-precision kernel
    # evaluation independently give the expected field. With 1.000 nC at a mesh node 1.000 cm from the
    # cathode, 1.000 µm cells and consistent manual bounds on a 32³ mesh, the horizontal field is 133.9
    # V/m instead of zero, beside an expected axial field of −2.247 × 10⁴ V/m. Errors are negligible for
    # sources within about 100 cell widths of the cathode, but fine grids with distant images suffer
    # cancellation and FFT contamination from unused kernel planes; the shipped example reaches this
    # regime.
    #
    # Part A puts 1 nC at a corner node 1 cm from the cathode on a 32³ mesh with 1 µm cells and
    # consistent manual bounds, and compares the public solver's field with the field of the image cell
    # from 14³-point Gauss–Legendre quadrature; repeating the package's image convolution with only the
    # unused last kernel planes zeroed removes most of the error. Part B spreads 1 nC over a 16³ mesh
    # for three cell sizes and compares the public image field with the same kernel formula evaluated in
    # 128-bit arithmetic. A final check evaluates one far kernel cell in Float64, in 128-bit arithmetic
    # and by direct quadrature.
    using SpaceCharge, LinearAlgebra, Random
    const SC = SpaceCharge

    function boxfield(r, d, Q)                   # field of a uniformly charged box of side d
        nq = 14
        ev = eigen(SymTridiagonal(zeros(nq), [j / sqrt(4j * j - 1) for j in 1:nq-1]))
        x = ev.values; w = 2 .* ev.vectors[1, :] .^ 2
        e = zeros(3)
        for i in 1:nq, j in 1:nq, k in 1:nq
            dvec = r .- [x[i] * d[1], x[j] * d[2], x[k] * d[3]] ./ 2
            e .+= w[i] * w[j] * w[k] / 8 .* dvec ./ norm(dvec)^3
        end
        return SC.FPEI * Q .* e
    end

    # Part A: consistent manual bounds (so the automatic-bounds image shift is absent), 32^3 nodes, 1 um
    # spacing, charge Q at node (1,1,1) = (0,0,z0), z0 = 1 cm. The expected image field at the charge
    # is the Coulomb field of a uniformly charged cell of -Q centred at (0,0,-z0), computed by 14^3
    # Gauss-Legendre quadrature; by symmetry Ex = Ey = 0.
    n = 32; d = 1e-6; z0 = 0.01; Q = 1e-9
    mesh = Mesh3D((n, n, n), (0.0, 0.0, z0), ((n - 1) * d, (n - 1) * d, z0 + (n - 1) * d))
    deposit!(mesh, [0.0], [0.0], [z0], [Q])
    solve!(mesh; at_cathode = true, fft_flags = SC.FFTW.ESTIMATE)
    public = vec(mesh.efield[1, 1, 1, :])
    expected = boxfield([0.0, 0.0, 2z0], collect(mesh.delta), -Q)

    # Same image convolution with the package's own kernel, workspace and FFT plans,
    # once as-is and once with only the three unused last planes set to zero.
    # get_green_function! fills the whole doubled array (2n)^3 with antiderivative values but
    # overwrites only 1:2n-1 per axis with cell integrals. The retained convolution never reads index
    # 2n (rho in 1:n, output in n:2n-1), but the FFT of the whole array does, so these undifferenced
    # values inflate its round-off. The free-space path zeros those planes explicitly.
    w = SC._get_workspace(mesh; fft_flags = SC.FFTW.ESTIMATE)
    offset = (0.0, 0.0, mesh.min_bounds[3] + mesh.max_bounds[3])
    SC._charge_transform!(mesh, w, nothing, true)
    g = zeros(2n, 2n, 2n); raw = zeros(3); clean = zeros(3); padmax = zeros(3); usemax = zeros(3)
    for comp in 1:3
        SC.get_green_function!(g, mesh.delta, mesh.gamma, comp; offset)
        padmax[comp] = max(maximum(abs, g[end, :, :]), maximum(abs, g[:, end, :]), maximum(abs, g[:, :, end]))
        usemax[comp] = maximum(abs, g[1:end-1, 1:end-1, 1:end-1])
        for zero_pad in (false, true)
            if zero_pad
                g[end, :, :] .= 0; g[:, end, :] .= 0; g[:, :, end] .= 0
            end
            mul!(w.product, w.forward, g); w.product .*= w.rhohat
            mul!(w.result, w.backward, w.product)
            (zero_pad ? clean : raw)[comp] = SC.FPEI / length(w.pad) * w.result[n, n, n]
        end
    end
    relerr(v) = norm(v - expected) / norm(expected)
    println("max |kernel| in unused padding planes = ", padmax, "   in used region = ", usemax)
    println("expected (cell quadrature)  E = ", expected)
    println("public solve!(at_cathode)   E = ", public, "  rel. vector error = ", relerr(public))
    println("image FFT, kernel as built  E = ", raw, "  rel. vector error = ", relerr(raw))
    println("image FFT, padding zeroed   E = ", clean, "  rel. vector error = ", relerr(clean))
    partA = relerr(public) > 100 * relerr(clean)

    # ---- Part B ----
    # With charge spread over the mesh, the in-use image kernel entries themselves are dominated by
    # Float64 cancellation in the eight-corner antiderivative difference. The reference is the same
    # kernel formula evaluated in 128-bit arithmetic and rounded once.
    setprecision(128)
    function exact_kernel!(g, delta, comp, offset)     # package formula, 128-bit, rounded once
        N = size(g, 1); dl = big.(delta); o = big.(offset)
        f = 1 / (dl[1] * dl[2] * dl[3])
        c(i, l) = (i - 1) * dl[l] + (1 - N) * dl[l] / 2 + o[l]
        C = [(x = c(i, 1); y = c(j, 2); z = c(k, 3);
              comp == 1 ? SC.field_green_function(x, y, z) : comp == 2 ? SC.field_green_function(y, z, x) :
              SC.field_green_function(z, x, y)) for i in 1:N, j in 1:N, k in 1:N]
        fill!(g, 0)
        for k in 1:N-1, j in 1:N-1, i in 1:N-1
            g[i, j, k] = Float64(f * (C[i+1, j+1, k+1] - C[i, j+1, k+1] - C[i+1, j, k+1] - C[i+1, j+1, k] -
                                      C[i, j, k] + C[i, j, k+1] + C[i, j+1, k] + C[i+1, j, k]))
        end
        return g
    end
    nb = 16; worstB = 0.0
    for d in (1e-6, 1e-5, 1e-4)
        m = Mesh3D((nb, nb, nb), (0.0, 0.0, z0), ((nb - 1) * d, (nb - 1) * d, z0 + (nb - 1) * d))
        Random.seed!(3); m.rho .= Q .* rand(nb, nb, nb) ./ nb^3          # 1 nC spread over the mesh
        solve!(m; fft_flags = SC.FFTW.ESTIMATE); free = copy(m.efield)
        solve!(m; at_cathode = true, fft_flags = SC.FFTW.ESTIMATE); img = m.efield .- free   # public image part
        wk = SC._get_workspace(m; fft_flags = SC.FFTW.ESTIMATE)
        off = (0.0, 0.0, m.min_bounds[3] + m.max_bounds[3])
        SC._charge_transform!(m, wk, nothing, true)
        gb = zeros(2nb, 2nb, 2nb)
        ref = map(1:3) do comp
            exact_kernel!(gb, m.delta, comp, off)
            mul!(wk.product, wk.forward, gb); wk.product .*= wk.rhohat; mul!(wk.result, wk.backward, wk.product)
            SC.FPEI / length(wk.pad) .* wk.result[nb:2nb-1, nb:2nb-1, nb:2nb-1]
        end
        s = maximum(abs, ref[3])
        eB = [maximum(abs, img[:, :, :, c] .- ref[c]) / s for c in 1:3]
        println("B: d = $d m (z0/d = ", round(z0 / d), "): max|Ex|,|Ez| exact = ", maximum(abs, ref[1]), ", ", s,
                "; public image error / max|Ez| (x,y,z) = ", eB)
        d == 1e-6 && (global worstB = eB[1])
    end
    # Independent recheck of part B: one Ex kernel cell centred at (1 um, 0, 2 cm), from the package
    # antiderivative in Float64 and in 128-bit arithmetic, and from direct Gauss-Legendre quadrature
    # of x/r^3 over the cell (no antiderivative).
    cellF(f, c, dd) = (s = zero(c[1]); for p in (-1, 1), q in (-1, 1), r in (-1, 1)
        s += p * q * r * f(c[1] + p * dd / 2, c[2] + q * dd / 2, c[3] + r * dd / 2); end; s / dd^3)
    glcell = let nq = 14, ev = eigen(SymTridiagonal(zeros(nq), [j / sqrt(4j * j - 1) for j in 1:nq-1]))
        xs = ev.values; ws = 2 .* ev.vectors[1, :] .^ 2
        sum(ws[i] * ws[j] * ws[k] / 8 * (1e-6 + xs[i] * 0.5e-6) / norm([1e-6 + xs[i] * 0.5e-6, xs[j] * 0.5e-6, 0.02 + xs[k] * 0.5e-6])^3
            for i in 1:nq, j in 1:nq, k in 1:nq)
    end
    println("recheck B: Ex cell integral Float64 = ", cellF(SC.field_green_function, (1e-6, 0.0, 0.02), 1e-6),
            "   128-bit = ", Float64(cellF(SC.field_green_function, big.((1e-6, 0.0, 0.02)), big(1e-6))),
            "   Gauss-Legendre = ", glcell)
    println(partA && worstB > 1e-2 ? "DEFECT REPRODUCED" : "NOT REPRODUCED")

    ```

## Provenance

| Role | Model | Effort | Date | Code examined |
|---|---|---|---|---|
| Found (unused padding planes of the shifted kernel contaminate the FFT (cause A)) | `gpt-6-astra` | xhigh | 2026-10-22 | SpaceCharge 1.3.0 (`ae3046330572`), source, Julia 1.13.2, CPU, Float64, serial |
| Found (cancellation in the eight-corner difference of the shifted kernel (cause B)) | `gpt-6-sol` | xhigh | 2026-10-22 | SpaceCharge 1.3.0 (`ae3046330572`), source, Julia 1.13.2, CPU, Float64, serial |
| Confirmed | `claude-opus-5-5` | high | 2026-10-22 | SpaceCharge 1.3.0 (`ae3046330572`), source, Julia 1.13.2, CPU, Float64, serial |

