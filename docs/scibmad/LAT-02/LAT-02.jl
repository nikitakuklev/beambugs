# LAT-02: Relative reference getters return wrong units or repeat the upstream increment
#
# The relative reference getters `dE_ref`, `dpc_ref` and `dp_over_q_ref` should each give the change
# from the upstream Beamline in its own unit, but fail in two cases. In the first Beamline of a
# `Branch`, where the code takes the upstream reference as zero so that relative equals absolute,
# each returns whichever reference is stored; a later Beamline that sets no reference repeats the
# upstream increment instead of 0. For a proton segment storing `E_ref` = 2.000 GeV, `dpc_ref` is
# 2.000×10⁹ instead of 1.766 GeV/c and `dp_over_q_ref` is 2.000×10⁹ instead of 5.892 T·m. Segments
# that keep `pc_ref` at 1.200 GeV/c report `dpc_ref` = 0.2000 GeV/c instead of 0. Code that reads
# energy changes from these getters gets them wrong without warning.
#
# The script builds a `Branch` whose only Beamline is a proton segment storing `E_ref` = 2 GeV and
# compares its relative getters with its absolute `pc_ref` and `p_over_q_ref`, which the
# upstream-zero convention makes equal. It then builds a four-segment Branch (`pc_ref` = 1 GeV/c,
# then `dpc_ref` = +0.2 GeV/c, then a segment with no reference parameters and one that sets only
# `species_ref`) and checks that the last two report `dpc_ref` = 0.
using SciBmad
p = Species("proton")
m = massof(p)
bad = false

# (a) First Branch segment storing E_ref = 2 GeV.
# The code's own first-segment convention is "upstream = 0" (comment at :723/:755, and the
# absolute-from-relative branch :722-723 returns pc_ref = dpc_ref). Hence dpc_ref must equal pc_ref
# and dp_over_q_ref must equal p_over_q_ref, each in its own units.
br = Branch([Marker(species_ref=p, E_ref=2e9)])
b = br.beamlines[1]
println("(a) E_ref = ", b.E_ref, ", pc_ref = ", b.pc_ref, ", p_over_q_ref = ", b.p_over_q_ref)
println("    observed dE_ref = ", b.dE_ref, ", dpc_ref = ", b.dpc_ref, ", dp_over_q_ref = ", b.dp_over_q_ref)
pc_exp = sqrt(2e9^2 - m^2)
println("    expected dE_ref = 2.0e9, dpc_ref = sqrt(E^2-m^2) = ", pc_exp, ", dp_over_q_ref = pc/c = ", pc_exp / C_LIGHT)
bad |= !(isapprox(b.dpc_ref, pc_exp; rtol=1e-14) && isapprox(b.dp_over_q_ref, pc_exp / C_LIGHT; rtol=1e-14))

# (b) Segments: pc = 1e9, then dpc = +2e8, then no reference change (two variants of "unchanged":
# no InitialBeamlineParams at all, and InitialBeamlineParams with only species).
br2 = Branch([Beamline([Marker()], species_ref=p, pc_ref=1e9),
              Beamline([Marker()], dpc_ref=2e8),
              Beamline([Drift(L=1.0)]),
              Beamline([Marker()], species_ref=p)])
for (i, s) in enumerate(br2.beamlines)
    println("(b) segment $i: pc_ref = ", s.pc_ref, ", observed dpc_ref = ", s.dpc_ref,
            i > 1 ? ", expected pc_i - pc_{i-1} = $(s.pc_ref - br2.beamlines[i-1].pc_ref)" : "")
end
bad |= !(br2.beamlines[3].dpc_ref == 0 && br2.beamlines[4].dpc_ref == 0)
println(bad ? "DEFECT REPRODUCED" : "NOT REPRODUCED")
