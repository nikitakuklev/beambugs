"""XLT-12: Ordinary MAD-X aperture import unpacks the wrong array slice

Importing a MAD-X ellipse or rectangle with two half-apertures crashes the translator. It always selects the third and fourth aperture values, leaving an empty slice for ordinary two-value shapes. An ellipse with half-apertures 0.01 m and 0.02 m raises ValueError instead of producing those Bmad limits. MAD-X accepts the input, and Bmad accepts the corresponding explicit collimator.

The script invokes the importer from BMAD_SOURCE on ellipse and rectangle apertures accepted by MAD-X. It requires the specific unpacking error and checks equivalent Bmad collimators as controls.
"""

from _bmad import verdict
from _lattice_import_export import HEADER, inspect, madx, translate, valid

defect = False
for shape, kind in [("ellipse", "ecollimator"), ("rectangle", "rcollimator")]:
    lattice = f"beam,particle=electron,pc=3; c:collimator,l=1,apertype={shape},aperture={{.01,.02}}; ll:line=(c); use,period=ll;"
    with madx() as mad:
        mad.input(lattice)
        assert list(mad.elements.c.aperture) == [0.01, 0.02]
    valid(inspect(HEADER + f"c:{kind},l=1,x_limit=.01,y_limit=.02\nll:line=(c)\nuse,ll\n"))
    status, log, output = translate(lattice)
    bad = status == 1 and "ValueError: not enough values to unpack (expected 2, got 0)" in log
    if status and not bad:
        raise RuntimeError(log)
    if not bad:
        valid(inspect(output))
    print(
        f"{shape}: unpacking ValueError={bad}; expected limits=[0.01,0.02] m; MAD-X and Bmad controls accepted"
    )
    defect |= bad
verdict(defect)
