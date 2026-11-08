"""IO-01: `Bunch.addParticlesTo` writes the source particle attributes onto the target's first rows

`Bunch.addParticlesTo(target)` appends the source particles to a target bunch with the same particle
attributes, but writes every source attribute value (per-particle `macrosize`, IDs, initial
coordinates) to target rows 0…N−1, N being the number of source particles, instead of to the
appended rows. When the target already holds particles, its first N particles silently take the
source values and the appended particles keep the zero initialization. Merging a two-particle bunch
with weights 100 and 101 into a three-particle bunch with weights 1, 2 and 3 gives weights 100, 101,
3, 0, 0 instead of 1, 2, 3, 100, 101: the total weight drops from 207 to 204 and sits on the wrong
coordinates. Coordinates are appended correctly; an empty target is unaffected.

The script builds a three-particle target and a two-particle source, both with a `macrosize`
attribute, calls `source.addParticlesTo(target)` and compares each target row's (x, macrosize) with
the concatenation of the target and source rows. The control merges the same source into an empty
target, where the appended rows start at index 0.
"""

from orbit.core.bunch import Bunch


def make(xs, ws):
    b = Bunch()
    b.addPartAttr("macrosize")
    for i, (x, w) in enumerate(zip(xs, ws)):
        b.addParticle(x, 0.0, 0.0, 0.0, 0.0, 0.0)
        b.partAttrValue("macrosize", i, 0, w)
    return b


def rows(b):
    return [(b.x(i), b.partAttrValue("macrosize", i, 0)) for i in range(b.getSize())]


src_x, src_w = [10.0, 11.0], [100.0, 101.0]
dst_x, dst_w = [0.0, 1.0, 2.0], [1.0, 2.0, 3.0]

# Appending keeps the target rows and adds the source rows after them, each particle with its own
# weight: the expected rows are the concatenation of the two particle lists.
dst = make(dst_x, dst_w)
make(src_x, src_w).addParticlesTo(dst)
obs = rows(dst)
exp = list(zip(dst_x + src_x, dst_w + src_w))
print(
    f"nonempty target rows (x, macrosize): observed {obs}; expected {exp} (target rows, then source rows)"
)
print(
    f"nonempty target total macrosize: observed {sum(w for _, w in obs)}; expected {sum(dst_w + src_w)}"
)

# Control: in an empty target the appended rows are rows 0..N-1, so the index mix-up cannot show.
ctl = make([], [])
make(src_x, src_w).addParticlesTo(ctl)
obs_c = rows(ctl)
exp_c = list(zip(src_x, src_w))
print(f"empty target (control): observed {obs_c}; expected {exp_c} (source rows)")

print("DEFECT REPRODUCED" if obs != exp and obs_c == exp_c else "NOT REPRODUCED")
