"""Independent MAD-X parsing, PTC tracking and MAD-NG 4D and 6D tracking for deterministic interchange reproducers.

MAD-X and MAD-NG run in children inheriting CPU affinity and thread limits. Their
scratch files live beside this helper and are removed when the service exits.
"""

from importlib.metadata import version
from pathlib import Path
from tempfile import TemporaryDirectory

import _prebuilt  # noqa: F401 -- Set numerical threads before cpymad imports NumPy.
from cpymad.madx import Madx


def consume(source, sequence="testseq"):
    with (
        TemporaryDirectory(prefix="madx-io-", dir=Path(__file__).resolve().parent) as scratch,
        Madx(stdout=False, cwd=scratch) as mad,
    ):
        try:
            mad.input(source)
            mad.beam(particle="proton", pc=3)
            mad.use(sequence=sequence)
            seq = mad.sequence[sequence]
            rows = [
                (
                    e.name,
                    float(e.at),
                    float(e.l),
                    float(e.k1) if e.base_type.name == "quadrupole" else 0.0,
                )
                for e in seq.elements
                if e.name not in (sequence + "$start", sequence + "$end")
            ]
            result = {
                "status": "accepted",
                "length": float(seq.length),
                "elements": rows,
                "expanded_names": [e.name for e in seq.expanded_elements],
            }
            print(f"CONSUMER cpymad={version('cpymad')} MAD-X={mad.version}")
            return result
        except RuntimeError as exc:
            return {"status": "RuntimeError", "error": str(exc)}


def ptc_track(source, initial, pc, sequence="testseq"):
    """Parse `source`, read back each magnet's (name, L, KNL, KSL) and track one
    ray (x, px, y, py) through PTC (model 2, method 6, 20 steps, 4D, not exact)."""
    x, px, y, py = initial
    with (
        TemporaryDirectory(prefix="madx-io-", dir=Path(__file__).resolve().parent) as scratch,
        Madx(stdout=False, cwd=scratch) as mad,
    ):
        mad.input(source)
        mad.beam(particle="proton", pc=pc)
        mad.use(sequence=sequence)
        magnets = [
            (e.name, float(e.l), [float(v) for v in e.knl], [float(v) for v in e.ksl])
            for e in mad.sequence[sequence].elements
            if e.base_type.name not in ("marker", "drift")
        ]
        mad.input(
            "ptc_create_universe; "
            "ptc_create_layout, model=2, method=6, nst=20, exact=false; "
            f"ptc_start, x={x:.17g}, px={px:.17g}, y={y:.17g}, py={py:.17g}; "
            "ptc_track, icase=4, onetable=true, dump=false; ptc_track_end; ptc_end;"
        )
        table = mad.table.trackone
        end = [float(table.x[-1]), float(table.px[-1]), float(table.y[-1]), float(table.py[-1])]
        print(f"CONSUMER cpymad={version('cpymad')} MAD-X={mad.version} PTC s_end={float(table.s[-1])!r}")
        return magnets, end


def madng_track(line, initial):
    """Load `line` into MAD-NG through the public `Line.to_madng` and track one ray
    (x, px, y, py) with t = pt = 0 (method 6, 20 slices per element). Returns the
    exit (x, px, y, py) and the length of the sequence MAD-NG built."""
    x, px, y, py = initial
    with TemporaryDirectory(prefix="madng-io-", dir=Path(__file__).resolve().parent) as scratch:
        mng = line.to_madng(sequence_name="seq", temp_fname=str(Path(scratch) / "line"),
                            stdout=str(Path(scratch) / "stdout.txt"))
        mng.send(f"""
            local _, mflw = track {{ sequence=seq, method=6, nslice=20, save=false,
                X0={{ x={x:.17g}, px={px:.17g}, y={y:.17g}, py={py:.17g}, t=0, pt=0 }} }}
            py:send(MAD.vector {{ mflw[1].x, mflw[1].px, mflw[1].y, mflw[1].py, seq.l }})
            py:send(MAD.env.version)
        """)
        values = [float(v) for v in mng.recv().ravel()]
        ng_version = mng.recv()
        mng.close()
    print(f"CONSUMER pymadng={version('pymadng')} MAD-NG={ng_version} sequence length={values[4]!r}")
    return values[:4], values[4]


def madng_track_6d(line, initial):
    """Like `madng_track`, for a full ray (x, px, y, py, t, pt). MAD-NG's pt is the
    energy deviation over the reference p0c, so an energy gain is p0c * pt. Returns
    the exit (x, px, y, py, t, pt) and the length of the sequence MAD-NG built."""
    x, px, y, py, t, pt = initial
    with TemporaryDirectory(prefix="madng-io-", dir=Path(__file__).resolve().parent) as scratch:
        mng = line.to_madng(sequence_name="seq", temp_fname=str(Path(scratch) / "line"),
                            stdout=str(Path(scratch) / "stdout.txt"))
        mng.send(f"""
            local _, mflw = track {{ sequence=seq, method=6, nslice=20, save=false,
                X0={{ x={x:.17g}, px={px:.17g}, y={y:.17g}, py={py:.17g}, t={t:.17g}, pt={pt:.17g} }} }}
            py:send(MAD.vector {{ mflw[1].x, mflw[1].px, mflw[1].y, mflw[1].py, mflw[1].t, mflw[1].pt, seq.l }})
            py:send(MAD.env.version)
        """)
        values = [float(v) for v in mng.recv().ravel()]
        ng_version = mng.recv()
        mng.close()
    print(f"CONSUMER pymadng={version('pymadng')} MAD-NG={ng_version} sequence length={values[6]!r}")
    return values[:6], values[6]
