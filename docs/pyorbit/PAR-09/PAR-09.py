"""PAR-09: The MAD-8 reader exits with status 0 on an unknown element type

When `readMAD` meets an element type that is not in its list, it prints "There is no element with type" and ends the process with `sys.exit(0)`, the success status, without building a lattice. A batch job or test that reads a MAD-8 file with a misspelled type, or with a valid MAD-8 type the reader does not know, such as INSTRUMENT or ECOLLIMATOR, stops early but reports success. Reading a MAD-8 line with `q: quadrupol, l=1.0, k1=0.5` exits with status 0, and the statement after `readMAD` never runs. The MAD-X reader exits with status 1 for an unknown type, but with status 0 when a CALL statement names its file without quotes, which MAD-X accepts.

The script runs each case in a child interpreter, because the readers end the process, and prints the child's exit status, the reader's message and whether the statement after the read ran. The cases are MAD-8 lines with the misspelled type and with INSTRUMENT and ECOLLIMATOR, direct `getParameter` calls for a missing parameter on a MAD-8 and a MAD-X element, and a MAD-X file that loads its elements with an unquoted CALL. The expected status of a run that stops on an error is nonzero; the controls are the MAD-X reader on an unknown type, which exits with 1, and the quoted CALL, which reads.
"""

import os

for _v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ.setdefault(_v, "1")
import subprocess
import sys
import tempfile

READ = (
    "import sys\nfrom orbit.teapot import TEAPOT_Lattice\nlat = TEAPOT_Lattice()\n"
    "getattr(lat, sys.argv[2])(sys.argv[1], 'ring')\nprint('read ok', lat.getLength())\n"
)
# getParameter of a parsed element with a parameter it does not have; the class is MAD_LattElement or MADX_LattElement.
GET = (
    "import sys\nfrom orbit import parsers\n"
    "getattr(parsers, sys.argv[2])('q', 'quadrupole').getParameter('k1')\nprint('read ok')\n"
)
SUB = "qa: quadrupole, l=1, k1=0.1;\n"
MADX_SEQ = "ring: sequence, l=10;\nqa, at=2;\nendsequence;\n"
CASES = (  # (label, child code, reader, file text, defect case)
    (
        "readMAD, misspelled type quadrupol",
        READ,
        "readMAD",
        "q: quadrupol, l=1.0, k1=0.5\nring: line=(q)\n",
        True,
    ),
    ("readMAD, INSTRUMENT", READ, "readMAD", "q: instrument, l=1.0\nring: line=(q)\n", False),
    ("readMAD, ECOLLIMATOR", READ, "readMAD", "q: ecollimator, l=1.0\nring: line=(q)\n", False),
    ("MAD_LattElement.getParameter, missing k1", GET, "MAD_LattElement", "", False),
    ("MADX_LattElement.getParameter, missing k1", GET, "MADX_LattElement", "", False),
    ("readMADX, unquoted CALL", READ, "readMADX", "call, file=sub.madx;\n" + MADX_SEQ, False),
    (
        "control, readMADX, misspelled type quadrupol",
        READ,
        "readMADX",
        "q: quadrupol, l=1.0, k1=0.5;\n" + MADX_SEQ.replace("qa", "q"),
        False,
    ),
    (
        "control, readMADX, quoted CALL",
        READ,
        "readMADX",
        "call, file='sub.madx';\n" + MADX_SEQ,
        False,
    ),
)
bad = False
with tempfile.TemporaryDirectory() as d:
    with open(os.path.join(d, "sub.madx"), "w") as f:
        f.write(SUB)
    for tag, code, reader, text, is_main in CASES:
        path = os.path.join(d, "lattice.mad" if reader == "readMAD" else "main.madx")
        with open(path, "w") as f:
            f.write(text)
        r = subprocess.run(
            [sys.executable, "-I", "-c", code, path, reader],
            capture_output=True,
            text=True,
            check=False,
            cwd=d,
        )
        lines = [line.strip() for line in r.stdout.splitlines() if line.strip()]
        msg = next(
            (
                line
                for line in lines
                if "no element" in line
                or "Can not" in line
                or "Wrong line" in line
                or "does not have" in line
            ),
            "",
        )
        ran = any(line.startswith("read ok") for line in lines)
        print(f"[{tag}] exit {r.returncode}, message '{msg}', statement after the read ran: {ran}")
        if is_main:
            bad = r.returncode == 0 and not ran
print("DEFECT REPRODUCED" if bad else "NOT REPRODUCED")
