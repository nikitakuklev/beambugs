"""LAT-16: The beam-writer header documents the wrong default format

Calling write_beam_file without a format argument writes an extensionless output as ASCII, although the API header labels HDF5 as the default. The header also calls the format argument logical, while the implementation accepts an integer enumeration. The extensionless output is byte-identical to explicitly selected ASCII, and both default.h5 and default.hdf5 produce HDF5 files. The header should describe filename-dependent selection and the integer argument type.

The script exercises omitted and explicit format arguments on the same deterministic beam and checks the file signatures. BMAD_SOURCE supplies the API header whose type and default claims are compared with those results.
"""

from _bmad import verdict
from _s3_12_particle_contracts import beam_files, document_claim

files = beam_files()
assert files["default_beam"] == files["beam.txt"]
for name in ("default.h5", "default.hdf5"):
    assert files[name].startswith(b"\x89HDF\r\n\x1a\n")
print(
    "omitted format: extensionless=ASCII (byte-identical to explicit ASCII); .h5=HDF5; .hdf5=HDF5"
)
verdict(
    document_claim(
        "bmad/multiparticle/beam_file_io.f90",
        "file_format -- logical, optional: ascii$, or hdf5$ (default).",
    )
)
