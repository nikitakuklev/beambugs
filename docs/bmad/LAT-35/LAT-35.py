"""LAT-35: Repeated HDF5 suffixes make the beam writer emit ASCII

Calling write_beam_file without file_format writes ASCII when the final .h5 or .hdf5 suffix also occurs earlier in the path. The resulting file has an HDF5 filename but is byte-identical to explicitly selected ASCII. The documented reader selects HDF5 from the final suffix and rejects the file, whereas the same particle data in genuine HDF5 reads successfully under that name. Plain suffixes, mixed .h5/.hdf5 suffixes and explicit HDF5 selection work.

The script writes a deterministic three-electron bunch under plain, repeated and mixed suffixes, including occurrences in directory names. It identifies formats from file bytes, checks round trips against the input coordinates and renames identical payloads to isolate filename dispatch. The shared _bmad.py runner supplies compilation and temporary-file cleanup.
"""

from pathlib import PurePosixPath

from _bmad import fortran, verdict

HDF5_SIGNATURE = b"\x89HDF\r\n\x1a\n"
AUTO_NAMES = (
    "plain.txt",
    "prior.h5.txt",
    "plain.h5",
    "plain.hdf5",
    "repeat.h5.h5",
    "repeat.hdf5.hdf5",
    "mixed.h5.hdf5",
    "mixed.hdf5.h5",
    "archive.h5/beam.h5",
    "archive.hdf5/beam.hdf5",
)
REPEATED_NAMES = (
    "repeat.h5.h5",
    "repeat.hdf5.hdf5",
    "archive.h5/beam.h5",
    "archive.hdf5/beam.hdf5",
)
EXPLICIT_NAMES = ("explicit.h5.h5", "explicit.hdf5.hdf5")

# Nonzero, exactly representable coordinates keep the round-trip check independent
# of random beam generation and reference-momentum shifts. All particles are alive.
SETUP = """
program suffix_io
use bmad
use beam_file_io
use, intrinsic :: ieee_arithmetic, only: ieee_is_finite
implicit none
type(beam_struct) beam
type(ele_struct) ele
integer i
call init_ele(ele, drift$)
ele%value(p0c$) = 8e6_rp
ele%value(p0c_start$) = 8e6_rp
ele%value(e_tot$) = hypot(8e6_rp, m_electron)
ele%value(e_tot_start$) = ele%value(e_tot$)
call reallocate_beam(beam, 1, 3)
do i = 1, 3
  call init_coord(beam%bunch(1)%particle(i), &
       real(i,rp)*[1.0_rp/1024, 1.0_rp/512, -1.0_rp/2048, -1.0_rp/1024, &
                   1.0_rp/256, 1.0_rp/64], ele, downstream_end$, electron$)
  beam%bunch(1)%particle(i)%charge = -1e-12_rp
enddo
beam%bunch(1)%charge_tot = -3e-12_rp
beam%bunch(1)%charge_live = -3e-12_rp
beam%bunch(1)%n_live = 3
"""

READ_SUBROUTINE = """
contains
subroutine check_read(name)
character(*) name
type(beam_struct) restored
type(beam_init_struct) bi
real(rp) delta, charge_delta
integer j
logical err
bi%n_particle = 0
bi%n_bunch = 0
write(*, '(2a)') 'BEGIN ', name
call read_beam_file(name, restored, bi, err, ele)
if (err) then
  write(*, '(2a)') 'REJECT ', name
else
  if (size(restored%bunch) /= 1) error stop 'Wrong bunch count'
  if (size(restored%bunch(1)%particle) /= 3) error stop 'Wrong particle count'
  delta = 0
  charge_delta = 0
  do j = 1, 3
    if (restored%bunch(1)%particle(j)%state /= alive$) error stop 'Wrong particle state'
    if (restored%bunch(1)%particle(j)%species /= electron$) error stop 'Wrong species'
    if (restored%bunch(1)%particle(j)%p0c /= 8e6_rp) error stop 'Wrong reference momentum'
    if (.not. all(ieee_is_finite(restored%bunch(1)%particle(j)%vec))) error stop 'Nonfinite coordinates'
    if (.not. ieee_is_finite(restored%bunch(1)%particle(j)%charge)) error stop 'Nonfinite charge'
    delta = max(delta, maxval(abs(restored%bunch(1)%particle(j)%vec - beam%bunch(1)%particle(j)%vec)))
    charge_delta = max(charge_delta, abs(restored%bunch(1)%particle(j)%charge + 1e-12_rp))
  enddo
  ! ASCII uses ES23.15; 2e-15 also accommodates double precision
  ! HDF5 momentum normalization. The tolerance is far below the coordinate spacing.
  if (delta > 2e-15_rp) error stop 'Wrong particle coordinates'
  if (charge_delta > 2e-27_rp) error stop 'Wrong particle charges'
  write(*, '(2a)') 'ACCEPT ', name
endif
write(*, '(2a)') 'END ', name
end subroutine check_read
end program suffix_io
"""


def read_files(files: dict[str, bytes]) -> dict[str, bool]:
    """Decode each payload in a fresh beam structure; recognize only the HDF5-open error."""
    calls = "".join(f"call check_read('{name}')\n" for name in files)
    log = fortran(SETUP + calls + READ_SUBROUTINE, files=files).log
    accepted: dict[str, bool] = {}
    for name in files:
        start, end = f"BEGIN {name}\n", f"END {name}\n"
        if log.count(start) != 1 or log.count(end) != 1:
            raise RuntimeError(log)
        block = log.split(start, 1)[1].split(end, 1)[0]
        ok, failed = f"ACCEPT {name}\n" in block, f"REJECT {name}\n" in block
        if ok == failed:
            raise RuntimeError(block)
        if failed:
            if (
                "HDF5_OPEN_FILE" not in block.upper()
                or f"CANNOT OPEN FILE FOR READING: {name}" not in block
            ):
                raise RuntimeError(block)
        elif any(marker in block for marker in ("[ERROR", "[FATAL", "[ABORT")):
            raise RuntimeError(block)
        accepted[name] = ok
    return accepted


def main() -> None:
    calls = "".join(f"call write_beam_file('{name}', beam)\n" for name in AUTO_NAMES)
    calls += "call write_beam_file('explicit.txt', beam, file_format=ascii$)\n"
    calls += "".join(
        f"call write_beam_file('{name}', beam, file_format=hdf5$)\n" for name in EXPLICIT_NAMES
    )
    names = (*AUTO_NAMES, "explicit.txt", *EXPLICIT_NAMES)
    result = fortran(
        SETUP + calls + "print *, 'WRITE_COMPLETE'\nend program suffix_io\n",
        files={"archive.h5/.keep": "", "archive.hdf5/.keep": ""},
        outputs=names,
    )
    if "WRITE_COMPLETE" not in result.log or any(
        marker in result.log for marker in ("[ERROR", "[FATAL", "[ABORT")
    ):
        raise RuntimeError(result.log)
    ascii_bytes = result.outputs["explicit.txt"]
    if not ascii_bytes.startswith(b"# ix_bunch"):
        raise RuntimeError("Explicit ASCII control has an unexpected header")

    formats: dict[str, str] = {}
    for name, data in result.outputs.items():
        if data.startswith(HDF5_SIGNATURE):
            formats[name] = "HDF5"
        elif data == ascii_bytes:
            formats[name] = "ASCII"
        else:
            raise RuntimeError(f"Unrecognized format or different ASCII payload: {name}")

    accepted = read_files(result.outputs)
    bad = False
    for name in AUTO_NAMES:
        # The documented reader contract depends only on the final suffix. Use
        # Python's suffix parser, independently of Fortran INDEX and file contents.
        expected = "HDF5" if PurePosixPath(name).suffix in (".h5", ".hdf5") else "ASCII"
        mismatch = formats[name] != expected
        if name not in REPEATED_NAMES and (mismatch or not accepted[name]):
            raise RuntimeError(f"Control failed: {name}")
        if accepted[name] != (formats[name] == expected):
            raise RuntimeError(f"Reader disagrees with the identified format: {name}")
        bad |= mismatch
        print(
            f"auto {name}: format={formats[name]} expected={expected} "
            f"read={'accepted' if accepted[name] else 'rejected'} expected=accepted"
        )
    for name in EXPLICIT_NAMES:
        if formats[name] != "HDF5" or not accepted[name]:
            raise RuntimeError(f"Explicit HDF5 control failed: {name}")
        print(f"explicit HDF5 {name}: format=HDF5 read=accepted; 3 particles match input")
    if not accepted["explicit.txt"]:
        raise RuntimeError("Explicit ASCII control cannot be read")

    # Only the names change: the same valid HDF5 bytes must decode under every
    # repeated-suffix path. Give the actual output its content-matching suffix so
    # this control also passes when the writer's suffix dispatch is corrected.
    renamed = {name: result.outputs["plain.h5"] for name in REPEATED_NAMES}
    for i, name in enumerate(REPEATED_NAMES):
        suffix = ".txt" if formats[name] == "ASCII" else ".h5"
        renamed[f"recovered-{i}{suffix}"] = result.outputs[name]
    if not all(read_files(renamed).values()):
        raise RuntimeError("Renamed-payload controls failed")
    print("renamed HDF5: 4/4 repeated paths accepted; 3 particles match input")
    print("renamed output: 4/4 content-matching suffixes accepted; 3 particles match input")
    print("format oracle: final suffix; ASCII identity: exact bytes; coordinate tolerance: 2e-15")
    verdict(bad)


if __name__ == "__main__":
    main()
