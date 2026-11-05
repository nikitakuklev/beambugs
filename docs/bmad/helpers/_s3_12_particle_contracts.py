"""Inline particle-contract fixtures for the LAT-13 through LAT-23 reproducers."""

import io
import re

import h5py
import numpy as np
from _bmad import fortran, source

BEAM = """
program beam_io
  use bmad
  use beam_file_io
  implicit none
  type(ele_struct) ele
  type(beam_struct), target :: beam, back
  type(beam_init_struct) init
  type(coord_struct), pointer :: p
  integer i, ib, mode
  logical err
  character(30) arg, fname, label
  call get_command_argument(1, arg)
  read(arg, *) mode
  call init_ele(ele, drift$)
  ele%value(p0c$) = 1e6_rp
  ele%value(p0c_start$) = 1e6_rp
  ele%value(e_tot$) = hypot(1e6_rp, m_electron)
  if (mode == 0) then
    call reallocate_beam(beam, 2, 3)
    do ib = 1, 2
      do i = 1, 3
        p => beam%bunch(ib)%particle(i)
        call init_coord(p, [0.001_rp*i, 0.04_rp*i, -0.002_rp*i, -0.02_rp*i, &
                        0.01_rp*i, 0.1_rp*i], ele, upstream_end$, electron$)
        p%charge = i*1e-12_rp
        p%t = ib*4e-9_rp + i*1e-10_rp
        p%s = 0.7_rp*ib
        p%ix_ele = 10+ib
        p%ix_branch = ib
        p%ix_user = 100*ib+i
        p%spin = [0.3_rp, -0.4_rp, 0.5_rp]
        if (i == 2) p%state = lost$
        if (i == 3) p%direction = -1
      enddo
      beam%bunch(ib)%charge_tot = 6e-12_rp
      beam%bunch(ib)%charge_live = 4e-12_rp
      beam%bunch(ib)%n_live = 2
    enddo
    call write_beam_file('beam.txt', beam, file_format=ascii$)
    call hdf5_write_beam('beam.h5', beam%bunch, .false., err)
    if (err) error stop 'hdf5 write failed'
    call hdf5_write_beam('patches.h5', beam%bunch, .false., err, as_patches=.true.)
    if (err) error stop 'patch write failed'
    call write_beam_file('default_beam', beam)
    call write_beam_file('default.h5', beam)
    call write_beam_file('default.hdf5', beam)
    print *, 'WRITE COMPLETE'
    stop
  endif
  call get_command_argument(2, fname)
  call read_beam_file(trim(fname), back, init, err)
  if (err) error stop 'beam read failed'
  write(*, '(a, 2i8)') 'RESULT constants ', electron$, lost$
  do ib = 1, size(back%bunch)
    write(label, '(a,i0)') 'b', ib
    write(*, '(3a, 3es26.17e3)') 'RESULT ', trim(label), ' ', back%bunch(ib)%charge_tot, &
       back%bunch(ib)%charge_live, real(back%bunch(ib)%n_live,rp)
    do i = 1, size(back%bunch(ib)%particle)
      p => back%bunch(ib)%particle(i)
      write(label, '(a,i0,a,i0)') 'b', ib, 'p', i
      write(*, '(3a, 30es26.17e3)') 'RESULT ', trim(label), ' ', p%vec, p%p0c, &
        p%beta, real(p%t,rp), p%charge, p%s, real(p%direction,rp), real(p%state,rp), &
        real(p%species,rp), real(p%ix_user,rp), real(p%ix_ele,rp), real(p%ix_branch,rp), p%spin
    enddo
  enddo
end program
"""

INIT = """
program initialization
  use bmad
  use beam_utils
  use random_mod
  implicit none
  type(ele_struct) ele
  type(lat_param_struct) param
  type(beam_init_struct) bi, used
  type(bunch_struct) bunch
  logical err
  integer i, iu
  character(32) kind
  call get_command_argument(1, kind)
  call init_ele(ele, drift$)
  ele%value(p0c$) = 1.0e6_rp
  ele%value(p0c_start$) = ele%value(p0c$)
  ele%value(e_tot$) = hypot(ele%value(p0c$), m_electron)
  ele%a%beta = 3.0_rp; ele%a%alpha = 0.7_rp
  ele%b%beta = 5.0_rp; ele%b%alpha = -0.4_rp
  ele%a%eta = 0.02_rp; ele%a%etap = -0.003_rp
  ele%b%eta = -0.01_rp; ele%b%etap = 0.002_rp
  param%particle = electron$
  bi%n_particle = 2048
  bi%a_norm_emit = 2.0e-6_rp
  bi%b_norm_emit = 3.0e-6_rp
  bi%sig_z = 0.004_rp; bi%sig_pz = 0.002_rp; bi%dpz_dz = 0.2_rp
  bi%bunch_charge = 2.3e-12_rp
  bi%center = [0.001_rp, -0.0002_rp, -0.002_rp, 0.0003_rp, 0.003_rp, 0.01_rp]
  bi%random_gauss_converter = 'exact'
  bi%renorm_center = .false. ! renorm_sigma itself centers the generated random sample.
  call ran_seed_put(314159)
  call init_bunch_distribution(ele, param, bi, 0, bunch, err, beam_init_used=used)
  if (err) error stop 'initialization failed'
  open(newunit=iu, file='particles.dat', status='replace')
  do i = 1, size(bunch%particle)
    write(iu, '(9es25.16e3)') bunch%particle(i)%vec, bunch%particle(i)%charge, &
      bunch%particle(i)%beta, bunch%particle(i)%t
  enddo
  close(iu)
  write(*, '(a,5es25.16e3)') 'RESULT init ', used%a_emit, used%b_emit, used%a_norm_emit, &
    used%b_norm_emit, m_electron
end program
"""

TIME = """
program initial_time
  use bmad
  use beam_utils
  implicit none
  type(lat_struct) lat
  type(beam_init_struct) bi
  type(bunch_struct) bunch
  logical err
  call bmad_parser('lat.bmad',lat)
  bi%n_particle = 1
  bi%use_particle_start = .true.
  bi%t_offset = 3.0e-9_rp
  lat%ele(0)%a%beta = 1; lat%ele(0)%b%beta = 1
  call init_bunch_distribution(lat%ele(0),lat%param,bi,0,bunch,err)
  if (err) error stop 'beam initialization failed'
  write(*,'(a,4es25.16e3)') 'RESULT time ',bunch%particle(1)%t,bunch%particle(1)%vec(5),bunch%particle(1)%beta,m_electron
end program
"""


def rows(log: str) -> dict[str, np.ndarray]:
    result = {}
    for line in log.splitlines():
        if line.startswith("RESULT "):
            _, label, *values = line.split()
            if label in result:
                raise RuntimeError(f"Duplicate result {label}")
            result[label] = np.array([float(v) for v in values])
    if not result or not all(np.isfinite(v).all() for v in result.values()):
        raise RuntimeError(f"Missing or nonfinite result: {log}")
    return result


def beam_files() -> dict[str, bytes]:
    names = ("beam.txt", "beam.h5", "patches.h5", "default_beam", "default.h5", "default.hdf5")
    result = fortran(BEAM, ["0"], outputs=names)
    if "WRITE COMPLETE" not in result.log:
        raise RuntimeError(result.log)
    with h5py.File(io.BytesIO(result.outputs["beam.h5"]), "r") as file:
        for ib in (1, 2):
            record = file[f"data/{ib:05}/particles/electron"]
            np.testing.assert_allclose(values(record, "id"), 100 * ib + np.arange(1, 4), rtol=0)
            np.testing.assert_allclose(values(record, "particleStatus"), [1, 2, 1], rtol=0)
            if values(record, "momentum/z")[2] >= 0:
                raise RuntimeError("Writer did not preserve the negative input momentum")
    return result.outputs


def read_beam(data: bytes, name: str = "beam.h5") -> dict[str, np.ndarray]:
    result = fortran(BEAM, ["1", name], files={name: data})
    parsed = rows(result.log)
    expected = {"constants", "b1", "b2"} | {f"b{ib}p{i}" for ib in (1, 2) for i in (1, 2, 3)}
    if parsed.keys() != expected:
        raise RuntimeError(result.log)
    np.testing.assert_array_equal(parsed["constants"], [-1, 2])
    return parsed


def values(record, name: str) -> np.ndarray:
    obj = record[name]
    if isinstance(obj, h5py.Dataset):
        return np.asarray(obj[()])
    if isinstance(obj, h5py.Group):
        return np.full(tuple(obj.attrs["shape"]), obj.attrs["value"])
    raise TypeError(name)


def document_claim(path: str, phrase: str) -> bool:
    """Print whether the cited wording remains in the supplied source tree."""
    text = re.sub(r"\s+", " ", (source() / path).read_text())
    present = phrase in text
    print(f"document {path}: {phrase!r}; present={present}")
    return present
