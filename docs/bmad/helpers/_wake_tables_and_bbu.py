"""Inline wake-table cases and independent sums for the wake-tables-and-bbu chunk."""

import re

import numpy as np
from _bmad import compare, run


def rows(log: str) -> dict[str, np.ndarray]:
    """Read unique numeric RESULT records; absence and duplicates are harness failures."""
    result = {}
    for line in log.splitlines():
        if not line.startswith("RESULT "):
            continue
        _, name, *values = line.split()
        if name in result or not values:
            raise RuntimeError(f"Invalid RESULT record: {line}")
        result[name] = np.array([float(x.replace("D", "E")) for x in values])
    if not result:
        raise RuntimeError(f"No RESULT records:\n{log}")
    return result


def require(label, observed, expected, *, atol=0.0, rtol=0.0):
    if compare(label, observed, expected, atol=atol, rtol=rtol):
        raise RuntimeError(f"Control failed: {label}")


def table_lattice():
    z = np.arange(-5, 6) * 0.01
    table = 1e18 * (1 + z / 0.05) ** 2
    text = ", ".join(f"{u:.17g} {w:.17g}" for u, w in zip(z, table, strict=True))
    lattice = (
        "parameter[particle]=electron\nparameter[p0c]=1e12\n"
        "w: lcavity,l=0.6,sr_wake={scale_with_length=F,z_long={w={"
        + text
        + ", }}}\nl: line=(w)\nuse,l\n"
    )
    return lattice, table


def table_kicks(z, charge, table, center):
    """Direct cloud-in-cell source/witness sum, zero outside the supplied wake table."""
    table = table.copy()
    half = len(table) // 2
    table[half] *= 0.5  # Fundamental theorem of beam loading: half the self wake.
    grid = (np.asarray(z) - center) / 0.01 + half
    low = np.floor(grid).astype(int)
    fraction = grid - low
    weights = np.stack([1 - fraction, fraction], axis=1)
    result = np.zeros(len(z))
    for i in range(len(z)):
        for j in range(len(z)):
            for a in (0, 1):
                for b in (0, 1):
                    index = low[i] + a - low[j] - b + half
                    if 0 <= index < len(table):
                        result[i] -= weights[i, a] * weights[j, b] * charge[j] * table[index] / 1e12
    return result


def bbu_control(init="control.init", label="BBU control HOM voltage gain"):
    result = run("bbu", [init], files=BBU_FILES, outputs=["for_py.txt"])
    text = result.outputs["for_py.txt"].decode()
    match = re.search(r"(?im)^\s*v_gain\s*=\s*([\d.eEdD+\-]+)", text)
    if match is None:
        raise RuntimeError(f"Missing BBU completion record: {text}\n{result.log}")
    if "lostbool = False" not in text:
        raise RuntimeError(f"BBU control lost a particle: {text}\n{result.log}")
    require(label, float(match[1].replace("D", "E")), 1.0, atol=2e-15)
    return result


TABLE_WAKE_SOURCE = r"""
program table_wake
  use bmad
  use wake_mod
  implicit none
  type(lat_struct) lat
  type(bunch_struct) bunch
  integer i
  logical err
  call bmad_parser('lat.bmad', lat, err_flag=err)
  if (err) error stop 'Parser failed'
  call reallocate_bunch(bunch, 3)
  do i = 1, 3
    call init_coord(bunch%particle(i), [0.0_rp,0.0_rp,0.0_rp,0.0_rp,(i-2)*0.01_rp,0.0_rp], &
      lat%ele(1), upstream_end$, electron$)
    bunch%particle(i)%charge = 0
  enddo
  bunch%particle(2)%charge = 2.0e-12_rp
  call sr_z_long_wake(lat%ele(1), bunch, 0.0_rp)
  write(*, '(a,3es25.16e3)') 'RESULT table ', bunch%particle%vec(6)
  write(*, '(a,9es25.16e3)') 'RESULT raw ', lat%ele(1)%wake%sr%z_long%w
end program
"""


WAKE_SUPPORT_SOURCE = r"""
program wake_support
  use bmad
  use wake_mod
  use output_mod, only: output_direct
  implicit none
  type(lat_struct) lat
  type(bunch_struct) bunch
  logical err
  integer i,j,which,perm(3),n
  real(rp) physical_z(3), physical_kick(3)
  character(16) arg,name
  call get_command_argument(1,arg); read(arg,*) which
  call bmad_parser('lat.bmad',lat,err_flag=err)
  if (err) error stop 'Parser failed'
  bmad_com%sr_wakes_on = .true.
  select case(which)
  case(1)
    physical_z = [0.013_rp,0.002_rp,-0.014_rp]
    do j=0,2
      call prepare(3)
      perm=[1,2,3]
      if (j==2) perm=[1,3,2]
      do i=1,3
        bunch%particle(i)%vec(5)=physical_z(perm(i))
        if (perm(i)==1) bunch%particle(i)%charge=2.0e-12_rp
      enddo
      call order_particles_in_z(bunch)
      if (j==0) then
        call sr_z_long_wake(lat%ele(1),bunch,-0.0005_rp)
      else
        call track1_sr_wake(bunch,lat%ele(1))
      endif
      do i=1,3
        physical_kick(perm(i))=bunch%particle(i)%vec(6)
      enddo
      write(name,'(a,i0)') 'permutation',j
      write(*,'(3a,3es25.16e3)') 'RESULT ',trim(name),' ',physical_kick
    enddo
  case(2)
    call prepare(2)
    bunch%particle(1)%vec(5)=0.09_rp
    bunch%particle(2)%charge=2.0e-12_rp
    open(unit=51,file='diagnostic.log',status='replace')
    call output_direct(file_unit=51,print_and_capture=.false.,min_level=s_error$,max_level=s_error$)
    call sr_z_long_wake(lat%ele(1),bunch,0.0_rp)
    close(51)
    write(*,'(a,2i4)') 'RESULT lost ',(merge(1,0,bunch%particle(i)%state==lost$),i=1,2)
    write(*,'(a,2es25.16e3)') 'RESULT kicks ',bunch%particle%vec(6)
  case(3)
    call prepare(101)
    bunch%particle(1)%charge=2.0e-12_rp
    bunch%particle(101)%vec(5)=-0.065_rp
    call sr_z_long_wake(lat%ele(1),bunch,0.0_rp)
    write(*,'(a,100es25.16e3)') 'RESULT core ',bunch%particle(1:100)%vec(6)
    write(*,'(a,es25.16e3)') 'RESULT outlier ',bunch%particle(101)%vec(6)
    write(*,'(a,i4)') 'RESULT live ',count(bunch%particle%state==alive$)
  end select
contains
  subroutine prepare(count_particles)
    integer count_particles,k
    call reallocate_bunch(bunch,count_particles)
    do k=1,count_particles
      call init_coord(bunch%particle(k),[0.0_rp,0.0_rp,0.0_rp,0.0_rp,0.0_rp,0.0_rp], &
        lat%ele(1),upstream_end$,electron$)
      bunch%particle(k)%charge=0
    enddo
  end subroutine
end program
"""

BBU_FILES = {
    "cutoff.bmad": "no_digested\nparameter[particle]=electron\nparameter[p0c]=1e9\nparameter[geometry]=open\nbeginning[beta_a]=10\nbeginning[beta_b]=10\nw: lcavity, l=1, rf_frequency=1.3e9, lr_wake={ &\n  mode={1e9,100,0,0,1,unpolarized}, mode={2e9,100,0,0,2,unpolarized}}\nln: line=(w)\nuse,ln\n",
    "cutoff_all.bmad": "no_digested\nparameter[particle]=electron\nparameter[p0c]=1e9\nparameter[geometry]=open\nbeginning[beta_a]=10\nbeginning[beta_b]=10\nw1: lcavity, l=1, rf_frequency=1.3e9, lr_wake={mode={1e9,100,0,0,1,unpolarized}}\nw2: lcavity, l=1, rf_frequency=1.3e9, lr_wake={mode={2e9,100,0,0,2,unpolarized}}\nln: line=(w1,w2)\nuse,ln\n",
    "control.init": "&bbu_params\nbbu_param%lat_filename='cutoff.bmad'\nbbu_param%hom_order_cutoff=2\nbbu_param%hybridize=False\nbbu_param%simulation_turns_max=12\nbbu_param%ran_seed=123\n/\n",
    "ramp.init": "&bbu_params\nbbu_param%lat_filename='cutoff.bmad'\nbbu_param%ramp_on=True\nbbu_param%ramp_pattern=1,2,3\nbbu_param%hybridize=False\nbbu_param%simulation_turns_max=12\nbbu_param%ran_seed=123\n/\n",
    "cutoff.init": "&bbu_params\nbbu_param%lat_filename='cutoff.bmad'\nbbu_param%hom_order_cutoff=1\nbbu_param%hybridize=False\nbbu_param%simulation_turns_max=12\nbbu_param%ran_seed=123\n/\n",
    "cutoff_all.init": "&bbu_params\nbbu_param%lat_filename='cutoff_all.bmad'\nbbu_param%hom_order_cutoff=1\nbbu_param%hybridize=False\nbbu_param%simulation_turns_max=12\nbbu_param%ran_seed=123\n/\n",
}

# Match the two-cavity complete-removal case while retaining every mode.
BBU_FILES["control_all.init"] = BBU_FILES["control.init"].replace("cutoff.bmad", "cutoff_all.bmad")
