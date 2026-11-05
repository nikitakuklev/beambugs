"""Synthetic converter distribution and build-derived diagnostic runner for XLT-01–04."""

from pathlib import Path

import numpy as np
from _bmad import _case, _execute, _recipe, _scratch, fortran, source

PROGRAM = "program converter_check\nuse bmad\nuse random_mod\nimplicit none\ntype(lat_struct), target :: lat\ntype(ele_struct), pointer :: e\ntype(coord_struct) :: a,b,d\nreal(rp) :: zexpected\ninteger :: i\nlogical :: err\ncharacter(100) :: mode\ncall get_command_argument(1,mode)\ncall bmad_parser('case.bmad',lat,err_flag=err)\nif(err) error stop 'lattice parsing failed'\n  e=>lat%ele(1)\n  if(trim(mode)=='direct'.or.trim(mode)=='global') bmad_com%spin_tracking_on=.true.\n  call init_coord(a,[0.0_rp,0.0_rp,0.0_rp,0.0_rp,0.023_rp,0.0_rp],e,upstream_end$)\n  a%spin=[0,0,1]\n  d=a\n  call track_a_drift(d,e%value(l$))\n  call ran_seed_put(46731)\n  do i=1,128\n    if(trim(mode)=='direct') then\n      b=a\n      call track_a_converter(b,e,lat%param)\n    else\n      call track1(a,e,lat%param,b)\n    endif\n    if(b%state/=alive$) error stop 'converter lost particle'\n    if(i==1) then\n      print '(a,l2)', 'GLOBAL_SPIN_ENABLED ',bmad_com%spin_tracking_on\n      zexpected=d%vec(5)*b%beta/d%beta\n      print '(a,8es25.16)', 'CONVERTER_CLOCK ',d%beta,b%beta,d%t,b%t,d%vec(5), &\n           b%vec(5),zexpected,b%vec(5)/b%beta-d%vec(5)/d%beta\n      print '(a,3es25.16)', 'CONVERTER_ABSOLUTE_CLOCK ',e%ref_time,-b%beta*c_light*(b%t-e%ref_time), &\n           zexpected+b%beta*c_light*(b%t-e%ref_time)\n      if(abs(zexpected+b%beta*c_light*(b%t-e%ref_time))>2e-17_rp) error stop 'converter absolute clock oracle failed'\n      print '(a,2es25.16)', 'SPIN_TABLE ',minval(e%converter%dist(1)%sub_dist(1)%prob_pc_r%spin_z), &\n           maxval(e%converter%dist(1)%sub_dist(1)%prob_pc_r%spin_z)\n    endif\n    print '(a,i4,3es25.16)', 'CONVERTER_SAMPLE ',i,b%p0c*(1+b%vec(6)),b%charge,b%spin(3)\n  enddo\nprint '(a)', 'COMPLETED'\nend program\n"


def lattice(tail: str = "") -> str:
    """Unit-integral outgoing density, two incoming-momentum tables and fixed spin transfers."""
    header = "parameter[geometry]=open\nbeginning[beta_a]=10\nbeginning[beta_b]=10\n"
    coefficients = {
        "alpha_x": 0,
        "alpha_y": 2,
        "beta": 0,
        "c_x": 0,
        "dxds_min": -0.8,
        "dxds_max": 0.8,
        "dyds_max": 0.6,
    }
    direction = ",\n".join(
        f"{key} = {{"
        + ("C=0, " if key == "dxds_min" else "")
        + f"fit_1d_r={{pc_out=2e5, poly=[{value}]}}, "
        f"fit_1d_r={{pc_out=1e6, poly=[{value}]}}, "
        f"fit_2d_pc={{k=0, poly=[1]}}, fit_2d_r={{k=0, poly=[{value}]}}}}"
        for key, value in coefficients.items()
    )

    def sub(inc, spin):
        prob = ",\n".join(
            f"row={{pc_out={pc},prob=[0,0.0003125,0.000625]}}" for pc in (200000, 600000, 1000000)
        )
        spins = ",\n".join(
            f"row={{pc_out={pc},spin_z=[{spin},{spin}]}}" for pc in (200000, 600000, 1000000)
        )
        return (
            f"sub_distribution={{pc_in={inc},spin_in=[0,0,1],\n"
            f"prob_pc_r={{r_values=[0,0.002,0.004],{prob}}},\n"
            f"spin_z_out={{r_values=[0.002,0.004],{spins}}},\n"
            f"direction_out={{{direction}}}}}"
        )

    base = header + (
        "parameter[particle]=electron\nparameter[p0c]=4e6\n"
        "cv: converter,l=0.0025,p0c=6e5,species_out=positron,distribution={material=tungsten,species_out=positron,thickness=0.0025,\n"
        + sub(3000000, 0.25)
        + ",\n"
        + sub(5000000, 0.75)
        + "}\n"
        "channel: line=(cv)\nuse,channel\n"
    )
    return base + tail


def samples(
    tail: str = "", mode: str = "converter", diagnostic: bool = False, assigned: bool = False
):
    """Return 128 samples, clock records and source-matched diagnostic polarization if requested."""
    if diagnostic:
        code = (source() / "bmad/low_level/track_a_converter.f90").read_text()
        anchor = "out%weight = rsd2 * dist%sub_dist(ix_sd)%prob_pc_r%integrated_prob + rsd * dist%sub_dist(ix_sd+1)%prob_pc_r%integrated_prob"
        assignment = "out%rel_spin_z = rsd2*out1%rel_spin_z + rsd*out2%rel_spin_z"
        if assigned:
            if code.count(anchor) != 1:
                raise RuntimeError("Missing unique incoming-momentum interpolation anchor")
            code = code.replace(anchor, anchor + "\n" + assignment)
        compile_argv, link_argv = _recipe(False)
        with _scratch() as tmp:
            work = Path(tmp)
            (work / "probe.f90").write_text(PROGRAM)
            (work / "converter.f90").write_text(code)
            _execute(compile_argv, work, "", 290)
            diagnostic_argv = [
                x.replace("probe.f90", "converter.f90").replace("probe.o", "converter.o")
                for x in compile_argv
            ]
            diagnostic_argv.extend(["-O0", "-finit-derived", "-finit-real=nan"])
            _execute(diagnostic_argv, work, "", 290)
            link_argv.insert(link_argv.index("probe.o") + 1, "converter.o")
            _execute(link_argv, work, "", 290)
            result = _case(
                [str(work / "probe"), mode], work / "run", {"case.bmad": lattice(tail)}, (), "", 290
            )
    else:
        result = fortran(PROGRAM, [mode], files={"case.bmad": lattice(tail)})
    if "COMPLETED" not in result.log:
        raise RuntimeError(result.log)
    rows = np.array(
        [
            [float(x) for x in line.split()[2:]]
            for line in result.log.splitlines()
            if line.startswith("CONVERTER_SAMPLE")
        ]
    )
    clock = np.array(
        [
            float(x)
            for x in next(
                line for line in result.log.splitlines() if line.startswith("CONVERTER_CLOCK")
            ).split()[1:]
        ]
    )
    absolute = np.array(
        [
            float(x)
            for x in next(
                line
                for line in result.log.splitlines()
                if line.startswith("CONVERTER_ABSOLUTE_CLOCK")
            ).split()[1:]
        ]
    )
    if rows.shape != (128, 3) or not np.isfinite(rows[:, :2]).all():
        raise RuntimeError("Invalid converter sample records")
    if not np.all(rows[:, 1] == rows[0, 1]):
        raise RuntimeError("Identical inputs received different physical weights")
    return rows, clock, absolute
