"""Cylindrical-map fixtures and analytic curl for s3-19."""

import cmath
import math
from typing import TypedDict

import numpy as np

C = 299792458.0


class Mode(TypedDict):
    frequency: float
    index: int
    m: int
    family: str
    coefficient: complex
    dz: float
    theta: float
    phase: float
    terms: int


class SampleGroup(TypedDict):
    ie: int
    point: int
    pos: np.ndarray
    center: int
    stencils: list[list[tuple[float, list[int]]]]


def modes_and_samples():
    modes: list[Mode] = []
    for frequency in (8e7, 7e8):
        for index in (1, 7):
            for m in (1, 2, 4):
                for family in ("b", "e"):
                    modes.append(
                        dict(
                            frequency=frequency,
                            index=index,
                            m=m,
                            family=family,
                            coefficient=complex(2e5, -1.2e5) * 0.01 ** (1 - m),
                            dz=0.11,
                            theta=0.43,
                            phase=0.11,
                            terms=8,
                        )
                    )
                    if family == "b" and m == 2:
                        scaled = modes[-1].copy()
                        scaled["coefficient"] *= -0.37
                        modes.append(scaled)
    lines = [
        "parameter[particle]=electron",
        "parameter[p0c]=8e8",
        "parameter[geometry]=open",
        "beginning[beta_a]=10",
        "beginning[beta_b]=10",
    ]
    for ie, mode in enumerate(modes, 1):
        arrays = {key: [0.0] * 8 for key in ("e_coef_re", "e_coef_im", "b_coef_re", "b_coef_im")}
        arrays[mode["family"] + "_coef_re"][mode["index"]] = mode["coefficient"].real
        arrays[mode["family"] + "_coef_im"][mode["index"]] = mode["coefficient"].imag
        attrs = ", ".join(
            f"{key}=(" + ",".join(map(str, value)) + ")" for key, value in arrays.items()
        )
        lines.append(
            f"f{ie}: em_field, l=.7, field_calc=fieldmap, rf_frequency={mode['frequency']}, "
            f"cylindrical_map={{m={mode['m']}, harmonic=1, dz={mode['dz']}, "
            f"theta0_azimuth={mode['theta']}, phi0_fieldmap={mode['phase']}, "
            f"ele_anchor_pt=beginning, {attrs}}}"
        )
    lines.extend(
        ["lat: line=(" + ",".join(f"f{i}" for i in range(1, len(modes) + 1)) + ")", "use,lat"]
    )
    lattice = "\n".join(lines) + "\n"
    samples = []
    groups: list[SampleGroup] = []
    for ie, mode in enumerate(modes, 1):
        for ip, xyz in enumerate(
            ((0.011, -0.007, 0.19), (-0.008, 0.014, 0.37), (0.017, 0.009, 0.51))
        ):
            pos = np.array([*xyz, (0.173 + ip * 0.079) / mode["frequency"]])
            center = len(samples)
            samples.append((ie, pos.copy()))
            group: SampleGroup = dict(ie=ie, point=ip, pos=pos, center=center, stencils=[])
            for divisor in (1, 2, 4):
                stencils = []
                for coordinate in range(4):
                    h = (2e-5 if coordinate < 3 else 1e-4 / mode["frequency"]) / divisor
                    ids = []
                    for offset in (-2, -1, 1, 2):
                        shifted = pos.copy()
                        shifted[coordinate] += offset * h
                        ids.append(len(samples))
                        samples.append((ie, shifted))
                    stencils.append((h, ids))
                group["stencils"].append(stencils)
            groups.append(group)
    sample_file = "".join(
        f"{ident} {ie} " + " ".join(f"{v:.17g}" for v in pos) + "\n"
        for ident, (ie, pos) in enumerate(samples)
    )
    return modes, samples, groups, {"fields.bmad": lattice, "field_samples.dat": sample_file}


def tilde_bessel(m, kappa_squared, radius):
    # I_m(kappa*r)/kappa**m as an entire series in kappa**2.
    term = complex((radius / 2) ** m / math.factorial(m))
    total = term
    for j in range(1, 80):
        term *= kappa_squared * radius**2 / (4 * j * (m + j))
        total += term
        if abs(term) < 1e-17 * abs(total):
            return total
    raise ArithmeticError("Bessel series did not converge")


def oracle(mode, pos):
    x, y, z, time = pos
    radius, phi = math.hypot(x, y), math.atan2(y, x)
    m, frequency = mode["m"], mode["frequency"]
    fourier_index = mode["index"]
    if 2 * fourier_index >= mode["terms"]:
        fourier_index -= mode["terms"]
    k = 2 * math.pi * fourier_index / (mode["terms"] * mode["dz"])
    omega = 2 * math.pi * frequency
    k2 = k * k - (omega / C) ** 2
    f, g, h = [tilde_bessel(order, k2, radius) for order in (m, m - 1, m + 1)]
    df, dg, dh = g - m * f / radius, k2 * f + (m - 1) * g / radius, f - (m + 1) * h / radius
    e = mode["coefficient"] if mode["family"] == "e" else 0j
    b = mode["coefficient"] if mode["family"] == "b" else 0j
    cosine, sine = math.cos(m * phi - mode["theta"]), math.sin(m * phi - mode["theta"])
    er = -1j * (k * e * h + b * f / radius) * cosine
    ep = -1j * (k * e * h + b * (f / radius - g / m)) * sine
    ez = e * f * cosine
    # Cylindrical curl of the documented E; curl(E)=i*omega*B for exp(-i*omega*t).
    dep_dr = -1j * (k * e * dh + b * (df / radius - f / radius**2 - dg / m)) * sine
    der_dphi = 1j * m * (k * e * h + b * f / radius) * sine
    curl = np.array(
        [
            -m * e * f * sine / radius - 1j * k * ep,
            1j * k * er - e * df * cosine,
            ep / radius + dep_dr - der_dphi / radius,
        ]
    )
    bcyl = curl / (1j * omega)
    phase = cmath.exp(1j * (k * z - omega * time - 2 * math.pi * mode["phase"]))
    cp, sp = math.cos(phi), math.sin(phi)
    rotation = np.array([[cp, -sp, 0], [sp, cp, 0], [0, 0, 1]])
    expected_e = (rotation @ np.array([er, ep, ez]) * phase).real
    expected_b = (rotation @ bcyl * phase).real
    extra_br = (1j * b * k * (m - 1) * f / radius * sine / omega * phase).real
    return expected_e, expected_b, extra_br * np.array([cp, sp, 0])


PROGRAM = r"""program field_check
use bmad
implicit none
type(lat_struct), target :: lat
type(coord_struct) :: orb
type(em_field_struct) :: field
real(rp) :: pos(4), v(6)
integer :: ident, ie
logical :: err
call bmad_parser('fields.bmad', lat, err_flag=err)
if(err) error stop 'parse failed'
open(10, file='field_samples.dat', status='old', action='read')
do
  read(10, *, end=100) ident, ie, pos
  v = 0
  v(1) = pos(1)
  v(3) = pos(2)
  call init_coord(orb, v, lat%ele(ie), upstream_end$)
  call em_field_calc(lat%ele(ie), lat%param, pos(3), orb, .true., field, &
                    err_flag=err, rf_time=pos(4), use_overlap=.false.)
  if (err) error stop 'field error'
  write(*,'(a,2i8,*(es25.16e3,1x))') 'SAMPLE', ident,ie,pos,field%E,field%B
enddo
100 close(10)
print '(a)', 'DONE'
end program
"""
