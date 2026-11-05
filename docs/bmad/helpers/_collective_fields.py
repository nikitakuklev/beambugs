"""Parsing and independent cell quadrature for the collective-fields reproducers."""

import re

import numpy as np
from _bmad import run


def rows(log: str) -> dict[str, np.ndarray]:
    """Require unique labelled numerical records and reject native diagnostics."""
    if re.search(r"\[(?:ERROR|FATAL|ABORT|SEVERE)\b|Fortran runtime error|ERROR STOP", log):
        raise RuntimeError(log)
    result = {}
    for line in log.splitlines():
        if not line.startswith("RESULT "):
            continue
        _, label, *values = line.split()
        if label in result or not values:
            raise ValueError(f"Invalid result record: {line}")
        result[label] = np.array([float(v.replace("D", "E")) for v in values])
    if not result:
        raise ValueError(f"No result records: {log}")
    return result


def cell_field(target, center, charge, spacing, order=40):
    """Coulomb field of a uniform cell, by tensor-product Gauss–Legendre integration."""
    nodes, weights = np.polynomial.legendre.leggauss(order)
    xyz = np.stack(np.meshgrid(nodes, nodes, nodes, indexing="ij"), axis=-1)
    weight = np.einsum("i,j,k->ijk", weights, weights, weights) / 8
    displacement = np.asarray(target) - np.asarray(center) - xyz * np.asarray(spacing) / 2
    radius2 = np.sum(displacement**2, axis=-1)
    # The weights average over the cell: charge is the total cell charge, in C.
    return (
        299792458.0**2
        * 1.00000000055e-7
        * charge
        * np.sum(displacement / radius2[..., None] ** 1.5 * weight[..., None], axis=(0, 1, 2))
    )


def tao_covariance(method: str, images: int = 0) -> tuple[np.ndarray, bool]:
    """Fresh deterministic Tao bunch; return covariances and configuration rejection."""
    lattice = f"""no_digested
parameter[geometry] = open
parameter[particle] = electron
parameter[p0c] = 10e6
beginning[beta_a] = 100
beginning[beta_b] = 100
bmad_com[csr_and_space_charge_on] = T
space_charge_com[n_bin] = 20
space_charge_com[n_shield_images] = {images}
space_charge_com[beam_chamber_height] = 0.01
d: drift, l = 0.01, space_charge_method = {method}, csr_ds_step = 0.01
lat: line = (d)
use, lat
"""
    commands = """set beam_init n_particle = 2000
set beam_init random_engine = quasi
set beam_init a_emit = 1e-9
set beam_init b_emit = 1e-9
set beam_init sig_z = 1e-3
set beam_init bunch_charge = 1e-11
set global track_type = beam
pipe bunch_params end
quit
"""
    log = run(
        "tao",
        ["-noplot", "-noinit", "-lat", "lat.bmad", "-startup", "run.tao"],
        files={"lat.bmad": lattice, "run.tao": commands},
    ).log
    if "Tao: quit" not in log:
        raise RuntimeError(log)
    diagnostic = re.search(r"\[(?:ERROR|FATAL|ABORT|SEVERE|WARNING)\b", log)
    # A future rejection must identify this unsupported combination, not an unrelated failure.
    rejected = bool(
        diagnostic
        and method == "cathode_fft_3d"
        and "CATHODE_FFT_3D" in log
        and "TRACKING" in log.upper()
    )
    if diagnostic and not rejected:
        raise RuntimeError(log)
    if rejected:
        return np.array([]), True
    values = {}
    for key in ("sigma_12", "sigma_56", "n_particle_live"):
        matches = re.findall(rf"^{key};[^;]+;[^;]+;\s*([^;\n]+)", log, re.MULTILINE)
        if len(matches) != 1:
            raise ValueError(f"Missing or duplicate {key}: {log}")
        values[key] = float(matches[0])
    if values["n_particle_live"] != 2000:
        raise ValueError(f"Particle-count control failed: {values}")
    return np.array([values["sigma_12"], values["sigma_56"]]), False
