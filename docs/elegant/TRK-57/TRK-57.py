"""TRK-57: CSBEND inclined-face quantum diffusion includes field outside the magnet

With inclined physical faces, CSBEND accumulates Gaussian quantum excitation through the whole sector body while correcting classical radiation loss for the face geometry. For a uniform rectangular magnet, translating a parallel incoming ray leaves its field-filled path and diffusion unchanged. A 1.200 m, 0.2400 rad bend instead gives a fractional diffusion derivative of 0.2010 m⁻¹; the variance at +4 mm is 0.1609% above that at −4 mm. An exact circle-and-plane calculation establishes the constant path, and moments and Gaussian tracking reproduce the error. Physical-end-preserving subdivision leaves it unchanged; rectangular CCBEND and centered diffusion controls pass.

The script compares deterministic moments and paired-seed Gaussian tracking with an exact Cartesian circle-and-plane path calculation. It refines and divides a rectangular CSBEND while retaining its physical ends, and checks perpendicular faces, centered diffusion, classical loss and rectangular CCBEND.
"""

from _slicing import close, discrepancy, native

# isort: split
# _slicing sets numerical thread limits before importing the native runner and NumPy.
import numpy as np
from scipy.optimize import brentq

LENGTH, ANGLE, P0, DX = 1.2, 0.24, 800.0, 0.004


def circle_path(offset):
    # Parallel face normals are (-sin(e), cos(e)); the entry displacement
    # (x, x*tan(e)) lies in that plane. Solve the exit intersection separately.
    e, radius = ANGLE / 2, LENGTH / ANGLE
    normal = np.array([-np.sin(e), np.cos(e)])
    entry = np.array([offset, offset * np.tan(e)])
    exit_point = radius * np.array([np.cos(ANGLE) - 1, np.sin(ANGLE)])

    def boundary(phi):
        point = entry + radius * np.array([np.cos(phi) - 1, np.sin(phi)])
        return np.dot(normal, point - exit_point)

    return radius * brentq(boundary, ANGLE / 2, 1.5 * ANGLE, xtol=5e-15)


def lattice(kind, slices, pieces, faces, offset, isr):
    definitions = [f"OFFSET: MALIGN,DX={offset:.17g}"]
    names = ["OFFSET"]
    for i in range(pieces):
        if kind == "CSBEND":
            e1 = ANGLE / 2 if faces and i == 0 else 0
            e2 = ANGLE / 2 if faces and i == pieces - 1 else 0
            options = (
                f",E1={e1:.17g},E2={e2:.17g},EDGE1_EFFECTS={int(i == 0)},"
                f"EDGE2_EFFECTS={int(i == pieces - 1)},EDGE_ORDER=1,EXPAND_HAMILTONIAN=0"
            )
        else:
            options = ",OPTIMIZE_FSE=1,OPTIMIZE_DX=1"
        definitions.append(
            f"B{i}: {kind},L={LENGTH / pieces:.17g},ANGLE={ANGLE / pieces:.17g},"
            f"N_SLICES={slices // pieces},INTEGRATION_ORDER=4,SYNCH_RAD=1,ISR={isr}{options}"
        )
        names.append(f"B{i}")
    definitions.append("END: MARK")
    names.append("END")
    return "\n".join(definitions) + f"\nBL: LINE=({','.join(names)})\n"


def simulate(source, *, count=1, moments=0):
    beam = {"x": np.zeros(count), "p": np.full(count, P0), "particleID": np.arange(1, count + 1)}
    commands = (
        "&global_settings SR_gaussian_limit=10 &end\n"
        f"&run_setup lattice=lattice.lte,use_beamline=BL,p_central={P0},"
        "default_order=1,output=out.sdds,random_number_seed=498173 &end\n"
        "&run_control n_steps=1 &end\n"
    )
    if moments:
        commands += (
            "&moments_output filename=mom.sdds,matched=0,equilibrium=0,"
            f"radiation=1,beta_x=3,beta_y=4,beta_z=1,n_slices={moments} &end\n"
        )
    commands += "&sdds_beam input=beam.sdds &end\n&track soft_failure=0 &end\n"
    result = native.run(
        {"lattice.lte": source, "run.ele": commands},
        beams={"beam.sdds": beam},
        outputs=("out.sdds", "mom.sdds") if moments else ("out.sdds",),
    )
    data = native.particles(result["out.sdds"])
    if not np.array_equal(data["particleID"], beam["particleID"]):
        raise RuntimeError("Particles were lost or particle IDs changed")
    variance = float(result["mom.sdds"][0, "s6"][-1] ** 2) if moments else None
    if variance is not None and (not np.isfinite(variance) or variance <= 0):
        raise RuntimeError("Expected positive finite diffusion")
    return data["p"] / P0, variance


def main():
    paths = [circle_path(x) for x in (-DX, 0, DX)]
    close(paths, [LENGTH] * 3, 3e-15, "Cartesian field paths")
    print(f"circle-plane path (m): observed={paths} expected={LENGTH} for every offset")
    h = ANGLE / LENGTH
    # Sands diffusion at frozen energy: (55/(24*sqrt(3)))*r_e*lambda_bar*p0^5*|h|^3*L.
    variance0 = 55 / (24 * np.sqrt(3)) * 2.8179403205e-15 * 3.8615926744e-13 * P0**5 * h**3 * LENGTH
    faults = []
    for kind, faces, slices, pieces in [
        ("CSBEND", True, 128, 1),
        ("CSBEND", True, 512, 1),
        ("CSBEND", True, 2048, 1),
        ("CSBEND", True, 2048, 8),
        ("CSBEND", False, 2048, 1),
        ("CCBEND", True, 512, 1),
    ]:
        vv = []
        for x in (-DX, 0, DX):
            _, variance = simulate(
                lattice(kind, slices, pieces, faces, x, 1), moments=slices // pieces
            )
            vv.append(variance)
        slope = (vv[2] - vv[0]) / (2 * DX * vv[1])
        expected = 0 if faces else np.sin(ANGLE) / LENGTH
        # Sector path's linear coefficient includes the entrance edge lens.
        sector = (
            np.sin(ANGLE) + (np.tan(ANGLE / 2) if faces else 0) * (1 - np.cos(ANGLE))
        ) / LENGTH
        close(vv[1] / variance0, 1, 1e-6, "Centered Sands diffusion")
        if kind == "CSBEND" and faces:
            faults.append(discrepancy(slope, expected, sector, 1e-6))
        else:
            close(slope, expected, 1e-6, "Physical path control")
        print(
            f"moments {kind} faces={int(faces)} slices={slices} pieces={pieces}: "
            f"fractional dV/dx={slope:.16g} expected={expected:.16g} m^-1 "
            f"(physical path); centered V={vv[1]:.16g} expected={variance0:.16g} (Sands)"
        )
    for kind in ("CSBEND", "CCBEND"):
        vv = []
        for x in (-DX, DX):
            p, _ = simulate(lattice(kind, 256, 1, True, x, 1), count=8192)
            vv.append(float(np.var(p)))
        ratio = vv[1] / vv[0]
        a = 2 * np.tan(ANGLE / 2) / LENGTH
        sector_ratio = (1 + a * DX) / (1 - a * DX)
        if kind == "CSBEND":
            faults.append(discrepancy(ratio, 1, sector_ratio, 1e-6))
        else:
            close(ratio, 1, 1e-14, "CCBEND Gaussian invariance")
        print(
            f"Gaussian {kind} 8192 particles/offset: V(+x)/V(-x)={ratio:.16g} "
            "expected=1 (parallel-face translation, paired seed 498173)"
        )
    for slices in (256, 1024, 4096):
        losses = []
        for x in (-DX, 0, DX):
            p, _ = simulate(lattice("CSBEND", slices, 1, True, x, 0))
            losses.append(float(1 - p[0]))
        slope = (losses[2] - losses[0]) / (2 * DX * losses[1])
        close(slope, 0, 1e-6, "Classical loss translation")
        print(
            f"classical ISR=0 slices={slices}: fractional d(loss)/dx={slope:.16g} "
            "expected=0 m^-1 (parallel-face path)"
        )
    native.verdict(any(faults))


if __name__ == "__main__":
    main()
