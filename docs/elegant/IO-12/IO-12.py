"""IO-12: MAD8 and SAD export delete embedded QUAD KQUAD and KSEXT steering

Exporting a QUAD, KQUAD or KSEXT with embedded HKICK or VKICK through elegantto's MAD8 or SAD output retains the host but silently deletes its steering. A 0.6300 m zero-gradient magnet with horizontal and vertical kicks of +0.1700 and −0.2300 mrad exports as an unsteered host, giving zero on-axis deflection. A closed-form constant-force integral agrees with native tracking, while the exported MAD8 model gives zero. Finite-gradient and zero-kick controls retain the host length and focusing, and dedicated MAD8 correctors preserve the requested kicks.

Setup: `elegantto` of the Elegant build on `PATH`; the script exports lattices with it.

The script exports both kick signs and zero-kick controls for three host types to MAD8 and SAD, then executes the MAD8 host maps. It compares native orbits with a closed-form constant-force integral and checks finite-gradient focusing and dedicated MAD8 correctors.
"""

import re

from _slicing import close, native, track

# isort: split
# _slicing sets numerical thread limits before importing the native runner and NumPy.
import numpy as np
from _misalign import export


def serialized_host(text, mode):
    """Read the generated model and require a host-only line before inferring zero steering."""
    if mode == "mad8":
        definitions = re.findall(r"(?im)^HOST:\s*(QUAD|SEXT),([^\n]+)$", text)
        assignments = [
            line for line in text.splitlines() if ":" in line and "LINE" not in line.upper()
        ]
    else:
        definitions = re.findall(r"(?im)^(QUAD|SEXT)\s+HOST\s*=\s*\(([^)]+)\)", text)
        assignments = [line for line in text.splitlines() if not line.upper().startswith("LINE")]
    if len(definitions) != 1 or len(assignments) != 1:
        raise RuntimeError(f"Expected exactly one exported host definition: {text}")
    element, body = definitions[0]
    coefficients = dict(re.findall(r"([A-Z][A-Z0-9]*)\s*=\s*([+\-0-9.eE]+)", body))
    allowed = {"L", "K1" if element == "QUAD" else "K2", "TILT" if mode == "mad8" else "ROTATE"}
    if set(coefficients) != allowed or not all(
        np.isfinite(float(v)) for v in coefficients.values()
    ):
        raise RuntimeError(f"Unexpected host parameters: {text}")
    if len(re.findall(r"\bHOST\b", text, re.IGNORECASE)) != 2:
        raise RuntimeError(f"Expected the host once in the selected line: {text}")
    return element, {k: float(v) for k, v in coefficients.items()}


def main():
    length, hk, vk = 0.63, 0.00017, -0.00023
    root = np.sqrt(1 - hk * hk - vk * vk)
    # p_x(z)=HKICK*z/L, p_y(z)=VKICK*z/L. Integrate x'=p_x/p_s exactly.
    exact = np.array([length * hk / (1 + root), hk / root, length * vk / (1 + root), vk / root])
    paraxial = np.array([length * hk / 2, hk, length * vk / 2, vk])
    defects = []
    for kind in ("QUAD", "KQUAD", "KSEXT"):
        strength = "K2=0" if kind == "KSEXT" else "K1=0"
        integration = ",N_SLICES=256,INTEGRATION_ORDER=6" if kind != "QUAD" else ""
        for scale in (1.0, -1.0, 0.0):
            lattice = (
                f"HOST: {kind},L={length},{strength},HKICK={scale * hk},VKICK={scale * vk}{integration}\n"
                "BL: LINE=(HOST)\n"
            )
            expected = scale * (paraxial if kind == "QUAD" else exact)
            source = track("native host", lattice, np.zeros((1, 6)))["q"][0, :4]
            close(source, expected, 2e-12 if kind == "QUAD" else 1e-16, "Native force integral")
            for mode in ("mad8", "sad"):
                text = export(lattice, mode)
                _, coefficients = serialized_host(text, mode)
                close(coefficients["L"], length, 0, "Exported host length")
                if mode == "mad8":
                    target = track("MAD8 target", text, np.zeros((1, 6)), matrix=True)
                    observed = target["q"][0, :4]
                    close(target["R"][[0, 2], [1, 3]], length, 1e-15, "Host drift matrix")
                else:
                    # Executed serialization contains only L, main strength and roll.
                    # At zero main strength it specifies a drift, whose axial orbit is zero.
                    close(coefficients["K2" if kind == "KSEXT" else "K1"], 0, 0, "SAD strength")
                    observed = np.zeros(4)
                close(observed, np.zeros(4), 0, "Unsteered target")
                defects.append(bool(scale and np.max(np.abs(observed - expected)) > 1e-4))
                print(
                    f"{kind} {mode} kick_scale={scale:g}: native={tuple(map(float, source))} "
                    f"expected={tuple(map(float, expected))} (constant-force integral); "
                    f"target={tuple(map(float, observed))}; host L={coefficients['L']:.16g} expected={length}"
                )
    # x''+K1*x=HKICK/L: solve the forced oscillator at infinitesimal steering.
    k1, hk = 1.3, 1e-7
    k = np.sqrt(k1)
    expected = np.array(
        [hk * (1 - np.cos(k * length)) / (length * k1), hk * np.sin(k * length) / (length * k)]
    )
    lattice = (
        f"HOST: KQUAD,L={length},K1={k1},HKICK={hk},N_SLICES=128,INTEGRATION_ORDER=6\n"
        "BL: LINE=(HOST)\n"
    )
    source = track("forced oscillator", lattice, np.zeros((1, 6)))["q"][0, :2]
    close(source, expected, 1e-18, "Finite-gradient force integral")
    text = export(lattice, "mad8")
    serialized_host(text, "mad8")
    target = track("focused target", text, np.zeros((1, 6)), matrix=True)
    expected_r21 = -k * np.sin(k * length)
    close(target["R"][1, 0], expected_r21, 3e-15, "Retained host focusing")
    close(target["q"][0, :4], 0, 0, "Missing finite-gradient steering")
    print(
        f"finite-gradient KQUAD: native={tuple(map(float, source))} expected={tuple(map(float, expected))} "
        f"(forced oscillator); MAD8 affine={tuple(map(float, target['q'][0, :2]))}; "
        f"R21={target['R'][1, 0]:.16g} expected={expected_r21:.16g} m^-1"
    )
    correctors = "H: HKICK,L=0,KICK=.00017\nV: VKICK,L=0,KICK=-.00023\nBL: LINE=(H,V)\n"
    source = track("native correctors", correctors, np.zeros((1, 6)))["q"][0, :4]
    target = track("MAD8 correctors", export(correctors, "mad8"), np.zeros((1, 6)))["q"][0, :4]
    close(target, source, 0, "Dedicated MAD8 correctors")
    close(target[[1, 3]], [0.00017, -0.00023], 0, "Requested corrector kicks")
    print(
        f"dedicated MAD8 correctors: slopes={tuple(map(float, target[[1, 3]]))} expected=(0.00017, -0.00023)"
    )
    native.verdict(any(defects))


if __name__ == "__main__":
    main()
