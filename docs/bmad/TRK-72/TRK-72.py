"""TRK-72: PTC applies RF cavity fringe kicks without their spin rotation

With tracking_method=symp_lie_ptc and spin tracking, PTC applies the transverse RF kick at lcavity and rfcavity fringes but leaves the spin unchanged across them, and spin_fringe_on has no effect. For a 5.110 MeV positron 1.000 mm off axis, a traveling-wave cavity's entrance fringe changes the transverse momentum Pₓ by −1.957×10⁻⁹ mc per volt of cavity voltage but rotates the spin by zero instead of −1.891×10⁻¹⁰ rad per volt, the rotation that the manual's RF fringe fields and the Thomas–BMT equation give for that kick. Both ends and the traveling-wave, standing-wave and PTC-standard models are affected. Body spin tracking and PTC's solenoid fringe spin continue to act.

Setup: `NPROC`, `PJRT_NPROC`, `OMP_NUM_THREADS`, `MKL_NUM_THREADS`, `OPENBLAS_NUM_THREADS` and `NUMEXPR_NUM_THREADS` all set to 1; the script refuses to run otherwise.

Track a positron at Lorentz factor 10, 1 mm off axis with spin (0,0,1), through one PTC-tracked cavity in fresh native processes, with one fringe enabled or none. The odd response to ±10 V of the difference between those runs isolates each boundary's orbital kick and spin rotation; the expected rotation follows from that kick through the manual's integrated RF fringe fields and the Thomas–BMT equation. Both cavity keys and the traveling-wave, standing-wave and PTC-standard models are covered, plus spin_fringe_on = False; a PTC solenoid's magnetic fringe spin is the passing control.
"""

from __future__ import annotations

import math
import os
from dataclasses import dataclass

from _bmad import compare, fortran, verdict

PROGRAM = r"""
program trk72
use bmad
implicit none
type(lat_struct) :: lat
type(coord_struct) :: start, finish
logical :: err
call bmad_parser('lat.bmad', lat, err_flag=err)
if (err) error stop 'parse failure'
bmad_com%spin_tracking_on = .true.
call init_coord(start, [1.0e-3_rp, 0.0_rp, 0.0_rp, 0.0_rp, 0.0_rp, 0.0_rp], &
                lat%ele(1), upstream_end$)
start%spin = [0.0_rp, 0.0_rp, 1.0_rp]
call track1(start, lat%ele(1), lat%param, finish, err_flag=err)
if (err .or. finish%state /= alive$) error stop 'tracking failure'
write(*,'(a,3es26.17)') 'CONST ', mass_of(start%species), anomalous_moment_of(start%species), &
  hypot(start%p0c, mass_of(start%species))/mass_of(start%species)
write(*,'(a,8es26.17)') 'ORBIT ', finish%p0c, finish%beta, finish%vec
write(*,'(a,3es26.17)') 'SPIN ', finish%spin
end program
"""

GAMMA = 10.0
MASS_EV = 510998.95069
P0C = MASS_EV * math.sqrt(GAMMA**2 - 1)
BETA = math.sqrt(1 - 1 / GAMMA**2)
X0 = 1e-3


@dataclass(frozen=True)
class Track:
    mass: float
    anomaly: float
    orbit: tuple[float, ...]
    spin: tuple[float, ...]

    @property
    def theta_y(self) -> float:
        # Rotation about y takes S=(0,0,1) to (sin θ, 0, cos θ).
        return math.atan2(self.spin[0], self.spin[2])

    @property
    def theta_x(self) -> float:
        # Rotation about x takes S=(0,0,1) to (0, −sin θ, cos θ).
        return math.atan2(-self.spin[1], self.spin[2])

    @property
    def ux(self) -> float:
        # u_x = P_x/(mc); Bmad stores P_x/P0 with the exit reference momentum.
        return self.orbit[3] * self.orbit[0] / self.mass


def track(element: str) -> Track:
    lattice = f"""no_digested
parameter[geometry]=open
parameter[particle]=positron
parameter[p0c]={P0C:.17g}
parameter[no_end_marker]=T
c: {element}
ln: line=(c)
use, ln
"""
    log = fortran(PROGRAM, files={"lat.bmad": lattice}).log
    if any(mark in log for mark in ("[ERROR", "[FATAL", "[ABORT", "[WARNING")):
        raise RuntimeError(log)
    rows: dict[str, tuple[float, ...]] = {}
    for line in log.splitlines():
        fields = line.split()
        if fields and fields[0] in ("CONST", "ORBIT", "SPIN"):
            if fields[0] in rows:
                raise RuntimeError(f"Duplicate output record:\n{log}")
            rows[fields[0]] = tuple(float(value) for value in fields[1:])
    if {key: len(value) for key, value in rows.items()} != {"CONST": 3, "ORBIT": 8, "SPIN": 3} or not all(
        math.isfinite(value) for row in rows.values() for value in row
    ):
        raise RuntimeError(f"Missing or nonfinite output:\n{log}")
    mass, anomaly, gamma = rows["CONST"]
    if not math.isclose(gamma, GAMMA, rel_tol=0, abs_tol=2e-15):
        raise RuntimeError(f"Unexpected entrance Lorentz factor: {gamma}")
    return Track(mass, anomaly, rows["ORBIT"], rows["SPIN"])


def cavity(key: str, model: str, voltage: float, fringe: str, extra: str = "") -> str:
    # The rfcavity waveform is a sine of the phase; phi0 = 0.25 puts its boundary field at a maximum.
    # An explicit ds_step is required for a PTC lcavity at parse time.
    phi0 = 0.25 if key == "rfcavity" else 0.0
    return (
        f"{key}, l=1, voltage={voltage:.17g}, rf_frequency=1.3e9, phi0={phi0}, cavity_type={model}, "
        f"autoscale_amplitude=F, autoscale_phase=F, fringe_at={fringe}, ds_step=0.1, "
        f"tracking_method=symp_lie_ptc, spin_tracking_method=tracking{extra}"
    )


def rf_oracle(du: float, a: float) -> float:
    # Manual RF fringe integrals at the entrance: ∫E_r ds = −r Ê/4, ∫B_φ ds = r Ê/(4c); the exit
    # negates both. Their Lorentz impulse at speed βc is ΔP_x c = −q x Ê (1+β)/(4β), and
    # Thomas–BMT with spin (0,0,1) gives
    #   Θ_y = −q x Ê/(4 mc² β) [1/γ + a + β(a + 1/(1+γ))] = Δu_x [1/γ + a + β(a + 1/(1+γ))]/(1+β).
    # PTC's own impulse sets Δu_x; any physical E/B split of it gives a rotation of this order.
    return du * (1 / GAMMA + a + BETA * (a + 1 / (1 + GAMMA))) / (1 + BETA)


def main() -> None:
    for variable in (
        "NPROC", "PJRT_NPROC", "OMP_NUM_THREADS", "MKL_NUM_THREADS",
        "OPENBLAS_NUM_THREADS", "NUMEXPR_NUM_THREADS",
    ):
        if os.environ.get(variable) != "1":
            raise RuntimeError(f"{variable} must be 1")
    defect = False
    for key, model in (
        ("lcavity", "traveling_wave"), ("lcavity", "standing_wave"), ("lcavity", "ptc_standard"),
        ("rfcavity", "traveling_wave"),
    ):
        runs = {(end, v): track(cavity(key, model, v, end))
                for end in ("entrance_end", "exit_end", "no_end") for v in (10.0, -10.0)}
        for end in ("entrance_end", "exit_end"):
            # The no-fringe run removes the body; the odd part in V removes terms of order V².
            def odd(observable, end: str = end, runs: dict = runs) -> float:
                return ((observable(runs[end, 10.0]) - observable(runs["no_end", 10.0]))
                        - (observable(runs[end, -10.0]) - observable(runs["no_end", -10.0]))) / 2
            du = odd(lambda r: r.ux)
            rotation = odd(lambda r: r.theta_y)
            label = f"PTC {key} {model} {end} at +/-10 V"
            print(f"{label}: orbital fringe kick odd Delta u_x={du:.17g}")
            if abs(du) < 1e-9:
                raise RuntimeError("PTC applied no orbital fringe kick")
            defect |= compare(f"{label} fringe spin rotation, rad; Thomas-BMT of manual fringe fields",
                              rotation, rf_oracle(du, runs[end, 10.0].anomaly), atol=1e-17, rtol=0)
        off = track(cavity(key, model, 10.0, "entrance_end", ", spin_fringe_on=F"))
        on = runs["entrance_end", 10.0]
        print(f"PTC {key} {model} entrance spin_fringe_on=F vs T at 10 V: spin identical={off.spin == on.spin}, "
              f"orbit identical={off.orbit == on.orbit}")

    # Magnetic hard-edge control: a PTC solenoid's entrance fringe, ∫B_r ds = −r B_s/2. With
    # q B_s c = k_s P0 c, Thomas–BMT gives Θ_x = (1 + aγ) k_s x/2.
    ks = 1e-3
    sol = {(end, k): track(f"solenoid, l=1, ks={k}, fringe_at={end}, ds_step=0.1, "
                           "tracking_method=symp_lie_ptc, spin_tracking_method=tracking")
           for end in ("entrance_end", "no_end") for k in (ks, -ks)}
    rotation = ((sol["entrance_end", ks].theta_x - sol["no_end", ks].theta_x)
                - (sol["entrance_end", -ks].theta_x - sol["no_end", -ks].theta_x)) / 2
    expected = (1 + sol["no_end", ks].anomaly * GAMMA) * ks * X0 / 2
    # Composition with the body rotation and the fringe's orbital kick enter at about 1e-6 relative.
    if compare("control PTC solenoid entrance fringe spin at +/-ks, rad; hard-edge Thomas-BMT",
               rotation, expected, atol=0, rtol=5e-6):
        raise RuntimeError("PTC solenoid fringe control failed")
    verdict(defect)


if __name__ == "__main__":
    main()
