"""TRK-70: Stepped native lcavities ignore spin_fringe_on

An lcavity tracked with tracking_method=bmad_standard and positive n_rf_steps (default 10) ignores spin_fringe_on = False: the RF fringe still rotates the spin, and the result is bit-for-bit identical to the default True setting, although the manual says the switch turns off the fringe fields' effect on spin. For an electron at γ = 10 entering a 1.3 GHz cavity 1.000 mm off axis, each fringe still rotates the spin by 9.766×10⁻¹¹ rad per volt of cavity voltage, equal to the rotation implied by its own transverse kick, instead of zero. Runge–Kutta tracking and n_rf_steps = 0 honor the switch.

Setup: `NPROC`, `PJRT_NPROC`, `OMP_NUM_THREADS`, `MKL_NUM_THREADS`, `OPENBLAS_NUM_THREADS` and `NUMEXPR_NUM_THREADS` all set to 1; the script refuses to run otherwise.

Track an electron at Lorentz factor 10 through a one-metre cavity with each RF fringe enabled separately, using both wave types and the parser defaults. Antisymmetrize the spin response to ±10 V to remove the leading body term, and compare it with the Thomas–BMT angle from the transverse fringe impulse and with zero when spin_fringe_on is False. Check complete output spin and orbit bits against the default, then exercise Runge–Kutta and n_rf_steps = 0 switch controls.
"""

from __future__ import annotations

import math
import os
import struct
from dataclasses import dataclass

from _bmad import compare, fortran, verdict

PROGRAM = r"""
program fringe_switch
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
write(*,'(a,3es26.17)') 'CONST ', mass_of(start%species), &
  anomalous_moment_of(start%species), hypot(start%p0c, mass_of(start%species))/mass_of(start%species)
write(*,'(a,10es26.17)') 'ORBIT ', finish%p0c, finish%beta, finish%vec, finish%t, finish%s
write(*,'(a,3es26.17)') 'SPIN ', finish%spin
write(*,'(a,3es26.17)') 'SETUP ', lat%ele(1)%value(l_active$), &
  lat%ele(1)%value(n_rf_steps$), lat%ele(1)%value(spin_fringe_on$)
end program
"""

GAMMA = 10.0
MASS_EV = 510998.95069
P0C = MASS_EV * math.sqrt(GAMMA**2 - 1)


@dataclass(frozen=True)
class Track:
    mass: float
    anomaly: float
    gamma: float
    orbit: tuple[float, ...]
    spin: tuple[float, ...]

    @property
    def angle(self) -> float:
        return math.atan2(self.spin[0], self.spin[2])

    @property
    def px(self) -> float:
        # Bmad stores Px/P0; each endpoint carries its own reference momentum.
        return self.orbit[0] * self.orbit[3]

    @property
    def bits(self) -> bytes:
        # Decimal output has 18 significant digits, enough to recover binary64 bits.
        values = (*self.orbit, *self.spin)
        return struct.pack(f"!{len(values)}d", *values)


def track(
    voltage: float,
    wave: str,
    fringe: str,
    switch: bool | None,
    *,
    method: str = "bmad_standard",
    steps: int | None = None,
) -> Track:
    attributes = [
        "lcavity", "l=1", f"voltage={voltage:.17g}", "rf_frequency=1.3e9", "phi0=0",
        f"cavity_type={wave}", "autoscale_amplitude=F", "autoscale_phase=F",
        f"fringe_at={fringe}", f"tracking_method={method}", "spin_tracking_method=tracking",
    ]
    # Omit both attributes in the baseline to exercise the actual parser defaults.
    if switch is not None:
        attributes.append(f"spin_fringe_on={'T' if switch else 'F'}")
    if steps is not None:
        attributes.append(f"n_rf_steps={steps}")
    lattice = f"""no_digested
parameter[geometry]=open
parameter[particle]=electron
parameter[p0c]={P0C:.17g}
parameter[no_end_marker]=T
c: {', '.join(attributes)}
ln: line=(c)
use, ln
"""
    log = fortran(PROGRAM, files={"lat.bmad": lattice}).log
    if any(mark in log for mark in ("[ERROR", "[FATAL", "[ABORT", "[WARNING")):
        raise RuntimeError(log)
    rows: dict[str, tuple[float, ...]] = {}
    for line in log.splitlines():
        fields = line.split()
        if fields and fields[0] in ("CONST", "ORBIT", "SPIN", "SETUP"):
            if fields[0] in rows:
                raise RuntimeError(f"Duplicate output record:\n{log}")
            rows[fields[0]] = tuple(float(value) for value in fields[1:])
    if {key: len(value) for key, value in rows.items()} != {
        "CONST": 3, "ORBIT": 10, "SPIN": 3, "SETUP": 3,
    } or not all(math.isfinite(value) for row in rows.values() for value in row):
        raise RuntimeError(f"Missing or nonfinite output:\n{log}")
    active, n_steps, sfo = rows["SETUP"]
    if active <= 0 or n_steps != (10 if steps is None else steps) or sfo != (switch is not False):
        raise RuntimeError(f"Element setup differs from request:\n{log}")
    mass, anomaly, gamma = rows["CONST"]
    if not math.isclose(gamma, GAMMA, rel_tol=0, abs_tol=2e-15):
        raise RuntimeError(f"Unexpected entrance Lorentz factor: {gamma}")
    return Track(mass, anomaly, gamma, rows["ORBIT"], rows["SPIN"])


def pair(
    wave: str,
    fringe: str,
    switch: bool | None,
    *,
    method: str = "bmad_standard",
    steps: int | None = None,
) -> tuple[float, float, tuple[Track, Track]]:
    plus = track(10.0, wave, fringe, switch, method=method, steps=steps)
    minus = track(-10.0, wave, fringe, switch, method=method, steps=steps)
    odd = (plus.angle - minus.angle) / 2
    # At x=1 mm, Px=0, a weak RF fringe creates Px proportional to V.
    # Thomas–BMT for its transverse electric impulse gives
    # theta_odd = (a + 1/(1+gamma)) * (Delta Px_odd)/(mc).
    # Body spin starts at V^2 (the fringe creates Px and RF changes energy),
    # so antisymmetrization removes it; the residual starts at V^3.
    kick_angle = (plus.anomaly + 1 / (1 + plus.gamma)) * (plus.px - minus.px) / (2 * plus.mass)
    return odd, kick_angle, (plus, minus)


def main() -> None:
    for variable in (
        "NPROC", "PJRT_NPROC", "OMP_NUM_THREADS", "MKL_NUM_THREADS",
        "OPENBLAS_NUM_THREADS", "NUMEXPR_NUM_THREADS",
    ):
        if os.environ.get(variable) != "1":
            raise RuntimeError(f"{variable} must be 1")
    defect = False
    for wave in ("traveling_wave", "standing_wave"):
        for fringe in ("entrance_end", "exit_end"):
            label = f"{wave} {fringe}, default n_rf_steps=10, odd response at +/-10 V"
            on, kick, records_on = pair(wave, fringe, None)
            off, _, records_off = pair(wave, fringe, False)
            if compare(
                label + " default spin angle, rad; BMT from tracked fringe kick",
                on, kick, atol=3e-24, rtol=2e-11,
            ):
                raise RuntimeError("Enabled fringe does not match the independent impulse oracle")
            defect |= compare(
                label + " spin_fringe_on=F angle, rad; zero fringe action",
                off, 0.0, atol=1e-17,
            )
            same = all(a.bits == b.bits for a, b in zip(records_on, records_off, strict=True))
            print(f"{label}: disabled/default spin and orbit bitwise equal={same}; "
                  f"|default angle|/voltage={abs(on)/10:.10g} rad/V")
    # These controls establish switch behavior, rather than calibrating their
    # different native-RF trajectory models against the stepped-cavity oracle.
    for method, steps in (("runge_kutta", None), ("bmad_standard", 0)):
        on, _, records_on = pair("traveling_wave", "entrance_end", None, method=method, steps=steps)
        off, _, records_off = pair("traveling_wave", "entrance_end", False, method=method, steps=steps)
        same_orbit = all(a.orbit == b.orbit for a, b in zip(records_on, records_off, strict=True))
        responds = abs(on - off) > 1e-10
        print(f"control {method} n_rf_steps={10 if steps is None else steps}: "
              f"default={on:.17g} disabled={off:.17g} rad; "
              f"switch changes spin={responds} expected=True; "
              f"orbit unchanged={same_orbit} expected=True")
        if not (responds and same_orbit):
            raise RuntimeError("Control path failed to toggle fringe spin alone")
    verdict(defect)


if __name__ == "__main__":
    main()
