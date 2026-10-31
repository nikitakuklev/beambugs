"""TRK-13: Exciter activates before its stated starting time

`Exciter` applies its first waveform sample to particles arriving less than one sample period before
`start_turn`. The documented start time, and an independent calculation of which sample covers each
arrival time, require no excitation before then. With `start_turn` = 5, revolution frequency 1 MHz,
sampling frequency 0.25 MHz, `knl=[2.5e-6]` and first sample 0.75, the reference particle receives
normalized horizontal momentum −1.875 × 10⁻⁶ at turn 4 instead of zero. Early arrivals are affected,
and when sampling is slower than revolution the kick can start a whole turn early; samples within
the waveform, arrivals more than one sample period early and times after the waveform ends are
handled correctly.

The script scans arrival times from 1.25 sample periods before the start to 3.25 after it, through
an exciter with three single-precision samples at 400 MHz, and compares each horizontal kick with
zero-order hold applied only where the real-valued sample index lies inside the waveform. A second
exciter, sampled at a quarter of the revolution frequency, kicks the reference particle one turn
before `start_turn`. Arrivals inside the waveform, after it and more than one period early must
match; the three arrivals just before the start and the early reference particle show the bug. It
runs serially on the installed prebuilt CPU kernels with kernel compilation forbidden.
"""

from _prebuilt import CTX, identity, kernels, np, particles, report, xt

identity()
cases = {}
fs, frev, start = 4e8, 1e6, 5
k = 2.5e-6
samples = np.array([0.75, -0.5, 0.25], dtype=np.float32)
element = xt.Exciter(
    _context=CTX,
    knl=[k],
    samples=samples,
    sampling_frequency=fs,
    frev=frev,
    start_turn=start,
)
# Real-valued sample coordinate u = fs ((at_turn - start_turn)/frev - zeta/(beta0 c)), support
# [0, nsamples); inside it the kick is -knl*samples[floor(u)], outside it zero. The kernel
# converts u to int64 first, truncating negative values toward zero.
sample_times = np.array([-1.25, -0.99, -0.75, -0.25, 0.0, 0.25, 0.75, 1.25, 2.25, 3.25])
p = particles(zeta=np.zeros(len(sample_times)), at_turn=start)
p.zeta = -sample_times * p.beta0[0] * 299792458.0 / fs
# Derive expected waveform after checking the real-valued half-open support;
# use floor for in-domain zero-order hold, without converting negatives to int.
expected = np.zeros(len(sample_times))
active = (sample_times >= 0) & (sample_times < len(samples))
# The kernel promotes stored Float32 samples to double for multiplication by
# its double-strength coefficient. NumPy otherwise keeps this product Float32.
expected[active] = -k * samples.astype(np.float64)[np.floor(sample_times[active]).astype(int)]
element.track(p)
print("TIME_FROM_START in sample periods=", sample_times)
print("PX observed=", np.asarray(p.px))
print("PX expected=", expected)
np.testing.assert_array_equal(p.px[[0, 4, 5, 6, 7, 8, 9]], expected[[0, 4, 5, 6, 7, 8, 9]])
cases["before-start px"] = (np.asarray(p.px)[1:4], expected[1:4])
np.testing.assert_array_equal(expected[1:4], [0.0, 0.0, 0.0])

# The same early kick occurs with zeta=0 one full revolution before start
# when sampling is slower than the revolution frequency. No oversized zeta.
slow = xt.Exciter(
    _context=CTX,
    knl=[k],
    samples=[0.75],
    sampling_frequency=frev / 4,
    frev=frev,
    start_turn=start,
)
early_reference = particles(at_turn=start - 1, zeta=0.0)
slow.track(early_reference)
print(
    "EARLY_REFERENCE turn=",
    early_reference.at_turn[0],
    "start=",
    start,
    "sample_time=-0.25 px=",
    early_reference.px[0],
    "expected=0",
)
cases["early reference px"] = (early_reference.px[0], 0.0)
kernels()
report(cases, rtol=4e-15, atol=0)
