# Method

## What I built

`run_inference.py` is a classical signal-processing cleaner. It takes raw wrist PPG plus the
wrist accelerometer and gyroscope channels and returns the cleaned PPG. The output is a 1-D
array with exactly the input length on the original 256 Hz grid. The core is IMU-referenced
adaptive noise cancellation (ANC): the motion sensors predict the artifact in the PPG, and the
prediction is subtracted. The model uses only the wrist PPG and IMU channels, and does not
use record names or activity labels.

Pipeline:

1. **Repair.** Channels are selected by name. PPG samples that are non-finite, or stuck at one
   value for 0.1 s or more (an ADC rail), are bridged by linear interpolation on the existing
   time grid.
2. **Cardiac band.** The PPG and IMU are band-passed to 0.4-4 Hz. The filter is a zero-phase
   Butterworth, order 3.
3. **Adaptive noise cancellation.** In overlapping 8-s blocks, a ridge-regularised 16-tap FIR
   model per IMU axis predicts the motion part of the PPG. The prediction is subtracted only if
   it passes both checks:
   - the fit transfers to held-out 1-s chunks of the same block;
   - it does not explain almost the whole block.

   Cancellation is skipped for the whole record if the PPG and IMU are not coherent.
4. **Output band.** The signal is band-passed to 1.4-4 Hz to remove the remaining
   low-frequency motion.
5. **Two-peak correction.** At low pulse rates the 1.4 Hz edge can split one pulse into two
   bumps. The local pulse period is tracked each second from the PPG spectrum, with bins that
   carry IMU power down-weighted. When two peaks are closer than half a period, the smaller
   one is flattened with a smooth bridge. Peaks are never merged if they are more than 0.34 s
   apart or if the pulse track is not confident.

The output is always a waveform derived from the recorded PPG, not beat indices or templates.

## Key decisions and why

- **Zero-phase filtering throughout.** Peaks are scored within ±150 ms, and a causal filter
  would delay every peak. Forward-backward filtering adds no delay. The detected peaks show no
  systematic offset: the median error is -12 ms, and no constant shift improves F1.
- **ANC with safeguards.** The IMU predicts arm motion well during running: switching ANC off
  costs s3_run 18 F1 points. During walking, arm swing is close to the pulse rate, so an
  unchecked fit can remove the pulse. The held-out check and the coherence gate keep uncertain
  fits from being subtracted. Mean development F1 is 0.610 with the checks and 0.606 without.
- **1.4-4 Hz output band.** Most motion energy lies below the pulse band. This band alone takes
  raw PPG from 0.432 to 0.590 mean F1, the largest single gain.
- **Two-peak correction with limits.** The original rule merged any peaks closer than half the
  tracked period. On walks, where the tracker can lock onto the step rate, it merged real
  beats. The 0.34 s cap and the confidence condition fixed this: s1_walk went from 0.28 to 0.35.
  Both constants are insensitive: ±20% changes pooled F1 by at most 0.0008, and
  leave-one-subject-out tuning did not beat the defaults.
- **Robust inputs.** The cleaner tolerates missing or reordered IMU channels, NaN gaps, flat
  sensors and ADC rails, and always returns a finite array of the right length.
- **Tested and rejected** (each change alone, development subjects only):
  - a pulse-adaptive high-pass edge;
  - standardised or decimated cancellation inputs;
  - a motion-energy gate;
  - amplitude normalisation;
  - cadence notch filters;
  - spectral suppression;
  - other filter orders and band edges.

  None beat the pipeline above. Changes that helped walks tended to hurt runs and bikes by a
  similar amount.

## Results

All tuning used the development subjects s1-s6. The holdout column uses s8 and s9, the only
holdout subjects available here (s7 was not available); they were not used for tuning.

| Mean F1 (±150 ms) | Walk | Run | Bike low | Bike high | All development | Holdout (s8, s9) |
|---|---|---|---|---|---|---|
| Raw PPG | 0.435 | 0.572 | 0.401 | 0.295 | 0.432 | - |
| This cleaner | 0.439 | 0.666 | 0.643 | 0.706 | 0.610 | 0.460 |

**Limitations.**
- Walking is the weakest activity, because arm swing near the pulse rate is only partly
  removed.
- On some records the offset between detected peaks and the reference timing drifts steadily
  over the recording, for example from -74 to -176 ms on s9_walk. This looks like clock drift
  between the recording devices, and no cleaning step can correct it.

## Runtime and reproducibility

A 5-minute record takes about 0.34 s on a laptop CPU. The method is deterministic and needs no
training, pre-trained models or network access. Dependencies are pinned in `requirements.txt`.

## AI use and what I verified

I used AI coding assistance (Claude, via Claude Code) for four things:
- reviewing failure cases;
- implementing numerical safeguards, the two-peak correction and its limits, and saturation
  handling;
- running the one-change-at-a-time comparison experiments;
- writing the robustness tests.

I verified these myself:
- method's research;
- data eda and read
- the final algorithm;
- the output contract: 1-D, finite, exact input length, 256 Hz;
- that the model reads only the wrist channels;
- the scores, using the assignment's evaluation protocol;
- the runtime and the dependency list.

## References

- Jarchi, D. and Casson, A. J. (2017). Description of a Database Containing Wrist PPG Signals
  Recorded during Physical Exercise with Both Accelerometer and Gyroscope Measures of Motion.
  *Data*, 2(1). https://doi.org/10.3390/data2010001
- Dataset: Wrist PPG During Exercise (WRIST) v1.0.0, PhysioNet,
  https://physionet.org/content/wrist/1.0.0/ (ODC-BY 1.0).
- Goldberger, A. L. et al. (2000). PhysioBank, PhysioToolkit, and PhysioNet. *Circulation*,
  101(23), e215-e220.
