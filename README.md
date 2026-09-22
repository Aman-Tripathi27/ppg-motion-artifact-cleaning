# Wrist PPG motion-artifact cleaning

Cleans motion-contaminated wrist PPG using the co-located accelerometer and gyroscope,
through IMU-referenced adaptive noise cancellation with zero-phase filtering. See
[method.md](method.md) for how it works and the results.

## Setup

```bash
pip install -r requirements.txt
```

## Usage

```bash
python run_inference.py <record> <output.npy|output.csv>
```

`<record>` is a WFDB record path from the
[WRIST dataset](https://physionet.org/content/wrist/1.0.0/), with or without `.hea`.
The output is the cleaned PPG as a 1-D array (`.npy`) or a single-column `.csv`. It has
the same length as the input, sampled at 256 Hz.

Example:

```bash
python run_inference.py wrist/1.0.0/s1_walk cleaned_s1_walk.npy
```

## Files

- `run_inference.py`: the cleaner, which runs as a command-line tool or via `clean_ppg()`
- `requirements.txt`: pinned dependencies
- `method.md`: the method, key decisions, results and AI-use declaration
