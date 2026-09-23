# Earthquake coherence calibration experiments

Reproducible regional alarm-area calibration for Sentinel-1 coherence time series.

## Contents

- `src/`: data acquisition, calibration, validation, and numerical experiments.
- `data/`: immutable cropped LiCSAR rasters, georeferencing, source URLs, and checksums; independently interpreted Copernicus grading data.
- `results/`: numerical reference outputs and fitted recurrent-model weights.
- `config.json`: fixed earthquake regions and acquisition pairs.
- `requirements.txt`: numerical environment.
- `LICENSE`: MIT license for original code.
- `THIRD_PARTY_NOTICES.md`: data and upstream model attribution.

## Reproduce

Python 3.12 is recommended. Create a virtual environment and install `requirements.txt`.

```sh
python -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
python src/test_calibration.py
python src/verify_data.py
python src/monte_carlo.py --workers 8 --repetitions 50000
python src/evaluate_real.py --workers 3
python src/evaluate_reference.py
python src/recurrent_baseline.py --event ridgecrest --device cpu
python src/recurrent_baseline.py --event turkey --device cpu
python src/recurrent_baseline.py --event morocco --device cpu
python src/summarize_results.py
python src/verify_results.py
```

The included data are sufficient for offline reruns. `download_data.py --workers 8` reacquires the same products from the public archive when network access is available. It requests only the TIFF metadata and compressed strips intersecting each crop, then preserves the original integer coherence values. Data are not synthetic unless explicitly described as simulated or injected.

The recurrent commands use the included frozen weights. Add `--retrain --device mps` to train on an Apple GPU, or `--retrain --device cpu` on other systems. Fixed seeds support repeatability; GPU training is not promised to be bitwise identical across devices or library versions. Primary statistical results are deterministic given the included arrays and configuration. The included weights were independently checked with CPU inference. `verify_results.py` compares regenerated summaries with immutable reference values at a numerical tolerance of 1e-6 absolute and 1e-5 relative.

Results are written under `results/`. All analyses use fixed acquisition-separated fitting, calibration, and later background-test partitions. Ridgecrest uses 29 calibration candidates because its usable history is shorter; the other two sites use 39. Masks and candidate partition indices are recorded in each event summary. The earthquake observations and optical labels never tune the thresholds. The scripts also evaluate violations of temporal exchangeability. A background alarm is an earthquake-screening false alarm, not evidence that the ground surface remained physically unchanged.

`Regional-max` controls the probability of excessive flagged area in any predefined region under its stated map-exchangeability assumptions. It does not control the fraction of flagged buildings that are structurally damaged, and it does not provide a distribution-free guarantee under arbitrary temporal shift.

## Hardware

The experiments target an Apple M3 Pro with 18 GB unified memory. Eight CPU workers run independent Monte Carlo repetitions; real-data jobs use three workers to limit simultaneous raster-array allocations. GPU computation is reserved for the recurrent predictor. Worker counts can be reduced without changing numerical inputs or seeds.

## GitHub use

Unzip this archive and commit its contents as the repository root. The repository contains experimental code, inputs, and numerical outputs only. Individual data crops remain below GitHub's ordinary file-size limit. The data licenses remain distinct from the MIT code license.
