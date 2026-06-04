# Data

This repository expects the AEC-Challenge synthetic dataset to be downloaded outside the repo code.

Expected folder:

```text
data/AEC-Challenge/datasets/synthetic/
├── farend_speech/
├── nearend_mic_signal/
├── echo_signal/
└── nearend_speech/
```

Prepare the train/validation/test CSV files:

```cmd
python scripts\prepare_data.py --root data\AEC-Challenge\datasets\synthetic
```

Outputs:

```text
data/processed/train_pairs.csv
data/processed/val_pairs.csv
data/processed/test_pairs.csv
```

Each CSV row contains:

```csv
file_id,x_path,y_path,d_path,s_path
```

- `x_path`: far-end speech x(n)
- `y_path`: microphone signal y(n)
- `d_path`: clean echo target d(n)
- `s_path`: near-end speech s(n), optional for analysis
