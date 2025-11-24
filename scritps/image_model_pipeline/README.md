# Image Model Pipeline

This folder contains an isolated training/validation loop for the pure image
feature tower so the existing scripts remain untouched. The workflow mirrors
`debug_text_model.py` but swaps the text encoder for the `ImageModel` tower.

## Files

- `train_image_model.py` – full dataset, model, trainer, and evaluation logic.

## Quick start

```powershell
python scritps\image_model_pipeline\train_image_model.py `
  --train-csv data\train_val_test_dir\train.csv `
  --val-csv data\train_val_test_dir\val.csv `
  --all-csv data\train_val_test_dir\all_ratings.csv `
  --text-features feature_output\movie_text_features_20251103_1457.npz `
  --img-features feature_output\movie_img_features_20251103_1457.npz `
  --batch-size 32 `
  --epochs 10 `
  --lr 5e-4 `
  --proj-dim 256 `
  --dropout-rate 0.3
```

For quick smoke tests (e.g., to ensure the script runs end-to-end without
computing the expensive Top-K metrics), append `--max-train-batches 2
--max-val-batches 2 --skip-topk --num-workers 0` to the command.

Use `--max-train-batches` / `--max-val-batches` for quick smoke tests and set
`--topk-max-users` if you need to cap the expensive Top-K evaluation.
