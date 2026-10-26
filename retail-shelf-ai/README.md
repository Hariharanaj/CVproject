# Retail Shelf AI

An industry-style retail shelf compliance prototype. It detects products on
store shelves, recognizes a configurable set of frequent SKUs, groups them by
shelf row, estimates visual gaps, and optionally compares the observations to
a manually authored planogram.

The project uses **SKUs on Shelves PL** as its primary dataset. The preparation
script keeps the most frequent SKU classes and maps every unsupported or
ambiguous product to `other_product`. This makes the first model useful and
measurable without pretending that a long-tailed 8,000-class detector is
reliable from the first run.

## Architecture

```text
Shelf image / video
        ↓
YOLO SKU detector
        ↓
Frequent SKU labels + other_product
        ↓
Shelf rows and visual-gap estimates
        ↓
Optional manual planogram comparison
        ↓
Annotated image + JSON report + Streamlit dashboard
```

This prototype does not infer backroom inventory, and a visual gap is not a
confirmed stockout. Price OCR can be added later as a separate component.

## Project structure

```text
retail-shelf-ai/
├── configs/
│   ├── example_planogram.json
│   └── train_config.yaml
├── data/
│   └── sample_shelf.jpg
├── models/                         # trained checkpoints, kept out of Git
├── scripts/
│   ├── download_skus_shelves_pl.py
│   └── prepare_skus_shelves_pl.py
├── src/
│   ├── app.py                      # Streamlit dashboard
│   ├── device_utils.py             # CUDA/MPS/CPU selection
│   ├── inference.py                # single-image inference
│   ├── planogram.py                # manual planogram rules
│   ├── report.py                   # JSON/CSV reports
│   ├── shelf_analysis.py           # row and gap analysis
│   ├── train.py                    # YOLO training entry point
│   └── video_inference.py          # webcam/video inference
└── tests/
```

## Local Mac setup

Use Python 3.10 or newer. The Mac is appropriate for inference and small
experiments; use MoLab for the large dataset preparation and GPU training.

```bash
cd /path/to/retail-shelf-ai
python3 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
python src/device_utils.py
```

On an M4 Mac, the selected device should normally be `mps`. If an operation is
not implemented by Metal, use:

```bash
export PYTORCH_ENABLE_MPS_FALLBACK=1
```

## MoLab: download and prepare the dataset

Run the following on MoLab, where the NVIDIA RTX 6000 is available. The source
archive is large, so store it in persistent remote storage if your MoLab
session is temporary.

```bash
git clone <your-repository-url> retail-shelf-ai
cd retail-shelf-ai
python -m pip install -r requirements.txt
```

Download and extract the public Hugging Face archive:

```bash
python scripts/download_skus_shelves_pl.py \
  --download-dir data/raw \
  --extract-dir data/raw/SKUs_on_Shelves_PL \
  --keep-archive
```

Prepare a first 50-SKU model. `--link-mode hardlink` avoids making a second
full copy of the images when the extracted archive and output are on the same
filesystem. Use `copy` if hardlinks are unavailable.

```bash
python scripts/prepare_skus_shelves_pl.py \
  --source-root data/raw/SKUs_on_Shelves_PL \
  --output data/SKUs_on_Shelves_PL \
  --top-k 50 \
  --min-instances 100 \
  --seed 42 \
  --link-mode hardlink \
  --overwrite
```

The script creates:

- `data/SKUs_on_Shelves_PL/SKUs_on_Shelves_PL.yaml`
- `class_map.json`, which contains the selected SKU names and IDs
- `dataset_summary.json`, which records class counts and split sizes
- YOLO `images/{train,val,test}` and `labels/{train,val,test}` directories

Before training, inspect `class_map.json` and `dataset_summary.json`. If the
frequent products are not useful for your intended demo, rerun with another
`--top-k` or `--min-instances` value.

## MoLab: train on the RTX 6000

First verify CUDA:

```bash
python src/device_utils.py
python -c "import torch; print(torch.cuda.is_available(), torch.cuda.get_device_name(0))"
```

Run a short smoke test before committing to a long job:

```bash
python src/train.py \
  --data data/SKUs_on_Shelves_PL/SKUs_on_Shelves_PL.yaml \
  --model yolo11m.pt \
  --device 0 \
  --epochs 3 \
  --imgsz 960 \
  --batch -1 \
  --workers 8 \
  --name skus_pl_smoke
```

If the smoke test uses the GPU and the labels look correct, run the main
training job:

```bash
python src/train.py \
  --data data/SKUs_on_Shelves_PL/SKUs_on_Shelves_PL.yaml \
  --model yolo11m.pt \
  --device 0 \
  --epochs 100 \
  --imgsz 960 \
  --batch -1 \
  --workers 8 \
  --patience 25 \
  --name skus_pl_top50
```

The best checkpoint is saved at:

```text
models/skus_pl_top50/weights/best.pt
```

Validate on the held-out test split after choosing the model:

```bash
python - <<'PY'
from ultralytics import YOLO

model = YOLO("models/skus_pl_top50/weights/best.pt")
metrics = model.val(
    data="data/SKUs_on_Shelves_PL/SKUs_on_Shelves_PL.yaml",
    split="test",
    imgsz=960,
    device=0,
    plots=True,
)
print("mAP50:", metrics.box.map50)
print("mAP50-95:", metrics.box.map)
print("precision:", metrics.box.mp)
print("recall:", metrics.box.mr)
PY
```

Keep the test split untouched until you have selected the checkpoint and
confidence threshold. Report overall mAP as well as per-class performance and
product-count error; a high overall score can hide poor results for rare SKUs.

## Integrate the MoLab model with Antigravity

The model and dataset do not need to be trained inside Antigravity. Antigravity
should run the application and use the exported checkpoint.

1. In MoLab, download or upload `models/skus_pl_top50/weights/best.pt` to a
   persistent location such as Hugging Face, Drive, or your computer.
2. In Antigravity, open the local folder `retail-shelf-ai`.
3. Copy the checkpoint into `models/skus_pl_top50/weights/best.pt`.
4. Also copy `data/SKUs_on_Shelves_PL/class_map.json` if you want to inspect
   the class mapping locally. The full 11 GB training dataset is not required
   for inference.
5. Install dependencies and launch:

   ```bash
   python -m pip install -r requirements.txt
   streamlit run src/app.py
   ```

6. Upload a shelf image, confirm the class names and confidence scores, and
   optionally enter a planogram JSON path in the sidebar.

If you use GitHub as the bridge, commit only source/configuration files and
the small class map. Keep the dataset archive and model checkpoint in release
storage or a private artifact store; they are excluded by `.gitignore`.

## Image and video inference

After copying `best.pt` locally:

```bash
python src/inference.py \
  --image data/sample_shelf.jpg \
  --model models/skus_pl_top50/weights/best.pt \
  --device mps \
  --conf 0.35
```

For prerecorded video:

```bash
python src/video_inference.py \
  --source path/to/shelf_video.mp4 \
  --model models/skus_pl_top50/weights/best.pt \
  --device mps \
  --skip 1 \
  --save
```

Skipped video frames deliberately show no old boxes, preventing detections
from a previous camera position from being displayed in the wrong location.

## Planogram rules

Copy [configs/example_planogram.json](configs/example_planogram.json), replace
the placeholder names using `class_map.json`, and configure expected counts per
row:

```json
{
  "name": "Aisle 4 shelf 2",
  "rows": [
    {
      "row_index": 0,
      "expected_counts": {
        "actual_sku_name_from_class_map": 4,
        "other_product": 1
      }
    }
  ]
}
```

The app reports missing, extra, and unexpected products for each configured
row. These are visual compliance findings for human review, not inventory
transactions.

## Testing

```bash
python -m pytest -q
```

The tests cover device selection, report serialization, row/gap analysis,
planogram comparison, and deterministic dataset splitting. Model quality still
requires evaluation on the held-out test split and visual review of predictions.

## Accuracy checklist

1. Inspect the generated class map and label visualizations.
2. Start with 20–50 frequent SKUs and an `other_product` class.
3. Keep related images together when creating splits; avoid near-duplicate
   leakage.
4. Compare 640px and 960px training on the same validation set.
5. Tune the confidence threshold on validation data, not the test set.
6. Review false positives between visually similar package designs.
7. Test on shelf imagery from the target camera/store domain before making
   operational claims.
8. Keep low-confidence and `other_product` detections visible in reports.

## Data and model notices

The Hugging Face dataset card identifies SKUs on Shelves PL as CC BY 4.0;
retain attribution when using it. Check the licenses of the dataset, model
framework, and any downstream model before commercial deployment. Do not use
the dashboard to make automated inventory or employee decisions without human
review and appropriate operational validation.
