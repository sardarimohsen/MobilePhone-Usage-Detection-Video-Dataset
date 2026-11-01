# MobilePhone-Usage-Detection-Video-Dataset

A repository on **detecting mobile-phone usage of people in indoor surveillance video**. It contains sample videos with their annotations, the code for data reading and preprocessing, training and end-to-end inference, and the initial trained model.

## Table of Contents

- [Repository Structure](#repository-structure)
- [Data](#data)
- [Model](#model)
- [Installation](#installation)
- [Usage](#usage)
- [Pipeline](#pipeline)

## Repository Structure

```text
.
├── README.md
├── README.fa.md
├── requirements.txt
├── .gitignore
├── src/                          # main source code
│   ├── build_clips.py            # reads videos and annotations, builds the cropped-clip cache
│   ├── dataset.py                # base dataset code used by dataset_stp.py
│   ├── dataset_stp.py            # dataset / data loading
│   ├── models_v3.py              # model definitions
│   ├── models_multimodal.py      # multimodal model components
│   ├── eval_tools.py             # evaluation utilities
│   ├── train_v3.py               # training script
│   └── viz/
│       ├── __init__.py
│       ├── config.py             # paths and constants (env: PHONE_VAULT, PHONE_RUNS)
│       ├── e2e_core.py           # crop, tracker stitching, temporal filter
│       └── e2e.py                # CLI: run / eval
├── samples/
│   ├── videos/                   # sample videos
│   └── annotations/              # matching annotation files
├── runs_final/
│   └── A_final_f0_s0/            # released model (head.pt, deploy.json)
└── weights/
    └── A_final_args.json         # training arguments of the released model
```

## Data

The full dataset contains 171 videos recorded at 19 locations. Video resolutions range from 1280x800 to 3072x2048 and frame rates from about 12.5 to 30 fps (most videos are 25 fps).

This repository includes only a few sample videos (one or two per location) from 6 of the 19 locations, each with its annotation files:

| Location |
|---|
| Laboratory Corridor |
| Logic Circuit Lab |
| Microprocessor Laboratory |
| Polymer Laboratory |
| Faculty Parking |
| Physics Laboratory |

The sample videos are enough to run the demo (`src.viz.e2e run`). Training and `eval` need the full dataset, which is not included.

**To request the full dataset, please send a message to:**

- 📧 comp.sardari@gmail.com
- 📧 armankhalili1381@gmail.com

## Model

Only the initial model is released at this stage: model `A`, a DINOv2 ViT-S/14 backbone (fine-tuned with LoRA) with a simple temporal head. It is stored in `runs_final/A_final_f0_s0/`. Its full training arguments are in `weights/A_final_args.json`.

The person detector `yolo11n.pt` is downloaded automatically by Ultralytics on first use.

## Installation

```bash
git clone https://github.com/sardarimohsen/MobilePhone-Usage-Detection-Video-Dataset.git
cd MobilePhone-Usage-Detection-Video-Dataset
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
```

For anything that reads the full dataset or the crop cache, set the paths in `src/viz/config.py`, or the environment variables `PHONE_VAULT` (data and cache) and `PHONE_RUNS` (run folders).

## Usage

**Run the demo on a sample video**

```bash
python -m src.viz.e2e run \
  --video "samples/videos/<location>/<video>.mp4" \
  --out output/demo.mp4 \
  --run A --ckpts 0:0 \
  --person-model yolo11n.pt
```

The output video shows a colour-coded box per tracked person (using / not using / not yet decided).

**Build the crop cache (needs the full dataset)**

```bash
python src/build_clips.py --dataset-root <DATASET_ROOT> --ann-root <ANNOTATION_ROOT> \
  --split-map <SPLIT_MAP> --out <CACHE> --size 224
# crop options (--margin, --mask-outside, --top-frac, ...): see --help
```

**Train (needs the full dataset)**

```bash
python src/train_v3.py --cache <CACHE> --folds <FOLDS_JSON> --fold 0 --seed 0 \
  --out runs_final --tag A_final --head simple \
  --backbone vit_small_patch14_dinov2.lvd142m --size 224 \
  --lora-r 16 --lora-alpha 32 --lora-target "qkv,proj" \
  --d-model 256 --layers 2 --heads 4
# remaining hyper-parameters: weights/A_final_args.json
```

**Evaluate the full pipeline (needs the full dataset)**

```bash
python -m src.viz.e2e eval --run A --fold 0 --seed 0 --modes det gt \
  --out output/e2e_eval --person-model yolo11n.pt
```

## Pipeline

1. Person detection and tracking (Ultralytics tracker, with optional re-identification).
2. Track stitching to reduce identity switches (IoU, spatial distance and time gap).
3. Masked square crop around each person.
4. Clip-level classification with model `A`.
5. Temporal filter (exponential moving average with on/off hysteresis) producing the final per-person state.
