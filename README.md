# MobilePhone-Usage-Detection-Video-Dataset

[English](#english) | [فارسی](#فارسی)

---

<a name="english"></a>

## English

### Overview

This repository is about **detecting mobile-phone usage of people in indoor surveillance video**. It contains:

- sample videos from the dataset, with their annotation files (a small subset only),
- the code for data reading and preprocessing, training, and end-to-end inference (detection, tracking, crop, phone-feature extraction, classification, temporal filtering),
- the initial trained models.

### Repository layout

```
.
├── README.md
├── requirements.txt
├── build_clips.py           # reads videos, builds the cropped-frame cache
├── dataset.py               # base dataset code used by dataset_stp.py
├── dataset_stp.py           # dataset / data loading
├── models_v3.py             # model definitions
├── models_multimodal.py     # multimodal model components
├── eval_tools.py            # evaluation utilities
├── train_v3.py              # training script
├── viz/
│   ├── __init__.py
│   ├── config.py            # paths and constants (env: PHONE_VAULT, PHONE_RUNS)
│   ├── e2e_core.py          # crop, FPI features, tracker stitching, temporal filter
│   └── e2e.py               # CLI: run / eval / parity
├── samples/
│   ├── videos/              # sample videos
│   └── annotations/         # matching annotation files
└── weights/                 # small weights or download instructions
```

### Data

The full dataset contains 171 videos recorded at 19 locations. This repository includes only a few sample videos (one or two per location) from 6 of the 19 locations, each with its annotation file:

| Location |
|---|
| Laboratory_Corridor |
| Logic Circuit Lab |
| Microprocessor Laboratory |
| Polymer Laboratory |
| Faculty Parking |
| Physics Laboratory |

Video resolutions range from 1280x800 to 3072x2048 and frame rates from about 12.5 to 30 fps (most videos are 25 fps).

The sample videos are enough to run the demo (`viz.e2e run`). Training, `parity` and `eval` need the full dataset and the crop cache, which are not included.

**To request the full dataset, please send a message to:**

- 📧 comp.sardari@gmail.com
- 📧 armankhalili1381@gmail.com

### Models

| Model | Purpose | Location |
|---|---|---|
| `A` (`head.pt`), initial baseline model | DINOv2 ViT-S/14 backbone (LoRA) with a simple temporal head, without FPI features | included in this repository: `runs_final/A_final_f0_s0/head.pt` |
| FPI detector (YOLO, 832 px) | detects phone-related items inside the crop; include only if the released model uses it | `TODO` (Release / external link; too large for a normal commit) |
| `yolo11n.pt` | person detector | downloaded automatically by Ultralytics |

Only initial models are released at this stage. The weights of the final, best-performing model are not included.

Expected path of the checkpoint: `runs_final/A_final_f0_s0/head.pt`. The full training arguments of this model are stored in `weights/A_final_args.json`.

### Installation

```bash
git clone https://github.com/sardarimohsen/MobilePhone-Usage-Detection-Video-Dataset.git
cd MobilePhone-Usage-Detection-Video-Dataset
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
```

Edit the paths in `viz/config.py` (or set the corresponding environment variables) so they point to your local data and cache folders.

### Usage

**1. Build the crop cache (training only, needs the full dataset)**

```bash
python build_clips.py --dataset-root <DATASET_ROOT> --ann-root <ANNOTATION_ROOT> \
  --split-map <SPLIT_MAP> --out <CACHE> --size 224
# crop options (--margin, --mask-outside, --top-frac, ...): see --help
```

**2. Train (needs the full dataset)**

```bash
python train_v3.py --cache <CACHE> --folds <FOLDS_JSON> --fold 0 --seed 0 \
  --out runs_final --tag A_final --head simple \
  --backbone vit_small_patch14_dinov2.lvd142m --size 224 \
  --lora-r 16 --lora-alpha 32 --lora-target "qkv,proj" \
  --d-model 256 --layers 2 --heads 4
# remaining hyper-parameters: weights/A_final_args.json
```

**3. Verify that inference matches training preprocessing (needs the full dataset)**

```bash
python -m viz.e2e parity --n 40 \
  --fpi-model weights/fpi_best.pt
```

**4. Evaluate the full pipeline (needs the full dataset)**

```bash
python -m viz.e2e eval \
  --run A --fold 0 --seed 0 \
  --modes det gt \
  --out output/e2e_eval \
  --person-model yolo11n.pt \
  --fpi-model weights/fpi_best.pt
```

**5. Run on a sample video**

```bash
python -m viz.e2e run \
  --video samples/videos/<location>/.../<video>.mp4 \
  --out output/demo.mp4 \
  --run A --ckpts 0:0 \
  --person-model yolo11n.pt \
  --fpi-model weights/fpi_best.pt \
  --neutral-warmup
```

The output video shows a colour-coded box per tracked person (using / not using / not yet decided).

### Pipeline

1. Person detection and tracking (Ultralytics tracker, with optional re-identification).
2. Track stitching to reduce identity switches (IoU, spatial distance and time gap).
3. Masked square crop around each person.
4. Phone-feature extraction with the FPI detector.
5. Clip-level classification (released initial model: `A`).
6. Temporal filter (exponential moving average with on/off hysteresis) producing the final per-person state.

---

<a name="فارسی"></a>

<div dir="rtl">

## فارسی

### معرفی

این مخزن درباره‌ی **تشخیص استفاده از تلفن همراه توسط افراد در ویدئوهای دوربین مداربسته‌ی داخلی** است و شامل موارد زیر می‌شود:

- نمونه‌ویدئوهایی از دیتاست همراه با فایل‌های برچسب (annotation) متناظر (فقط زیرمجموعه‌ی کوچکی از آن)،
- کدهای خواندن و پیش‌پردازش داده، آموزش مدل و اجرای انتها‌به‌انتها (تشخیص، ردیابی، برش، استخراج ویژگی گوشی، طبقه‌بندی و فیلتر زمانی)،
- مدل‌های اولیه‌ی آموزش‌دیده.

### ساختار مخزن

ساختار پوشه‌ها مانند نسخه‌ی انگلیسی بالا است. ساختار اصلی پروژه عمداً حفظ شده تا دستورهای `python -m viz.e2e ...` بدون تغییر کار کنند.

### داده

دیتاست کامل شامل ۱۷۱ ویدئو است که در ۱۹ موقعیت مکانی ضبط شده‌اند. در این مخزن فقط چند ویدئوی نمونه (یک یا دو ویدئو برای هر موقعیت) از ۶ موقعیت از ۱۹ موقعیت، همراه با فایل برچسب متناظر هر ویدئو قرار دارد:

| موقعیت مکانی |
|---|
| Laboratory_Corridor |
| Logic Circuit Lab |
| Microprocessor Laboratory |
| Polymer Laboratory |
| Faculty Parking |
| Physics Laboratory |

رزولوشن ویدئوها از 1280x800 تا 3072x2048 و نرخ فریم از حدود ۱۲٫۵ تا ۳۰ فریم بر ثانیه متغیر است (بیشتر ویدئوها ۲۵ فریم بر ثانیه‌اند).

نمونه‌ویدئوها برای اجرای دمو (`viz.e2e run`) کافی هستند. آموزش، `parity` و `eval` به دیتاست کامل و کش برش‌ها نیاز دارند که در این مخزن قرار داده نشده‌اند.

**برای دریافت دیتاست کامل، لطفاً به یکی از ایمیل‌های زیر پیام دهید:**

- 📧 comp.sardari@gmail.com
- 📧 armankhalili1381@gmail.com

### مدل‌ها

| مدل | کاربرد | محل |
|---|---|---|
| `A` (فایل `head.pt`)، مدل اولیه | بک‌بون DINOv2 ViT-S/14 (با LoRA) و هد زمانی ساده، بدون ویژگی‌های FPI | موجود در همین مخزن: `runs_final/A_final_f0_s0/head.pt` |
| آشکارساز FPI (یولو، ۸۳۲ پیکسل) | تشخیص اشیای مرتبط با گوشی داخل برش؛ فقط اگر مدل منتشرشده از آن استفاده می‌کند | `TODO` (Release یا لینک خارجی؛ برای کامیت معمولی بزرگ است) |
| `yolo11n.pt` | آشکارساز شخص | به‌طور خودکار توسط Ultralytics دانلود می‌شود |

در این مرحله فقط مدل‌های اولیه منتشر می‌شوند و وزن‌های مدل نهایی (بهترین دقت) در مخزن قرار ندارد.

مسیر مورد انتظار چک‌پوینت: `runs_final/A_final_f0_s0/head.pt`. آرگومان‌های کامل آموزش این مدل در `weights/A_final_args.json` ذخیره شده است.

### نصب

```bash
git clone https://github.com/sardarimohsen/MobilePhone-Usage-Detection-Video-Dataset.git
cd MobilePhone-Usage-Detection-Video-Dataset
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
```

مسیرهای داخل `viz/config.py` را (یا متغیرهای محیطی متناظر) طوری تنظیم کنید که به پوشه‌های داده و کش روی سیستم شما اشاره کنند.

### نحوه‌ی استفاده

دستورهای مراحل ۱ تا ۵ همان دستورهای بخش انگلیسی هستند. مراحل ساخت کش، آموزش، `parity` و `eval` به دیتاست کامل نیاز دارند؛ اجرای دمو روی نمونه‌ویدئوها ممکن است. خروجی ویدئو برای هر شخص ردیابی‌شده یک کادر رنگی نشان می‌دهد (در حال استفاده / عدم استفاده / هنوز نامشخص).

### خط لوله‌ی پردازش

۱. تشخیص و ردیابی افراد (ردیاب Ultralytics، با قابلیت اختیاری بازشناسی ظاهری).
۲. اتصال دوباره‌ی ترک‌ها برای کاهش تغییر شناسه (بر پایه‌ی IoU، فاصله‌ی مکانی و فاصله‌ی زمانی).
۳. برش مربعی ماسک‌شده اطراف هر شخص.
۴. استخراج ویژگی گوشی با آشکارساز FPI.
۵. طبقه‌بندی در سطح کلیپ (مدل اولیه‌ی منتشرشده: `A`).
۶. فیلتر زمانی (میانگین متحرک نمایی با آستانه‌های روشن/خاموش مجزا) برای تعیین وضعیت نهایی هر شخص.

</div>
