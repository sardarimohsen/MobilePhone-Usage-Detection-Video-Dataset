<div dir="rtl">
    
**🌐 [English](README.md) | [فارسی](README.fa.md)**

<div dir="rtl" align="center">

# 📱 MobilePhone-Usage-Detection-Video-Dataset

**تشخیص استفاده از تلفن همراه توسط افراد در ویدئوهای دوربین مداربسته‌ی داخلی**

[![Demo](https://placehold.co/800x400?text=Demo+Video+Coming+Soon)](https://github.com/sardarimohsen/MobilePhone-Usage-Detection-Video-Dataset)

</div>

---

## 📖 معرفی

این مخزن یک خط لوله‌ی کامل برای **تشخیص استفاده از تلفن همراه توسط افراد در ویدئوهای دوربین مداربسته‌ی داخلی** ارائه می‌دهد. شامل نمونه‌ویدئوها همراه با برچسب‌ها، ابزارهای پیش‌پردازش داده، کد آموزش و اجرای انتها‌به‌انتها، و یک مدل اولیه‌ی آموزش‌دیده است.

## 📂 ساختار مخزن

```text
.
├── README.md
├── README.fa.md
├── requirements.txt
├── .gitignore
├── src/                          # کدهای اصلی پروژه
│   ├── build_clips.py            # ساخت کش کلیپ‌های برش‌خورده
│   ├── dataset.py                # کد پایه‌ی دیتاست
│   ├── dataset_stp.py            # دیتاست / بارگذاری داده
│   ├── models_v3.py              # تعریف مدل‌ها
│   ├── models_multimodal.py      # اجزای مدل چندوجهی
│   ├── eval_tools.py             # ابزارهای ارزیابی
│   ├── train_v3.py               # اسکریپت آموزش
│   └── viz/
│       ├── config.py             # مسیرها و ثابت‌ها (متغیرهای محیطی: PHONE_VAULT و PHONE_RUNS)
│       ├── e2e_core.py           # برش، اتصال دوباره‌ی ترک‌ها، فیلتر زمانی
│       └── e2e.py                # خط فرمان: run / eval
├── samples/
│   ├── videos/                   # نمونه‌ویدئوها
│   └── annotations/              # فایل‌های برچسب متناظر
├── runs_final/
│   └── A_final_f0_s0/            # مدل منتشرشده (head.pt و deploy.json)
└── weights/
    └── A_final_args.json         # آرگومان‌های آموزش مدل منتشرشده
```

## 📊 داده

دیتاست کامل شامل **۱۷۱ ویدئو** است که در **۱۹ موقعیت مکانی** ضبط شده‌اند. رزولوشن ویدئوها از **۱۲۸۰×۸۰۰ تا ۳۰۷۲×۲۰۴۸** و نرخ فریم از حدود **۱۲٫۵ تا ۳۰ فریم بر ثانیه** متغیر است (بیشتر ویدئوها ۲۵ فریم بر ثانیه‌اند).

در این مخزن فقط چند ویدئوی نمونه (یک یا دو ویدئو برای هر موقعیت) از **۶ موقعیت از ۱۹ موقعیت**، همراه با فایل‌های برچسب هر ویدئو قرار دارد:

| موقعیت مکانی |
|---|
| Laboratory Corridor |
| Logic Circuit Lab |
| Microprocessor Laboratory |
| Polymer Laboratory |
| Faculty Parking |
| Physics Laboratory |

نمونه‌ویدئوها برای اجرای دمو (`src.viz.e2e run`) کافی هستند. آموزش و `eval` به دیتاست کامل نیاز دارند که در این مخزن قرار داده نشده است.

### 📥 نحوه‌ی درخواست

برای دریافت دیتاست کامل با رزولوشن بالا، لطفاً به نویسنده‌ی مسئول ایمیل بزنید:

- **تماس:** Dr. Sardari
- **ایمیل:** [sardarimohsen@gmail.com](mailto:sardarimohsen@gmail.com)
- **موضوع ایمیل:** `MPUD High-Resolution Data Request`

## 🧠 مدل

در این مرحله فقط مدل اولیه منتشر شده است: مدل **A** با بک‌بون **DINOv2 ViT-S/14** (تنظیم‌شده با LoRA) و یک هد زمانی ساده. این مدل در پوشه‌ی `runs_final/A_final_f0_s0/` قرار دارد و آرگومان‌های کامل آموزش آن در `weights/A_final_args.json` است.

آشکارساز شخص `yolo11n.pt` در اولین اجرا به‌طور خودکار توسط Ultralytics دانلود می‌شود.

## ⚙️ نصب

```bash
git clone https://github.com/sardarimohsen/MobilePhone-Usage-Detection-Video-Dataset.git
cd MobilePhone-Usage-Detection-Video-Dataset
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
```

برای هر کاری که دیتاست کامل یا کش برش‌ها را می‌خواند، مسیرها را در `src/viz/config.py` یا با متغیرهای محیطی `PHONE_VAULT` (داده و کش) و `PHONE_RUNS` (پوشه‌ی اجراها) تنظیم کنید.

## 🚀 نحوه‌ی استفاده

> **توجه:** اجرای دمو روی یک نمونه‌ویدئو فقط به فایل‌های موجود در این مخزن نیاز دارد. اما ساخت کش برش‌ها، آموزش مدل و ارزیابی خط لوله‌ی کامل نیازمند دسترسی به دیتاست کامل است.

### 🎬 اجرای دمو روی یک نمونه‌ویدئو

```bash
python -m src.viz.e2e run \
  --video "samples/videos/<location>/<video>.mp4" \
  --out output/demo.mp4 \
  --run A --ckpts 0:0 \
  --person-model yolo11n.pt
```

ویدئوی خروجی برای هر شخص ردیابی‌شده یک کادر رنگی نشان می‌دهد (🟢 در حال استفاده / 🔴 عدم استفاده).

### 🗂️ ساخت کش برش‌ها

```bash
python src/build_clips.py --dataset-root <DATASET_ROOT> --ann-root <ANNOTATION_ROOT> \
  --split-map <SPLIT_MAP> --out <CACHE> --size 224
# گزینه‌های برش (--margin و --mask-outside و --top-frac و ...): خروجی --help
```

### 🏋️ آموزش

```bash
python src/train_v3.py --cache <CACHE> --folds <FOLDS_JSON> --fold 0 --seed 0 \
  --out runs_final --tag A_final --head simple \
  --backbone vit_small_patch14_dinov2.lvd142m --size 224 \
  --lora-r 16 --lora-alpha 32 --lora-target "qkv,proj" \
  --d-model 256 --layers 2 --heads 4
# سایر هایپرپارامترها: weights/A_final_args.json
```

### 📈 ارزیابی خط لوله‌ی کامل

```bash
python -m src.viz.e2e eval --run A --fold 0 --seed 0 --modes det gt \
  --out output/e2e_eval --person-model yolo11n.pt
```

## 🔄 خط لوله‌ی پردازش

۱. **تشخیص و ردیابی افراد** (ردیاب Ultralytics، با قابلیت اختیاری بازشناسی ظاهری).
۲. **اتصال دوباره‌ی ترک‌ها** برای کاهش تغییر شناسه (بر پایه‌ی IoU، فاصله‌ی مکانی و فاصله‌ی زمانی).
۳. **برش مربعی ماسک‌شده** اطراف هر شخص.
۴. **طبقه‌بندی در سطح کلیپ** با مدل `A`.
۵. **فیلتر زمانی** (میانگین متحرک نمایی با آستانه‌های روشن/خاموش مجزا) برای تعیین وضعیت نهایی هر شخص.
