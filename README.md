# Sortify — AI Waste Identification & Disposal Guidance (YOLO11)

Sortify is an AI-powered waste detection and smart disposal guidance platform built with **Ultralytics YOLO11** and **FastAPI**. It identifies discarded items from photos/camera feeds and maps detections to **5 municipal waste disposal categories**:
1. 🟢 **Recyclable**
2. ⚪ **General waste**
3. 🟡 **Organic**
4. 🟣 **E-waste**
5. 🔴 **Hazardous**

---

## 🏆 1. Active Production Model

- **Active Production Checkpoint:** [`weights/yolo11s_8cls_v3_generalization.pt`](weights/yolo11s_8cls_v3_generalization.pt) (also symlinked as [`YOLOv11/model/best.pt`](YOLOv11/model/best.pt))
- **Base Architecture:** Ultralytics `YOLO11s` (9.4M parameters, 19.2 MB)
- **Dataset:** `v3 Generalization Dataset` (8 Classes + Hard Negative Background Supervision)
- **Overall Performance:** **`75.67%` mAP@50**, **`61.41%` mAP@50-95**, **`79.08%` Precision**, **`69.50%` Recall**
- **Key Advantage:** Hard negative background supervision reduces false positives by **`>80%`** when pointed at everyday surfaces, desks, floors, or hands.

---

## 📊 2. All Models Benchmark & Comparison

Comprehensive held-out test split evaluation across all trained Sortify model checkpoints:

| Model Checkpoint | Arch | Classes | Dataset Used | Precision | Recall | **mAP@50** | **mAP@50-95** | Status |
| :--- | :---: | :---: | :--- | :---: | :---: | :---: | :---: | :---: |
| **`yolo11s_8cls_v3_generalization.pt`** | YOLO11s | 8 | **v3 Generalization** *(Hard negatives)* | **79.08%** | **69.50%** | **75.67%** | **61.41%** | **Production Champion** |
| **`yolo11s_8cls_v2_curated.pt`** | YOLO11s | 8 | **v2 Curated Dataset** | 79.15% | 70.16% | **76.27%** | 62.27% | Validated |
| **`yolo11s_8cls_real_organic.pt`** | YOLO11s | 8 | **Real Organic Augmentations** | 73.82% | 72.06% | **76.46%** | 63.05% | Validated |
| **`yolo11s_8cls_v1_streamlined_old_dataset.pt`** | YOLO11s | 8 | **Old 8-Class Streamlined Dataset** | 79.83% | 70.87% | **76.81%** | 63.58% | Archived (Old Dataset) |
| **`yolo11s_10cls_legacy_old_dataset.pt`** | YOLO11s | 10 | **Old 10-Class Legacy Dataset** | 82.80% | 68.84% | **76.67%** | 61.48% | Archived (Old Dataset) |
| **`sortify_yolo11s_calibrated_aug`** | YOLO11s | 10 | **Old 10-Class Legacy Dataset** | 80.72% | 68.90% | **75.84%** | 60.73% | Archived (Old Dataset) |
| **`yolo11m_10cls_calibrated_old_dataset.pt`** | YOLO11m | 10 | **Old 10-Class Legacy Dataset** | 79.84% | 69.04% | **75.30%** | 61.01% | Archived (Old Dataset) |

*Full comparative report with confusion matrix analysis is available in [`Summaries/Results/MODEL_BENCHMARK_AND_COMPARISON.md`](Summaries/Results/MODEL_BENCHMARK_AND_COMPARISON.md).*

---

## 🗂️ 3. Dataset Lineage & Class Structure

### 3.1 Class Mapping (8-Class Production Detector → 5 Sortify Categories)

| ID | Detector Class Name | Sortify Disposal Category | Default Catalog Item | UI Icon |
| :-: | :--- | :--- | :--- | :---: |
| **`0`** | `paper_cardboard` | **Recyclable** | `box` (Cardboard Box) | `recycle` |
| **`1`** | `plastic_bottle` | **Recyclable** | `bottle` (Plastic Bottle) | `recycle` |
| **`2`** | `metal` | **Recyclable** | `can` (Metal Can / Tin) | `recycle` |
| **`3`** | `flexible_plastic` | **General waste** | `plastic-bag` (Plastic Bag) | `trash` |
| **`4`** | `sanitary_ppe_waste` | **General waste** | `tissue` (Sanitary Tissue) | `trash` |
| **`5`** | `organic` | **Organic** | `scraps` (Food Scraps) | `leaf` |
| **`6`** | `battery_ewaste` | **E-waste** | `battery` (Household Battery) | `plug` |
| **`7`** | `hazardous_medical`| **Hazardous** | `medical-waste` (Medical Waste) | `shield` |

### 3.2 Dataset Evolution Summary
- **Old 10-Class Legacy Dataset (7,355 images, 21,886 boxes):** Original unification of TACO, Mendeley, Bangladesh, PackWISE, and RecyBat24. Dropped due to 13% confusion between `paper`/`cardboard` and 29% background miss rate on transparent `glass`.
- **Old 8-Class Streamlined Dataset (v1):** Unified `paper` + `cardboard` into `paper_cardboard`, renamed `plastic_bag` to `flexible_plastic`, and excluded transparent `glass`.
- **Real Organic Augmentations:** Augmented organic category with real un-tiled food scraps and fruit peels.
- **v2 Curated Dataset:** Re-annotated ambiguous multi-label items and tightened bounding boxes.
- **v3 Generalization Dataset (Current Production):** Introduced negative supervision with non-waste background images to suppress out-of-distribution hallucinations.

---

## 🚀 4. How to Run the Application

### 4.1 Setup Environment

```bash
# Clone the repository
git clone git@github.com:MankiratSingh-git/Trash-Detection.git
cd Trash-Detection

# Activate your Python virtual environment (Python 3.10+ recommended)
python -m venv .venv
.venv\Scripts\activate       # Windows
# source .venv/bin/activate  # Linux/macOS

pip install -r src/fast-api/requirements.txt
```

### 4.2 Launch the Sortify ML Studio & API

Start the FastAPI backend (serves REST inference endpoints, model registry, and web interface):

```bash
# Using the runner script:
python src/fast-api/run.py

# Or using Uvicorn directly:
python -m uvicorn main:app --app-dir src/fast-api --reload --port 8000
```

Browser Access:
* **Interactive Web UI:** [http://localhost:8000/](http://localhost:8000/)
* **Interactive Swagger API Docs:** [http://localhost:8000/docs](http://localhost:8000/docs)
* **Registered Models Endpoint:** [http://localhost:8000/api/v1/models](http://localhost:8000/api/v1/models)

---

## 📁 5. Cleaned Repository Organization

```text
Sortify-ML/
├── Dataset/                     # Active YOLOv11 dataset splits
├── weights/                     # Standardized model weights
│   ├── best.pt                  # Production Champion (v3 Generalization)
│   ├── yolo11s_8cls_v3_generalization.pt
│   ├── yolo11s_8cls_v2_curated.pt
│   ├── yolo11s_8cls_real_organic.pt
│   ├── yolo11s_8cls_v1_streamlined_old_dataset.pt
│   ├── yolo11s_10cls_legacy_old_dataset.pt
│   ├── yolo11m_10cls_calibrated_old_dataset.pt
│   └── pretrained/              # Ultralytics base weights (yolo11s.pt, yolo11m.pt)
├── YOLOv11/
│   ├── model/
│   │   ├── best.pt              # Active production weights
│   │   ├── sortify_class_mapping.json
│   │   └── archive/             # Labeled historical weights
│   └── runs/                    # Training and test evaluation artifacts
├── Summaries/
│   └── Results/
│       ├── MODEL_BENCHMARK_AND_COMPARISON.md  # Definitive multi-model benchmark
│       ├── YOLOv11_Run_Summary.md             # Training trajectory & run summary
│       ├── yolo11_test_evaluation.md          # Held-out test split evaluation
│       └── yolov11_training_run_summary.md    # Multi-epoch loss convergence
└── src/
    ├── fast-api/                # FastAPI application with dynamic model registry
    └── frontend/                # Sortify ML Studio web interface
```

---

## 📄 Key Documentation Links

* **Comprehensive Model Benchmark & Comparison:** [`Summaries/Results/MODEL_BENCHMARK_AND_COMPARISON.md`](Summaries/Results/MODEL_BENCHMARK_AND_COMPARISON.md)
* **Training & Epoch Progress Report:** [`Summaries/Results/YOLOv11_Run_Summary.md`](Summaries/Results/YOLOv11_Run_Summary.md)
* **Preprocessing & Curation Audit:** [`Summaries/Results/Preprocessing_Report.md`](Summaries/Results/Preprocessing_Report.md)
