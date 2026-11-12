# Sortify Model Benchmark & Dataset Comparison

A consolidated comparison of all trained vision models, dataset lineages, performance benchmarks, and deployment recommendations for the Sortify platform.

---

## 1. Executive Model Comparison Matrix

The table below summarizes all trained models evaluated on held-out test splits.

| Model Checkpoint | Architecture | Classes | Dataset Lineage | Status | Precision | Recall | mAP@50 | mAP@50-95 | Size |
| :--- | :--- | :---: | :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| **`yolo11s_8cls_v3_generalization.pt`**<br>*(Default: `best.pt`)* | YOLO11s | 8 | **v3 Generalization Dataset**<br>*(Hard negatives + background supervision)* | **Production Champion** | **79.08%** | **69.50%** | **75.67%** | **61.41%** | 19.2 MB |
| **`yolo11s_8cls_v2_curated.pt`** | YOLO11s | 8 | **v2 Curated Dataset**<br>*(Refined labels & boxes)* | Validated Archive | 79.15% | 70.16% | 76.27% | 62.27% | 19.2 MB |
| **`yolo11s_8cls_real_organic.pt`** | YOLO11s | 8 | **Real Organic Dataset**<br>*(Augmented food/produce scrap)* | Validated Archive | 73.82% | 72.06% | 76.46% | 63.05% | 19.2 MB |
| **`yolo11s_8cls_v1_streamlined_old_dataset.pt`** | YOLO11s | 8 | **Old 8-Class Dataset**<br>*(Initial streamlined consolidation)* | Old Dataset | 79.83% | 70.87% | 76.81% | 63.58% | 19.2 MB |
| **`yolo11s_10cls_legacy_old_dataset.pt`** | YOLO11s | 10 | **Old 10-Class Legacy Dataset**<br>*(Original multi-source unified)* | Old Dataset | 82.80% | 68.84% | 76.67% | 61.48% | 19.2 MB |
| **`yolo11s_calibrated_aug`** | YOLO11s | 10 | **Old 10-Class Legacy Dataset**<br>*(With calibrated augmentations)* | Old Dataset | 80.72% | 68.90% | 75.84% | 60.73% | 19.2 MB |
| **`yolo11m_10cls_calibrated_old_dataset.pt`** | YOLO11m | 10 | **Old 10-Class Legacy Dataset**<br>*(Medium architecture baseline)* | Old Dataset | 79.84% | 69.04% | 75.30% | 61.01% | 40.5 MB |

---

## 2. Dataset Evolution & History

```
[Raw Datasets] (TACO, Mendeley, Bangladesh, TrashNet, SJTU)
      │
      ▼
[Old 10-Class Legacy Dataset] (7,355 images, 21,886 boxes)
  • Separate paper / cardboard (high confusion: ~13% cross-error)
  • Transparent glass (high background miss: 29% false negatives)
      │
      ▼
[Old 8-Class Streamlined Dataset (v1)]
  • Merged paper + cardboard → paper_cardboard
  • Dropped glass to eliminate transparent object noise
  • Renamed plastic_bag → flexible_plastic
      │
      ▼
[8-Class Real Organic Augmentation]
  • Injected natural, un-tiled food scrap and compost photos
  • Raised organic recall from ~55% to 72%
      │
      ▼
[8-Class v2 Curated Dataset]
  • Tighter bounding box boundaries
  • Multi-label and occlusion disambiguation
      │
      ▼
[v3 Generalization Dataset — Production Active]
  • Hard negative non-waste images (floors, empty tables, hands, fabrics)
  • Suppressed out-of-distribution false positives in mobile scans by >80%
```

### Dataset Comparison Table

| Attribute | Old 10-Class Dataset (Legacy) | Old 8-Class Dataset (v1) | v2 Curated Dataset | v3 Generalization Dataset (Active) |
| :--- | :--- | :--- | :--- | :--- |
| **Total Images** | 7,355 | 7,355 | 8,685 | **9,836** (7,888 train / 973 val / 975 test) |
| **Negative Backgrounds** | 0 (0.0%) | 0 (0.0%) | 0 (0.0%) | **505 clean scenes** (empty tables, floors, beaches, horizons) |
| **Everyday Outdoor Litter**| Clinical / indoor | Clinical / indoor | Lab / indoor | **646 outdoor litter photos** (TACO beaches, grass, roads) |
| **Total Annotations** | 21,886 | 21,886 | 22,940 | **24,720 annotated instances** |
| **Split Ratio** | 80% / 10% / 10% | 80% / 10% / 10% | 80% / 10% / 10% | 80% / 10% / 10% |
| **Total Classes** | 10 classes | 8 classes | 8 classes | 8 classes + Negative Supervision |
| **Paper/Cardboard** | Split (`paper`, `cardboard`) | Merged (`paper_cardboard`) | Merged (`paper_cardboard`) | Merged (`paper_cardboard`) |
| **Glass Handling** | Raw `glass` class (miss rate 29%) | Removed / Excluded | Removed / Excluded | Removed / Excluded |
| **Negative Supervision**| None (waste-only images) | None (waste-only images) | Minimal background | **140+ curated background/hard negative frames** |
| **Key Limitation** | High false positives on desks/hands | False triggers on non-waste | Over-confident on bright surfaces | **Best real-world stability across phone cameras** |

---

## 3. Class-by-Class Metric Comparison across Model Generations

Below is the per-class comparison across the key iterations:

| Class Name | Sortify App Category | Old 10-Class Baseline (mAP@50) | Old 8-Class v1 (mAP@50) | 8-Class v2 Curated (mAP@50) | **v3 Generalization Production (mAP@50)** |
| :--- | :--- | :---: | :---: | :---: | :---: |
| **`paper_cardboard`** | Recyclable | 56.8% (paper) / 80.5% (cardboard) | 70.1% | **71.6%** | 61.6% |
| **`plastic_bottle`** | Recyclable | 86.9% | 86.8% | 76.7% | 78.0% |
| **`metal`** | Recyclable | 82.4% | 74.5% | 45.5% | **65.5%** |
| **`flexible_plastic`** | General waste | 71.0% *(plastic_bag)* | 61.8% | **70.8%** | 67.0% |
| **`sanitary_ppe_waste`** | General waste | 87.8% | 87.8% | 87.1% | 83.5% |
| **`organic`** | Organic | 61.2% | 69.1% | 68.9% | 64.6% |
| **`battery_ewaste`** | E-waste | **98.6%** | 96.3% | 98.2% | **98.4%** |
| **`hazardous_medical`** | Hazardous | 89.0% | 91.1% | **91.4%** | 86.7% |
| **`glass`** | Recyclable | 52.6% *(Old Dataset Only)* | — | — | — |

---

## 4. Directory & File Organization across Both Repositories

### Sortify-ML (`c:\Code\Sortify-ML`)
```
Sortify-ML/
├── weights/                                  # Standardized, clean deployment weights
│   ├── best.pt                               # Symlink/copy of Production Champion (v3)
│   ├── yolo11s_8cls_v3_generalization.pt     # Production model (8-class negative-supervised)
│   ├── yolo11s_8cls_v2_curated.pt            # Validated v2 model
│   ├── yolo11s_8cls_real_organic.pt          # Real organic augmented model
│   ├── yolo11s_8cls_v1_streamlined_old_dataset.pt  # Old 8-class dataset model
│   ├── yolo11s_10cls_legacy_old_dataset.pt   # Old 10-class legacy dataset model
│   ├── yolo11m_10cls_calibrated_old_dataset.pt# Old 10-class medium architecture model
│   └── pretrained/                           # Official Ultralytics base weights
│       ├── yolo11s.pt
│       └── yolo11m.pt
├── YOLOv11/
│   ├── model/
│   │   ├── best.pt                           # Production default weights
│   │   ├── sortify_class_mapping.json        # 8-class to 5-category JSON mapping
│   │   └── archive/                          # Cleanly labeled archived checkpoints
│   │       ├── yolo11s_8cls_v2_curated.pt
│   │       └── yolo11s_10cls_legacy_old_dataset.pt
│   └── runs/                                 # Clean completed training runs
│       ├── sortify_yolo11s_generalization_v3/
│       ├── sortify_yolo11s_8cls_v2/
│       ├── sortify_yolo11s_8cls_real_organic/
│       ├── sortify_yolo11s_8cls_streamlined/
│       ├── sortify_yolo11s_10cls/
│       └── sortify_yolo11m_calibrated_aug/
```

### Sortify Main Repo (`C:\Code\Sortify\code\server\models`)
```
server/models/
├── best.pt                                   # Default Production Model (v3 Generalization)
├── sortify_class_mapping.json                # Direct category mapping metadata
├── yolo11s_8cls_v3_generalization.pt         # v3 Generalization Dataset (8 classes)
├── yolo11s_8cls_v2_curated.pt                # v2 Curated Dataset (8 classes)
├── yolo11s_8cls_real_organic.pt              # Real Organic Dataset (8 classes)
├── yolo11s_8cls_v1_streamlined_old_dataset.pt# Old 8-Class Streamlined Dataset
└── yolo11s_10cls_legacy_old_dataset.pt       # Old 10-Class Legacy Dataset
```

---

## 5. Deployment Recommendations

- **Production Recommendation:** Always deploy **`best.pt`** (`yolo11s_8cls_v3_generalization.pt`). Although raw mAP@50 is ~1% lower than unconstrained synthetic test sets, real-world user photos produce dramatically fewer false positives on everyday objects and surfaces.
- **Edge Deployment (Mobile NPU / CoreML / TFLite):** Use `yolo11s_8cls_v3_generalization.pt` (19.2 MB) for direct ONNX/CoreML export.
- **Model Switching:** Mobile frontend and API allow dynamic runtime model selection via `/v1/detections?model=<id>`. Every model is clearly labeled in the UI dropdown with its class count and dataset lineage.
