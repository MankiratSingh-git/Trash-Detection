# Sortify YOLOv11 — Held-Out Test Split Evaluation Report

- **Evaluated Production Checkpoint:** `yolo11s_8cls_v3_generalization.pt` (`YOLOv11/model/best.pt`)
- **Dataset:** v3 Generalization Dataset (8 classes + Hard Negative Supervision)
- **Overall Precision (`P`):** `0.7908` (`79.08%`)
- **Overall Recall (`R`):** `0.6950` (`69.50%`)
- **Overall `mAP@50`:** **`0.7567`** (`75.67%`)
- **Overall `mAP@50-95`:** **`0.6141`** (`61.41%`)

---

## 1. Production Model Per-Class Metrics

| Class ID | Detector Class | Backend Disposal Category | Precision | Recall | mAP@50 | mAP@50-95 | Performance Tier |
| :-: | :--- | :--- | --: | --: | --: | --: | :--- |
| **`6`** | **`battery_ewaste`** | **E-waste** | `0.9627` | `0.9739` | **`0.9839`** | `0.8917` | Outstanding (`>95%`) |
| **`7`** | **`hazardous_medical`** | **Hazardous** | `0.8574` | `0.8471` | **`0.8670`** | `0.7377` | Excellent (`>85%`) |
| **`4`** | **`sanitary_ppe_waste`** | **General waste** | `0.8426` | `0.7553` | **`0.8352`** | `0.6935` | Strong (`80%–85%`) |
| **`1`** | **`plastic_bottle`** | **Recyclable** | `0.7862` | `0.7429` | **`0.7795`** | `0.6550` | Strong (`75%–80%`) |
| **`3`** | **`flexible_plastic`** | **General waste** | `0.7266` | `0.5870` | **`0.6700`** | `0.5024` | Good (`65%–70%`) |
| **`2`** | **`metal`** | **Recyclable** | `0.7131` | `0.6325` | **`0.6550`** | `0.5429` | Good (`65%–70%`) |
| **`5`** | **`organic`** | **Organic** | `0.7552` | `0.5198` | **`0.6462`** | `0.3711` | Moderate (`60%–65%`) |
| **`0`** | **`paper_cardboard`** | **Recyclable** | `0.6825` | `0.5013` | **`0.6163`** | `0.5189` | Moderate (`60%–65%`) |
| — | **OVERALL MEAN** | **All 5 Categories** | **`0.7908`** | **`0.6950`** | **`0.7567`** | **`0.6141`** | **Production Ready** |

---

## 2. Cross-Model Benchmark Comparison (All Evaluated Runs)

| Model Key | Classes | Dataset Tag | P | R | mAP@50 | mAP@50-95 | Notes |
| :--- | :---: | :--- | :---: | :---: | :---: | :---: | :--- |
| **`yolo11s_8cls_v3_generalization`** *(Active)* | 8 | v3 Generalization | 79.08% | 69.50% | **75.67%** | 61.41% | **Lowest false positive rate on mobile background/hands** |
| `yolo11s_8cls_v2_curated` | 8 | v2 Curated | 79.15% | 70.16% | 76.27% | 62.27% | Refined bounding boxes |
| `yolo11s_8cls_real_organic` | 8 | Real Organic Aug | 73.82% | 72.06% | 76.46% | 63.05% | High recall on food scraps |
| `yolo11s_8cls_v1_streamlined_old_dataset` | 8 | Old 8-Class Dataset | 79.83% | 70.87% | 76.81% | 63.58% | Baseline streamlined model |
| `yolo11s_10cls_legacy_old_dataset` | 10 | Old 10-Class Legacy | 82.80% | 68.84% | 76.67% | 61.48% | Includes separate paper/cardboard & glass |
| `yolo11s_calibrated_aug` | 10 | Old 10-Class Legacy | 80.72% | 68.90% | 75.84% | 60.73% | Calibrated augmentations |
| `yolo11m_10cls_calibrated_old_dataset` | 10 | Old 10-Class Legacy | 79.84% | 69.04% | 75.30% | 61.01% | Medium architecture (40.5 MB) |
