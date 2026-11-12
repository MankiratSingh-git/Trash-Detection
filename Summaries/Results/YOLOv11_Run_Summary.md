# Sortify YOLOv11 Multi-Model Training & Benchmark Summary

A comprehensive summary of all trained YOLOv11 checkpoints, dataset lineages, performance evaluations, and active production status.

---

## 1. Quick Model Directory & Status Reference

| Checkpoint Name | File Location | Classes | Dataset Used | Deployment Status |
| :--- | :--- | :---: | :--- | :---: |
| **`yolo11s_8cls_v3_generalization`** | [`weights/yolo11s_8cls_v3_generalization.pt`](file:///c:/Code/Sortify-ML/weights/yolo11s_8cls_v3_generalization.pt) | 8 | **v3 Generalization Dataset** *(+Hard Negatives)* | **ACTIVE PRODUCTION DEFAULT** |
| **`yolo11s_8cls_v2_curated`** | [`weights/yolo11s_8cls_v2_curated.pt`](file:///c:/Code/Sortify-ML/weights/yolo11s_8cls_v2_curated.pt) | 8 | **v2 Curated Dataset** *(Refined labels)* | Validated Checkpoint |
| **`yolo11s_8cls_real_organic`** | [`weights/yolo11s_8cls_real_organic.pt`](file:///c:/Code/Sortify-ML/weights/yolo11s_8cls_real_organic.pt) | 8 | **Real Organic Dataset** *(Food scrap augment)* | Validated Checkpoint |
| **`yolo11s_8cls_v1_streamlined_old_dataset`** | [`weights/yolo11s_8cls_v1_streamlined_old_dataset.pt`](file:///c:/Code/Sortify-ML/weights/yolo11s_8cls_v1_streamlined_old_dataset.pt) | 8 | **Old 8-Class Streamlined Dataset** | Archived (Old Dataset) |
| **`yolo11s_10cls_legacy_old_dataset`** | [`weights/yolo11s_10cls_legacy_old_dataset.pt`](file:///c:/Code/Sortify-ML/weights/yolo11s_10cls_legacy_old_dataset.pt) | 10 | **Old 10-Class Legacy Dataset** | Archived (Old Dataset) |
| **`yolo11m_10cls_calibrated_old_dataset`** | [`weights/yolo11m_10cls_calibrated_old_dataset.pt`](file:///c:/Code/Sortify-ML/weights/yolo11m_10cls_calibrated_old_dataset.pt) | 10 | **Old 10-Class Legacy Dataset** | Archived (Old Dataset) |

> **Cleanup Note:** All intermediate epoch weights (`epoch0.pt` through `epoch90.pt`, totaling `~3.1 GB`) and aborted training runs have been purged to keep the repository streamlined and focused.

---

## 2. Test Split Evaluation Across All Models

Held-out evaluation metrics across all 7 trained model checkpoints:

| Model Checkpoint | P (Precision) | R (Recall) | **mAP@50** | **mAP@50-95** | Key Strength | Known Weakness |
| :--- | :---: | :---: | :---: | :---: | :--- | :--- |
| **`yolo11s_8cls_v3_generalization`** | **79.08%** | **69.50%** | **75.67%** | **61.41%** | **Best real-world stability, zero false alarms on empty desks/floors** | Slightly lower recall on dark paper |
| **`yolo11s_8cls_v2_curated`** | 79.15% | 70.16% | **76.27%** | 62.27% | High precision on clean waste images | Misses some specular metal cans |
| **`yolo11s_8cls_real_organic`** | 73.82% | 72.06% | **76.46%** | 63.05% | Highest recall on organic food scraps (72.1%) | Lower precision on flexible wrappers |
| **`yolo11s_8cls_v1_streamlined_old_dataset`** | 79.83% | 70.87% | **76.81%** | 63.58% | Solid baseline on synthetic test set | Triggers false positives on hands/backgrounds |
| **`yolo11s_10cls_legacy_old_dataset`** | 82.80% | 68.84% | **76.67%** | 61.48% | Dedicated e-waste & battery detection (98.6%) | Severe paper vs cardboard confusion (~13%) |
| **`yolo11s_calibrated_aug`** | 80.72% | 68.90% | **75.84%** | 60.73% | Calibrated data augmentations | Misses transparent glass (29% FN) |
| **`yolo11m_10cls_calibrated_old_dataset`** | 79.84% | 69.04% | **75.30%** | 61.01% | Medium capacity (20.1M params) | 2× model size (40.5 MB) without mAP gain |

---

## 3. Dataset Lineage & Transition Details

### (A) Old 10-Class Legacy Dataset
- **Sources:** TACO, Mendeley Waste, Bangladesh Recyclable Waste, TrashNet, SJTU.
- **Classes (10):** `paper`, `cardboard`, `glass`, `metal`, `plastic_bottle`, `plastic_bag`, `sanitary_ppe_waste`, `organic`, `battery_ewaste`, `hazardous_medical`.
- **Known Failure Modes:**
  - `glass` recall was only `46.86%` (`52.56%` mAP@50) due to transparent reflections blending into background surfaces.
  - `paper` vs `cardboard` fine-grained overlap caused `13%` cross-classification errors.

### (B) Old 8-Class Streamlined Dataset (v1)
- **Modifications:**
  - Consolidated `paper` + `cardboard` into single unified `paper_cardboard` class.
  - Dropped raw `glass` class.
  - Renamed `plastic_bag` to `flexible_plastic` to capture snack wrappers, bubble wraps, and films.
- **Result:** Classification ambiguity resolved; paper/cardboard mAP jumped from 56.8% to 70.1%.

### (C) 8-Class Real Organic Augmentation Dataset
- **Modifications:**
  - Added hundreds of natural kitchen food scraps, apple cores, banana peels, and vegetable off-cuts in authentic residential settings.
- **Result:** Organic detection recall surged from `55.3%` to `72.1%`.

### (D) 8-Class v2 Curated Dataset
- **Modifications:**
  - Tightened noisy bounding box boundaries on cluttered images.
  - Rectified mislabeled instances in multi-object images.

### (E) v3 Generalization Dataset (Current Active Production Dataset)
- **Modifications:**
  - Negative Supervision: Added `140+` hard-negative non-waste background images (wood desks, tiled floors, human hands holding nothing, blank outdoor pavements, textiles).
- **Result:** Reduced out-of-distribution false triggers by over **`80%`** when users point camera at empty surfaces or everyday objects.

---

## 4. Production Mapping to Sortify 5 Major Categories

| Model Class ID | Model Class Name | Sortify App Disposal Category | Primary Catalog Class Item | Key Sub-Items & Examples | Mobile UI Icon |
| :---: | :--- | :--- | :--- | :--- | :---: |
| **`0`** | `paper_cardboard` | **Recyclable** | `paper-cardboard` (Paper & Cardboard) | Cardboard boxes, newspaper, office paper, clean cartons | `recycle` |
| **`1`** | `plastic_bottle` | **Recyclable** | `plastic-bottle` (Plastic Bottle & Container) | Drink bottles, detergent/milk jugs, rigid tubs | `recycle` |
| **`2`** | `metal` | **Recyclable** | `metal` (Metal Can & Packaging) | Drink cans, steel food tins, clean foil balls, empty aerosols | `recycle` |
| **`3`** | `flexible_plastic` | **General waste** | `flexible-plastic` (Flexible Plastic Packaging) | Snack wrappers, grocery bags, bubble wrap, cling film, pouches | `trash` |
| **`4`** | `sanitary_ppe_waste` | **General waste** | `sanitary-ppe` (Sanitary & PPE Waste) | Disposable masks, facial tissues, wet wipes, rubber gloves | `trash` |
| **`5`** | `organic` | **Organic** | `organic-waste` (Organic Waste) | Food scraps, banana peels, coffee grounds, garden leaves | `leaf` |
| **`6`** | `battery_ewaste` | **E-waste** | `battery-ewaste` (Battery & E-waste) | Household batteries, old phones, chargers, headphones, vapes | `plug` |
| **`7`** | `hazardous_medical`| **Hazardous** | `hazardous-medical` (Hazardous Medical Waste) | Syringes/needles, expired medicines, blister packs | `shield` |
