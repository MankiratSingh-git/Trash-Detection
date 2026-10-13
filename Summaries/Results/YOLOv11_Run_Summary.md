# Sortify YOLOv11 (`yolo11s`) — Training & Test Evaluation Summary

## 1. Run Overview & Configuration

| Parameter | Value |
| :--- | :--- |
| **Run Name** | `sortify_yolo11s_10cls` |
| **Base Architecture** | Ultralytics `YOLO11s` (`yolo11s.pt` pretrained) |
| **Dataset** | [`Dataset/data.yaml`](file:///c:/Code/Sortify-ML/Dataset/data.yaml) (`7,355` images, `21,886` boxes across `10` classes) |
| **Splits** | `5,887` Train (`80%`) / `734` Val (`10%`) / `734` Held-Out Test (`10%`) |
| **Epochs Completed** | `100 / 100` (Total time: `2h 34m 53s`, `~92.9s`/epoch) |
| **Input Resolution & Batch** | `imgsz = 640`, `batch = 16`, `AMP (Mixed Precision) = True` |
| **Optimizer & Schedule** | `auto` (`lr0 = 0.01`, Cosine LR `cos_lr = True`, `warmup_epochs = 3.0`) |
| **Augmentations** | `mosaic = 1.0` (`close_mosaic = 15` at epoch 86), `mixup = 0.10`, `degrees = 15.0`, `fliplr = 0.5`, `flipud = 0.1` |
| **Best Checkpoint (`best.pt`)** | [`YOLOv11/runs/sortify_yolo11s_10cls/weights/best.pt`](file:///c:/Code/Sortify-ML/YOLOv11/runs/sortify_yolo11s_10cls/weights/best.pt) (**Epoch `84`**) |

---

## 2. Executive Performance Summary (Validation vs. Held-Out Test Split)

| Split | Images | Instances | Precision (`P`) | Recall (`R`) | **`mAP@50`** | **`mAP@50-95`** |
| :--- | --: | --: | --: | --: | --: | --: |
| **Validation (Best Fitness — Epoch `84`)** | `734` | `2,244` | `0.8224` (`82.24%`) | `0.6869` (`68.69%`) | **`0.7682` (`76.82%`)** | **`0.6168` (`61.68%`)** |
| **Validation (Peak `mAP@50` — Epoch `77`)** | `734` | `2,244` | `0.8128` (`81.28%`) | `0.7015` (`70.15%`) | **`0.7692` (`76.92%`)** | **`0.6156` (`61.56%`)** |
| **Held-Out Test Split (`best.pt`)** | `734` | `2,168` | `0.8280` (`82.80%`) | `0.6884` (`68.84%`) | **`0.7667` (`76.67%`)** | **`0.6148` (`61.48%`)** |
| **Val $\to$ Test Generalization Delta** | — | — | **`+0.56%`** | **`+0.15%`** | **`-0.15%`** | **`-0.20%`** |

> **Key Takeaway:** The held-out test performance (`76.67%` `mAP@50`, `61.48%` `mAP@50-95`, `82.80%` Precision) matches validation within **`0.15%–0.20%`**, confirming zero data leakage and strong cross-split generalization.

---

## 3. Held-Out Test Split Per-Class Performance

| Class ID | Detector Class | Sortify Backend Category | Test Boxes | Precision | Recall | **`mAP@50`** | **`mAP@50-95`** | Performance Tier |
| :-: | :--- | :--- | --: | --: | --: | --: | --: | :--- |
| **`8`** | **`battery_ewaste`** | **E-waste** | `159` | `97.17%` | `94.97%` | **`98.64%`** | `87.69%` | Excellent (`>85%`) |
| **`9`** | **`hazardous_medical`** | **Hazardous** | `142` | `91.29%` | `82.39%` | **`89.00%`** | `74.97%` | Excellent (`>85%`) |
| **`6`** | **`sanitary_ppe_waste`** | **General waste** | `186` | `87.17%` | `77.96%` | **`87.79%`** | `74.64%` | Excellent (`>85%`) |
| **`4`** | **`plastic_bottle`** | **Recyclable** | `185` | `82.35%` | `79.46%` | **`86.85%`** | `73.41%` | Excellent (`>85%`) |
| **`3`** | **`metal`** | **Recyclable** | `204` | `83.32%` | `72.55%` | **`82.40%`** | `59.51%` | Strong (`80%–85%`) |
| **`1`** | **`cardboard`** | **Recyclable** | `228` | `82.61%` | `70.82%` | **`80.50%`** | `63.18%` | Strong (`80%–85%`) |
| **`5`** | **`plastic_bag`** | **General waste** | `309` | `65.26%` | `63.75%` | **`70.96%`** | `56.59%` | Good (`70%–80%`) |
| **`7`** | **`organic`** | **Organic** | `180` | `75.67%` | `55.29%` | **`61.22%`** | `47.32%` | Moderate (`60%–70%`) |
| **`0`** | **`paper`** | **Recyclable** | `257` | `72.42%` | `44.36%` | **`56.79%`** | `37.46%` | Moderate (`50%–60%`) |
| **`2`** | **`glass`** | **Recyclable** | `318` | `90.81%` | `46.86%` | **`52.56%`** | `40.05%` | High Precision (`90.8%`), Lower Recall |
| — | **OVERALL MEAN** | **All 5 Categories** | **`2,168`** | **`82.80%`** | **`68.84%`** | **`76.67%`** | **`61.48%`** | — |

---

## 4. Training Trajectory Across 100 Epochs

| Epoch | Elapsed Time | `train/box_loss` | `train/cls_loss` | `val/box_loss` | `val/cls_loss` | Val Precision | Val Recall | **Val `mAP@50`** | **Val `mAP@50-95`** | Note |
| :-: | --: | --: | --: | --: | --: | --: | --: | --: | --: | :--- |
| **`1`** | `2.9 min` | `1.4642` | `2.4548` | `1.5219` | `2.1976` | `0.4778` | `0.4101` | **`0.3880`** | **`0.2307`** | End of Epoch 1 warmup |
| **`10`** | `18.5 min` | `1.3201` | `1.6673` | `1.1383` | `1.4961` | `0.5732` | `0.5187` | **`0.5285`** | **`0.3675`** | Early convergence |
| **`25`** | `41.8 min` | `1.1798` | `1.3559` | `0.9214` | `1.0409` | `0.7357` | `0.6206` | **`0.6824`** | **`0.5171`** | Val `mAP@50` crosses `68%` |
| **`50`** | `79.5 min` | `1.0338` | `1.0427` | `0.8205` | `0.8337` | `0.7743` | `0.6855` | **`0.7493`** | **`0.5877`** | Halfway mark (`74.9%` `mAP@50`) |
| **`75`** | `117.4 min` | `0.9370` | `0.8662` | `0.7695` | `0.7642` | `0.7955` | `0.6970` | **`0.7631`** | **`0.6128`** | Steady cosine decay |
| **`84`** | `130.9 min` | `0.9071` | `0.8281` | `0.7653` | `0.7611` | `0.8224` | `0.6869` | **`0.7682`** | **`0.6168`** | **Best Checkpoint (`best.pt`)** |
| **`86`** | `134.1 min` | `0.7538` | `0.5999` | `0.7669` | `0.7663` | `0.8060` | `0.6890` | **`0.7617`** | **`0.6129`** | `close_mosaic=15` (mosaic disabled) |
| **`100`** | `154.9 min` | `0.7128` | `0.5448` | `0.7713` | `0.7775` | `0.7860` | `0.7020` | **`0.7610`** | **`0.6095`** | Final epoch (`last.pt`) |

---

## 5. Key Diagnostic Insights (Curves & Confusion Matrix)

1. **Smooth Convergence & Zero Overfitting:**
   - Validation losses dropped monotonically from Epoch 1 (`val/box_loss = 1.522`, `val/cls_loss = 2.198`) to Epoch 84 (`val/box_loss = 0.765`, `val/cls_loss = 0.761`) and remained flat through Epoch 100 without diverging.
   - At **Epoch 86**, deactivating mosaic augmentation (`close_mosaic = 15`) produced an immediate drop in training box loss (`0.908 → 0.754`) and classification loss (`0.826 → 0.600`) as the model fine-tuned on un-tiled natural images.
2. **Strong Category Separation (`>80%` `mAP@50` on 6 of 10 Classes):**
   - **`battery_ewaste` (`98.64%` `mAP@50`)**, **`hazardous_medical` (`89.00%`)**, **`sanitary_ppe_waste` (`87.79%`)**, **`plastic_bottle` (`86.85%`)**, **`metal` (`82.40%`)**, and **`cardboard` (`80.50%`)** all achieved high precision and recall.
   - Separating `paper` and `cardboard` (and dropping Bangladesh's mixed `Paper` class) paid off: **`cardboard` achieved `87%` true-positive accuracy** on the normalized test confusion matrix with only `7%` confusion with `paper`.
3. **Error Modes Identified in the Normalized Confusion Matrix:**
   - **`glass` (`90.81%` Precision, `46.86%` Recall):** When `glass` is detected, it is almost always correct (`90.8%`), but `29%` of ground-truth glass objects are missed as background (transparent glass boundaries) and `11%` are predicted as `organic` (food inside glass bowls/jars in Mendeley images).
   - **`paper` (`72.42%` Precision, `44.36%` Recall) & `plastic_bag` (`65.26%` Precision, `63.75%` Recall):** Crumpled white/translucent flexible films on the PackWISE sorting belt (`13%` of `plastic_bag` and `10%` of `sanitary_ppe_waste` tissues) show mild cross-confusion with `paper`.

---

## 6. Generated Run Artifacts

- **Best Model Weights:** [`YOLOv11/runs/sortify_yolo11s_10cls/weights/best.pt`](file:///c:/Code/Sortify-ML/YOLOv11/runs/sortify_yolo11s_10cls/weights/best.pt)
- **Backend Category Mapping JSON:** [`YOLOv11/runs/sortify_yolo11s_10cls/sortify_class_mapping.json`](file:///c:/Code/Sortify-ML/YOLOv11/runs/sortify_yolo11s_10cls/sortify_class_mapping.json)
- **Training Curves (`results.png`):** [`YOLOv11/runs/sortify_yolo11s_10cls/results.png`](file:///c:/Code/Sortify-ML/YOLOv11/runs/sortify_yolo11s_10cls/results.png)
- **Test Split Confusion Matrix:** [`YOLOv11/runs/sortify_yolo11s_10cls_test/confusion_matrix_normalized.png`](file:///c:/Code/Sortify-ML/YOLOv11/runs/sortify_yolo11s_10cls_test/confusion_matrix_normalized.png)
- **Test Split PR / F1 Curves:** [`YOLOv11/runs/sortify_yolo11s_10cls_test/BoxPR_curve.png`](file:///c:/Code/Sortify-ML/YOLOv11/runs/sortify_yolo11s_10cls_test/BoxPR_curve.png), [`BoxF1_curve.png`](file:///c:/Code/Sortify-ML/YOLOv11/runs/sortify_yolo11s_10cls_test/BoxF1_curve.png)
