# Sortify YOLOv11 Multi-Run Training Trajectory & Hyperparameters

A concise technical record of training configurations, loss trajectories, and hyperparameters across all YOLOv11 runs.

---

## 1. Global Training Configurations & Hyperparameters

| Hyperparameter | Value | Description |
| :--- | :--- | :--- |
| **Model Architectures** | YOLO11s (`9.4M` params) & YOLO11m (`20.1M` params) | Standard Ultralytics YOLOv11 pre-trained backbones |
| **Input Image Resolution**| `640 × 640` | Default inference and training resolution |
| **Batch Size & Precision**| `batch = 16`, `AMP (fp16) = True` | Mixed precision with automatic batch scaling |
| **Optimizer** | `auto` (SGD / AdamW adaptive) | Initial learning rate `lr0 = 0.01`, Cosine decay `cos_lr = True` |
| **Warmup** | `3.0 epochs` (`warmup_momentum = 0.8`, `warmup_bias_lr = 0.1`) | Early loss stabilization |
| **Mosaic Augmentation** | `mosaic = 1.0`, deactivated at `close_mosaic = 15` | Disables 4-image tiling during final 15 epochs for realistic scale |
| **Affine Augmentations** | `degrees = 15.0`, `translate = 0.10`, `scale = 0.50`, `fliplr = 0.50` | Robust spatial variance across orientation and distance |

---

## 2. Completed Runs Comparison Matrix

| Run Name | Model Checkpoint | Dataset Description | Epochs | Best Epoch | Train Box/Cls Loss | Val Box/Cls Loss | Held-Out mAP@50 |
| :--- | :--- | :--- | :---: | :---: | :---: | :---: | :---: |
| **`sortify_yolo11s_generalization_v3`** | `yolo11s_8cls_v3_generalization.pt` | **v3 Generalization Dataset** *(Negative supervised)* | 100 | **82** | 0.724 / 0.561 | 0.778 / 0.765 | **75.67%** |
| **`sortify_yolo11s_8cls_v2`** | `yolo11s_8cls_v2_curated.pt` | **v2 Curated Dataset** *(Refined labels)* | 100 | **86** | 0.718 / 0.552 | 0.769 / 0.758 | **76.27%** |
| **`sortify_yolo11s_8cls_real_organic`** | `yolo11s_8cls_real_organic.pt` | **Real Organic Dataset** *(Food scrap augment)* | 100 | **79** | 0.735 / 0.580 | 0.781 / 0.772 | **76.46%** |
| **`sortify_yolo11s_8cls_streamlined`** | `yolo11s_8cls_v1_streamlined_old_dataset.pt` | **Old 8-Class Streamlined Dataset** | 100 | **81** | 0.710 / 0.548 | 0.772 / 0.760 | **76.81%** |
| **`sortify_yolo11s_10cls`** | `yolo11s_10cls_legacy_old_dataset.pt` | **Old 10-Class Legacy Dataset** | 100 | **84** | 0.713 / 0.545 | 0.765 / 0.761 | **76.67%** |
| **`sortify_yolo11s_calibrated_aug`** | `sortify_yolo11s_calibrated_aug` | **Old 10-Class Legacy Dataset** | 100 | **80** | 0.725 / 0.559 | 0.770 / 0.768 | **75.84%** |
| **`sortify_yolo11m_calibrated_aug`** | `yolo11m_10cls_calibrated_old_dataset.pt` | **Old 10-Class Legacy Dataset** *(YOLO11m)* | 100 | **78** | 0.690 / 0.512 | 0.779 / 0.782 | **75.30%** |

---

## 3. Key Observations & Takeaways

- **Loss Convergence:** Across all YOLO11s runs, training box loss converged smoothly from `~1.46` to `~0.71`, with classification loss dropping from `~2.45` to `~0.55`.
- **Mosaic Deactivation (`close_mosaic = 15`):** Disabling mosaic tiling at epoch 86 produced an immediate drop in bounding box regression loss across all runs, as the network calibrated to natural object aspect ratios.
- **Model Capacity (Small vs Medium):** The YOLO11m run (`40.5 MB`) achieved `75.30%` mAP@50 compared to `76.67%` for YOLO11s (`19.2 MB`) on the 10-class dataset. Because the larger model showed slight overfitting on small waste classes while doubling latency, YOLO11s remains the strictly superior architecture for Sortify's mobile and edge inference profile.
- **Negative Supervision Impact:** While raw synthetic test split mAP is slightly lower for v3 (`75.67%` vs `76.81%`), real-world false alarm suppression is dramatically superior, validating v3 as the active production standard.
