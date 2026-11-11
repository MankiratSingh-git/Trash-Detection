# Sortify YOLOv11 — Held-Out Test Split Evaluation Report

- **Evaluated Checkpoint:** `C:\Code\Sortify-ML\YOLOv11\runs\sortify_yolo11s_generalization_v3\weights\best.pt`
- **Overall Precision (`P`):** `0.7908`
- **Overall Recall (`R`):** `0.6950`
- **Overall `mAP@50`:** **`0.7567`**
- **Overall `mAP@50-95`:** **`0.6141`**

## Per-Class Test Metrics

| Class ID | Detector Class | Backend Disposal Category | Precision | Recall | mAP@50 | mAP@50-95 |
| :-: | :--- | :--- | --: | --: | --: | --: |
| **`0`** | **`paper_cardboard`** | **Recyclable** | `0.6825` | `0.5013` | **`0.6163`** | `0.5189` |
| **`1`** | **`plastic_bottle`** | **Recyclable** | `0.7862` | `0.7429` | **`0.7795`** | `0.6550` |
| **`2`** | **`metal`** | **Recyclable** | `0.7131` | `0.6325` | **`0.6550`** | `0.5429` |
| **`3`** | **`flexible_plastic`** | **General waste** | `0.7266` | `0.5870` | **`0.6700`** | `0.5024` |
| **`4`** | **`sanitary_ppe_waste`** | **General waste** | `0.8426` | `0.7553` | **`0.8352`** | `0.6935` |
| **`5`** | **`organic`** | **Organic** | `0.7552` | `0.5198` | **`0.6462`** | `0.3711` |
| **`6`** | **`battery_ewaste`** | **E-waste** | `0.9627` | `0.9739` | **`0.9839`** | `0.8917` |
| **`7`** | **`hazardous_medical`** | **Hazardous** | `0.8574` | `0.8471` | **`0.8670`** | `0.7377` |
