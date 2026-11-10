# Sortify YOLOv11 — Held-Out Test Split Evaluation Report

- **Evaluated Checkpoint:** `C:\Code\Sortify-ML\YOLOv11\runs\sortify_yolo11s_8cls_real_organic\weights\best.pt`
- **Overall Precision (`P`):** `0.7382`
- **Overall Recall (`R`):** `0.7206`
- **Overall `mAP@50`:** **`0.7646`**
- **Overall `mAP@50-95`:** **`0.6305`**

## Per-Class Test Metrics

| Class ID | Detector Class | Backend Disposal Category | Precision | Recall | mAP@50 | mAP@50-95 |
| :-: | :--- | :--- | --: | --: | --: | --: |
| **`0`** | **`paper_cardboard`** | **Recyclable** | `0.7078` | `0.5968` | **`0.6908`** | `0.5864` |
| **`1`** | **`plastic_bottle`** | **Recyclable** | `0.7748` | `0.8366` | **`0.8894`** | `0.7810` |
| **`2`** | **`metal`** | **Recyclable** | `0.6574` | `0.5961` | **`0.6506`** | `0.5618` |
| **`3`** | **`flexible_plastic`** | **General waste** | `0.5841` | `0.6098` | **`0.6137`** | `0.4873` |
| **`4`** | **`sanitary_ppe_waste`** | **General waste** | `0.7814` | `0.8354` | **`0.8770`** | `0.7485` |
| **`5`** | **`organic`** | **Organic** | `0.6084` | `0.4712` | **`0.5485`** | `0.3130` |
| **`6`** | **`battery_ewaste`** | **E-waste** | `0.9632` | `0.9231` | **`0.9501`** | `0.8251` |
| **`7`** | **`hazardous_medical`** | **Hazardous** | `0.8285` | `0.8958` | **`0.8963`** | `0.7412` |
