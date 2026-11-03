# Sortify YOLOv11 — Held-Out Test Split Evaluation Report

- **Evaluated Checkpoint:** `C:\Code\Trash-Detection\YOLOv11\model\best.pt`
- **Overall Precision (`P`):** `0.7983`
- **Overall Recall (`R`):** `0.7087`
- **Overall `mAP@50`:** **`0.7696`**
- **Overall `mAP@50-95`:** **`0.6423`**

## Per-Class Test Metrics

| Class ID | Detector Class | Backend Disposal Category | Precision | Recall | mAP@50 | mAP@50-95 |
| :-: | :--- | :--- | --: | --: | --: | --: |
| **`0`** | **`paper_cardboard`** | **Recyclable** | `0.6982` | `0.4412` | **`0.5457`** | `0.4577` |
| **`1`** | **`plastic_bottle`** | **Recyclable** | `0.8184` | `0.8084` | **`0.8492`** | `0.7032` |
| **`2`** | **`metal`** | **Recyclable** | `0.7059` | `0.5939` | **`0.7006`** | `0.5878` |
| **`3`** | **`flexible_plastic`** | **General waste** | `0.6417` | `0.5585` | **`0.6184`** | `0.4687` |
| **`4`** | **`sanitary_ppe_waste`** | **General waste** | `0.8794` | `0.8235` | **`0.8781`** | `0.7485` |
| **`5`** | **`organic`** | **Organic** | `0.8037` | `0.5910` | **`0.6910`** | `0.5100` |
| **`6`** | **`battery_ewaste`** | **E-waste** | `0.9563` | `0.9613` | **`0.9630`** | `0.8494` |
| **`7`** | **`hazardous_medical`** | **Hazardous** | `0.8830` | `0.8921` | **`0.9108`** | `0.8135` |
