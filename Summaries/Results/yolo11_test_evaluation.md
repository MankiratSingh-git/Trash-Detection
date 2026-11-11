# Sortify YOLOv11 — Held-Out Test Split Evaluation Report

- **Evaluated Checkpoint:** `C:\Code\Sortify-ML\YOLOv11\runs\sortify_yolo11s_8cls_v2\weights\best.pt`
- **Overall Precision (`P`):** `0.7915`
- **Overall Recall (`R`):** `0.7016`
- **Overall `mAP@50`:** **`0.7627`**
- **Overall `mAP@50-95`:** **`0.6227`**

## Per-Class Test Metrics

| Class ID | Detector Class | Backend Disposal Category | Precision | Recall | mAP@50 | mAP@50-95 |
| :-: | :--- | :--- | --: | --: | --: | --: |
| **`0`** | **`paper_cardboard`** | **Recyclable** | `0.7353` | `0.6254` | **`0.7161`** | `0.6169` |
| **`1`** | **`plastic_bottle`** | **Recyclable** | `0.7855` | `0.6847` | **`0.7668`** | `0.6613` |
| **`2`** | **`metal`** | **Recyclable** | `0.6624` | `0.3980` | **`0.4549`** | `0.3459` |
| **`3`** | **`flexible_plastic`** | **General waste** | `0.7129` | `0.6673` | **`0.7077`** | `0.5396` |
| **`4`** | **`sanitary_ppe_waste`** | **General waste** | `0.8233` | `0.8315` | **`0.8706`** | `0.7569` |
| **`5`** | **`organic`** | **Organic** | `0.7507` | `0.5856` | **`0.6894`** | `0.4074` |
| **`6`** | **`battery_ewaste`** | **E-waste** | `0.9552` | `0.9677` | **`0.9823`** | `0.8676` |
| **`7`** | **`hazardous_medical`** | **Hazardous** | `0.9068` | `0.8525` | **`0.9140`** | `0.7858` |
