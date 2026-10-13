# Sortify YOLOv11 — Held-Out Test Split Evaluation Report

- **Evaluated Checkpoint:** `C:\Code\Sortify-ML\YOLOv11\runs\sortify_yolo11s_10cls\weights\best.pt`
- **Overall Precision (`P`):** `0.8280`
- **Overall Recall (`R`):** `0.6884`
- **Overall `mAP@50`:** **`0.7667`**
- **Overall `mAP@50-95`:** **`0.6148`**

## Per-Class Test Metrics

| Class ID | Detector Class | Backend Disposal Category | Precision | Recall | mAP@50 | mAP@50-95 |
| :-: | :--- | :--- | --: | --: | --: | --: |
| **`0`** | **`paper`** | **Recyclable** | `0.7242` | `0.4436` | **`0.5679`** | `0.3746` |
| **`1`** | **`cardboard`** | **Recyclable** | `0.8261` | `0.7082` | **`0.8050`** | `0.6318` |
| **`2`** | **`glass`** | **Recyclable** | `0.9081` | `0.4686` | **`0.5256`** | `0.4005` |
| **`3`** | **`metal`** | **Recyclable** | `0.8332` | `0.7255` | **`0.8240`** | `0.5951` |
| **`4`** | **`plastic_bottle`** | **Recyclable** | `0.8235` | `0.7946` | **`0.8685`** | `0.7341` |
| **`5`** | **`plastic_bag`** | **General waste** | `0.6526` | `0.6375` | **`0.7096`** | `0.5659` |
| **`6`** | **`sanitary_ppe_waste`** | **General waste** | `0.8717` | `0.7796` | **`0.8779`** | `0.7464` |
| **`7`** | **`organic`** | **Organic** | `0.7567` | `0.5529` | **`0.6122`** | `0.4732` |
| **`8`** | **`battery_ewaste`** | **E-waste** | `0.9717` | `0.9497` | **`0.9864`** | `0.8769` |
| **`9`** | **`hazardous_medical`** | **Hazardous** | `0.9129` | `0.8239` | **`0.8900`** | `0.7497` |
