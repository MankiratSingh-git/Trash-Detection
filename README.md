# Sortify — AI Waste Identification & Disposal Guidance (YOLO11)

Sortify is an AI-powered waste detection and smart disposal guidance application built with **Ultralytics YOLO11** and **FastAPI**. It identifies discarded items from photos/camera feeds and maps detections to **5 municipal waste disposal categories**:
1. 🟢 **Recyclable**
2. ⚪ **General waste**
3. 🟡 **Organic**
4. 🟣 **E-waste**
5. 🔴 **Hazardous**

---

## 📊 1. Dataset Composition & Curation

The training dataset was consolidated from **5 public waste datasets** spanning `13,108` raw images and `56,162` raw bounding boxes, distilled into a balanced, leak-free **10-class YOLOv11 dataset** (`7,355` images, `21,886` bounding boxes).

### 1.1 Source Datasets Used

| # | Dataset Name | Domain / Imaging Conditions | Raw Images | Raw Instances | Raw Classes | Source Link |
| :-: | :--- | :--- | --: | --: | --: | :--- |
| **1** | **Bangladesh Recyclable Waste** (`data_canonical_75-15-10`) | Outdoor public spaces (`640×640`) | `2,842` | `6,076` | `4` | [Roboflow Universe](https://universe.roboflow.com/waste-detections-by-fu/waste-detections-c0mkz/dataset/12) |
| **2** | **MedBin Dataset 2** (`v1i.yolov11`) | Clinical, pharmaceutical & PPE waste (`640×640`) | `1,796` | `6,410` | `37` | [Roboflow Universe](https://universe.roboflow.com/medical-waste-e3vov/medbin_dataset-2) / [GitHub](https://github.com/Wayne3918/MedbinNet) |
| **3** | **Mendeley Trash Detection** (`CUSTOM_DATASET`) | Household & catalog waste (`640×640`) | `5,049` | `34,082` | `6` | [Mendeley Data (`z732f9pwxt/1`)](https://data.mendeley.com/datasets/z732f9pwxt/1) |
| **4** | **PackWISE Packaging Waste** | Industrial sorting belt (`4096×4096`) | `586` | `6,713` | `23` | [Fraunhofer Fordatis (`463.2`)](https://fordatis.fraunhofer.de/handle/fordatis/463.2?locale=en) |
| **5** | **RecyBat24 Battery Dataset** | Waste batteries (`110` physical batteries, `640×640`) | `2,835` | `2,881` | `3` | [Zenodo (`13126689`)](https://zenodo.org/records/13126689) |
| — | **Total Raw Pool** | — | **`13,108`** | **`56,162`** | **`73`** | — |

### 1.2 Format Standardization & Curation Methodology

1. **Polygon-to-Box Conversion:** Standardized `80` mixed label files across Bangladesh, MedBin, and Mendeley where polygon lines triggered bounding-box coordinate corruption in Ultralytics.
2. **COCO JSON to YOLO Conversion:** Converted PackWISE and RecyBat24 annotations to standard YOLO format (avoiding RecyBat24's $3\times$ triplicated segmentation file).
3. **Strict `Specific → General` Hierarchy:**
   - **Dropped Bangladesh `Paper` (`1,575`):** Mixed cardboard boxes and paper into one class. Excluded to keep `paper` (`0`) and `cardboard` (`1`) strictly distinct.
   - **Dropped Mendeley `PLASTIC` (`3,950`) & Bangladesh `Plastic waste` (`1,830`):** Broad catch-all classes mixing rigid containers and flexible shopping bags.
   - **Dropped Ambiguous & Specialized Labware (`40` classes total):** Excluded `aluminium-coated-foil`, `undefined` edge slivers, `beverage-carton` Tetra Paks, and 15 clinical lab instruments.
4. **Pure-Image Filtering:** Excluded multi-object images containing dropped classes so that unannotated items never appear as false background negatives.
5. **Geometry & Density Filtering:** Removed full-image macro/texture crops ($>90\%$ width & height) and capped produce collages ($>15$ boxes/image in Mendeley `BIODEGRADABLE`), reducing `organic` from `19,019` noisy micro-boxes to `2,446` clean instances.
6. **Leak-Free Object Grouping:** Grouped RecyBat24 by the **`110` physical batteries** so no battery appears in both train and validation/test splits.

### 1.3 Final 10-Class Dataset Distribution (`Dataset/`)

| ID | Class Name | Sortify Backend Category | Train Boxes (`80%`) | Val Boxes (`10%`) | Test Boxes (`10%`) | **Total Boxes** | **Total Images** |
| :-: | :--- | :--- | --: | --: | --: | --: | --: |
| **`0`** | **`paper`** | **Recyclable** | `1,967` | `272` | `257` | **`2,496`** | `879` |
| **`1`** | **`cardboard`** | **Recyclable** | `2,391` | `244` | `228` | **`2,863`** | `1,013` |
| **`2`** | **`glass`** | **Recyclable** | `1,653` | `232` | `318` | **`2,203`** | `860` |
| **`3`** | **`metal`** | **Recyclable** | `1,989` | `282` | `204` | **`2,475`** | `915` |
| **`4`** | **`plastic_bottle`** | **Recyclable** | `1,355` | `177` | `185` | **`1,717`** | `1,081` |
| **`5`** | **`plastic_bag`** | **General waste** | `2,348` | `315` | `309` | **`2,972`** | `1,049` |
| **`6`** | **`sanitary_ppe_waste`** | **General waste** | `1,431` | `171` | `186` | **`1,788`** | `1,015` |
| **`7`** | **`organic`** | **Organic** | `1,990` | `276` | `180` | **`2,446`** | `618` |
| **`8`** | **`battery_ewaste`** | **E-waste** | `1,184` | `139` | `159` | **`1,482`** | `1,446` |
| **`9`** | **`hazardous_medical`** | **Hazardous** | `1,166` | `136` | `142` | **`1,444`** | `983` |
| — | **TOTAL** | **All 5 Categories** | **`17,474`** | **`2,244`** | **`2,168`** | **`21,886`** | **`7,355`** |

> **Class Balance:** Instance counts range from **`1,444`** (`hazardous_medical`) to **`2,972`** (`plastic_bag`) — a tight **`2.06 : 1`** max-to-min balance ratio across all 10 classes (down from `1,358 : 1` in the raw data).

---

## 📈 2. Model Performance & Test Results

The detector was trained using **Ultralytics YOLO11s** (`yolo11s.pt` baseline) for **100 epochs** on `Dataset/data.yaml` with multi-domain augmentations (`mosaic=1.0`, `close_mosaic=15`, `mixup=0.10`, HSV jitter, and affine transforms).

### 2.1 Overall Performance (Validation vs. Held-Out Test Split)

| Split | Images | Instances | Precision (`P`) | Recall (`R`) | **`mAP@50`** | **`mAP@50-95`** |
| :--- | --: | --: | --: | --: | --: | --: |
| **Validation (Epoch 84 `best.pt`)** | `734` | `2,244` | `82.24%` | `68.69%` | **`76.82%`** | **`61.68%`** |
| **Held-Out Test Split (`best.pt`)** | `734` | `2,168` | `82.80%` | `68.84%` | **`76.67%`** | **`61.48%`** |
| **Generalization Delta** | — | — | **`+0.56%`** | **`+0.15%`** | **`-0.15%`** | **`-0.20%`** |

*The `0.15%` gap between validation and held-out test `mAP@50` demonstrates strong generalization and confirms zero data leakage across splits.*

### 2.2 Per-Class Held-Out Test Split Accuracy

| Class ID | Detector Class | Sortify Category | Test Boxes | Precision | Recall | **`mAP@50`** | **`mAP@50-95`** |
| :-: | :--- | :--- | --: | --: | --: | --: | --: |
| **`8`** | **`battery_ewaste`** | **E-waste** | `159` | `97.17%` | `94.97%` | **`98.64%`** | `87.69%` |
| **`9`** | **`hazardous_medical`** | **Hazardous** | `142` | `91.29%` | `82.39%` | **`89.00%`** | `74.97%` |
| **`6`** | **`sanitary_ppe_waste`** | **General waste** | `186` | `87.17%` | `77.96%` | **`87.79%`** | `74.64%` |
| **`4`** | **`plastic_bottle`** | **Recyclable** | `185` | `82.35%` | `79.46%` | **`86.85%`** | `73.41%` |
| **`3`** | **`metal`** | **Recyclable** | `204` | `83.32%` | `72.55%` | **`82.40%`** | `59.51%` |
| **`1`** | **`cardboard`** | **Recyclable** | `228` | `82.61%` | `70.82%` | **`80.50%`** | `63.18%` |
| **`5`** | **`plastic_bag`** | **General waste** | `309` | `65.26%` | `63.75%` | **`70.96%`** | `56.59%` |
| **`7`** | **`organic`** | **Organic** | `180` | `75.67%` | `55.29%` | **`61.22%`** | `47.32%` |
| **`0`** | **`paper`** | **Recyclable** | `257` | `72.42%` | `44.36%` | **`56.79%`** | `37.46%` |
| **`2`** | **`glass`** | **Recyclable** | `318` | `90.81%` | `46.86%` | **`52.56%`** | `40.05%` |
| — | **OVERALL MEAN** | **All 5 Categories** | **`2,168`** | **`82.80%`** | **`68.84%`** | **`76.67%`** | **`61.48%`** |

*Model weights are stored at:* [`YOLOv11/model/best.pt`](YOLOv11/model/best.pt).

---

## 🚀 3. How to Run the Application

### 3.1 Setup Environment

Clone the repository and install dependencies:

```bash
git clone git@github.com:MankiratSingh-git/Trash-Detection.git
cd Trash-Detection

# Activate your Python virtual environment (Python 3.10+ recommended)
python -m venv .venv
# Windows:
.venv\Scripts\activate
# Linux/macOS:
source .venv/bin/activate

pip install -r src/fast-api/requirements.txt
```

### 3.2 Launch the Sortify ML Studio Web App

Start the FastAPI server (serves both the REST API and the frontend UI):

```bash
# Using the runner script:
python src/fast-api/run.py

# Or using Uvicorn directly:
python -m uvicorn main:app --app-dir src/fast-api --reload --port 8000
```

Once running, open your browser:
* **Interactive Web UI:** [http://localhost:8000/](http://localhost:8000/)
* **Interactive Swagger API Docs:** [http://localhost:8000/docs](http://localhost:8000/docs)
* **ReDoc API Documentation:** [http://localhost:8000/redoc](http://localhost:8000/redoc)

### 3.3 Recommended Inference Settings in the Web UI
* **Confidence Threshold:** `0.25` to `0.35` *(setting below 0.20 exposes faint candidate background noise)*.
* **IoU (NMS) Threshold:** `0.45` to `0.50` *(standard non-maximum suppression)*.

### 3.4 Command-Line Inference & Evaluation

You can run predictions or evaluate checkpoints directly from the terminal:

```bash
# Evaluate best.pt on the held-out test split:
python YOLOv11/model.py --eval-only YOLOv11/model/best.pt

# Run prediction on an image or directory:
python YOLOv11/model.py --eval-only YOLOv11/model/best.pt --predict path/to/image.jpg --conf 0.30
```

---

## 📁 4. Project Directory Structure

```text
├── Dataset/                     # Unified 10-class YOLOv11 dataset (7,355 images)
│   ├── data.yaml                # YOLOv11 dataset configuration
│   ├── images/{train,val,test}  # 80% / 10% / 10% stratified image splits
│   └── labels/{train,val,test}  # 5-column YOLO bounding box labels
├── Preprocessing/               # Preprocessing & dataset builder scripts
│   ├── make_yolov11_compatible.py
│   └── build_sortify_dataset.py
├── Summaries/                   # Analysis scripts & audit reports
│   ├── Results/
│   │   ├── Preprocessing_Report.md
│   │   ├── YOLOv11_Run_Summary.md
│   │   └── final_dataset_summary.md
│   └── Scripts/
├── YOLOv11/                     # Model scripts & checkpoints
│   ├── model.py                 # Training, evaluation & inference script
│   └── model/
│       ├── best.pt              # Trained 10-class model checkpoint (Epoch 84)
│       └── sortify_class_mapping.json
└── src/
    ├── fast-api/                # Modular FastAPI backend & routers
    └── frontend/                # Sortify ML Studio web interface
```

---

## 📄 Documentation Reports

* **Full Preprocessing & Curation Audit:** [`Summaries/Results/Preprocessing_Report.md`](Summaries/Results/Preprocessing_Report.md)
* **Training & Epoch Progress Report:** [`Summaries/Results/YOLOv11_Run_Summary.md`](Summaries/Results/YOLOv11_Run_Summary.md)
