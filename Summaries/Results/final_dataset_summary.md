# Sortify Unified YOLOv11 Dataset — Final Build Summary & Verification Report

## 1. Dataset Overview & YOLOv11 Compatibility Status

- **Dataset Location:** [`Dataset/`](file:///c:/Code/Sortify-ML/Dataset)
- **Configuration File:** [`Dataset/data.yaml`](file:///c:/Code/Sortify-ML/Dataset/data.yaml)
- **Preprocessing Script:** [`Preprocessing/build_sortify_dataset.py`](file:///c:/Code/Sortify-ML/Preprocessing/build_sortify_dataset.py)
- **Total Curated Images:** **`7,355`** (`train`: `5,887`, `val`: `734`, `test`: `734`)
- **Total Bounding Boxes:** **`21,886`** (`train`: `17,474`, `val`: `2,244`, `test`: `2,168`)
- **1-to-1 Image/Label Pair Check:** `PASS` (`0` unmatched images, `0` unmatched labels)
- **5-Column YOLO Bounding-Box Geometry Check:** `PASS` (`0` malformed lines, `0` out-of-bounds boxes)
- **Cross-Split Data Leakage Check:** `PASS` (`0` duplicated Roboflow stems across splits; `0` physical batteries shared across `train`/`val`/`test`)

---

## 2. Class-by-Class Instance & Image Distribution Across Splits

| Class ID | Detector Class | Backend Category | Train Boxes | Val Boxes | Test Boxes | **Total Boxes** | Train Imgs | Val Imgs | Test Imgs | **Total Imgs** |
| :-: | :--- | :--- | --: | --: | --: | --: | --: | --: | --: | --: |
| **`0`** | **`paper`** | **Recyclable** | `1,967` | `272` | `257` | **`2,496`** | `703` | `88` | `88` | **`879`** |
| **`1`** | **`cardboard`** | **Recyclable** | `2,391` | `244` | `228` | **`2,863`** | `811` | `101` | `101` | **`1,013`** |
| **`2`** | **`glass`** | **Recyclable** | `1,653` | `232` | `318` | **`2,203`** | `688` | `86` | `86` | **`860`** |
| **`3`** | **`metal`** | **Recyclable** | `1,989` | `282` | `204` | **`2,475`** | `730` | `91` | `94` | **`915`** |
| **`4`** | **`plastic_bottle`** | **Recyclable** | `1,355` | `177` | `185` | **`1,717`** | `869` | `107` | `105` | **`1,081`** |
| **`5`** | **`plastic_bag`** | **General waste (or Film Drop-off)** | `2,348` | `315` | `309` | **`2,972`** | `843` | `103` | `103` | **`1,049`** |
| **`6`** | **`sanitary_ppe_waste`** | **General waste** | `1,431` | `171` | `186` | **`1,788`** | `804` | `105` | `106` | **`1,015`** |
| **`7`** | **`organic`** | **Organic** | `1,990` | `276` | `180` | **`2,446`** | `495` | `63` | `60` | **`618`** |
| **`8`** | **`battery_ewaste`** | **E-waste** | `1,184` | `139` | `159` | **`1,482`** | `1,160` | `139` | `147` | **`1,446`** |
| **`9`** | **`hazardous_medical`** | **Hazardous** | `1,166` | `136` | `142` | **`1,444`** | `799` | `87` | `97` | **`983`** |
| — | **TOTAL** | **All 5 Categories** | **`17,474`** | **`2,244`** | **`2,168`** | **`21,886`** | **`5,887`** | **`734`** | **`734`** | **`7,355`** |

---

## 3. Source Dataset Image Contribution by Split

| Source Dataset | Train Images (`80%`) | Val Images (`10%`) | Test Images (`10%`) | **Total Curated Images** |
| :--- | --: | --: | --: | --: |
| **Bangladesh Recyclable Waste (`data_canonical_75-15-10`)** | `874` | `110` | `110` | **`1,094`** |
| **Mendeley Trash Detection (`CUSTOM_DATASET`)** | `2,319` | `289` | `289` | **`2,897`** |
| **PackWISE Packaging Waste (`1280×1280` resized)** | `432` | `52` | `52` | **`536`** |
| **MedBin Dataset 2 (`v1i.yolov11`)** | `1,206` | `151` | `151` | **`1,508`** |
| **RecyBat24 Battery Dataset (`110` physical batteries grouped)** | `1,056` | `132` | `132` | **`1,320`** |
| **TOTAL** | **`5,887`** | **`734`** | **`734`** | **`7,355`** |

---

## 4. Training Command (Ultralytics YOLOv11)

You can train YOLOv11 directly on the new `Dataset/` directory using:

```powershell
python YOLOv11/trainEXAMPLE.py --data Dataset/data.yaml --model yolo11s.pt --epochs 100 --imgsz 640
```
