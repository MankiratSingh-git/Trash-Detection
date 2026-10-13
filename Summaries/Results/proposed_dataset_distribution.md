# Sortify YOLOv11 — Proposed 10-Class Generalized Taxonomy & Dataset Distribution

## 1. Executive Summary & Curation Principles

This document defines the proposed **10-class generalized object-detection taxonomy** for **Sortify**, distilled from the **73 source classes** across our 5 YOLOv11-compatible raw datasets (`13,108` raw images, `56,162` raw bounding boxes).

### Core Curation Rules Applied
1. **Strict `Specific → General` Hierarchy (Zero Broad-into-Specific Contamination):**
   - Because `paper` and `cardboard` are separate target classes, **Bangladesh `Paper`** (`1,575` boxes—which mixes cardboard boxes and paper) is **dropped**.
   - Because `plastic_bottle` and `plastic_bag` are separate target classes, **Mendeley `PLASTIC`** (`3,950` boxes—which mixes rigid bottles and shopping bags) and **Bangladesh `Plastic waste`** (`1,830` catch-all boxes) are **dropped**.
   - Ambiguous composite materials like **PackWISE `aluminium-coated-foil`** (`391` boxes), **`beverage-carton`** (`179` boxes), and **`undefined`** (`521` boxes) are **dropped**.
2. **Pure-Image Filtering (Preventing Unlabeled Foreground Conflicts):**
   - In **Bangladesh**, any multi-object image containing dropped `Paper` or `Plastic waste` is excluded so that unlabeled paper/cardboard or plastic items never appear as false background negatives alongside `Plastic Bottle` or `Plastic bag`.
   - In **Mendeley**, any multi-object image containing dropped `PLASTIC` (`5`) is excluded so unlabeled plastic bottles/bags never appear in the background of `paper`, `cardboard`, `glass`, `metal`, or `organic` images.
3. **Deduplication & Leak Prevention:**
   - **Bangladesh:** Deduplicates the `152` Roboflow augmented/leaked stems (`65` of which leaked across `train`/`valid`/`test`).
   - **MedBin:** Deduplicates the `5` cross-split leaked stems and removes the `19` contrived `Radioactive_objects` images.
   - **RecyBat24:** Groups all images by the **`110` unique physical battery IDs** (`battery001`–`battery110`) before splitting so the same physical battery never leaks across `train`, `val`, and `test`, and caps at `12` viewpoints per physical battery (`1,320` images / `1,341` boxes) so repetitive gray-tabletop photos do not dominate training.
4. **Geometry & Density Cleaning:**
   - **Mendeley:** Removes full-image macro/texture boxes (`w > 0.90` and `h > 0.90`, eliminating `251` cardboard texture crops and `313` paper macro/clipart boxes) and excludes dense stock-catalog produce piles (`>15` `BIODEGRADABLE` boxes per image), reducing `BIODEGRADABLE` from `19,019` noisy micro-boxes to **`2,446` clean `organic` instances**.

---

## 2. Proposed 10-Class Detector Taxonomy & Instance Distribution

| Class ID | Detector Class | Backend Disposal Category | Included Source Classes (Dataset : Class Name) | Raw Mapped Instances | Clean / Filtered Instances | Final Balanced Target Instances | Clean Images (w/ Class) |
| :-: | :--- | :--- | :--- | --: | --: | --: | --: |
| **`0`** | **`paper`** | **Recyclable** | Mendeley `PAPER` (`1,838` clean / `2,168` raw), PackWISE `paper-bag` (`150`), `paper-other` (`485`), `paper-cup/pot` (`23`) | `2,826` | `2,496` | **`2,496`** | `879` |
| **`1`** | **`cardboard`** | **Recyclable** | Mendeley `CARDBOARD` (`2,401` clean / `2,718` raw), MedBin `paperbox` (`372`), `covid_buffer_box` (`90`) | `3,180` | `2,863` | **`2,863`** | `1,013` |
| **`2`** | **`glass`** | **Recyclable** | Mendeley `GLASS` (`2,091` clean / `2,407` raw), MedBin `glass_bottle` (`112`) | `2,519` | `2,203` | **`2,203`** | `860` |
| **`3`** | **`metal`** | **Recyclable** | Mendeley `METAL` (`3,648` clean / `3,820` raw, capped to `2,200`), PackWISE `metal-can` (`104`), `metal-foil` (`73`), MedBin `Oxygen_cylinder` (`98`) | `4,095` | `3,923` | **`2,475`** | `1,333` |
| **`4`** | **`plastic_bottle`** | **Recyclable** | Bangladesh `Plastic Bottle` (`1,264` pure / `1,634` raw), MedBin `plastic_medical_bottle` (`310`), PackWISE `plastic-bottle` (`143`) | `2,087` | `1,717` | **`1,717`** | `1,081` |
| **`5`** | **`plastic_bag`** | **General waste (or Film Drop-off)** | PackWISE `plastic-foil/bag` (`2,229`), Bangladesh `Plastic bag` (`464` pure / `1,037` raw), MedBin `plastic_medical_bag` (`279`) | `3,545` | `2,972` | **`2,972`** | `1,049` |
| **`6`** | **`sanitary_ppe_waste`** | **General waste** | PackWISE `tissue` (`457`), MedBin `medical_gloves` (`336`), `Mask` (`242`), `N95` (`208`), `cotton_swab` (`210`), `gauze` (`122`), `iodine_swab` (`109`), `bandage` (`105`) | `1,789` | `1,788` | **`1,788`** | `1,015` |
| **`7`** | **`organic`** | **Organic** | Mendeley `BIODEGRADABLE` (`2,446` clean in images with <=15 bio boxes / `19,019` raw) | `19,019` | `2,446` | **`2,446`** | `618` |
| **`8`** | **`battery_ewaste`** | **E-waste** | RecyBat24 `cylindric` (`479`), `pouch` (`1,742`), `prismatic` (`660`) [capped to `12` imgs/battery = `1,341`], MedBin `Electronic_thermometer` (`126`) | `3,007` | `3,007` | **`1,482`** | `2,960` |
| **`9`** | **`hazardous_medical`** | **Hazardous** | MedBin `drug_packaging` (`375`), `syringe` (`266`), `Bloody_objects` (`148`), `unguent` (`146`), `mercury_thermometer` (`115`), `blade` (`104`), `scalpel` (`99`), `needle` (`98`), PackWISE `blister` (`95`) | `1,446` | `1,444` | **`1,444`** | `983` |
| — | **TOTAL** | **All 5 Categories Covered** | **33 Kept Source Subclasses (40 Dropped)** | **`43,513`** | **`24,859`** | **`21,886`** | — |

> **Class Balance Achievement:** Across all 10 detector classes, the final balanced instance counts range from **`1,444`** (`hazardous_medical`) to **`2,972`** (`plastic_bag`) — a tight **`2.06 : 1` max-to-min ratio** (down from `1,358 : 1` in the raw datasets).

---

## 3. Backend Disposal Category Coverage Summary

| Backend Category | Mapped Detector Classes | Total Final Balanced Instances | Share of Dataset (%) |
| :--- | :--- | --: | --: |
| **1. Recyclable** | `paper` (`2,496`), `cardboard` (`2,863`), `glass` (`2,203`), `metal` (`2,475`), `plastic_bottle` (`1,717`) | `11,754` | `53.7%` |
| **2. General Waste** | `plastic_bag` (`2,972`), `sanitary_ppe_waste` (`1,788`) | `4,760` | `21.7%` |
| **3. Organic** | `organic` (`2,446`) | `2,446` | `11.2%` |
| **4. E-Waste** | `battery_ewaste` (`1,482`) | `1,482` | `6.8%` |
| **5. Hazardous** | `hazardous_medical` (`1,444`) | `1,444` | `6.6%` |
| **TOTAL** | **10 Classes** | **`21,886`** | **`100.0%`** |

---

## 4. Dataset-by-Dataset Contribution & Dropped Classes Breakdown

### 4.1 Bangladesh Recyclable Waste Dataset (`data_canonical_75-15-10`)
- **Images Retained:** `1,094` pure deduplicated images (from `2,842` raw images).
- **Class Mapping & Filtering:**
  - `0: Paper` (`1,575` raw) $\to$ **DROPPED** *(mixes cardboard boxes and paper; any Bangladesh image containing `Paper` is excluded)*.
  - `1: Plastic Bottle` (`1,634` raw) $\to$ **`4: plastic_bottle`** (`1,264` clean instances in pure deduplicated images).
  - `2: Plastic bag` (`1,037` raw) $\to$ **`5: plastic_bag`** (`464` clean instances in pure deduplicated images).
  - `3: Plastic waste` (`1,830` raw) $\to$ **DROPPED** *(catch-all class mixing cups, straws, and wrappers; images containing `Plastic waste` are excluded to prevent unlabeled plastic clutter)*.

### 4.2 Mendeley Trash Detection Dataset (`CUSTOM_DATASET`)
- **Images Retained:** `3,315` clean images (from `5,049` unique images).
- **Class Mapping & Filtering:**
  - `0: BIODEGRADABLE` (`19,019` raw) $\to$ **`7: organic`** (`2,446` clean instances after removing images with `>15` bio boxes and full-image boxes).
  - `1: CARDBOARD` (`2,718` raw) $\to$ **`1: cardboard`** (`2,401` clean instances after removing `251` full-image texture crops).
  - `2: GLASS` (`2,407` raw) $\to$ **`2: glass`** (`2,091` clean instances).
  - `3: METAL` (`3,820` raw) $\to$ **`3: metal`** (`3,648` clean instances, subsampled/capped to **`2,200`**).
  - `4: PAPER` (`2,168` raw) $\to$ **`0: paper`** (`1,838` clean instances after removing `313` full-image macro/clipart boxes).
  - `5: PLASTIC` (`3,950` raw) $\to$ **DROPPED** *(broad class mixing bottles and bags; all images containing `PLASTIC` are excluded)*.

### 4.3 PackWISE Packaging Waste Dataset (`data`)
- **Images Retained:** `536` images containing kept target classes (from `586` raw images). *(Note: `4096×4096` images will be pre-resized to `1280×1280` during dataset construction to prevent disk/I/O bottlenecks.)*
- **Kept Classes (9):**
  - `3: paper-bag` (`150`), `15: paper-other` (`485`), `22: paper-cup/pot` (`23`) $\to$ **`0: paper`** (`658` instances).
  - `9: metal-can` (`104`), `10: metal-foil` (`73`) $\to$ **`3: metal`** (`177` instances).
  - `18: plastic-bottle` (`143`) $\to$ **`4: plastic_bottle`** (`143` instances).
  - `1: plastic-foil/bag` (`2,229`) $\to$ **`5: plastic_bag`** (`2,229` instances).
  - `13: tissue` (`457`) $\to$ **`6: sanitary_ppe_waste`** (`457` instances).
  - `8: blister` (`95`) $\to$ **`9: hazardous_medical`** (`95` instances).
- **Dropped Classes (14):**
  - `2: aluminium-coated-foil` (`391` — dropped per user instruction), `0: undefined` (`521` — unrecognizable edge slivers), `4: beverage-carton` (`179` — composite Tetra Pak), `5: plastic-other` (`458` — ambiguous plastic fragments), `6: plastic-lid` (`156`), `7: plastic-cup/pot` (`657` — optional rigid plastic, excluded to keep `plastic_bottle` strictly bottles), `11: net` (`32`), `12: metal-lid` (`162` — 51.2% tiny), `14: wood` (`32`), `16: foamed-plastic` (`226` — 50% tiny styrofoam crumbs), `17: textile` (`34`), `19: ribbon/tape` (`77`), `20: metal-other` (`15`), `21: tube` (`14`).

### 4.4 MedBin Dataset 2 (`MedBin_dataset 2.v1i.yolov11`)
- **Images Retained:** `1,508` deduplicated images containing kept classes (from `1,796` raw images).
- **Kept Classes (22):**
  - `23: paperbox` (`372`), `11: covid_buffer_box` (`90`) $\to$ **`1: cardboard`** (`462` instances).
  - `15: glass_bottle` (`112`) $\to$ **`2: glass`** (`112` instances).
  - `4: Oxygen_cylinder` (`98`) $\to$ **`3: metal`** (`98` instances).
  - `26: plastic_medical_bottle` (`310`) $\to$ **`4: plastic_bottle`** (`310` instances).
  - `25: plastic_medical_bag` (`279`) $\to$ **`5: plastic_bag`** (`279` instances).
  - `2: Mask` (`242`), `3: N95` (`208`), `19: medical_gloves` (`336`), `9: cotton_swab` (`210`), `18: iodine_swab` (`109`), `14: gauze` (`122`), `6: bandage` (`105`) $\to$ **`6: sanitary_ppe_waste`** (`1,331` deduplicated instances).
  - `1: Electronic_thermometer` (`126`) $\to$ **`8: battery_ewaste`** (`126` instances).
  - `13: drug_packaging` (`375`), `31: syringe` (`266`), `0: Bloody_objects` (`148`), `36: unguent` (`146`), `21: mercury_thermometer` (`115`), `7: blade` (`104`), `29: scalpel` (`99`), `22: needle` (`98`) $\to$ **`9: hazardous_medical`** (`1,349` deduplicated instances).
- **Dropped Classes (15):**
  - `5: Radioactive_objects` (`19` — contrived tin can with paper radiation symbol), `8: capsule` (`194` — 43% tiny), `24: pill` (`242` — 58% tiny), and specialized clinical/lab instruments (`10: covid_buffer`, `12: covid_test_case`, `16: harris_uni_core`, `17: harris_uni_core_cap`, `20: medical_infusion_bag`, `27: reagent_tube`, `28: reagent_tube_cap`, `30: single_channel_pipette`, `32: transferpettor_glass`, `33: transferpettor_plastic`, `34: tweezer_metal`, `35: tweezer_plastic`).

### 4.5 RecyBat24 Battery Dataset (`recybat24`)
- **Images Retained:** `1,320` subsampled images (`12` viewpoints per physical battery across all `110` unique physical batteries, down from `2,835` raw images).
- **Class Mapping & Leak-Free Split Strategy:**
  - `0: cylindric` (`479` raw), `1: pouch` (`1,742` raw), `2: prismatic` (`660` raw) $\to$ **`8: battery_ewaste`** (`1,356` balanced instances).
  - **Grouped Split Assignment:** The `110` physical battery IDs (`battery001`–`battery110`) are partitioned at the **physical battery level** (`80%` train = `88` batteries, `10%` val = `11` batteries, `10%` test = `11` batteries) to eliminate the cross-split leakage present in the raw dataset.

---

## 5. Summary of Unified Dataset Volume & Split Plan

| Metric | Count / Specification |
| :--- | :--- |
| **Total Curated Images** | **`~7,473` images** (`1,094` Bangladesh + `3,005` Mendeley (after metal cap) + `536` PackWISE + `1,508` MedBin + `1,320` RecyBat24) |
| **Total Curated Bounding Boxes** | **`21,886` instances** across `10` classes |
| **Target Train / Val / Test Split** | **`80%` Train / `10%` Val / `10%` Test** (stratified by class and grouped by physical object/stem ID so zero augmented or multi-angle copies leak across splits) |
| **Output Format** | Standard Ultralytics YOLOv11 directory (`images/{train,val,test}`, `labels/{train,val,test}`, `data.yaml`) |
