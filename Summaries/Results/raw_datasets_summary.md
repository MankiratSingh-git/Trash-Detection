# Raw Datasets Analysis & YOLOv11 Compatibility Report

This report summarizes all datasets located in `RAW Datasets/`, including split-wise image and instance counts, full class distributions, and YOLOv11 compatibility verification.

## 1. Executive Summary

| # | Dataset | Partition / Variant | Classes (`nc`) | Train Imgs | Val Imgs | Test Imgs | Total Imgs | Total Instances | YOLOv11 Status |
| :-: | :--- | :--- | :-: | --: | --: | --: | --: | --: | :-: |
| 1 | **Bangladesh Recyclable Waste Dataset** | `Canonical Partition (75-15-10)` | 4 | 2,132 | 426 | 284 | **2,842** | **6,076** | **PASS (Compatible)** |
| 2 | **Bangladesh Recyclable Waste Dataset** | `Robustness Partition (70-15-15)` | 4 | 1,989 | 426 | 427 | **2,842** | **6,076** | **PASS (Compatible)** |
| 3 | **MedBin Dataset (`MedBin_dataset 2.v1i.yolov11`)** | `Standard` | 37 | 1,410 | 357 | 29 | **1,796** | **6,410** | **PASS (Compatible)** |
| 4 | **Mendely Trash Detection Dataset (`CUSTOM_DATASET`)** | `Standard` | 6 | 3,536 | 1,008 | 505 | **5,049** | **34,082** | **PASS (Compatible)** |
| 5 | **Packwise (`PackWISE Packaging Waste Dataset`)** | `Standard` | 23 | 410 | 88 | 88 | **586** | **6,713** | **PASS (Compatible)** |
| 6 | **recybat24 (`RecyBat24 Battery Dataset`)** | `Standard` | 3 | 1,425 | 1,410 | 0 | **2,835** | **2,881** | **PASS (Compatible)** |

> **Note on Bangladesh Recyclable Waste Dataset:** Contains two alternative split partitions (`data_canonical_75-15-10` [v12] and `data_robustness_70-15-15` [v10]) over the exact same **2,842 raw images** and **6,076 object instances**. Counting it once, the 5 unique raw datasets comprise **13,108 unique images** and **56,162 annotated object instances** across **73 dataset-specific classes**.

## 2. YOLOv11 Compatibility Audit & Fixes Applied

All datasets were audited for strict Ultralytics YOLOv11 object-detection compatibility (`<class_id> <x_center> <y_center> <width> <height>` normalized in `[0, 1]`, 1-to-1 image/label matching, and valid `data.yaml` configuration) and standardized via `Preprocessing/make_yolov11_compatible.py`:

| Dataset | Initial Audit Findings (Before Preprocessing) | Fixes Applied (`Preprocessing/make_yolov11_compatible.py`) | Final Verification |
| :--- | :--- | :--- | :--- |
| **1. Bangladesh Recyclable Waste** (`canonical` & `robustness`) | 6,071 of 6,076 annotations were multi-point polygons while 5 were 5-column bboxes (with **4 files mixing both bbox and polygon lines**, which causes Ultralytics `verify_image_label` to misparse 5-column boxes as 2-point segments). Missing `path: .` in `data.yaml`. | Converted all 6,071 polygon annotations in both partitions into tight axis-aligned 5-column YOLOv11 bounding boxes (`xc yc w h`) and added `path: .` to `data.yaml`. | **100% YOLOv11 Compatible** (0 polygon/mixed lines, 0 out-of-bounds, 2,842/2,842 matched pairs). |
| **2. MedBin (`MedBin_dataset 2.v1i.yolov11`)** | Contained **62 polygon lines** across 50 files (**48 files mixing 5-column bboxes and polygons** in the same `.txt` file). `data.yaml` used `../train/images` relative paths without `path: .`. | Converted all 62 polygon annotations to 5-column YOLOv11 bounding boxes (`xc yc w h`) and standardized `data.yaml` paths (`path: .`, `train: train/images`, `val: valid/images`, `test: test/images`). | **100% YOLOv11 Compatible** (6,410 valid bboxes across 1,796 matched pairs). |
| **3. Mendely (`CUSTOM_DATASET`)** | `train` had **29 orphan `*(1).jpg` images** without labels and **30 orphan `*(1).txt` labels** without images (all 59 verified byte-for-byte identical copies of existing paired files in `train`). Also had **1,158 polygon annotations** (28 mixed files) and 2 stale `.cache` files. `data.yaml` used `../train/images`. | Removed the 29 duplicate orphan images, 30 duplicate orphan labels, and 2 stale `.cache` files; converted all 1,158 polygon annotations to 5-column YOLOv11 bboxes; standardized `data.yaml`. | **100% YOLOv11 Compatible** (5,049 matched image/label pairs, 34,082 valid bboxes). |
| **4. Packwise** | Provided only in **COCO JSON format** (`data/train.json`, `data/val.json`, `data/test.json`) with no YOLO `.txt` labels or `data.yaml`. Bundled conversion script sat inside `RAW Datasets/Packwise/`. | Moved conversion script to `Preprocessing/`, generated YOLOv11 `.txt` label files for all 586 images (`train`: 410, `val`: 88, `test`: 88, including 28 background images), and created `data.yaml` (`nc: 23`). | **100% YOLOv11 Compatible** (586 matched image/label pairs, 6,713 valid bboxes). |
| **5. recybat24** | Provided only in **COCO JSON format** (`train/annotations.json`, `val/annotations.json`). Note: `instance_segmentation_annotations.json` contained 3x duplicate bbox entries per object (4,323 train / 4,320 val). | Generated YOLOv11 `.txt` label files from ground-truth `annotations.json` for all 2,835 images (`train`: 1,425, `val`: 1,410, including 1 background image) and created `data.yaml` (`nc: 3`). | **100% YOLOv11 Compatible** (2,835 matched image/label pairs, 2,881 valid bboxes). |

## 3. Detailed Dataset & Class Breakdown

### 3.1 Bangladesh Recyclable Waste Dataset — Canonical Partition (75-15-10)

- **YAML Config:** `RAW Datasets/bangladesh_recyclable_waste_dataset/data_canonical_75-15-10/data.yaml`
- **Number of Classes (`nc`):** 4
- **Note:** Partition v12 (75% train, 15% val, 10% test) of the 2,842-image Bangladesh municipal solid waste dataset.
- **Note:** Dataset README.md lists classes in a different order than data.yaml; actual YOLO label files follow `data.yaml` (`0: Paper`, `1: Plastic Bottle`, `2: Plastic bag`, `3: Plastic waste`).

#### Split Statistics & Validation Checks

| Split | Images | Label Files | Background (Empty) Imgs | Bounding Boxes | Unmatched Imgs/Lbls | Polygon / Malformed / OOB | YOLOv11 Ready |
| :--- | --: | --: | --: | --: | :-: | :-: | :-: |
| `train` | 2,132 | 2,132 | 0 | 4,492 | 0 / 0 | 0 / 0 / 0 | **Yes** |
| `val` | 426 | 426 | 0 | 1,048 | 0 / 0 | 0 / 0 / 0 | **Yes** |
| `test` | 284 | 284 | 0 | 536 | 0 / 0 | 0 / 0 / 0 | **Yes** |
| **Total** | **2,842** | **2,842** | **0** | **6,076** | **0 / 0** | **0 / 0 / 0** | **Yes** |

#### Class-wise Instance & Image Counts

| Class ID | Class Name | Train Instances (Imgs) | Val Instances (Imgs) | Test Instances (Imgs) | Total Instances | Total Images | % of Dataset Instances |
| :-: | :--- | --: | --: | --: | --: | --: | --: |
| `0` | **Paper** | 1,153 (765) | 274 (150) | 148 (112) | **1,575** | 1,027 | 25.92% |
| `1` | **Plastic Bottle** | 1,234 (666) | 276 (134) | 124 (76) | **1,634** | 876 | 26.89% |
| `2` | **Plastic bag** | 768 (502) | 165 (98) | 104 (69) | **1,037** | 669 | 17.07% |
| `3` | **Plastic waste** | 1,337 (625) | 333 (143) | 160 (81) | **1,830** | 849 | 30.12% |

### 3.2 Bangladesh Recyclable Waste Dataset — Robustness Partition (70-15-15)

- **YAML Config:** `RAW Datasets/bangladesh_recyclable_waste_dataset/data_robustness_70-15-15/data.yaml`
- **Number of Classes (`nc`):** 4
- **Note:** Alternative partition v10 (70% train, 15% val, 15% test) of the same 2,842 images and 6,076 instances as the Canonical partition.

#### Split Statistics & Validation Checks

| Split | Images | Label Files | Background (Empty) Imgs | Bounding Boxes | Unmatched Imgs/Lbls | Polygon / Malformed / OOB | YOLOv11 Ready |
| :--- | --: | --: | --: | --: | :-: | :-: | :-: |
| `train` | 1,989 | 1,989 | 0 | 4,190 | 0 / 0 | 0 / 0 / 0 | **Yes** |
| `val` | 426 | 426 | 0 | 1,048 | 0 / 0 | 0 / 0 / 0 | **Yes** |
| `test` | 427 | 427 | 0 | 838 | 0 / 0 | 0 / 0 / 0 | **Yes** |
| **Total** | **2,842** | **2,842** | **0** | **6,076** | **0 / 0** | **0 / 0 / 0** | **Yes** |

#### Class-wise Instance & Image Counts

| Class ID | Class Name | Train Instances (Imgs) | Val Instances (Imgs) | Test Instances (Imgs) | Total Instances | Total Images | % of Dataset Instances |
| :-: | :--- | --: | --: | --: | --: | --: | --: |
| `0` | **Paper** | 1,061 (703) | 274 (150) | 240 (174) | **1,575** | 1,027 | 25.92% |
| `1` | **Plastic Bottle** | 1,139 (625) | 276 (134) | 219 (117) | **1,634** | 876 | 26.89% |
| `2` | **Plastic bag** | 719 (467) | 165 (98) | 153 (104) | **1,037** | 669 | 17.07% |
| `3` | **Plastic waste** | 1,271 (587) | 333 (143) | 226 (119) | **1,830** | 849 | 30.12% |

### 3.3 MedBin Dataset (`MedBin_dataset 2.v1i.yolov11`)

- **YAML Config:** `RAW Datasets/MedBin_dataset 2.v1i.yolov11/data.yaml`
- **Number of Classes (`nc`):** 37
- **Note:** 37-class medical waste dataset. Note that 11 classes have 0 instances in the small 29-image `test` split, while all 37 classes are represented in both `train` and `val`.

#### Split Statistics & Validation Checks

| Split | Images | Label Files | Background (Empty) Imgs | Bounding Boxes | Unmatched Imgs/Lbls | Polygon / Malformed / OOB | YOLOv11 Ready |
| :--- | --: | --: | --: | --: | :-: | :-: | :-: |
| `train` | 1,410 | 1,410 | 0 | 5,004 | 0 / 0 | 0 / 0 / 0 | **Yes** |
| `val` | 357 | 357 | 0 | 1,195 | 0 / 0 | 0 / 0 / 0 | **Yes** |
| `test` | 29 | 29 | 0 | 211 | 0 / 0 | 0 / 0 / 0 | **Yes** |
| **Total** | **1,796** | **1,796** | **0** | **6,410** | **0 / 0** | **0 / 0 / 0** | **Yes** |

#### Class-wise Instance & Image Counts

| Class ID | Class Name | Train Instances (Imgs) | Val Instances (Imgs) | Test Instances (Imgs) | Total Instances | Total Images | % of Dataset Instances |
| :-: | :--- | --: | --: | --: | --: | --: | --: |
| `0` | **Bloody_objects** | 110 (100) | 38 (34) | 0 (0) | **148** | 134 | 2.31% |
| `1` | **Electronic_thermometer** | 107 (107) | 18 (18) | 1 (1) | **126** | 126 | 1.97% |
| `2` | **Mask** | 186 (174) | 50 (39) | 6 (6) | **242** | 219 | 3.78% |
| `3` | **N95** | 160 (139) | 40 (35) | 8 (5) | **208** | 179 | 3.24% |
| `4` | **Oxygen_cylinder** | 83 (83) | 15 (15) | 0 (0) | **98** | 98 | 1.53% |
| `5` | **Radioactive_objects** | 14 (14) | 5 (5) | 0 (0) | **19** | 19 | 0.30% |
| `6` | **bandage** | 82 (79) | 23 (21) | 0 (0) | **105** | 100 | 1.64% |
| `7` | **blade** | 93 (93) | 11 (11) | 0 (0) | **104** | 104 | 1.62% |
| `8` | **capsule** | 157 (72) | 31 (13) | 6 (2) | **194** | 87 | 3.03% |
| `9` | **cotton_swab** | 157 (113) | 44 (32) | 9 (8) | **210** | 153 | 3.28% |
| `10` | **covid_buffer** | 75 (73) | 20 (19) | 5 (5) | **100** | 97 | 1.56% |
| `11` | **covid_buffer_box** | 71 (71) | 17 (17) | 2 (2) | **90** | 90 | 1.40% |
| `12` | **covid_test_case** | 126 (117) | 31 (26) | 5 (5) | **162** | 148 | 2.53% |
| `13` | **drug_packaging** | 285 (193) | 67 (47) | 23 (13) | **375** | 253 | 5.85% |
| `14` | **gauze** | 100 (91) | 18 (13) | 4 (4) | **122** | 108 | 1.90% |
| `15` | **glass_bottle** | 96 (89) | 16 (16) | 0 (0) | **112** | 105 | 1.75% |
| `16` | **harris_uni_core** | 89 (89) | 19 (19) | 1 (1) | **109** | 109 | 1.70% |
| `17` | **harris_uni_core_cap** | 86 (86) | 19 (19) | 2 (2) | **107** | 107 | 1.67% |
| `18` | **iodine_swab** | 87 (87) | 21 (21) | 1 (1) | **109** | 109 | 1.70% |
| `19` | **medical_gloves** | 272 (237) | 54 (47) | 10 (9) | **336** | 293 | 5.24% |
| `20` | **medical_infusion_bag** | 127 (125) | 20 (20) | 0 (0) | **147** | 145 | 2.29% |
| `21` | **mercury_thermometer** | 94 (92) | 21 (21) | 0 (0) | **115** | 113 | 1.79% |
| `22` | **needle** | 85 (82) | 13 (13) | 0 (0) | **98** | 95 | 1.53% |
| `23` | **paperbox** | 303 (233) | 56 (41) | 13 (6) | **372** | 280 | 5.80% |
| `24` | **pill** | 181 (57) | 48 (14) | 13 (3) | **242** | 74 | 3.78% |
| `25` | **plastic_medical_bag** | 193 (141) | 60 (42) | 26 (15) | **279** | 198 | 4.35% |
| `26` | **plastic_medical_bottle** | 223 (167) | 68 (45) | 19 (9) | **310** | 221 | 4.84% |
| `27` | **reagent_tube** | 213 (148) | 74 (49) | 15 (11) | **302** | 208 | 4.71% |
| `28` | **reagent_tube_cap** | 214 (150) | 75 (50) | 14 (11) | **303** | 211 | 4.73% |
| `29` | **scalpel** | 85 (85) | 14 (14) | 0 (0) | **99** | 99 | 1.54% |
| `30` | **single_channel_pipette** | 90 (90) | 20 (20) | 6 (6) | **116** | 116 | 1.81% |
| `31` | **syringe** | 217 (203) | 44 (40) | 5 (5) | **266** | 248 | 4.15% |
| `32` | **transferpettor_glass** | 109 (93) | 19 (15) | 0 (0) | **128** | 108 | 2.00% |
| `33` | **transferpettor_plastic** | 101 (87) | 20 (18) | 5 (5) | **126** | 110 | 1.97% |
| `34` | **tweezer_metal** | 83 (82) | 19 (18) | 1 (1) | **103** | 101 | 1.61% |
| `35` | **tweezer_plastic** | 148 (130) | 27 (26) | 7 (7) | **182** | 163 | 2.84% |
| `36` | **unguent** | 102 (101) | 40 (37) | 4 (4) | **146** | 142 | 2.28% |

### 3.4 Mendely Trash Detection Dataset (`CUSTOM_DATASET`)

- **YAML Config:** `RAW Datasets/Mendely/Trash Detection Dataset/CUSTOM_DATASET/data.yaml`
- **Number of Classes (`nc`):** 6
- **Note:** 6-class general recyclable waste dataset (`BIODEGRADABLE`, `CARDBOARD`, `GLASS`, `METAL`, `PAPER`, `PLASTIC`).
- **Note:** Pre-cleaning `train` split had 3,565 images (with 29 byte-identical `*(1).jpg` orphan duplicates) and 3,566 label files (with 30 byte-identical `*(1).txt` orphan duplicates totaling 23,625 raw lines). After removing those 59 unmatched duplicate copies, `train` has 3,536 matched image/label pairs and 23,512 bounding boxes (3 images have empty background labels).

#### Split Statistics & Validation Checks

| Split | Images | Label Files | Background (Empty) Imgs | Bounding Boxes | Unmatched Imgs/Lbls | Polygon / Malformed / OOB | YOLOv11 Ready |
| :--- | --: | --: | --: | --: | :-: | :-: | :-: |
| `train` | 3,536 | 3,536 | 3 | 23,512 | 0 / 0 | 0 / 0 / 0 | **Yes** |
| `val` | 1,008 | 1,008 | 0 | 7,284 | 0 / 0 | 0 / 0 / 0 | **Yes** |
| `test` | 505 | 505 | 0 | 3,286 | 0 / 0 | 0 / 0 / 0 | **Yes** |
| **Total** | **5,049** | **5,049** | **3** | **34,082** | **0 / 0** | **0 / 0 / 0** | **Yes** |

#### Class-wise Instance & Image Counts

| Class ID | Class Name | Train Instances (Imgs) | Val Instances (Imgs) | Test Instances (Imgs) | Total Instances | Total Images | % of Dataset Instances |
| :-: | :--- | --: | --: | --: | --: | --: | --: |
| `0` | **BIODEGRADABLE** | 13,024 (630) | 4,450 (171) | 1,545 (97) | **19,019** | 898 | 55.80% |
| `1` | **CARDBOARD** | 1,994 (607) | 451 (172) | 273 (92) | **2,718** | 871 | 7.97% |
| `2` | **GLASS** | 1,538 (600) | 432 (177) | 437 (84) | **2,407** | 861 | 7.06% |
| `3` | **METAL** | 2,595 (833) | 789 (233) | 436 (101) | **3,820** | 1,167 | 11.21% |
| `4` | **PAPER** | 1,617 (594) | 347 (168) | 204 (82) | **2,168** | 844 | 6.36% |
| `5` | **PLASTIC** | 2,744 (664) | 815 (179) | 391 (90) | **3,950** | 933 | 11.59% |

### 3.5 Packwise (`PackWISE Packaging Waste Dataset`)

- **YAML Config:** `RAW Datasets/Packwise/data/data.yaml`
- **Number of Classes (`nc`):** 23
- **Note:** 23-class conveyor-belt packaging waste dataset (4096x4096 RGB line-scan images) converted from COCO JSON (`train.json`, `val.json`, `test.json`) to YOLOv11 format.
- **Note:** COCO category IDs `1..23` are mapped to 0-indexed YOLO class IDs `0..22`.

#### Split Statistics & Validation Checks

| Split | Images | Label Files | Background (Empty) Imgs | Bounding Boxes | Unmatched Imgs/Lbls | Polygon / Malformed / OOB | YOLOv11 Ready |
| :--- | --: | --: | --: | --: | :-: | :-: | :-: |
| `train` | 410 | 410 | 20 | 4,669 | 0 / 0 | 0 / 0 / 0 | **Yes** |
| `val` | 88 | 88 | 4 | 990 | 0 / 0 | 0 / 0 / 0 | **Yes** |
| `test` | 88 | 88 | 4 | 1,054 | 0 / 0 | 0 / 0 / 0 | **Yes** |
| **Total** | **586** | **586** | **28** | **6,713** | **0 / 0** | **0 / 0 / 0** | **Yes** |

#### Class-wise Instance & Image Counts

| Class ID | Class Name | Train Instances (Imgs) | Val Instances (Imgs) | Test Instances (Imgs) | Total Instances | Total Images | % of Dataset Instances |
| :-: | :--- | --: | --: | --: | --: | --: | --: |
| `0` | **undefined** | 365 (182) | 73 (35) | 83 (42) | **521** | 259 | 7.76% |
| `1` | **plastic-foil/bag** | 1,511 (344) | 364 (74) | 354 (78) | **2,229** | 496 | 33.20% |
| `2` | **aluminium-coated-foil** | 268 (149) | 50 (25) | 73 (39) | **391** | 213 | 5.82% |
| `3` | **paper-bag** | 105 (85) | 25 (18) | 20 (16) | **150** | 119 | 2.23% |
| `4` | **beverage-carton** | 118 (101) | 27 (24) | 34 (28) | **179** | 153 | 2.67% |
| `5` | **plastic-other** | 335 (166) | 62 (33) | 61 (34) | **458** | 233 | 6.82% |
| `6` | **plastic-lid** | 104 (84) | 29 (24) | 23 (19) | **156** | 127 | 2.32% |
| `7` | **plastic-cup/pot** | 454 (228) | 100 (51) | 103 (55) | **657** | 334 | 9.79% |
| `8` | **blister** | 63 (52) | 17 (14) | 15 (13) | **95** | 79 | 1.42% |
| `9` | **metal-can** | 76 (60) | 15 (15) | 13 (12) | **104** | 87 | 1.55% |
| `10` | **metal-foil** | 56 (44) | 11 (11) | 6 (6) | **73** | 61 | 1.09% |
| `11` | **net** | 21 (19) | 6 (6) | 5 (5) | **32** | 30 | 0.48% |
| `12` | **metal-lid** | 122 (75) | 12 (10) | 28 (19) | **162** | 104 | 2.41% |
| `13` | **tissue** | 335 (128) | 48 (27) | 74 (24) | **457** | 179 | 6.81% |
| `14` | **wood** | 22 (19) | 5 (4) | 5 (5) | **32** | 28 | 0.48% |
| `15` | **paper-other** | 339 (184) | 63 (34) | 83 (44) | **485** | 262 | 7.22% |
| `16` | **foamed-plastic** | 158 (88) | 39 (20) | 29 (18) | **226** | 126 | 3.37% |
| `17` | **textile** | 25 (18) | 6 (6) | 3 (3) | **34** | 27 | 0.51% |
| `18` | **plastic-bottle** | 101 (78) | 22 (16) | 20 (18) | **143** | 112 | 2.13% |
| `19` | **ribbon/tape** | 56 (49) | 8 (5) | 13 (11) | **77** | 65 | 1.15% |
| `20` | **metal-other** | 11 (9) | 2 (2) | 2 (2) | **15** | 13 | 0.22% |
| `21` | **tube** | 10 (9) | 2 (2) | 2 (2) | **14** | 13 | 0.21% |
| `22` | **paper-cup/pot** | 14 (13) | 4 (3) | 5 (5) | **23** | 21 | 0.34% |

### 3.6 recybat24 (`RecyBat24 Battery Dataset`)

- **YAML Config:** `RAW Datasets/recybat24/data.yaml`
- **Number of Classes (`nc`):** 3
- **Note:** 3-class waste battery dataset (`cylindric`, `pouch`, `prismatic`) converted from COCO `annotations.json` to YOLOv11 format.
- **Note:** Provides `train` (1,425 images) and `val` (1,410 images) splits; no separate `test` split is provided in the raw dataset.

#### Split Statistics & Validation Checks

| Split | Images | Label Files | Background (Empty) Imgs | Bounding Boxes | Unmatched Imgs/Lbls | Polygon / Malformed / OOB | YOLOv11 Ready |
| :--- | --: | --: | --: | --: | :-: | :-: | :-: |
| `train` | 1,425 | 1,425 | 1 | 1,441 | 0 / 0 | 0 / 0 / 0 | **Yes** |
| `val` | 1,410 | 1,410 | 0 | 1,440 | 0 / 0 | 0 / 0 / 0 | **Yes** |
| **Total** | **2,835** | **2,835** | **1** | **2,881** | **0 / 0** | **0 / 0 / 0** | **Yes** |

#### Class-wise Instance & Image Counts

| Class ID | Class Name | Train Instances (Imgs) | Val Instances (Imgs) | Total Instances | Total Images | % of Dataset Instances |
| :-: | :--- | --: | --: | --: | --: | --: |
| `0` | **cylindric** | 239 (224) | 240 (210) | **479** | 434 | 16.63% |
| `1` | **pouch** | 872 (870) | 870 (870) | **1,742** | 1,740 | 60.47% |
| `2` | **prismatic** | 330 (330) | 330 (330) | **660** | 660 | 22.91% |

