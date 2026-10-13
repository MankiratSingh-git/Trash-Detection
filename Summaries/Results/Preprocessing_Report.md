# Sortify — Dataset Preprocessing & Curation Report (YOLOv11)

## 1. Source Datasets Used

The unified **Sortify** object-detection dataset was constructed from five publicly available waste and recycling datasets totaling **`13,108` raw images** and **`56,162` annotated instances** (after removing the duplicate alternate partition in the Bangladesh dataset):

| # | Dataset Name | Domain / Imaging Conditions | Original Format | Raw Images | Raw Instances |Raw Classes | Source Link |
| :-: | :--- | :--- | :--- | --: | --: | --: | :--- |
| **1** | **Bangladesh Recyclable Waste Dataset** (`data_canonical_75-15-10`) | In-situ outdoor municipal solid waste (`640×640`) | YOLO (mixed polygon + bbox) | `2,842` | `6,076` | `4` | [Roboflow Universe (v12)](https://universe.roboflow.com/waste-detections-by-fu/waste-detections-c0mkz/dataset/12) |
| **2** | **MedBin Dataset 2** (`v1i.yolov11`) | Clinical, pharmaceutical & PPE waste (`640×640`) | YOLO (bbox + 62 polygons) | `1,796` | `6,410` | `37` | [Roboflow Universe](https://universe.roboflow.com/medical-waste-e3vov/medbin_dataset-2) / [GitHub](https://github.com/Wayne3918/MedbinNet) |
| **3** | **Mendeley Trash Detection Dataset** (`CUSTOM_DATASET`) | Household & catalog waste (`640×640`) | YOLO (bbox + 1,166 polygons) | `5,049` | `34,082` | `6` | [Mendeley Data (`z732f9pwxt/1`)](https://data.mendeley.com/datasets/z732f9pwxt/1) |
| **4** | **PackWISE Packaging Waste Dataset** | High-res industrial sorting belt (`4096×4096`) | COCO JSON (`train`/`val`/`test.json`) | `586` | `6,713` | `23` | [Fraunhofer Fordatis (`463.2`)](https://fordatis.fraunhofer.de/handle/fordatis/463.2?locale=en) |
| **5** | **RecyBat24 Battery Dataset** | Waste batteries (`110` physical batteries, `640×640`) | COCO JSON (`annotations.json`) | `2,835` | `2,881` | `3` | [Zenodo (`13126689`)](https://zenodo.org/records/13126689) |
| — | **Total Raw Pool** | — | — | **`13,108`** | **`56,162`** | **`73`** | — |

---

## 2. Initial Raw Class Structure (73 Classes)

| Dataset | Raw Classes (`ID: Class Name (Instance Count)`) |
| :--- | :--- |
| **1. Bangladesh Recyclable Waste** (`4` classes) | `0: Paper (1,575)`, `1: Plastic Bottle (1,634)`, `2: Plastic bag (1,037)`, `3: Plastic waste (1,830)` |
| **2. Mendeley Trash Detection** (`6` classes) | `0: BIODEGRADABLE (19,019)`, `1: CARDBOARD (2,718)`, `2: GLASS (2,407)`, `3: METAL (3,820)`, `4: PAPER (2,168)`, `5: PLASTIC (3,950)` |
| **3. PackWISE Packaging Waste** (`23` classes) | `0: undefined (521)`, `1: plastic-foil/bag (2,229)`, `2: aluminium-coated-foil (391)`, `3: paper-bag (150)`, `4: beverage-carton (179)`, `5: plastic-other (458)`, `6: plastic-lid (156)`, `7: plastic-cup/pot (657)`, `8: blister (95)`, `9: metal-can (104)`, `10: metal-foil (73)`, `11: net (32)`, `12: metal-lid (162)`, `13: tissue (457)`, `14: wood (32)`, `15: paper-other (485)`, `16: foamed-plastic (226)`, `17: textile (34)`, `18: plastic-bottle (143)`, `19: ribbon/tape (77)`, `20: metal-other (15)`, `21: tube (14)`, `22: paper-cup/pot (23)` |
| **4. MedBin Dataset 2** (`37` classes) | `0: Bloody_objects (148)`, `1: Electronic_thermometer (126)`, `2: Mask (242)`, `3: N95 (208)`, `4: Oxygen_cylinder (98)`, `5: Radioactive_objects (19)`, `6: bandage (105)`, `7: blade (104)`, `8: capsule (194)`, `9: cotton_swab (210)`, `10: covid_buffer (100)`, `11: covid_buffer_box (90)`, `12: covid_test_case (162)`, `13: drug_packaging (375)`, `14: gauze (122)`, `15: glass_bottle (112)`, `16: harris_uni_core (109)`, `17: harris_uni_core_cap (107)`, `18: iodine_swab (109)`, `19: medical_gloves (336)`, `20: medical_infusion_bag (147)`, `21: mercury_thermometer (115)`, `22: needle (98)`, `23: paperbox (372)`, `24: pill (242)`, `25: plastic_medical_bag (279)`, `26: plastic_medical_bottle (310)`, `27: reagent_tube (302)`, `28: reagent_tube_cap (303)`, `29: scalpel (99)`, `30: single_channel_pipette (116)`, `31: syringe (266)`, `32: transferpettor_glass (128)`, `33: transferpettor_plastic (126)`, `34: tweezer_metal (103)`, `35: tweezer_plastic (182)`, `36: unguent (146)` |
| **5. RecyBat24** (`3` classes) | `0: cylindric (479)`, `1: pouch (1,742)`, `2: prismatic (660)` |

---

## 3. Format Standardization & YOLOv11 Conversion

Before merging, each raw dataset required structural and annotation fixes to ensure 100% Ultralytics YOLOv11 object-detection compatibility (`class_id x_center y_center width height` normalized to $[0, 1]$):

1. **Resolving Mixed Polygon & Bounding-Box Corruptions (Bangladesh, MedBin, Mendeley):**
   - **Issue:** `80` label files across Bangladesh (`4`), MedBin (`48`), and Mendeley (`28`) contained **both** 5-column bounding boxes and $>5$-column segmentation polygons within the same `.txt` file. In Ultralytics YOLO, mixed files cause `segments2boxes` to misinterpret 5-column `(x_center, y_center, w, h)` boxes as 2-point polygons, silently corrupting bounding-box coordinates.
   - **Fix:** Converted all polygon lines (`6,073` in Bangladesh, `62` in MedBin, `1,166` in Mendeley) into tight axis-aligned 5-column YOLO bounding boxes:
     $$x_c = \frac{x_{\min} + x_{\max}}{2}, \quad y_c = \frac{y_{\min} + y_{\max}}{2}, \quad w = x_{\max} - x_{\min}, \quad h = y_{\max} - y_{\min}$$
2. **COCO JSON to YOLOv11 Conversion (PackWISE & RecyBat24):**
   - **PackWISE:** Converted COCO `[x_min, y_min, width, height]` annotations from `train.json`, `val.json`, and `test.json` into normalized 0-indexed `.txt` YOLO label files (`category_id - 1`) and pre-resized `4096×4096` images to `1280×1280` to eliminate dataloader I/O bottlenecks.
   - **RecyBat24:** Converted `train/annotations.json` and `val/annotations.json` into YOLO `.txt` files. Avoided `instance_segmentation_annotations.json`, which was found to contain $3\times$ triplicated entries per object (`8,643` vs. `2,881` true instances).
3. **Orphan File Cleanup (Mendeley):**
   - Removed `29` byte-identical duplicate `*(1).jpg` images and `30` `*(1).txt` label files in `train/` that lacked matching pairs, restoring a strict 1-to-1 match (`5,049` image/label pairs).

---

## 4. Class Selection, Mapping & Curation Methodology

Sortify routes detected items into **5 backend disposal categories**: `Recyclable`, `General waste`, `Organic`, `E-waste`, and `Hazardous`. To support these categories without training instability, the `73` raw classes were distilled into **10 generalized detector classes** using four rules:

1. **Strict `Specific → General` Mapping (No Broad-into-Specific Conflicts):**
   - **Dropped Bangladesh `Paper` (`1,575`):** Bangladesh `Paper` mixes cardboard boxes and paper together. Because `paper` (`0`) and `cardboard` (`1`) are separate detector classes, Bangladesh `Paper` was excluded so cardboard boxes are never mislabeled as `paper`.
   - **Dropped Mendeley `PLASTIC` (`3,950`) & Bangladesh `Plastic waste` (`1,830`):** Both are broad/catch-all classes mixing rigid containers, cups, and flexible bags/wrappers. Keeping only explicit bottle classes (`plastic_bottle`) and bag classes (`plastic_bag`) prevents cross-class confusion.
   - **Dropped Ambiguous/Composite & Labware Classes (`40` total dropped):** Excluded PackWISE `aluminium-coated-foil` (`391`), `undefined` (`521` edge slivers), `beverage-carton` (`179`), and `15` MedBin specialized laboratory instruments (`pipettes`, `harris_uni_core`, `reagent_tubes`, `tweezers`, contrived `Radioactive_objects`, and tiny loose `pills`/`capsules`).
2. **Pure-Image Filtering (Zero Unlabeled Foreground Objects):**
   - Deleting a bounding box from a multi-object image without removing the image forces YOLO to treat the visible object as background (false negative). Therefore, any multi-object image in Bangladesh containing dropped `Paper` or `Plastic waste`, and any image in Mendeley containing dropped `PLASTIC`, was **excluded entirely**.
3. **Geometry & Class-Balance Filtering:**
   - **Removed Full-Image Macro Boxes (Mendeley):** Filtered out bounding boxes covering $>90\%$ of both width and height (`251` cardboard texture crops and `313` paper macro/clipart boxes).
   - **Filtered High-Density Produce Collages (Mendeley `BIODEGRADABLE`):** Raw `BIODEGRADABLE` had `19,019` boxes across `898` images (`21.2` boxes/image from stock grocery photos). Excluding images with $>15$ biodegradable boxes reduced `organic` to **`2,446` clean instances**.
   - **Balanced Metal & Batteries:** Subsampled pure-`METAL` images in Mendeley to cap `metal` at `2,475` total instances, and capped RecyBat24 at `12` viewpoints per physical battery (`1,320` images / `1,356` instances).

---

## 5. Final 10-Class Dataset Distribution & Validation

### 5.1 Final Taxonomy & Split Distribution (`80%` Train / `10%` Val / `10%` Test)

| ID | Detector Class | Backend Category | Mapped Source Classes | Train Boxes | Val Boxes | Test Boxes | **Total Boxes** | **Total Images** |
| :-: | :--- | :--- | :--- | --: | --: | --: | --: | --: |
| **`0`** | **`paper`** | **Recyclable** | Mendeley `PAPER`, PackWISE `paper-bag`, `paper-other`, `paper-cup/pot` | `1,967` | `272` | `257` | **`2,496`** | `879` |
| **`1`** | **`cardboard`** | **Recyclable** | Mendeley `CARDBOARD`, MedBin `paperbox`, `covid_buffer_box` | `2,391` | `244` | `228` | **`2,863`** | `1,013` |
| **`2`** | **`glass`** | **Recyclable** | Mendeley `GLASS`, MedBin `glass_bottle` | `1,653` | `232` | `318` | **`2,203`** | `860` |
| **`3`** | **`metal`** | **Recyclable** | Mendeley `METAL` (capped), PackWISE `metal-can`, `metal-foil`, MedBin `Oxygen_cylinder` | `1,989` | `282` | `204` | **`2,475`** | `915` |
| **`4`** | **`plastic_bottle`** | **Recyclable** | Bangladesh `Plastic Bottle`, MedBin `plastic_medical_bottle`, PackWISE `plastic-bottle` | `1,355` | `177` | `185` | **`1,717`** | `1,081` |
| **`5`** | **`plastic_bag`** | **General Waste** | PackWISE `plastic-foil/bag`, Bangladesh `Plastic bag`, MedBin `plastic_medical_bag` | `2,348` | `315` | `309` | **`2,972`** | `1,049` |
| **`6`** | **`sanitary_ppe_waste`** | **General Waste** | PackWISE `tissue`, MedBin `Mask`, `N95`, `medical_gloves`, `cotton_swab`, `iodine_swab`, `gauze`, `bandage` | `1,431` | `171` | `186` | **`1,788`** | `1,015` |
| **`7`** | **`organic`** | **Organic** | Mendeley `BIODEGRADABLE` ($\le 15$ boxes/image) | `1,990` | `276` | `180` | **`2,446`** | `618` |
| **`8`** | **`battery_ewaste`** | **E-Waste** | RecyBat24 `cylindric`, `pouch`, `prismatic` (`12` views/battery), MedBin `Electronic_thermometer` | `1,184` | `139` | `159` | **`1,482`** | `1,446` |
| **`9`** | **`hazardous_medical`** | **Hazardous** | MedBin `syringe`, `blade`, `scalpel`, `needle`, `Bloody_objects`, `mercury_thermometer`, `drug_packaging`, `unguent`, PackWISE `blister` | `1,166` | `136` | `142` | **`1,444`** | `983` |
| — | **TOTAL** | **All 5 Categories** | **33 Kept Subclasses (40 Dropped)** | **`17,474`** | **`2,244`** | **`2,168`** | **`21,886`** | **`7,355` unique** |

### 5.2 Dataset Integrity & Leak-Free Validation Checks

| Validation Check | Method Applied | Result |
| :--- | :--- | :--- |
| **Class Balance Ratio** | Reduced max-to-min instance ratio from `1,358 : 1` (raw) to **`2.06 : 1`** (`1,444` min vs. `2,972` max). | **PASS** |
| **Roboflow Stem Deduplication** | Deduplicated `152` rotated/flipped base stems in Bangladesh (`65` cross-split leaks) and `5` in MedBin prior to splitting. | **PASS (`0` leaked stems)** |
| **Physical Object Grouping (RecyBat24)** | Grouped all images by the **`110` unique physical batteries** (`88` train / `11` val / `11` test) so no battery appears in both train and val/test. | **PASS (`0` shared batteries)** |
| **1-to-1 Image / Label Pairing** | Verified all `7,355` images (`5,887` train, `734` val, `734` test) have matching `.txt` label files. | **PASS (`0` orphans)** |
| **Bounding-Box Geometry** | Verified all `21,886` annotations have 5 columns (`class xc yc w h`), valid class IDs `[0..9]`, $w, h > 0$, and coordinates in $[0, 1]$. | **PASS (`0` errors)** |
