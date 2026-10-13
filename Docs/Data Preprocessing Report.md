<h1 style="text-align: center">Sortify - Data Preprocessing Report</h1>

<h2 style="text-align: center">YOLOv11 Waste-Detection Dataset</h2>

## 1. Overview

Sortify needs a model that can recognise common waste items in images and connect each detection to one of five disposal groups: **Recyclable, General Waste, Organic, E-Waste,** or **Hazardous**. To prepare the dataset, five public datasets were combined, suitable object types were selected, and their labels were converted into one consistent format.

The original sources contained **13,108 images** and **56,162 labelled objects**. After filtering and cleaning, the final dataset contains **7,355 unique images** and **21,886 labelled objects** across **10 detector classes**.

## 2. Source Datasets

The sources cover different waste types and collection settings. Their labels and image formats were checked before combining them. The table below summarises the main classes in each original dataset; the full source class lists are longer, especially for MedBin and PackWISE.

| Source dataset                  | Image type / setting                       | Raw images | Raw labelled objects | Major original classes                                                                                                                                            | Source                                                                                                                                     |
| ------------------------------- | ------------------------------------------ | ---------: | -------------------: | ----------------------------------------------------------------------------------------------------------------------------------------------------------------- | ------------------------------------------------------------------------------------------------------------------------------------------ |
| **Bangladesh Recyclable Waste** | Outdoor and general waste                  |      2,842 |                6,076 | Paper, Plastic Bottle, Plastic bag, Plastic waste                                                                                                                 | [Roboflow Universe](https://universe.roboflow.com/waste-detections-by-fu/waste-detections-c0mkz/dataset/12)                                |
| **MedBin Dataset 2**            | Medical and clinical waste                 |      1,796 |                6,410 | Medical gloves, masks, N95 masks, syringes, needles, blades, scalpels, drug packaging, glass bottles, plastic medical bottles and bags, paper boxes, thermometers | [Roboflow Universe](https://universe.roboflow.com/medical-waste-e3vov/medbin_dataset-2) · [GitHub](https://github.com/Wayne3918/MedbinNet) |
| **Mendeley Trash Detection**    | Household waste and catalog-style images   |      5,049 |               34,082 | Biodegradable, Cardboard, Glass, Metal, Paper, Plastic                                                                                                            | [Mendeley Data](https://data.mendeley.com/datasets/z732f9pwxt/1)                                                                           |
| **PackWISE Packaging Waste**    | High-resolution industrial sorting belt    |        586 |                6,713 | Plastic foil/bags, paper bags and other paper, metal cans and foil, plastic bottles, tissues, plastic cups/lids, blister packs, foam and other packaging          | [Fraunhofer Fordatis](https://fordatis.fraunhofer.de/handle/fordatis/463.2?locale=en)                                                      |
| **RecyBat24**                   | Batteries photographed from multiple views |      2,835 |                2,881 | Cylindrical, pouch, and prismatic batteries                                                                                                                       | [Zenodo](https://zenodo.org/records/13126689)                                                                                              |
| **Total raw pool**              | **Five source datasets**                   | **13,108** |           **56,162** | **73 source classes in total**                                                                                                                                    | —                                                                                                                                          |

_Note:_ The Bangladesh source includes alternate partitions of the same underlying images. They were not counted as a separate dataset.

## 3. Preparing the Labels

Each image needs a label file that identifies its objects and gives the location of each object. The sources used different formats, so the labels were standardised for YOLOv11.

- **Converted shape outlines to boxes:** Some annotations traced the object's outline rather than drawing a rectangle. These were converted to rectangular bounding boxes.
- **Converted JSON annotations:** PackWISE and RecyBat24 stored their labels in JSON files. Their object locations and class IDs were converted into YOLO label files.
- **Resized large images:** PackWISE images were reduced from **4096 × 4096** to **1280 × 1280 pixels** to make them easier to load during training.
- **Removed duplicate or unmatched files:** Identified duplicate images and label files without matching images were removed.
- **Checked box coordinates:** Labels were checked for the required five values, valid class IDs, positive box sizes, and coordinates within the image.

## 4. Selecting and Combining Classes

The five sources originally contained **73 classes**. These were reduced to **10 classes** chosen for Sortify. The aim was to combine similar items without assigning objects to misleading labels.

- **Removed broad labels:** Labels such as `Plastic waste` and `PLASTIC` could include bottles, bags, cups, or wrappers. They were excluded because Sortify treats plastic bottles and plastic bags as separate classes.
- **Removed unclear or unsuitable labels:** Some labels described mixed materials, unclear fragments, specialised laboratory equipment, or examples unsuitable for the planned classes.
- **Removed affected images where needed:** If an image contained an object whose label was being excluded, the whole image was removed in the relevant cases. Otherwise, the model could see an object without being told what it was.
- **Filtered unusual examples:** Very large boxes covering nearly an entire image and dense produce collages were removed from selected Mendeley classes.
- **Limited repeated battery views:** RecyBat24 includes multiple photographs of the same physical battery. The number of views was limited, and all views of one battery were kept in the same data split.

These steps were intended to make the labels more consistent and reduce the chance of the model learning from contradictory examples.

## 5. Which Source Classes Were Used for Each Sortify Class?

This table shows which original dataset labels contributed to each final Sortify detector class. Some source classes were combined because they describe similar objects. Other source classes were left out because they were too broad, unclear, or did not fit the final classes.

| Final Sortify class  | Source dataset              | Original class labels used                                                                                    |
| -------------------- | --------------------------- | ------------------------------------------------------------------------------------------------------------- |
| `paper`              | Mendeley Trash Detection    | `PAPER`                                                                                                       |
| `paper`              | PackWISE                    | `paper-bag`, `paper-other`, `paper-cup/pot`                                                                   |
| `cardboard`          | Mendeley Trash Detection    | `CARDBOARD`                                                                                                   |
| `cardboard`          | MedBin Dataset 2            | `paperbox`, `covid_buffer_box`                                                                                |
| `glass`              | Mendeley Trash Detection    | `GLASS`                                                                                                       |
| `glass`              | MedBin Dataset 2            | `glass_bottle`                                                                                                |
| `metal`              | Mendeley Trash Detection    | `METAL`                                                                                                       |
| `metal`              | PackWISE                    | `metal-can`, `metal-foil`                                                                                     |
| `metal`              | MedBin Dataset 2            | `Oxygen_cylinder`                                                                                             |
| `plastic_bottle`     | Bangladesh Recyclable Waste | `Plastic Bottle`                                                                                              |
| `plastic_bottle`     | PackWISE                    | `plastic-bottle`                                                                                              |
| `plastic_bottle`     | MedBin Dataset 2            | `plastic_medical_bottle`                                                                                      |
| `plastic_bag`        | Bangladesh Recyclable Waste | `Plastic bag`                                                                                                 |
| `plastic_bag`        | PackWISE                    | `plastic-foil/bag`                                                                                            |
| `plastic_bag`        | MedBin Dataset 2            | `plastic_medical_bag`                                                                                         |
| `sanitary_ppe_waste` | PackWISE                    | `tissue`                                                                                                      |
| `sanitary_ppe_waste` | MedBin Dataset 2            | `Mask`, `N95`, `medical_gloves`, `cotton_swab`, `iodine_swab`, `gauze`, `bandage`                             |
| `organic`            | Mendeley Trash Detection    | `BIODEGRADABLE` (after filtering dense produce-collage images)                                                |
| `battery_ewaste`     | RecyBat24                   | `cylindric`, `pouch`, `prismatic`                                                                             |
| `battery_ewaste`     | MedBin Dataset 2            | `Electronic_thermometer`                                                                                      |
| `hazardous_medical`  | MedBin Dataset 2            | `drug_packaging`, `syringe`, `Bloody_objects`, `unguent`, `mercury_thermometer`, `blade`, `scalpel`, `needle` |
| `hazardous_medical`  | PackWISE                    | `blister`                                                                                                     |

### Important exclusions

Some source classes were deliberately not used in the final dataset:

- **Bangladesh Recyclable Waste:** `Paper` and `Plastic waste` were excluded because they combine different object types that Sortify treats as separate classes.
- **Mendeley Trash Detection:** `PLASTIC` was excluded because it mixes different plastic items; some unusually large paper/cardboard boxes and dense organic collages were also filtered.
- **PackWISE:** unclear or mixed classes such as `undefined`, `aluminium-coated-foil`, `beverage-carton`, and several small or ambiguous packaging classes were excluded.
- **MedBin Dataset 2:** specialised laboratory equipment and selected small classes, including loose pills/capsules and the dataset's `Radioactive_objects` examples, were excluded.
- **RecyBat24:** the battery shape labels were combined into one `battery_ewaste` class. Repeated views were limited, and photos of the same physical battery were kept in one split.

The final instance counts in Section 6 are the counts after filtering, merging, and splitting. Source-class counts may not add up exactly to the final class total because images and annotations were removed during cleaning and some counts were measured at different stages.

## 6. Final Classes and Distribution

The model predicts one of the 10 detector classes below. Sortify's backend maps each class to a default disposal group. The correct disposal route can still depend on local rules and whether an item is contaminated.

|  ID | Detector class       | Backend disposal group | Train boxes | Validation boxes | Test boxes | **Total boxes** | Images containing class |
| --: | -------------------- | ---------------------- | ----------: | ---------------: | ---------: | --------------: | ----------------------: |
|   0 | `paper`              | Recyclable             |       1,967 |              272 |        257 |       **2,496** |                     879 |
|   1 | `cardboard`          | Recyclable             |       2,391 |              244 |        228 |       **2,863** |                   1,013 |
|   2 | `glass`              | Recyclable             |       1,653 |              232 |        318 |       **2,203** |                     860 |
|   3 | `metal`              | Recyclable             |       1,989 |              282 |        204 |       **2,475** |                     915 |
|   4 | `plastic_bottle`     | Recyclable             |       1,355 |              177 |        185 |       **1,717** |                   1,081 |
|   5 | `plastic_bag`        | General Waste          |       2,348 |              315 |        309 |       **2,972** |                   1,049 |
|   6 | `sanitary_ppe_waste` | General Waste          |       1,431 |              171 |        186 |       **1,788** |                   1,015 |
|   7 | `organic`            | Organic                |       1,990 |              276 |        180 |       **2,446** |                     618 |
|   8 | `battery_ewaste`     | E-Waste                |       1,184 |              139 |        159 |       **1,482** |                   1,446 |
|   9 | `hazardous_medical`  | Hazardous              |       1,166 |              136 |        142 |       **1,444** |                     983 |
|   — | **TOTAL**            | **Five groups**        |  **17,474** |        **2,244** |  **2,168** |      **21,886** | **7,355 unique images** |

_Note:_ “Images containing class” counts images that include that class. A single image can contain more than one class, so these per-class image counts should not be added together to calculate the number of unique images.

## 7. Training, Validation, and Test Sets

| Split          |    Images | Labelled objects | Purpose                                                               |
| -------------- | --------: | ---------------: | --------------------------------------------------------------------- |
| **Training**   |     5,887 |           17,474 | Used to train the model to recognise the 10 classes.                  |
| **Validation** |       734 |            2,244 | Used during development to compare training runs and choose settings. |
| **Test**       |       734 |            2,168 | Kept separate for the final evaluation after model choices are made.  |
| **Total**      | **7,355** |       **21,886** | All unique images and labelled objects.                               |

The split is approximately **80% training, 10% validation, and 10% testing**. Battery images were split by physical battery ID, so different photographs of the same battery do not appear in more than one split.

## 8. Verification Results

The final dataset passed the following structural checks:

- **Image and label matching:** Every final image has a matching label file; no unmatched images or labels were found.
- **Label format:** All checked object labels contain the required five values: class ID, horizontal centre, vertical centre, width, and height.
- **Bounding boxes:** No malformed label lines or boxes extending outside the image were found.
- **Class IDs:** Labels use the expected class IDs, from 0 to 9.
- **Duplicate image stems:** No duplicate Roboflow image stems were found across the audited splits.
- **Battery separation:** No physical battery ID is shared between the training, validation, and test sets.

These checks show that the files are consistently organised and that the listed split checks passed. They do **not** prove that every object is labelled correctly or that the trained model will perform well on real-world images. Model performance must be measured after training.

## 9. Summary

The final Sortify dataset combines five sources into one YOLOv11-compatible dataset with **7,355 images, 21,886 labelled objects, and 10 detector classes**. Labels were standardised, unsuitable examples were filtered, and the final files passed checks for matching labels, valid bounding boxes, and separation between the data splits.
