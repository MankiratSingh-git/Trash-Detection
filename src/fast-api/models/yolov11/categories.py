"""Sortify Waste Taxonomy, Category Mappings & Visual Palettes."""

from __future__ import annotations

from typing import Any, Dict, List, Optional, Tuple, TypedDict


class CategoryInfo(TypedDict, total=False):
    name: str
    category: str
    guidance: str
    color: str
    bgr: tuple[int, int, int]
    display_name: str
    item_id: str
    examples: list[str]


SORTIFY_CATEGORIES: List[str] = [
    "Recyclable",
    "General waste",
    "Organic",
    "E-waste",
    "Hazardous",
]

# Standard category colors and OpenCV BGR tuples
CATEGORY_PALETTE: Dict[str, Dict[str, Any]] = {
    "Recyclable": {"color": "#10B981", "bgr": (129, 185, 16)},       # Emerald green
    "General waste": {"color": "#6B7280", "bgr": (128, 114, 107)},    # Slate gray
    "Organic": {"color": "#F59E0B", "bgr": (11, 158, 245)},          # Amber
    "E-waste": {"color": "#8B5CF6", "bgr": (246, 92, 139)},          # Purple
    "Hazardous": {"color": "#EF4444", "bgr": (68, 68, 239)},          # Red
}

# Specific visual colors per class
CLASS_COLORS: Dict[str, Dict[str, Any]] = {
    # 8-Class Streamlined Taxonomy
    "paper_cardboard": {"color": "#10B981", "bgr": (129, 185, 16)},
    "plastic_bottle": {"color": "#3B82F6", "bgr": (246, 130, 59)},
    "metal": {"color": "#2563EB", "bgr": (235, 99, 37)},
    "flexible_plastic": {"color": "#6B7280", "bgr": (128, 114, 107)},
    "sanitary_ppe_waste": {"color": "#4B5563", "bgr": (99, 85, 75)},
    "organic": {"color": "#F59E0B", "bgr": (11, 158, 245)},
    "battery_ewaste": {"color": "#8B5CF6", "bgr": (246, 92, 139)},
    "hazardous_medical": {"color": "#EF4444", "bgr": (68, 68, 239)},
    # Legacy 10-Class specific distinctions
    "paper": {"color": "#10B981", "bgr": (129, 185, 16)},
    "cardboard": {"color": "#059669", "bgr": (105, 150, 5)},
    "glass": {"color": "#0D9488", "bgr": (136, 148, 13)},
    "plastic_bag": {"color": "#6B7280", "bgr": (128, 114, 107)},
}


def get_category_color(category: str, class_name: Optional[str] = None) -> Tuple[str, Tuple[int, int, int]]:
    """Resolve hex color and BGR tuple for a category or class name."""
    if class_name and class_name.lower().strip() in CLASS_COLORS:
        entry = CLASS_COLORS[class_name.lower().strip()]
        return entry["color"], entry["bgr"]

    if category in CATEGORY_PALETTE:
        entry = CATEGORY_PALETTE[category]
        return entry["color"], entry["bgr"]

    return "#9CA3AF", (175, 163, 156)


# Canonical Streamlined 8-Class Mapping
SORTIFY_CLASSES_8CLS: Dict[int, CategoryInfo] = {
    0: {
        "name": "paper_cardboard",
        "category": "Recyclable",
        "display_name": "Paper & Cardboard",
        "item_id": "paper-cardboard",
        "guidance": "Clean dry paper, boxes, and cardboard packaging in paper recycling bin",
        "examples": ["Cardboard box", "Newspaper", "Office paper", "Carton", "Paper bag"],
        "color": "#10B981",
        "bgr": (129, 185, 16),
    },
    1: {
        "name": "plastic_bottle",
        "category": "Recyclable",
        "display_name": "Plastic Bottle & Container",
        "item_id": "plastic-bottle",
        "guidance": "Empty and crush rigid plastic bottles, jugs, and tubs for recycling",
        "examples": ["Water bottle", "Milk jug", "Detergent container", "Yogurt tub"],
        "color": "#3B82F6",
        "bgr": (246, 130, 59),
    },
    2: {
        "name": "metal",
        "category": "Recyclable",
        "display_name": "Metal Can & Packaging",
        "item_id": "metal",
        "guidance": "Empty drink cans, food tins, clean scrunched foil in metal recycling",
        "examples": ["Drink can", "Food tin", "Aluminium foil", "Aerosol can"],
        "color": "#2563EB",
        "bgr": (235, 99, 37),
    },
    3: {
        "name": "flexible_plastic",
        "category": "General waste",
        "display_name": "Flexible Plastic Packaging",
        "item_id": "flexible-plastic",
        "guidance": "General waste bin (or designated supermarket soft-plastic collection point)",
        "examples": ["Snack wrapper", "Plastic shopping bag", "Bubble wrap", "Cling film", "Pouch"],
        "color": "#6B7280",
        "bgr": (128, 114, 107),
    },
    4: {
        "name": "sanitary_ppe_waste",
        "category": "General waste",
        "display_name": "Sanitary & PPE Waste",
        "item_id": "sanitary-ppe",
        "guidance": "Seal disposable masks, gloves, tissues, and wipes securely in general waste",
        "examples": ["Face mask", "Used tissue", "Cleaning wipe", "Disposable gloves"],
        "color": "#4B5563",
        "bgr": (99, 85, 75),
    },
    5: {
        "name": "organic",
        "category": "Organic",
        "display_name": "Organic Waste",
        "item_id": "organic-waste",
        "guidance": "Compost or curbside green-lid FOGO organic waste bin",
        "examples": ["Vegetable scraps", "Banana peel", "Coffee grounds", "Garden leaves"],
        "color": "#F59E0B",
        "bgr": (11, 158, 245),
    },
    6: {
        "name": "battery_ewaste",
        "category": "E-waste",
        "display_name": "Battery & E-waste",
        "item_id": "battery-ewaste",
        "guidance": "Never bin; tape terminals and bring to battery or e-waste collection depot",
        "examples": ["Household battery", "Mobile phone", "Charger cable", "Headphones", "Vape"],
        "color": "#8B5CF6",
        "bgr": (246, 92, 139),
    },
    7: {
        "name": "hazardous_medical",
        "category": "Hazardous",
        "display_name": "Hazardous Medical Waste",
        "item_id": "hazardous-medical",
        "guidance": "Drop sharps into puncture-proof container; return expired pills to pharmacy",
        "examples": ["Syringe & needle", "Expired medicine", "Blister pack", "Clinical dressing"],
        "color": "#EF4444",
        "bgr": (68, 68, 239),
    },
}

# Legacy 10-Class Mapping
SORTIFY_CLASSES_10CLS: Dict[int, CategoryInfo] = {
    0: {
        "name": "paper",
        "category": "Recyclable",
        "display_name": "Paper",
        "item_id": "paper",
        "guidance": "Clean dry paper bin",
        "examples": ["Newspaper", "Office paper", "Envelopes", "Magazines"],
        "color": "#10B981",
        "bgr": (129, 185, 16),
    },
    1: {
        "name": "cardboard",
        "category": "Recyclable",
        "display_name": "Cardboard Box",
        "item_id": "box",
        "guidance": "Flatten and place in cardboard/paper recycling",
        "examples": ["Cardboard box", "Corrugated carton", "Shipping box"],
        "color": "#059669",
        "bgr": (105, 150, 5),
    },
    2: {
        "name": "glass",
        "category": "Recyclable",
        "display_name": "Glass Container",
        "item_id": "jar",
        "guidance": "Rinse glass bottle/jar and place in glass recycling",
        "examples": ["Glass jar", "Wine bottle", "Beverage bottle"],
        "color": "#0D9488",
        "bgr": (136, 148, 13),
    },
    3: {
        "name": "metal",
        "category": "Recyclable",
        "display_name": "Metal Can & Packaging",
        "item_id": "metal",
        "guidance": "Empty metal can/foil and place in metal recycling",
        "examples": ["Drink can", "Food tin", "Aluminium foil"],
        "color": "#2563EB",
        "bgr": (235, 99, 37),
    },
    4: {
        "name": "plastic_bottle",
        "category": "Recyclable",
        "display_name": "Plastic Bottle & Container",
        "item_id": "plastic-bottle",
        "guidance": "Empty/crush rigid plastic bottle and cap for recycling",
        "examples": ["Water bottle", "Milk jug", "Detergent container"],
        "color": "#3B82F6",
        "bgr": (246, 130, 59),
    },
    5: {
        "name": "plastic_bag",
        "category": "General waste",
        "display_name": "Plastic Bag & Flexible Film",
        "item_id": "flexible-plastic",
        "guidance": "General waste bin (or dedicated soft-plastic film drop-off)",
        "examples": ["Plastic shopping bag", "Snack wrapper", "Bubble wrap"],
        "color": "#6B7280",
        "bgr": (128, 114, 107),
    },
    6: {
        "name": "sanitary_ppe_waste",
        "category": "General waste",
        "display_name": "Sanitary & PPE Waste",
        "item_id": "sanitary-ppe",
        "guidance": "Seal masks, gloves, tissues, and dressings in general waste",
        "examples": ["Face mask", "Used tissue", "Cleaning wipe"],
        "color": "#4B5563",
        "bgr": (99, 85, 75),
    },
    7: {
        "name": "organic",
        "category": "Organic",
        "display_name": "Organic Waste",
        "item_id": "organic-waste",
        "guidance": "Compost or biodegradable organic waste bin",
        "examples": ["Vegetable scraps", "Banana peel", "Garden leaves"],
        "color": "#F59E0B",
        "bgr": (11, 158, 245),
    },
    8: {
        "name": "battery_ewaste",
        "category": "E-waste",
        "display_name": "Battery & E-waste",
        "item_id": "battery-ewaste",
        "guidance": "Never bin; take to battery / e-waste collection point",
        "examples": ["Household battery", "Mobile phone", "Charger cable"],
        "color": "#8B5CF6",
        "bgr": (246, 92, 139),
    },
    9: {
        "name": "hazardous_medical",
        "category": "Hazardous",
        "display_name": "Hazardous Medical Waste",
        "item_id": "hazardous-medical",
        "guidance": "Sharps / hazardous clinical or pharmaceutical disposal container",
        "examples": ["Syringe & needle", "Expired medicine", "Blister pack"],
        "color": "#EF4444",
        "bgr": (68, 68, 239),
    },
}

# Default canonical reference is the 8-class streamlined taxonomy
SORTIFY_CLASSES: Dict[int, CategoryInfo] = SORTIFY_CLASSES_8CLS

DEFAULT_INFO: CategoryInfo = {
    "name": "unknown_waste",
    "category": "General waste",
    "guidance": "Place in general waste if unsure",
    "color": "#9CA3AF",
    "bgr": (175, 163, 156),
}


def get_class_info(
    class_id: int,
    class_name: Optional[str] = None,
    custom_mapping: Optional[Dict[int, CategoryInfo]] = None,
) -> CategoryInfo:
    """Look up waste class info by ID or class name with custom mapping support."""
    # 1. Custom model-specific mapping takes highest priority
    if custom_mapping:
        if class_id in custom_mapping:
            return custom_mapping[class_id]
        if class_name:
            cname = class_name.lower().strip()
            for info in custom_mapping.values():
                if info["name"].lower() == cname:
                    return info

    # 2. Check by exact class name in known taxonomies (avoids 8 vs 10 class index collision)
    if class_name:
        clean_name = class_name.lower().strip()
        for mapping in (SORTIFY_CLASSES_8CLS, SORTIFY_CLASSES_10CLS):
            for info in mapping.values():
                if info["name"].lower() == clean_name:
                    return info

    # 3. Fallback to ID lookup in default taxonomy
    if class_id in SORTIFY_CLASSES:
        return SORTIFY_CLASSES[class_id]

    color, bgr = get_category_color("General waste", class_name)
    return {
        "name": class_name or f"class_{class_id}",
        "category": "General waste",
        "guidance": "Inspect item carefully or dispose in general waste",
        "color": color,
        "bgr": bgr,
    }
