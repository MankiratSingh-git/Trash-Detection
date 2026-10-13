"""Canonical 10-Class Sortify Waste Taxonomy and Mapping for YOLOv11."""

from __future__ import annotations

from typing import Dict, List, Optional, TypedDict


class CategoryInfo(TypedDict):
    name: str
    category: str
    guidance: str
    color: str
    bgr: tuple[int, int, int]


SORTIFY_CLASSES: Dict[int, CategoryInfo] = {
    0: {
        "name": "paper",
        "category": "Recyclable",
        "guidance": "Clean dry paper bin",
        "color": "#10B981",  # Emerald green
        "bgr": (129, 185, 16),
    },
    1: {
        "name": "cardboard",
        "category": "Recyclable",
        "guidance": "Flatten and place in cardboard/paper recycling",
        "color": "#059669",  # Darker emerald
        "bgr": (105, 150, 5),
    },
    2: {
        "name": "glass",
        "category": "Recyclable",
        "guidance": "Rinse glass bottle/jar and place in glass recycling",
        "color": "#0D9488",  # Teal
        "bgr": (136, 148, 13),
    },
    3: {
        "name": "metal",
        "category": "Recyclable",
        "guidance": "Empty metal can/foil and place in metal recycling",
        "color": "#2563EB",  # Royal Blue
        "bgr": (235, 99, 37),
    },
    4: {
        "name": "plastic_bottle",
        "category": "Recyclable",
        "guidance": "Empty/crush rigid plastic bottle and cap for recycling",
        "color": "#3B82F6",  # Light Blue
        "bgr": (246, 130, 59),
    },
    5: {
        "name": "plastic_bag",
        "category": "General waste",
        "guidance": "General waste bin (or dedicated soft-plastic film drop-off)",
        "color": "#6B7280",  # Gray
        "bgr": (128, 114, 107),
    },
    6: {
        "name": "sanitary_ppe_waste",
        "category": "General waste",
        "guidance": "Seal masks, gloves, tissues, and dressings in general waste",
        "color": "#4B5563",  # Slate Gray
        "bgr": (99, 85, 75),
    },
    7: {
        "name": "organic",
        "category": "Organic",
        "guidance": "Compost or biodegradable organic waste bin",
        "color": "#F59E0B",  # Amber / Orange
        "bgr": (11, 158, 245),
    },
    8: {
        "name": "battery_ewaste",
        "category": "E-waste",
        "guidance": "Never bin; take to battery / e-waste collection point",
        "color": "#8B5CF6",  # Purple
        "bgr": (246, 92, 139),
    },
    9: {
        "name": "hazardous_medical",
        "category": "Hazardous",
        "guidance": "Sharps / hazardous clinical or pharmaceutical disposal container",
        "color": "#EF4444",  # Red
        "bgr": (68, 68, 239),
    },
}

SORTIFY_CATEGORIES: List[str] = [
    "Recyclable",
    "General waste",
    "Organic",
    "E-waste",
    "Hazardous",
]

DEFAULT_INFO: CategoryInfo = {
    "name": "unknown_waste",
    "category": "General waste",
    "guidance": "Place in general waste if unsure",
    "color": "#9CA3AF",
    "bgr": (175, 163, 156),
}


def get_class_info(class_id: int, class_name: Optional[str] = None) -> CategoryInfo:
    """Look up waste class info by ID or fallback to class name matching."""
    if class_id in SORTIFY_CLASSES:
        return SORTIFY_CLASSES[class_id]

    if class_name:
        clean_name = class_name.lower().strip()
        for info in SORTIFY_CLASSES.values():
            if info["name"].lower() == clean_name:
                return info

    return {
        "name": class_name or f"class_{class_id}",
        "category": "General waste",
        "guidance": "Inspect item carefully or dispose in general waste",
        "color": "#9CA3AF",
        "bgr": (175, 163, 156),
    }
