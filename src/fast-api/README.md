# Sortify Multi-Model FastAPI Backend

A modular, extensible Computer Vision API for Sortify waste detection and disposal categorization. Built with **FastAPI**, **Ultralytics YOLO11**, and **OpenCV**, designed to scale to multiple models with zero friction.

---

## 📁 Directory Structure

```text
src/
├── fast-api/
│   ├── core/
│   │   ├── base_model.py       # Abstract BaseVisionModel interface
│   │   ├── config.py           # Settings, weights paths, and CORS
│   │   ├── registry.py         # Dynamic ModelRegistry for model discovery
│   │   └── schemas.py          # Pydantic schemas (DetectionResult, BoundingBox, etc.)
│   ├── models/
│   │   ├── yolov11/
│   │   │   ├── categories.py   # Sortify 8-class & 10-class taxonomy & disposal bins
│   │   │   └── service.py      # YOLOv11 detector implementation with auto-mapping
│   │   └── __init__.py         # Registers available models in registry
│   ├── routers/
│   │   ├── health.py           # /health and /api/v1/models discovery
│   │   ├── yolov11.py          # Dedicated YOLOv11 endpoints
│   │   └── models.py           # Dynamic endpoints for ANY registered model (/api/v1/{model_name}/...)
│   ├── main.py                 # FastAPI app, middleware, and frontend mount
│   ├── run.py                  # Standalone execution script
│   └── requirements.txt
└── frontend/
    ├── index.html              # Modern, responsive test interface with model selection
    ├── style.css               # Clean dark-themed CSS
    └── app.js                  # Interactive canvas overlay, sliders, and API client
```

---

## 🎯 Available Models & Options

The backend supports multiple model checkpoints registered in the ModelRegistry:

| Model ID | Architecture | Classes | Checkpoint Path | Description |
|---|---|---|---|---|
| **`yolov11`** *(Default)* | YOLO11s | **8 classes** | `YOLOv11/model/best.pt` | Production deployed streamlined model |
| **`yolov11s_8cls`** | YOLO11s | **8 classes** | `YOLOv11/runs/sortify_yolo11s_8cls_streamlined/weights/best.pt` | Streamlined 8-class detector |
| **`yolov11s_10cls`** | YOLO11s | **10 classes**| `YOLOv11/runs/sortify_yolo11s_10cls/weights/best.pt` | Legacy 10-class detector |
| **`yolov11m_calibrated`**| YOLO11m | **10 classes**| `YOLOv11/runs/sortify_yolo11m_calibrated_aug/weights/best.pt` | Calibrated medium 10-class detector |

---

## 🚀 Quickstart: Running the Server

### Option 1: Using the runner script
```bash
python src/fast-api/run.py
```

### Option 2: Using Uvicorn directly
```bash
python -m uvicorn main:app --app-dir src/fast-api --reload --port 8000
```

Once running:
- **Testing Frontend UI:** [http://localhost:8000/](http://localhost:8000/)
- **Interactive Swagger Docs:** [http://localhost:8000/docs](http://localhost:8000/docs)
- **ReDoc:** [http://localhost:8000/redoc](http://localhost:8000/redoc)

---

## 📡 API Endpoints

### 1. Dynamic Model Endpoints (Works for ANY registered model)
| Method | Endpoint | Description |
|---|---|---|
| `POST` | `/api/v1/{model_name}/predict` | Upload an image, get bounding boxes, confidence, Sortify waste category, and disposal guidance |
| `POST` | `/api/v1/{model_name}/annotate` | Upload an image, receive the annotated JPEG directly |
| `POST` | `/api/v1/{model_name}/predict-batch`| Upload multiple images for batch inference |
| `GET`  | `/api/v1/{model_name}/info` | Inspect model metadata, classes, categories, and device |
| `POST` | `/api/v1/{model_name}/reload` | Dynamically reload weights or point to a custom checkpoint |

*Examples:*
- `/api/v1/yolov11/predict` (Default deployed model)
- `/api/v1/yolov11s_8cls/predict` (Streamlined 8-class model)
- `/api/v1/yolov11s_10cls/predict` (Legacy 10-class model)

### 2. System Endpoints
| Method | Endpoint | Description |
|---|---|---|
| `GET` | `/health` | Check service health and loaded model instances |
| `GET` | `/api/v1/models` | List all registered vision models and metadata |

---

## ♻️ Streamlined 8-Class Waste Taxonomy

| Class ID | Class Name | Sortify Category | Guidance |
|---|---|---|---|
| 0 | `paper_cardboard` | Recyclable | Clean dry paper, box, or cardboard recycling bin |
| 1 | `plastic_bottle` | Recyclable | Empty/crush rigid plastic bottle and cap for recycling |
| 2 | `metal` | Recyclable | Empty metal can, foil, or lid and place in metal recycling |
| 3 | `flexible_plastic` | General waste | General waste bin (or dedicated soft-plastic film drop-off) |
| 4 | `sanitary_ppe_waste` | General waste | Seal masks, gloves, tissues, and dressings in general waste |
| 5 | `organic` | Organic | Compost or biodegradable organic waste bin |
| 6 | `battery_ewaste` | E-waste | Never bin; take to battery / e-waste collection point |
| 7 | `hazardous_medical` | Hazardous | Sharps / hazardous clinical or pharmaceutical disposal container |
