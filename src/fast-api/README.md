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
│   │   │   ├── categories.py   # Sortify 10-class waste taxonomy & disposal bins
│   │   │   └── service.py      # YOLOv11 detector implementation
│   │   └── __init__.py         # Registers available models in registry
│   ├── routers/
│   │   ├── health.py           # /health and /api/v1/models discovery
│   │   └── yolov11.py          # YOLOv11 endpoints (/predict, /annotate, etc.)
│   ├── main.py                 # FastAPI app, middleware, and frontend mount
│   ├── run.py                  # Standalone execution script
│   └── requirements.txt
└── frontend/
    ├── index.html              # Modern, responsive test interface
    ├── style.css               # Clean dark-themed CSS
    └── app.js                  # Interactive canvas overlay, sliders, and API client
```

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

### 1. YOLOv11 Endpoints
| Method | Endpoint | Description |
|---|---|---|
| `POST` | `/api/v1/yolov11/predict` | Upload an image, get bounding boxes, confidence, Sortify waste category, and disposal guidance |
| `POST` | `/api/v1/yolov11/annotate` | Upload an image, receive the annotated JPEG directly |
| `POST` | `/api/v1/yolov11/predict-batch`| Upload multiple images for batch inference |
| `GET`  | `/api/v1/yolov11/info` | Inspect model metadata, classes, categories, and device |
| `POST` | `/api/v1/yolov11/reload` | Dynamically reload weights or point to a custom checkpoint |

### 2. System Endpoints
| Method | Endpoint | Description |
|---|---|---|
| `GET` | `/health` | Check service health and loaded model instances |
| `GET` | `/api/v1/models` | List all registered vision models and metadata |

---

## 🛠️ Adding New Models in the Future

The architecture is built according to the **Open-Closed Principle (OCP)**. To add a new model (e.g., `yolov8`, `resnet`, or `clip`):

### Step 1: Implement the Model Service
Create a new directory in `src/fast-api/models/<model_name>/service.py`:
```python
from ...core.base_model import BaseVisionModel
from ...core.schemas import PredictionResponse, ModelMetadata

class MyNewModel(BaseVisionModel):
    def __init__(self):
        super().__init__(name="mynewmodel", version="1.0.0", description="...")

    def load(self, weights_path=None):
        ...

    def predict(self, image, conf=0.25, iou=0.45, **kwargs) -> PredictionResponse:
        ...

    def annotate(self, image, **kwargs):
        ...

    def get_metadata(self) -> ModelMetadata:
        ...
```

### Step 2: Register in `src/fast-api/models/__init__.py`
```python
from ..core.registry import model_registry
from .mynewmodel.service import MyNewModel

model_registry.register("mynewmodel", lambda: MyNewModel())
```

### Step 3: Add Router in `src/fast-api/routers/mynewmodel.py`
Create endpoint handlers and include them in `src/fast-api/main.py`:
```python
app.include_router(mynewmodel_router)
```
The model will automatically appear in `/api/v1/models` and in the frontend dropdown!

---

## ♻️ Sortify Waste Categories

| Class ID | Class Name | Sortify Category | Guidance |
|---|---|---|---|
| 0 | paper | Recyclable | Clean dry paper bin |
| 1 | cardboard | Recyclable | Flatten and place in cardboard/paper recycling |
| 2 | glass | Recyclable | Rinse glass bottle/jar and place in glass recycling |
| 3 | metal | Recyclable | Empty metal can/foil and place in metal recycling |
| 4 | plastic_bottle | Recyclable | Empty/crush rigid plastic bottle and cap for recycling |
| 5 | plastic_bag | General waste | General waste bin (or dedicated soft-plastic drop-off) |
| 6 | sanitary_ppe_waste | General waste | Seal masks, gloves, tissues in general waste |
| 7 | organic | Organic | Compost or biodegradable organic waste bin |
| 8 | battery_ewaste | E-waste | Never bin; take to battery / e-waste collection point |
| 9 | hazardous_medical | Hazardous | Sharps / hazardous medical container |
