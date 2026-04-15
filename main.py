import os
import io
import logging
import time
import numpy as np
import requests
from PIL import Image
from fastapi import FastAPI, HTTPException, UploadFile, File
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import List, Optional

# ── Structured Logging ──
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%Y-%m-%dT%H:%M:%S",
)
logger = logging.getLogger("snapfind-ai")

# ── FastAPI App ──
app = FastAPI(
    title="SnapFind AI - Face Processing Service",
    version="1.0.0",
    description="DeepFace-powered face detection and embedding extraction",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# ── Lazy Model Loading Singleton ──
class ModelManager:
    _instance = None
    _model_loaded = False

    def __new__(cls):
        if cls._instance is None:
            cls._instance = super().__new__(cls)
        return cls._instance

    def ensure_loaded(self):
        if not self._model_loaded:
            logger.info("Loading DeepFace model (first use)...")
            start = time.time()
            try:
                from deepface import DeepFace
                # Warm up with a dummy image to pre-load weights
                dummy = np.zeros((100, 100, 3), dtype=np.uint8)
                dummy_img = Image.fromarray(dummy)
                buf = io.BytesIO()
                dummy_img.save(buf, format="PNG")
                buf.seek(0)
                try:
                    DeepFace.represent(
                        img_path=np.array(dummy_img),
                        model_name="Facenet",
                        enforce_detection=False,
                        detector_backend="opencv",
                    )
                except Exception:
                    pass  # Expected for dummy image
                self._model_loaded = True
                elapsed = time.time() - start
                logger.info(f"Model loaded in {elapsed:.2f}s")
            except Exception as e:
                logger.error(f"Failed to load model: {e}")
                raise

    def extract_faces(self, image_array: np.ndarray):
        self.ensure_loaded()
        from deepface import DeepFace

        results = DeepFace.represent(
            img_path=image_array,
            model_name="Facenet",
            enforce_detection=False,
            detector_backend="opencv",
        )
        return results

model_manager = ModelManager()

# ── Pydantic Models ──
class FaceBox(BaseModel):
    x: int
    y: int
    w: int
    h: int

class FaceResult(BaseModel):
    embedding: List[float]
    box: FaceBox

class ExtractRequest(BaseModel):
    imageUrl: str
    photoId: str
    eventId: str

class ExtractResponse(BaseModel):
    photoId: str
    eventId: str
    faces: List[FaceResult]
    processingTime: float

class CompareResponse(BaseModel):
    embedding: Optional[List[float]]
    faceDetected: bool

class HealthResponse(BaseModel):
    status: str
    modelLoaded: bool
    uptime: float

# ── State ──
startup_time = time.time()

# ── Startup Event ──
@app.on_event("startup")
async def startup_event():
    """Warm up model on startup for faster first request"""
    logger.info("Starting model warmup...")
    try:
        model_manager.ensure_loaded()
        logger.info("Model warmup complete")
    except Exception as e:
        logger.warning(f"Model warmup failed (will retry on first request): {e}")

# ── Endpoints ──

@app.get("/health", response_model=HealthResponse)
async def health_check():
    return HealthResponse(
        status="healthy",
        modelLoaded=model_manager._model_loaded,
        uptime=time.time() - startup_time,
    )

@app.post("/extract-faces", response_model=ExtractResponse)
async def extract_faces(request: ExtractRequest):
    """Extract face embeddings from an image URL"""
    start = time.time()

    try:
        # Download image with timeout
        logger.info(f"Downloading image for photo {request.photoId}")
        response = requests.get(request.imageUrl, timeout=30, stream=True)
        response.raise_for_status()

        # Check content length
        content_length = int(response.headers.get("content-length", 0))
        if content_length > 15 * 1024 * 1024:  # 15MB safety margin
            raise HTTPException(status_code=400, detail="Image too large")

        # Memory-safe image loading
        image_data = io.BytesIO(response.content)
        image = Image.open(image_data).convert("RGB")

        # Resize if too large (memory optimization)
        max_dim = 2000
        if max(image.size) > max_dim:
            ratio = max_dim / max(image.size)
            new_size = (int(image.size[0] * ratio), int(image.size[1] * ratio))
            image = image.resize(new_size, Image.LANCZOS)

        image_array = np.array(image)

        # Extract faces
        results = model_manager.extract_faces(image_array)

        faces = []
        for result in results:
            if isinstance(result, dict):
                embedding = result.get("embedding", [])
                facial_area = result.get("facial_area", {})

                if embedding and len(embedding) > 0:
                    faces.append(FaceResult(
                        embedding=embedding,
                        box=FaceBox(
                            x=facial_area.get("x", 0),
                            y=facial_area.get("y", 0),
                            w=facial_area.get("w", 0),
                            h=facial_area.get("h", 0),
                        ),
                    ))

        elapsed = time.time() - start
        logger.info(
            f"Extracted {len(faces)} faces from photo {request.photoId} "
            f"in {elapsed:.2f}s"
        )

        return ExtractResponse(
            photoId=request.photoId,
            eventId=request.eventId,
            faces=faces,
            processingTime=elapsed,
        )

    except requests.exceptions.Timeout:
        logger.error(f"Timeout downloading image for photo {request.photoId}")
        raise HTTPException(status_code=408, detail="Image download timed out")
    except requests.exceptions.RequestException as e:
        logger.error(f"Failed to download image: {e}")
        raise HTTPException(status_code=400, detail=f"Failed to download image: {str(e)}")
    except Exception as e:
        logger.error(f"Face extraction failed for photo {request.photoId}: {e}")
        raise HTTPException(status_code=500, detail=f"Face extraction failed: {str(e)}")


@app.post("/compare-faces", response_model=CompareResponse)
async def compare_faces(file: UploadFile = File(...)):
    """Extract embedding from uploaded face image for comparison"""
    try:
        # Validate file size
        contents = await file.read()
        if len(contents) > 5 * 1024 * 1024:
            raise HTTPException(status_code=400, detail="File too large. Max 5MB.")

        # Load image
        image = Image.open(io.BytesIO(contents)).convert("RGB")

        # Resize for efficiency
        max_dim = 800
        if max(image.size) > max_dim:
            ratio = max_dim / max(image.size)
            new_size = (int(image.size[0] * ratio), int(image.size[1] * ratio))
            image = image.resize(new_size, Image.LANCZOS)

        image_array = np.array(image)

        results = model_manager.extract_faces(image_array)

        if not results:
            return CompareResponse(embedding=None, faceDetected=False)

        # Use the first detected face
        first_face = results[0] if isinstance(results[0], dict) else {}
        embedding = first_face.get("embedding", [])

        if not embedding:
            return CompareResponse(embedding=None, faceDetected=False)

        return CompareResponse(embedding=embedding, faceDetected=True)

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Face comparison failed: {e}")
        raise HTTPException(status_code=500, detail=f"Face comparison failed: {str(e)}")


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
