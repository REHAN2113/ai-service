import os
import io
import logging
import time
import boto3
import numpy as np
import requests
from PIL import Image
from fastapi import FastAPI, HTTPException, UploadFile, File
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import List, Optional
from botocore.exceptions import ClientError

# ── Structured Logging ──
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%Y-%m-%dT%H:%M:%S",
)
logger = logging.getLogger("snapfind-ai")

# ── AWS Rekognition Client ──
rekognition = boto3.client(
    'rekognition',
    region_name=os.environ.get('AWS_REGION', 'us-east-1'),
    aws_access_key_id=os.environ.get('AWS_ACCESS_KEY_ID'),
    aws_secret_access_key=os.environ.get('AWS_SECRET_ACCESS_KEY')
)

COLLECTION_ID = os.environ.get('AWS_REKOGNITION_COLLECTION_ID', 'snapfind-faces')

# ── FastAPI App ──
app = FastAPI(
    title="SnapFind AI - Face Processing Service",
    version="1.0.0",
    description="AWS Rekognition-powered face detection and embedding extraction",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# ── AWS Collection Manager ──
class CollectionManager:
    _instance = None
    _initialized = False

    def __new__(cls):
        if cls._instance is None:
            cls._instance = super().__new__(cls)
        return cls._instance

    def ensure_collection_exists(self):
        """Ensure the Rekognition collection exists, create if needed"""
        if self._initialized:
            return
        
        logger.info(f"Ensuring AWS Rekognition collection '{COLLECTION_ID}' exists...")
        try:
            try:
                rekognition.describe_collection(CollectionId=COLLECTION_ID)
                logger.info(f"Collection '{COLLECTION_ID}' already exists")
            except ClientError as e:
                if e.response['Error']['Code'] == 'ResourceNotFoundException':
                    logger.info(f"Creating collection '{COLLECTION_ID}'...")
                    rekognition.create_collection(CollectionId=COLLECTION_ID)
                    logger.info(f"Collection '{COLLECTION_ID}' created successfully")
                else:
                    raise
            
            self._initialized = True
        except Exception as e:
            logger.error(f"Failed to initialize collection: {e}")
            raise

    def extract_faces_from_image(self, image_bytes: bytes) -> List[dict]:
        """Extract faces from image bytes using Rekognition detect_faces"""
        try:
            response = rekognition.detect_faces(
                Image={'Bytes': image_bytes},
                Attributes=['ALL']
            )
            
            results = []
            for face_detail in response.get('FaceDetails', []):
                bounding_box = face_detail['BoundingBox']
                embedding = self._face_attributes_to_embedding(face_detail)
                
                results.append({
                    'embedding': embedding,
                    'bounding_box': bounding_box,
                    'confidence': face_detail.get('Confidence', 0),
                })
            
            return results
        except Exception as e:
            logger.error(f"Face detection failed: {e}")
            raise

    def _face_attributes_to_embedding(self, face_detail: dict) -> List[float]:
        """Convert AWS Rekognition face attributes to embedding-like list"""
        embedding = []
        
        # Add confidence score
        embedding.append(face_detail.get('Confidence', 0) / 100.0)
        
        # Add quality metrics
        quality = face_detail.get('Quality', {})
        embedding.append(quality.get('Brightness', 0) / 255.0)
        embedding.append(quality.get('Sharpness', 0) / 255.0)
        
        # Add pose angles (normalized from -180/+180 to 0-1)
        pose = face_detail.get('Pose', {})
        embedding.append((pose.get('Roll', 0) + 180) / 360.0)
        embedding.append((pose.get('Yaw', 0) + 180) / 360.0)
        embedding.append((pose.get('Pitch', 0) + 180) / 360.0)
        
        # Add eye information
        eyes_open = face_detail.get('EyesOpen', {})
        embedding.append(eyes_open.get('Confidence', 0) / 100.0)
        
        # Add mouth open
        mouth_open = face_detail.get('MouthOpen', {})
        embedding.append(mouth_open.get('Confidence', 0) / 100.0)
        
        # Add emotion values as features
        emotions = face_detail.get('Emotions', [])
        emotion_values = [e.get('Confidence', 0) / 100.0 for e in emotions[:5]]
        embedding.extend(emotion_values)
        
        # Add face landmarks if available
        landmarks = face_detail.get('Landmarks', [])
        for landmark in landmarks[:10]:
            embedding.append(landmark.get('X', 0))
            embedding.append(landmark.get('Y', 0))
        
        # Pad to 128 dimensions (standard embedding size)
        while len(embedding) < 128:
            embedding.append(0.0)
        
        return embedding[:128]

collection_manager = CollectionManager()

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
    collectionInitialized: bool
    uptime: float

# ── State ──
startup_time = time.time()

# ── Startup Event ──
@app.on_event("startup")
async def startup_event():
    """Initialize AWS Rekognition collection on startup"""
    logger.info("Initializing AWS Rekognition collection...")
    try:
        collection_manager.ensure_collection_exists()
        logger.info("Collection initialization complete")
    except Exception as e:
        logger.warning(f"Collection initialization failed (will retry on first request): {e}")

# ── Endpoints ──

@app.get("/health", response_model=HealthResponse)
async def health_check():
    return HealthResponse(
        status="healthy",
        collectionInitialized=collection_manager._initialized,
        uptime=time.time() - startup_time,
    )

@app.post("/extract-faces", response_model=ExtractResponse)
async def extract_faces(request: ExtractRequest):
    """Extract face embeddings from an image URL using AWS Rekognition"""
    start = time.time()

    try:
        # Ensure collection exists
        collection_manager.ensure_collection_exists()
        
        # Download image with timeout
        logger.info(f"Downloading image for photo {request.photoId}")
        response = requests.get(request.imageUrl, timeout=30, stream=True)
        response.raise_for_status()

        # Check content length
        content_length = int(response.headers.get("content-length", 0))
        if content_length > 15 * 1024 * 1024:  # 15MB safety margin
            raise HTTPException(status_code=400, detail="Image too large")

        # Validate and load image
        image_bytes = response.content
        try:
            image = Image.open(io.BytesIO(image_bytes)).convert("RGB")
        except Exception:
            raise HTTPException(status_code=400, detail="Invalid image format")

        # Resize if too large (AWS Rekognition supports up to 4096x4096)
        max_dim = 4096
        if max(image.size) > max_dim:
            ratio = max_dim / max(image.size)
            new_size = (int(image.size[0] * ratio), int(image.size[1] * ratio))
            image = image.resize(new_size, Image.LANCZOS)

        # Convert to bytes for Rekognition API
        img_byte_arr = io.BytesIO()
        image.save(img_byte_arr, format='JPEG')
        image_bytes = img_byte_arr.getvalue()

        # Extract faces using Rekognition
        face_data = collection_manager.extract_faces_from_image(image_bytes)

        faces = []
        for face_info in face_data:
            embedding = face_info.get('embedding', [])
            bbox = face_info.get('bounding_box', {})
            
            # Convert relative bounding box coordinates to pixel values
            x = int(bbox.get('Left', 0) * image.width)
            y = int(bbox.get('Top', 0) * image.height)
            w = int(bbox.get('Width', 0) * image.width)
            h = int(bbox.get('Height', 0) * image.height)
            
            if embedding and len(embedding) > 0:
                faces.append(FaceResult(
                    embedding=embedding,
                    box=FaceBox(x=x, y=y, w=w, h=h),
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
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Face extraction failed for photo {request.photoId}: {e}")
        raise HTTPException(status_code=500, detail=f"Face extraction failed: {str(e)}")


@app.post("/compare-faces", response_model=CompareResponse)
async def compare_faces(file: UploadFile = File(...)):
    """Extract embedding from uploaded face image for comparison using AWS Rekognition"""
    try:
        # Ensure collection exists
        collection_manager.ensure_collection_exists()
        
        # Validate file size
        contents = await file.read()
        if len(contents) > 5 * 1024 * 1024:
            raise HTTPException(status_code=400, detail="File too large. Max 5MB.")

        # Validate image format
        try:
            image = Image.open(io.BytesIO(contents)).convert("RGB")
        except Exception:
            raise HTTPException(status_code=400, detail="Invalid image format")

        # Resize for efficiency
        max_dim = 800
        if max(image.size) > max_dim:
            ratio = max_dim / max(image.size)
            new_size = (int(image.size[0] * ratio), int(image.size[1] * ratio))
            image = image.resize(new_size, Image.LANCZOS)

        # Convert to bytes for Rekognition API
        img_byte_arr = io.BytesIO()
        image.save(img_byte_arr, format='JPEG')
        image_bytes = img_byte_arr.getvalue()

        # Extract faces using Rekognition detect_faces
        face_data = collection_manager.extract_faces_from_image(image_bytes)

        if not face_data:
            return CompareResponse(embedding=None, faceDetected=False)

        # Use the first detected face
        first_face = face_data[0]
        embedding = first_face.get('embedding', [])

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
