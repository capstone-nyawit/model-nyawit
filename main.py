import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

import tensorflow as tf
import keras_cv
from fastapi import FastAPI, UploadFile, File
import cv2
import numpy as np
from pathlib import Path
from pydantic import BaseModel
import urllib.request

app = FastAPI(title="Palm Detection ML Service")

# Setup constants
IMG_SIZE = 800
BBOX_FORMAT = "xywh"
CKPT_PATH = Path("working/runs/palm_detection/retinanet_v2/best.weights.h5")

# Global reference to the loaded model
model = None

def get_model():
    global model
    if model is None:
        print("Loading model architecture...")
        model = keras_cv.models.RetinaNet.from_preset(
            "resnet50_imagenet", 
            num_classes=5, 
            bounding_box_format=BBOX_FORMAT
        )
        model.prediction_decoder = keras_cv.layers.NonMaxSuppression(
            bounding_box_format=BBOX_FORMAT, 
            from_logits=True, 
            iou_threshold=0.5, 
            confidence_threshold=0.05
        )

        if CKPT_PATH.exists():
            print(f"Loading weights from {CKPT_PATH}...")
            model.load_weights(str(CKPT_PATH))
            print("Weights loaded successfully.")
        else:
            print(f"WARNING: Checkpoint {CKPT_PATH} not found!")
    return model

@app.get("/")
def health_check():
    return {"status": "ok", "message": "Palm Detection ML Service is running"}

class PredictRequest(BaseModel):
    image_url: str

@app.post("/predict")
async def predict(payload: PredictRequest):
    try:
        # Download image from Cloudinary URL
        req = urllib.request.urlopen(payload.image_url)
        nparr = np.asarray(bytearray(req.read()), dtype=np.uint8)
        img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
    except Exception as e:
        return {"error": f"Failed to download image: {str(e)}"}
        
    if img is None:
        return {"error": "Invalid image"}
        
    # Preprocess image
    original_h, original_w = img.shape[:2]
    img_resized = cv2.resize(img, (IMG_SIZE, IMG_SIZE))
    img_rgb = cv2.cvtColor(img_resized, cv2.COLOR_BGR2RGB)
    
    # Expand dims
    input_tensor = tf.expand_dims(img_rgb, axis=0)
    input_tensor = tf.cast(input_tensor, tf.float32)
    
    # Inference (using lazily loaded model)
    y_pred = get_model().predict(input_tensor)
    
    # Process results
    boxes = np.array(y_pred["boxes"][0])
    confidence = np.array(y_pred["confidence"][0])
    classes = np.array(y_pred["classes"][0])
    
    valid_indices = confidence > 0
    valid_boxes = boxes[valid_indices]
    valid_confidence = confidence[valid_indices]
    valid_classes = classes[valid_indices]
    
    results = []
    for i in range(len(valid_boxes)):
        box = valid_boxes[i]
        x, y, w, h = box
        
        norm_x = float(x / IMG_SIZE)
        norm_y = float(y / IMG_SIZE)
        norm_w = float(w / IMG_SIZE)
        norm_h = float(h / IMG_SIZE)
        
        results.append({
            "box": [norm_x, norm_y, norm_w, norm_h],
            "confidence": float(valid_confidence[i]),
            "class_id": int(valid_classes[i])
        })
        
    return {
        "predictions": results,
        "original_shape": [original_h, original_w]
    }

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8001)
