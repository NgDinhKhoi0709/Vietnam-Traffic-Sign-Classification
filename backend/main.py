from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.middleware.cors import CORSMiddleware
import os
import numpy as np
import cv2
import joblib
import pickle
from tensorflow.keras.models import load_model
from skimage.feature import hog
from skimage import exposure
import sys

# Add parent directory to path to allow importing modules from root
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(BASE_DIR)

try:
    from histogram_feature import extract_histogram_features
    from ccv_feature import extract_ccv_features
    # HOG is simple enough to keep inline or we could import if we wanted
except ImportError as e:
    print(f"Warning: Could not import feature extraction modules: {e}")

app = FastAPI()

# Configure CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Allow all origins for dev
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# --- CONFIGURATION ---
# 1. HOG Config
HOG_TARGET_SIZE = (64, 64)
HOG_ORIENTATIONS = 6
HOG_PIXELS_PER_CELL = (8, 8)
HOG_CELLS_PER_BLOCK = (3, 3)

# 2. Histogram Config
HIST_TARGET_SIZE = (64, 64)
HIST_BINS = (8, 8, 8)
HIST_COLOR_SPACE = 'BGR'

# 3. CCV Config
CCV_TARGET_SIZE = (64, 64)
CCV_BINS = 4
CCV_THRESHOLD = 200
CCV_COLOR_SPACE = 'BGR' # Based on user prompt image (BGR)

# Paths
SVM_HOG_DIR = os.path.join(BASE_DIR, "hog_results", "svm_kernel-rbf_C-10_gamma-scale")
SVM_HIST_DIR = os.path.join(BASE_DIR, "histogram_results", "svm_kernel-rbf_C-10_gamma-scale")
SVM_CCV_DIR = os.path.join(BASE_DIR, "ccv_results", "svm_kernel-rbf_C-10_gamma-scale")

VGG16_MODEL_PATH = os.path.join(BASE_DIR, "vgg16_results_128x128", "traffic_sign_cnn_model.keras")

# Global variables for models
models = {
    "hog": {"model": None, "scaler": None, "le": None, "loaded": False},
    "hist": {"model": None, "scaler": None, "le": None, "loaded": False},
    "ccv": {"model": None, "scaler": None, "le": None, "loaded": False},
    "vgg16": {"model": None, "le": None, "loaded": False}
}

# --- HELPER FUNCTIONS ---
def load_svm_model(model_dir, key):
    global models
    try:
        print(f"Loading SVM ({key}) from {model_dir}...")
        if not os.path.exists(model_dir):
             print(f"❌ Directory not found: {model_dir}")
             return

        models[key]["model"] = joblib.load(os.path.join(model_dir, "svm_model.pkl"))
        try:
            models[key]["scaler"] = joblib.load(os.path.join(model_dir, "scaler.pkl"))
        except:
            with open(os.path.join(model_dir, "scaler.pkl"), 'rb') as f:
                models[key]["scaler"] = pickle.load(f)
        
        models[key]["le"] = joblib.load(os.path.join(model_dir, "label_encoder.pkl"))
        models[key]["loaded"] = True
        print(f"✅ SVM ({key}) loaded.")
    except Exception as e:
        print(f"❌ Failed to load SVM ({key}): {e}")

def load_vgg16_resources():
    global models
    try:
        print(f"Loading VGG16 from {VGG16_MODEL_PATH}...")
        if os.path.exists(VGG16_MODEL_PATH):
            models["vgg16"]["model"] = load_model(VGG16_MODEL_PATH)
            # Try to load VGG16 specific label encoder if exists
            cnn_le_path = os.path.join(os.path.dirname(VGG16_MODEL_PATH), "label_encoder.pkl")
            if os.path.exists(cnn_le_path):
                try:
                    models["vgg16"]["le"] = joblib.load(cnn_le_path)
                except:
                    with open(cnn_le_path, 'rb') as f:
                        models["vgg16"]["le"] = pickle.load(f)
            print("✅ VGG16 loaded.")
            models["vgg16"]["loaded"] = True
        else:
            print(f"❌ VGG16 model file not found at {VGG16_MODEL_PATH}")
    except Exception as e:
        print(f"❌ Failed to load VGG16: {e}")

def extract_hog_features_single(image, orientations=9, pixels_per_cell=(8, 8), 
                        cells_per_block=(2, 2), visualize=False, multichannel=False):
    # Convert to gray if needed
    if len(image.shape) == 3 and not multichannel:
        image_processed = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    else:
        image_processed = image
    
    features = hog(
        image_processed,
        orientations=orientations,
        pixels_per_cell=pixels_per_cell,
        cells_per_block=cells_per_block,
        visualize=False,
        channel_axis=-1 if multichannel and len(image.shape) == 3 else None
    )
    return features

# --- STARTUP ---
@app.on_event("startup")
async def startup_event():
    load_svm_model(SVM_HOG_DIR, "hog")
    load_svm_model(SVM_HIST_DIR, "hist")
    load_svm_model(SVM_CCV_DIR, "ccv")
    load_vgg16_resources()

# --- ENDPOINTS ---
@app.get("/")
async def root():
    return {"status": "ok", "message": "Traffic Sign Recognition API is running"}

@app.post("/predict")
async def predict(file: UploadFile = File(...)):
    # Read image
    contents = await file.read()
    nparr = np.frombuffer(contents, np.uint8)
    image = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
    
    if image is None:
        raise HTTPException(status_code=400, detail="Invalid image file")

    results = {
        "svm": {},
        "vgg16": {}
    }

    # --- 1. SVM Predictions ---
    # (SVM prediction logic removed as requested to only focus on VGG16 for frontend display, 
    # although the backend logic is kept in case you want to revert or use it later, 
    # but let's comment it out to save processing time if frontend doesn't need it)
    
    # A. HOG
    # if models["hog"]["loaded"]: ...
    
    # --- 2. VGG16 Prediction ---
    if models["vgg16"]["loaded"]:
        try:
            # Preprocess for VGG16 (128x128)
            img_cnn = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            img_cnn = cv2.resize(img_cnn, (128, 128)) # Changed to 128x128
            img_cnn = img_cnn.astype('float32') / 255.0
            img_cnn = np.expand_dims(img_cnn, axis=0)
            
            # Predict
            predictions = models["vgg16"]["model"].predict(img_cnn)
            pred_idx = np.argmax(predictions[0])
            confidence = float(np.max(predictions[0]))
            
            # Label
            pred_label = "Unknown"
            if models["vgg16"]["le"]:
                pred_label = models["vgg16"]["le"].inverse_transform([pred_idx])[0]
            elif models["hog"]["le"]: # Fallback
                pred_label = models["hog"]["le"].inverse_transform([pred_idx])[0]
            
            # All classes probabilities
            # Get all class names and their probabilities
            all_probs = []
            if models["vgg16"]["le"]:
                class_names = models["vgg16"]["le"].classes_
                for idx, prob in enumerate(predictions[0]):
                    all_probs.append({
                        "class": class_names[idx],
                        "confidence": float(prob)
                    })
            else:
                # Fallback if no label encoder
                for idx, prob in enumerate(predictions[0]):
                    all_probs.append({
                        "class": f"Class {idx}",
                        "confidence": float(prob)
                    })
            
            # Sort by confidence descending
            all_probs.sort(key=lambda x: x["confidence"], reverse=True)

            results["vgg16"] = {
                "class": pred_label,
                "confidence": confidence,
                "top3": all_probs, # Returning ALL probs, frontend can slice top 5
                "status": "success"
            }
        except Exception as e:
            print(e)
            results["vgg16"] = {"status": "error", "message": str(e)}
    else:
        results["vgg16"] = {"status": "error", "message": "Model not loaded"}

    return results
