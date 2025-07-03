import onnxruntime as ort
import numpy as np
import cv2

# --------- USER SETTINGS ---------
MODEL_PATH = "Models/resnet18_cropped_webcam.onnx"
IMAGE_PATH = "Photo-204.jpeg"   # <-- Change this to your image path
CLASS_NAMES = ["Claude Monet", "Da Vinci", "Picasso", "Van Gogh"]  # In correct order!
# ---------------------------------

# 1. Load and preprocess the image
img = cv2.imread(IMAGE_PATH)
if img is None:
    raise ValueError(f"Could not load image at {IMAGE_PATH}")

# EfficientNet expects 224x224 and float32, RGB, normalized
img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
img = cv2.resize(img, (224, 224))
img = img.astype(np.float32) / 255.0

# EfficientNet-B0 normalization (mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
mean = np.array([0.485, 0.456, 0.406])
std = np.array([0.229, 0.224, 0.225])
img = (img - mean) / std

# Add batch dimension
img = np.expand_dims(img, axis=0)

# 2. Load ONNX model
session = ort.InferenceSession(MODEL_PATH)

# 3. Get input and output names
input_name = session.get_inputs()[0].name
output_name = session.get_outputs()[0].name

# 4. Run inference
outputs = session.run([output_name], {input_name: img.astype(np.float32)})
pred = outputs[0][0]  # First (and only) batch element

# 5. Get predicted class
predicted_idx = np.argmax(pred)
predicted_label = CLASS_NAMES[predicted_idx]
confidence = pred[predicted_idx]

print(f"Predicted Artist: {predicted_label} (confidence: {confidence:.2f})")
