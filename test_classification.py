import onnxruntime as ort
import numpy as np
from PIL import Image
import torchvision.transforms as transforms
import os  # Added to check file existence

# Load the ONNX model
model_path = 'models/resnet18-artistrecco-best-final.onnx'
session = ort.InferenceSession(model_path)

# Define image transformations
test_transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

# Class names
classes = ['Claude Monet', 'Da Vinci', 'Picasso', 'Van Gogh']

# Load and preprocess the image
def predict_image(image_path):
    image = Image.open(image_path).convert('RGB')
    image_tensor = test_transform(image).unsqueeze(0).numpy()  # Add batch dimension

    # Run inference
    input_name = session.get_inputs()[0].name
    output_name = session.get_outputs()[0].name
    output = session.run([output_name], {input_name: image_tensor})[0]

    # Get probabilities and prediction
    probabilities = np.exp(output) / np.sum(np.exp(output), axis=1)  # Softmax
    predicted_class_idx = np.argmax(probabilities)
    confidence = probabilities[0][predicted_class_idx] * 100

    # Print results
    predicted_class = classes[predicted_class_idx]
    print(f"Predicted Artist: {predicted_class} (Confidence: {confidence:.2f}%)")
    print(f"Confidence for all classes: {[f'{cls}: {prob*100:.2f}%' for cls, prob in zip(classes, probabilities[0])]}")

# Test with an uploaded image
default_path = 'test/IMG_4411.jpg'  # Default path
image_path = input(f"Enter the path to your image (e.g., '{default_path}' or press Enter for default): ")
if not image_path.strip():  # If user presses Enter without typing
    image_path = default_path
if os.path.exists(image_path):  # Check if the path is valid
    predict_image(image_path)
else:
    print(f"Error: The file {image_path} does not exist. Please provide a valid path.")