from ultralytics import YOLO
import cv2
import matplotlib.pyplot as plt
import os

# Load the YOLOv8-OBB model
model_path = 'models/best.pt'  # Adjust path if needed
model = YOLO(model_path)

# Load and run inference on an image
image_path = input("Enter the path to your image (e.g., 'testvideo/0051.png' or press Enter for default): ")
if not image_path.strip():
    image_path = 'test/yolo/IMG_4418.jpg'  # Default path
if not os.path.exists(image_path):
    print(f"Error: The file {image_path} does not exist. Please provide a valid path.")
else:
    # Run inference
    results = model(image_path, conf=0.5, iou=0.5)  # Adjust thresholds as needed

    # Visualize results
    for result in results:
        # Plot the image with oriented bounding boxes
        im_array = result.plot()  # This includes OBBs
        im = cv2.cvtColor(im_array, cv2.COLOR_BGR2RGB)  # Convert BGR to RGB for matplotlib
        plt.figure(figsize=(10, 10))
        plt.imshow(im)
        plt.axis('off')
        plt.title('Detection Results with Oriented Bounding Boxes')
        plt.savefig('output_detection.jpg')  # Save the image
        plt.show()  # Display the image

    print("Inference completed. Check the plot or 'output_detection.jpg' for visualization.")