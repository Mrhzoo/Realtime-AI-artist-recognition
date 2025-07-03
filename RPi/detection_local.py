# import cv2
# from ultralytics import YOLO
# import time

# # Load the YOLOv8-OBB model
# model_path = 'Models/best.pt'  # Adjusted for RPi/Models
# model = YOLO(model_path)

# # Initialize the webcam
# cap = cv2.VideoCapture(0)
# if not cap.isOpened():
#     print("Error: Webcam not accessible.")
#     exit()

# # Set lower resolution
# cap.set(cv2.CAP_PROP_FRAME_WIDTH, 320)  # Reduced to 320x240
# cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 240)

# # Process frames
# frame_count = 0
# skip_frames = 2  # Process every 3rd frame (more skipping)

# while True:
#     success, frame = cap.read()
#     if not success:
#         break
#     frame_count += 1
#     if frame_count % (skip_frames + 1) != 0:  # Skip frames
#         continue

#     # Run detection
#     start_time = time.time()
#     results = model(frame, conf=0.6, iou=0.5)  # Increased confidence to 0.6
#     annotated_frame = results[0].plot()  # Add oriented bounding boxes
#     inference_time = time.time() - start_time
#     print(f"Inference time: {inference_time:.2f}s")

#     # Display the frame
#     cv2.imshow('Detection Window', annotated_frame)

#     # Break loop with 'q'
#     if cv2.waitKey(1) & 0xFF == ord('q'):
#         break

# # Cleanup
# cap.release()
# cv2.destroyAllWindows()












# import cv2
# from ultralytics import YOLO
# import time

# # Load the YOLOv8n-OBB custom model in ONNX format
# model_path = 'Models/best.onnx'  # Updated to use the ONNX model
# model = YOLO(model_path)

# # Initialize the webcam
# cap = cv2.VideoCapture(0)
# if not cap.isOpened():
#     print("Error: Webcam not accessible.")
#     exit()

# # Set resolution
# cap.set(cv2.CAP_PROP_FRAME_WIDTH, 320)
# cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 240)

# # Process frames
# frame_count = 0
# skip_frames = 1  # Process every 2nd frame

# while True:
#     success, frame = cap.read()
#     if not success:
#         break
#     frame_count += 1
#     if frame_count % (skip_frames + 1) != 0:  # Skip frames
#         continue

#     # Run detection
#     start_time = time.time()
#     results = model(frame, conf=0.4, iou=0.5)  # Keep confidence at 0.4
#     annotated_frame = results[0].plot()  # Add oriented bounding boxes
#     inference_time = time.time() - start_time
#     print(f"Inference time: {inference_time:.2f}s")

#     # Display the frame
#     cv2.imshow('Detection Window', annotated_frame)

#     # Break loop with 'q'
#     if cv2.waitKey(1) & 0xFF == ord('q'):
#         break

# # Cleanup
# cap.release()
# cv2.destroyAllWindows()




# import cv2
# from ultralytics import YOLO
# import time
# import numpy as np

# # Load the YOLOv8n-OBB custom model in ONNX format
# model_path = 'Models/best.onnx'  # Current paper model
# model = YOLO(model_path)

# # Initialize the webcam
# cap = cv2.VideoCapture(0)
# if not cap.isOpened():
#     print("Error: Webcam not accessible.")
#     exit()

# # Set resolution
# cap.set(cv2.CAP_PROP_FRAME_WIDTH, 320)
# cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 240)

# # Process frames
# frame_count = 0
# skip_frames = 1  # Process every 2nd frame

# print("Press 'q' or 'Q' to quit. Use Ctrl+C as a fallback.")
# while True:
#     success, frame = cap.read()
#     if not success:
#         break
#     frame_count += 1
#     if frame_count % (skip_frames + 1) != 0:  # Skip frames
#         continue

#     # Preprocess frame to reduce glare (optional, adjust if helpful)
#     alpha = 1.2  # Contrast control
#     beta = 10    # Brightness control
#     frame_adjusted = cv2.convertScaleAbs(frame, alpha=alpha, beta=beta)

#     # Run detection
#     start_time = time.time()
#     results = model(frame_adjusted, conf=0.4, iou=0.5)  # Keep confidence at 0.4
#     annotated_frame = results[0].plot()  # Add oriented bounding boxes
#     inference_time = time.time() - start_time
#     print(f"Inference time: {inference_time:.2f}s")

#     # Display the frame
#     cv2.imshow('Detection Window', annotated_frame)

#     # Improved exit condition with timeout
#     key = cv2.waitKey(1) & 0xFF
#     if key == ord('q') or key == ord('Q'):  # Allow 'q' or 'Q' to exit
#         break
#     elif key == 27:  # Allow Esc key as alternative
#         break

# # Cleanup
# cap.release()
# cv2.destroyAllWindows()
# print("Program terminated gracefully.")






















# import cv2
# from ultralytics import YOLO
# import time
# import numpy as np

# # Load the YOLOv8n-OBB custom phone model in ONNX format
# model_path = 'Models/Phonebest.onnx'  # Updated to use the new phone model
# model = YOLO(model_path)

# # Initialize the webcam
# cap = cv2.VideoCapture(0)
# if not cap.isOpened():
#     print("Error: Webcam not accessible.")
#     exit()

# # Set resolution
# cap.set(cv2.CAP_PROP_FRAME_WIDTH, 320)
# cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 240)

# # Process frames
# frame_count = 0
# skip_frames = 1  # Process every 2nd frame

# print("Press 'q' or 'Q' to quit. Use Ctrl+C as a fallback.")
# while True:
#     success, frame = cap.read()
#     if not success:
#         break
#     frame_count += 1
#     if frame_count % (skip_frames + 1) != 0:  # Skip frames
#         continue

#     # Preprocess frame to reduce glare (optional, adjust if helpful)
#     alpha = 1.2  # Contrast control
#     beta = 10    # Brightness control
#     frame_adjusted = cv2.convertScaleAbs(frame, alpha=alpha, beta=beta)

#     # Run detection
#     start_time = time.time()
#     results = model(frame_adjusted, conf=0.4, iou=0.5)  # Keep confidence at 0.4
#     annotated_frame = results[0].plot()  # Add oriented bounding boxes
#     inference_time = time.time() - start_time
#     print(f"Inference time: {inference_time:.2f}s")

#     # Display the frame
#     cv2.imshow('Detection Window', annotated_frame)

#     # Improved exit condition with timeout
#     key = cv2.waitKey(1) & 0xFF
#     if key == ord('q') or key == ord('Q'):  # Allow 'q' or 'Q' to exit
#         break
#     elif key == 27:  # Allow Esc key as alternative
#         break

# # Cleanup
# cap.release()
# cv2.destroyAllWindows()
# print("Program terminated gracefully.")






























# import cv2
# from ultralytics import YOLO
# import time
# import numpy as np

# # Load both YOLOv8n-OBB custom models
# paper_model_path = 'Models/best.onnx'       # Paper-specific model
# phone_model_path = 'Models/Phonebest.onnx'  # Phone-specific model
# paper_model = YOLO(paper_model_path)
# phone_model = YOLO(phone_model_path)

# # Initialize the webcam
# cap = cv2.VideoCapture(0)
# if not cap.isOpened():
#     print("Error: Webcam not accessible.")
#     exit()

# # Set resolution
# cap.set(cv2.CAP_PROP_FRAME_WIDTH, 320)
# cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 240)

# # Process frames
# frame_count = 0
# skip_frames = 1  # Process every 2nd frame

# print("Press 'q' or 'Q' to quit. Use Ctrl+C as a fallback.")
# while True:
#     success, frame = cap.read()
#     if not success:
#         break
#     frame_count += 1
#     if frame_count % (skip_frames + 1) != 0:  # Skip frames
#         continue

#     # Preprocess frame to reduce glare
#     alpha = 1.2  # Contrast control
#     beta = 10    # Brightness control
#     frame_adjusted = cv2.convertScaleAbs(frame, alpha=alpha, beta=beta)

#     # Run detection with paper model first
#     start_time = time.time()
#     paper_results = paper_model(frame_adjusted, conf=0.4, iou=0.5)
#     paper_annotated = paper_results[0].plot()
#     paper_detections = paper_results[0].boxes

#     # If no or weak detections, try phone model
#     if len(paper_detections) == 0 or max(paper_detections.conf, default=0) < 0.3:
#         phone_results = phone_model(frame_adjusted, conf=0.5, iou=0.5)
#         phone_annotated = phone_results[0].plot()
#         if len(phone_results[0].boxes) > 0 and max(phone_results[0].boxes.conf, default=0) >= 0.3:
#             paper_annotated = phone_annotated  # Use phone model result if better

#     inference_time = time.time() - start_time
#     print(f"Inference time: {inference_time:.2f}s")

#     # Display the frame
#     cv2.imshow('Detection Window', paper_annotated)

#     # Exit condition
#     key = cv2.waitKey(1) & 0xFF
#     if key == ord('q') or key == ord('Q'):  # Allow 'q' or 'Q' to exit
#         break
#     elif key == 27:  # Allow Esc key as alternative
#         break

# # Cleanup
# cap.release()
# cv2.destroyAllWindows()
# print("Program terminated gracefully.")













# -------------------------------------------------------------------------------------------------------------





# import cv2
# from ultralytics import YOLO
# import time
# import numpy as np

# # Load both YOLOv8n-OBB custom models
# paper_model_path = 'Models/best.onnx'       # Paper-specific model
# phone_model_path = 'Models/Phonebest.onnx'  # Phone-specific model
# paper_model = YOLO(paper_model_path)
# phone_model = YOLO(phone_model_path)

# # Load the classification model (ResNet-18 ONNX)
# class_model_path = 'Models/resnet18-artistrecco-best-final.onnx'
# class_net = cv2.dnn.readNetFromONNX(class_model_path)
# class_input_size = (224, 224)  # Typical for ResNet-18, adjust if different

# # Define artist names (confirm with your training data)
# artist_names = ['Claude Monet', 'Da Vinci', 'Picasso', 'Van Gogh']  # Verify order

# # Initialize the webcam
# cap = cv2.VideoCapture(0)
# if not cap.isOpened():
#     print("Error: Webcam not accessible.")
#     exit()

# # Set resolution
# cap.set(cv2.CAP_PROP_FRAME_WIDTH, 320)
# cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 240)

# # Process frames
# frame_count = 0
# skip_frames = 2  # Process every 3rd frame to reduce lag

# print("Press 'q' or 'Q' to quit. Use Ctrl+C as a fallback.")
# while True:
#     success, frame = cap.read()
#     if not success:
#         break
#     frame_count += 1
#     if frame_count % (skip_frames + 1) != 0:  # Skip frames
#         continue

#     # Preprocess frame to reduce glare and match ResNet
#     frame_adjusted = cv2.convertScaleAbs(frame, alpha=1.2, beta=10)  # Keep glare reduction
#     frame_rgb = cv2.cvtColor(frame_adjusted, cv2.COLOR_BGR2RGB)  # Convert to RGB

#     # Run detection with paper model first
#     start_time = time.time()
#     paper_results = paper_model(frame_rgb, conf=0.4, iou=0.5)
#     paper_annotated = paper_results[0].plot()
#     paper_detections = paper_results[0].boxes

#     # If no or weak detections, try phone model
#     final_annotated = paper_annotated
#     if len(paper_detections) == 0 or max(paper_detections.conf, default=0) < 0.3:
#         phone_results = phone_model(frame_rgb, conf=0.5, iou=0.5)
#         phone_annotated = phone_results[0].plot()
#         if len(phone_results[0].boxes) > 0 and max(phone_results[0].boxes.conf, default=0) >= 0.3:
#             final_annotated = phone_annotated
#             paper_detections = phone_results[0].boxes

#     # Classify detected regions and collect artist names with percentages
#     detected_artists = []
#     detections = paper_detections
#     for box in detections:
#         x1, y1, x2, y2 = map(int, box.xyxy[0])  # Detection bounding box
#         crop = frame_rgb[y1:y2, x1:x2]
#         if crop.size == 0:
#             continue

#         # Resize and preprocess for ResNet
#         crop_resized = cv2.resize(crop, class_input_size)
#         blob = cv2.dnn.blobFromImage(crop_resized, 1.0 / 255.0, class_input_size, (0.485, 0.456, 0.406), swapRB=False, crop=False)
#         class_net.setInput(blob)
#         out = class_net.forward()

#         # Get artist name and percentage
#         class_id = np.argmax(out)
#         confidence = out[0][class_id]
#         if confidence > 0.5:  # Confidence threshold
#             artist_name = artist_names[class_id] if 0 <= class_id < len(artist_names) else 'Unknown'
#             percentage = int(confidence * 100)  # Convert to percentage
#             detected_artists.append(f"{artist_name} ({percentage}%)")

#     # Overlay artist names at the bottom
#     if detected_artists:
#         text = ", ".join(detected_artists)
#         cv2.putText(final_annotated, text, (10, 230), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

#     inference_time = time.time() - start_time
#     print(f"Inference time: {inference_time:.2f}s")

#     # Display the frame
#     cv2.imshow('Detection Window', final_annotated)

#     # Exit condition
#     key = cv2.waitKey(1) & 0xFF
#     if key == ord('q') or key == ord('Q'):  # Allow 'q' or 'Q' to exit
#         break
#     elif key == 27:  # Allow Esc key as alternative
#         break

# # Cleanup
# cap.release()
# cv2.destroyAllWindows()
# print("Program terminated gracefully.")















#--------------------------------------------------------------------------------------------------------------------------------------------


# import cv2
# from ultralytics import YOLO
# import time
# import numpy as np

# # Load both YOLOv8n-OBB custom models
# paper_model_path = 'Models/best.onnx'       # Paper-specific model
# phone_model_path = 'Models/Phonebest.onnx'  # Phone-specific model
# paper_model = YOLO(paper_model_path)
# phone_model = YOLO(phone_model_path)

# # Load the classification model (ResNet-18 ONNX)
# class_model_path = 'Models/resnet18-artistrecco-best-final.onnx'
# class_net = cv2.dnn.readNetFromONNX(class_model_path)
# class_input_size = (224, 224)  # Typical for ResNet-18

# # Define artist names (check that order matches your training labels)
# artist_names = ['Claude Monet', 'Da Vinci', 'Picasso', 'Van Gogh']

# # Initialize the webcam
# cap = cv2.VideoCapture(0)
# if not cap.isOpened():
#     print("Error: Webcam not accessible.")
#     exit()

# # Set webcam resolution
# cap.set(cv2.CAP_PROP_FRAME_WIDTH, 320)
# cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 240)

# frame_count = 0
# skip_frames = 2  # Process every 3rd frame

# print("Press 'q' or 'Q' to quit. Use Ctrl+C as fallback.")
# while True:
#     success, frame = cap.read()
#     if not success:
#         break

#     frame_count += 1
#     if frame_count % (skip_frames + 1) != 0:
#         continue

#     # Preprocess frame
#     frame_adjusted = cv2.convertScaleAbs(frame, alpha=1.2, beta=10)
#     frame_rgb = cv2.cvtColor(frame_adjusted, cv2.COLOR_BGR2RGB)

#     # Detection with paper model
#     start_time = time.time()
#     paper_results = paper_model(frame_rgb, conf=0.4, iou=0.5)
#     paper_annotated = paper_results[0].plot()
#     paper_detections = paper_results[0].boxes

#     # Try phone model if paper model fails
#     final_annotated = paper_annotated
#     if len(paper_detections) == 0 or max(paper_detections.conf, default=0) < 0.3:
#         phone_results = phone_model(frame_rgb, conf=0.5, iou=0.5)
#         phone_annotated = phone_results[0].plot()
#         if len(phone_results[0].boxes) > 0 and max(phone_results[0].boxes.conf, default=0) >= 0.3:
#             final_annotated = phone_annotated
#             paper_detections = phone_results[0].boxes

#     # Classification
#     detected_artists = []
#     height, width = frame_rgb.shape[:2]
#     for box in paper_detections:
#         x1, y1, x2, y2 = map(int, box.xyxy[0])
#         # Clip coordinates to be within image bounds
#         x1, y1 = max(0, x1), max(0, y1)
#         x2, y2 = min(width, x2), min(height, y2)

#         crop = frame_rgb[y1:y2, x1:x2]
#         if crop.size == 0:
#             continue

#         # Resize and preprocess crop
#         crop_resized = cv2.resize(crop, class_input_size)
#         blob = cv2.dnn.blobFromImage(
#             crop_resized, 1.0 / 255.0, class_input_size,
#             (0.485, 0.456, 0.406), swapRB=False, crop=False
#         )
#         class_net.setInput(blob)
#         out = class_net.forward()

#         # Apply softmax to convert to probabilities
#         exp = np.exp(out - np.max(out))
#         probs = exp / np.sum(exp)
#         class_id = np.argmax(probs)
#         confidence = probs[0][class_id]

#         if confidence > 0.5:
#             artist_name = artist_names[class_id] if 0 <= class_id < len(artist_names) else 'Unknown'
#             percentage = int(confidence * 100)
#             detected_artists.append(f"{artist_name} ({percentage}%)")

#     # Display results
#     if detected_artists:
#         text = ", ".join(detected_artists)
#         cv2.putText(final_annotated, text, (10, 230), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

#     inference_time = time.time() - start_time
#     print(f"Inference time: {inference_time:.2f}s")

#     # Show the result
#     cv2.imshow('Detection Window', final_annotated)

#     key = cv2.waitKey(1) & 0xFF
#     if key in [ord('q'), ord('Q'), 27]:  # 'q', 'Q' or Esc
#         break

# # Cleanup
# cap.release()
# cv2.destroyAllWindows()
# print("Program terminated gracefully.")




# ===================================================================================================================================================================


# import cv2
# from ultralytics import YOLO
# import time
# import numpy as np

# # Load YOLO models
# paper_model_path = 'Models/best.onnx'
# phone_model_path = 'Models/Phonebest.onnx'
# paper_model = YOLO(paper_model_path)
# phone_model = YOLO(phone_model_path)

# # Load classification model
# class_model_path = 'Models/artist_classifier_resnet18.onnx'
# class_net = cv2.dnn.readNetFromONNX(class_model_path)
# class_input_size = (224, 224)

# # Define artist labels
# artist_names = ['Claude Monet', 'Da Vinci', 'Picasso', 'Van Gogh']

# # Start webcam
# cap = cv2.VideoCapture(0)
# if not cap.isOpened():
#     print("Error: Webcam not accessible.")
#     exit()

# cap.set(cv2.CAP_PROP_FRAME_WIDTH, 320)
# cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 240)

# frame_count = 0
# skip_frames = 2

# print("Press 'q' or 'Q' to quit.")
# while True:
#     success, frame = cap.read()
#     if not success:
#         break

#     frame_count += 1
#     if frame_count % (skip_frames + 1) != 0:
#         continue

#     # Use raw frame with correct colors
#     frame_rgb = frame.copy()

#     # Run detection (start with paper)
#     start_time = time.time()
#     paper_results = paper_model(frame_rgb, conf=0.4, iou=0.5)
#     paper_detections = paper_results[0].boxes
#     final_annotated = paper_results[0].plot()

#     # Fallback to phone model if no good detection
#     if len(paper_detections) == 0 or max(paper_detections.conf, default=0) < 0.3:
#         phone_results = phone_model(frame_rgb, conf=0.5, iou=0.5)
#         phone_detections = phone_results[0].boxes
#         if len(phone_detections) > 0 and max(phone_detections.conf, default=0) >= 0.3:
#             final_annotated = phone_results[0].plot()
#             paper_detections = phone_detections

#     # Classification for each detected region
#     detected_artists = []
#     for box in paper_detections:
#         x1, y1, x2, y2 = map(int, box.xyxy[0])
#         crop = frame_rgb[y1:y2, x1:x2]
#         if crop.size == 0:
#             continue

#         # Resize & preprocess for ResNet
#         crop_resized = cv2.resize(crop, class_input_size)
#         blob = cv2.dnn.blobFromImage(
#             crop_resized, scalefactor=1.0 / 255.0,
#             size=class_input_size, mean=(0.485, 0.456, 0.406),
#             swapRB=True, crop=False
#         )
#         class_net.setInput(blob)
#         out = class_net.forward()

#         # Get prediction
#         class_id = np.argmax(out)
#         # Apply softmax to get probabilities
#         exp_scores = np.exp(out[0] - np.max(out[0]))  # Subtract max for numerical stability
#         probs = exp_scores / exp_scores.sum()
#         confidence = float(probs[class_id])

#         if confidence > 0.5:
#             artist_name = artist_names[class_id] if 0 <= class_id < len(artist_names) else 'Unknown'
#             percentage = int(confidence * 100)
#             detected_artists.append(f"{artist_name} ({percentage}%)")

#     # Show result text
#     if detected_artists:
#         text = ", ".join(detected_artists)
#         cv2.putText(final_annotated, text, (10, 230), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

#     # FPS
#     inference_time = time.time() - start_time
#     print(f"Inference time: {inference_time:.2f}s")

#     # Show
#     cv2.imshow('Detection Window', final_annotated)
#     key = cv2.waitKey(1) & 0xFF
#     if key in (ord('q'), ord('Q'), 27):  # q, Q, or Esc
#         break

# cap.release()
# cv2.destroyAllWindows()
# print("Program ended.")






# ++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
























































# bbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbb





# import cv2
# from ultralytics import YOLO
# import time
# import numpy as np
# import onnxruntime as ort

# # Load YOLO models
# paper_model_path = 'Models/best.onnx'
# phone_model_path = 'Models/Phonebest.onnx'
# paper_model = YOLO(paper_model_path)
# phone_model = YOLO(phone_model_path)

# # Load ResNet18 ONNX classification model with multi-threading
# class_model_path = 'Models/artist_classifier_resnet18.onnx'
# so = ort.SessionOptions()
# so.intra_op_num_threads = 4  # Use all Pi 5 cores
# class_session = ort.InferenceSession(class_model_path, sess_options=so)
# class_input_name = class_session.get_inputs()[0].name

# # Preprocessing parameters (MUST match training)
# MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32)
# STD = np.array([0.229, 0.224, 0.225], dtype=np.float32)
# INPUT_SIZE = (224, 224)

# # Artist labels (VERIFY ORDER MATCHES TRAINING)
# artist_names = ['Claude Monet', 'Da Vinci', 'Picasso', 'Van Gogh']

# # Webcam setup
# cap = cv2.VideoCapture(0)
# cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
# cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

# def preprocess_image(image):
#     # Convert BGR to RGB
#     image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
#     # Resize and normalize
#     image = cv2.resize(image, INPUT_SIZE)
#     image = image.astype(np.float32) / 255.0
#     image = (image - MEAN) / STD
#     # Change from HWC to CHW format
#     image = image.transpose(2, 0, 1)
#     return np.expand_dims(image, axis=0)  # Add batch dimension

# frame_skip = 4  # Process every 5th frame for speed
# frame_count = 0

# print("Press Q to quit")
# while True:
#     success, frame = cap.read()
#     if not success:
#         break

#     key = cv2.waitKey(1) & 0xFF
#     if key in [ord('q'), ord('Q')]:
#         break

#     frame_count += 1
#     if frame_count % frame_skip != 0:
#         continue

#     # Detection
#     results = paper_model(frame, conf=0.4, iou=0.5)[0]
#     final_frame = results.plot()

#     # Fallback to phone detection
#     if len(results.boxes) == 0 or results.boxes.conf.max() < 0.3:
#         phone_results = phone_model(frame, conf=0.5, iou=0.5)[0]
#         if len(phone_results.boxes) > 0:
#             final_frame = phone_results.plot()
#             results = phone_results

#     # Classification
#     detected_artists = []
#     for box in results.boxes:
#         x1, y1, x2, y2 = map(int, box.xyxy[0].tolist())
#         crop = frame[y1:y2, x1:x2]
#         if crop.size == 0:
#             continue

#         input_data = preprocess_image(crop)
#         outputs = class_session.run(None, {class_input_name: input_data.astype(np.float32)})
#         logits = np.squeeze(outputs[0])
#         # Apply softmax to get probabilities
#         exp_scores = np.exp(logits - np.max(logits))
#         probabilities = exp_scores / exp_scores.sum()
#         class_id = np.argmax(probabilities)
#         confidence = probabilities[class_id]


#         if confidence > 0.5:
#             label = f"{artist_names[class_id]} ({confidence * 100:.0f}%)"
#             detected_artists.append(label)

#     # Show artist names at the bottom
#     if detected_artists:
#         text = ", ".join(detected_artists)
#         cv2.putText(final_frame, text, (10, final_frame.shape[0] - 10),
#                     cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)

#     cv2.imshow('Art Recognizer', final_frame)

# cap.release()
# cv2.destroyAllWindows()
# print("Program ended.")


# bbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbb





# import cv2
# from ultralytics import YOLO
# import numpy as np
# import onnxruntime as ort
# import smbus
# import time







# # ---------------------------
# # LCD Configuration (Add this section)
# # ---------------------------
# bus = smbus.SMBus(1)  # I2C bus initialization
# LCD_ADDRESS = 0x27     # Verify this matches your LCD's I2C address
# LCD_WIDTH = 16

# def lcd_byte(bits, mode):
#     bits_high = mode | (bits & 0xF0) | 0x08
#     bits_low  = mode | ((bits << 4) & 0xF0) | 0x08
#     bus.write_byte(LCD_ADDRESS, bits_high)
#     time.sleep(0.0005)
#     bus.write_byte(LCD_ADDRESS, (bits_high | 0x04))
#     time.sleep(0.0005)
#     bus.write_byte(LCD_ADDRESS, (bits_high & ~0x04))
#     time.sleep(0.0005)
#     bus.write_byte(LCD_ADDRESS, bits_low)
#     time.sleep(0.0005)
#     bus.write_byte(LCD_ADDRESS, (bits_low | 0x04))
#     time.sleep(0.0005)
#     bus.write_byte(LCD_ADDRESS, (bits_low & ~0x04))
#     time.sleep(0.0005)

# def lcd_init():
#     lcd_byte(0x33, 0)
#     lcd_byte(0x32, 0)
#     lcd_byte(0x06, 0)
#     lcd_byte(0x0C, 0)
#     lcd_byte(0x28, 0)
#     lcd_byte(0x01, 0)
#     time.sleep(0.005)

# def lcd_string(message, line):
#     line_address = 0xC0 if line == 2 else 0x80
#     # Only use printable characters, replace anything else with space
#     safe_msg = ''.join(c if 32 <= ord(c) <= 126 else ' ' for c in message)
#     # Pad to 16 characters to overwrite any leftover characters
#     padded_msg = safe_msg.ljust(LCD_WIDTH)
#     lcd_byte(line_address, 0)
#     for char in padded_msg:
#         lcd_byte(ord(char), 1)



# def lcd_clear_line(line):
#     lcd_string(" " * LCD_WIDTH, line)



# def display_artist_on_lcd(artist_name, confidence):
#     lcd_clear_line(1)
#     lcd_clear_line(2)
#     lcd_string(f"Artist: {artist_name[:LCD_WIDTH]}", 1)
#     lcd_string(f"Conf: {int(confidence):02d} %", 2)





# # Initialize LCD at program start
# # Initialize LCD at program start
# lcd_init()
# lcd_clear_line(1)
# lcd_clear_line(2)
# lcd_string("Art Recognizer", 1)
# lcd_string("Initializing...", 2)
# time.sleep(2)





# # --- Model Paths ---
# paper_model_path = 'Models/best.onnx'
# phone_model_path = 'Models/Phonebest.onnx'
# class_model_path = 'Models/artist_classifier_resnet18.onnx'

# # --- Load YOLO Models ---
# paper_model = YOLO(paper_model_path)
# phone_model = YOLO(phone_model_path)

# # --- Load ResNet18 ONNX Classification Model with Multi-threading ---
# so = ort.SessionOptions()
# so.intra_op_num_threads = 4  # Use all Pi 5 cores
# class_session = ort.InferenceSession(class_model_path, sess_options=so)
# class_input_name = class_session.get_inputs()[0].name

# # --- Preprocessing Parameters (MUST match training) ---
# MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32)
# STD = np.array([0.229, 0.224, 0.225], dtype=np.float32)
# INPUT_SIZE = (224, 224)

# # --- Artist Labels (VERIFY ORDER MATCHES TRAINING) ---
# artist_names = ['Claude Monet', 'Da Vinci', 'Picasso', 'Van Gogh']

# # --- Webcam Setup (Higher Resolution) ---
# cap = cv2.VideoCapture(0)
# cap.set(cv2.CAP_PROP_FRAME_WIDTH, 320)
# cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 240)

# def preprocess_image(image):
#     # Convert BGR to RGB
#     image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
#     # Resize and normalize
#     image = cv2.resize(image, INPUT_SIZE)
#     image = image.astype(np.float32) / 255.0
#     image = (image - MEAN) / STD
#     # Change from HWC to CHW format
#     image = image.transpose(2, 0, 1)
#     return np.expand_dims(image, axis=0)  # Add batch dimension

# frame_skip = 5  # Process every 5th frame for speed
# frame_count = 0

# print("Press Q to quit")
# while True:
#     success, frame = cap.read()
#     if not success:
#         break

#     key = cv2.waitKey(1) & 0xFF
#     if key in [ord('q'), ord('Q')]:
#         break

#     frame_count += 1
#     if frame_count % frame_skip != 0:
#         continue

#     # Detection
#     results = paper_model(frame, conf=0.6, iou=0.5)[0]
#     final_frame = results.plot()

#     # Fallback to phone detection if no good detection
#     if len(results.boxes) == 0 or results.boxes.conf.max() < 0.3:
#         phone_results = phone_model(frame, conf=0.5, iou=0.5)[0]
#         if len(phone_results.boxes) > 0:
#             final_frame = phone_results.plot()
#             results = phone_results



            

#     # Classification
#     detected_artists = []
#     for box in results.boxes:
#         x1, y1, x2, y2 = map(int, box.xyxy[0].tolist())
#         crop = frame[y1:y2, x1:x2]
#         if crop.size == 0:
#             continue

#         input_data = preprocess_image(crop)
#         outputs = class_session.run(None, {class_input_name: input_data.astype(np.float32)})
#         logits = np.squeeze(outputs[0])
#         # Apply softmax to get probabilities
#         exp_scores = np.exp(logits - np.max(logits))
#         probabilities = exp_scores / exp_scores.sum()
#         class_id = np.argmax(probabilities)
#         confidence = probabilities[class_id]

#         if confidence > 0.5:
#             label = f"{artist_names[class_id]} ({confidence * 100:.0f}%)"
#             detected_artists.append(label)

#             display_artist_on_lcd(artist_names[class_id], confidence * 100)

#     # Clear LCD when no detections
#     if not detected_artists:
#         lcd_string("No artwork", 1)
#         lcd_string("detected :(", 2)
        

#     # Show artist names at the bottom
#     if detected_artists:
#         text = ", ".join(detected_artists)
#         cv2.putText(final_frame, text, (10, final_frame.shape[0] - 10),
#                     cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)

#     cv2.imshow('Art Recognizer', final_frame)

# cap.release()
# cv2.destroyAllWindows()
# print("Program ended.")































































# import cv2
# from ultralytics import YOLO
# import numpy as np
# import onnxruntime as ort
# import smbus
# import time

# # ---------------------------
# # LCD Configuration
# # ---------------------------
# bus = smbus.SMBus(1)  # I2C bus initialization
# LCD_ADDRESS = 0x27    # Verify this matches your LCD's I2C address
# LCD_WIDTH = 16

# def lcd_byte(bits, mode):
#     bits_high = mode | (bits & 0xF0) | 0x08
#     bits_low = mode | ((bits << 4) & 0xF0) | 0x08
#     bus.write_byte(LCD_ADDRESS, bits_high)
#     time.sleep(0.0001)
#     bus.write_byte(LCD_ADDRESS, (bits_high | 0x04))
#     time.sleep(0.0001)
#     bus.write_byte(LCD_ADDRESS, (bits_high & ~0x04))
#     time.sleep(0.0001)
#     bus.write_byte(LCD_ADDRESS, bits_low)
#     time.sleep(0.0001)
#     bus.write_byte(LCD_ADDRESS, (bits_low | 0x04))
#     time.sleep(0.0001)
#     bus.write_byte(LCD_ADDRESS, (bits_low & ~0x04))
#     time.sleep(0.0001)

# def lcd_init():
#     lcd_byte(0x33, 0)
#     lcd_byte(0x32, 0)
#     lcd_byte(0x06, 0)
#     lcd_byte(0x0C, 0)
#     lcd_byte(0x28, 0)
#     lcd_byte(0x01, 0)
#     time.sleep(0.005)

# def lcd_string(message, line):
#     line_address = 0xC0 if line == 2 else 0x80
#     safe_msg = ''.join(c if 32 <= ord(c) <= 126 else ' ' for c in message)
#     padded_msg = safe_msg.ljust(LCD_WIDTH)
#     lcd_byte(line_address, 0)
#     for char in padded_msg:
#         lcd_byte(ord(char), 1)

# def lcd_clear_line(line):
#     lcd_string(" " * LCD_WIDTH, line)

# def display_artist_on_lcd(artist_name, confidence):
#     lcd_clear_line(1)
#     lcd_clear_line(2)
#     lcd_string(f"Artist: {artist_name[:LCD_WIDTH]}", 1)
#     lcd_string(f"Conf: {int(confidence):02d} %", 2)

# def display_unknown_on_lcd():
#     lcd_clear_line(1)
#     lcd_clear_line(2)
#     lcd_string("Unknown artist", 1)
#     lcd_string("Conf: -- %", 2)

# # Initialize LCD
# lcd_init()
# lcd_clear_line(1)
# lcd_clear_line(2)
# lcd_string("Art Recognizer", 1)
# lcd_string("Initializing...", 2)
# time.sleep(2)

# # --- Model Paths ---
# paper_model_path = 'Models/best.onnx'
# phone_model_path = 'Models/Phonebest.onnx'
# class_model_path = 'Models/artist_classifier_resnet18.onnx'

# # --- Load YOLO Models ---
# paper_model = YOLO(paper_model_path)
# phone_model = YOLO(phone_model_path)

# # --- Load ResNet18 ONNX Classification Model ---
# so = ort.SessionOptions()
# so.intra_op_num_threads = 2
# class_session = ort.InferenceSession(class_model_path, sess_options=so)
# class_input_name = class_session.get_inputs()[0].name

# # --- Preprocessing Parameters ---
# MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32)
# STD = np.array([0.229, 0.224, 0.225], dtype=np.float32)
# INPUT_SIZE = (224, 224)

# # --- Artist Labels ---
# artist_names = ['Claude Monet', 'Da Vinci', 'Picasso', 'Van Gogh']

# # --- Webcam Setup ---
# cap = cv2.VideoCapture(0)
# cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
# cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

# def preprocess_image(image):
#     image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
#     image = cv2.resize(image, INPUT_SIZE)
#     image = image.astype(np.float32) / 255.0
#     image = (image - MEAN) / STD
#     image = image.transpose(2, 0, 1)
#     return np.expand_dims(image, axis=0)

# frame_skip = 15
# frame_count = 0

# print("Press Q to quit")
# while True:
#     success, frame = cap.read()
#     if not success:
#         print("Failed to read frame.")
#         break

#     key = cv2.waitKey(1) & 0xFF
#     if key in [ord('q'), ord('Q')]:
#         break

#     frame_count += 1
#     if frame_count % frame_skip != 0:
#         continue

#     # Detection
#     start_time = time.time()
#     results = paper_model(frame, conf=0.5, iou=0.5)[0]
#     final_frame = results.plot() if results.boxes is not None else frame.copy()

#     # Fallback to phone detection
#     if len(results.boxes) == 0 or results.boxes.conf.max() < 0.3:
#         phone_results = phone_model(frame, conf=0.5, iou=0.5)[0]
#         if len(phone_results.boxes) > 0 and phone_results.boxes.conf.max() >= 0.3:
#             final_frame = phone_results.plot() if phone_results.boxes is not None else frame.copy()
#             results = phone_results

#     # Classification
#     detected_artists = []
#     if len(results.boxes) > 0:
#         box = results.boxes[0]
#         x1, y1, x2, y2 = map(int, box.xyxy[0].tolist())
#         crop = frame[y1:y2, x1:x2]
#         if crop.size == 0:
#             continue

#         input_data = preprocess_image(crop)
#         outputs = class_session.run(None, {class_input_name: input_data.astype(np.float32)})
#         logits = np.squeeze(outputs[0])
#         exp_scores = np.exp(logits - np.max(logits))
#         probabilities = exp_scores / exp_scores.sum()
#         class_id = np.argmax(probabilities)
#         confidence = probabilities[class_id]

#         if confidence > 0.5:
#             label = f"{artist_names[class_id]} ({confidence * 100:.0f}%)"
#             detected_artists.append(label)
#             display_artist_on_lcd(artist_names[class_id], confidence * 100)
#         else:
#             detected_artists.append("Unknown artist")
#             display_unknown_on_lcd()
#             cv2.putText(final_frame, "Unknown artist", (10, final_frame.shape[0] - 10),
#                         cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)  # Red text

#     # Clear LCD when no detections
#     elif len(results.boxes) == 0:
#         lcd_string("No artwork", 1)
#         lcd_string("detected :(", 2)

#     # Show artist names at the bottom
#     if detected_artists:
#         text = ", ".join(detected_artists)
#         cv2.putText(final_frame, text, (10, final_frame.shape[0] - 10),
#                     cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)

#     print(f"Inference time: {time.time() - start_time:.2f}s, FPS: {1/(time.time() - start_time):.1f}")
#     cv2.imshow('Art Recognizer', final_frame)

# cap.release()
# cv2.destroyAllWindows()
# print("Program ended.")




































# import cv2
# from ultralytics import YOLO
# import time
# import numpy as np
# import onnxruntime as ort
# import smbus

# # ---------------------------
# # LCD Configuration
# # ---------------------------
# bus = smbus.SMBus(1)  # I2C bus initialization
# LCD_ADDRESS = 0x27    # Verify this matches your LCD's I2C address
# LCD_WIDTH = 16

# def lcd_byte(bits, mode):
#     bits_high = mode | (bits & 0xF0) | 0x08
#     bits_low = mode | ((bits << 4) & 0xF0) | 0x08
#     bus.write_byte(LCD_ADDRESS, bits_high)
#     time.sleep(0.0001)
#     bus.write_byte(LCD_ADDRESS, (bits_high | 0x04))
#     time.sleep(0.0001)
#     bus.write_byte(LCD_ADDRESS, (bits_high & ~0x04))
#     time.sleep(0.0001)
#     bus.write_byte(LCD_ADDRESS, bits_low)
#     time.sleep(0.0001)
#     bus.write_byte(LCD_ADDRESS, (bits_low | 0x04))
#     time.sleep(0.0001)
#     bus.write_byte(LCD_ADDRESS, (bits_low & ~0x04))
#     time.sleep(0.0001)

# def lcd_init():
#     lcd_byte(0x33, 0)
#     lcd_byte(0x32, 0)
#     lcd_byte(0x06, 0)
#     lcd_byte(0x0C, 0)
#     lcd_byte(0x28, 0)
#     lcd_byte(0x01, 0)
#     time.sleep(0.005)

# def lcd_string(message, line):
#     line_address = 0xC0 if line == 2 else 0x80
#     safe_msg = ''.join(c if 32 <= ord(c) <= 126 else ' ' for c in message)
#     padded_msg = safe_msg.ljust(LCD_WIDTH)
#     lcd_byte(line_address, 0)
#     for char in padded_msg:
#         lcd_byte(ord(char), 1)

# def lcd_clear_line(line):
#     lcd_string(" " * LCD_WIDTH, line)

# def display_artist_on_lcd(artist_name, confidence):
#     lcd_clear_line(1)
#     lcd_clear_line(2)
#     lcd_string(f"Artist: {artist_name[:LCD_WIDTH]}", 1)
#     lcd_string(f"Conf: {int(confidence):02d} %", 2)

# def display_unknown_on_lcd():
#     lcd_clear_line(1)
#     lcd_clear_line(2)
#     lcd_string("Unknown artist", 1)
#     lcd_string("Conf: -- %", 2)

# # Initialize LCD
# lcd_init()
# lcd_clear_line(1)
# lcd_clear_line(2)
# lcd_string("Art Recognizer", 1)
# lcd_string("Initializing...", 2)
# time.sleep(2)

# # Load YOLO models
# paper_model_path = 'Models/best.onnx'
# phone_model_path = 'Models/Phonebest.onnx'
# paper_model = YOLO(paper_model_path)
# phone_model = YOLO(phone_model_path)

# # Load ResNet18 ONNX classification model with multi-threading
# class_model_path = 'Models/artist_classifier_resnet18.onnx'
# so = ort.SessionOptions()
# so.intra_op_num_threads = 4  # Use all Pi 5 cores
# class_session = ort.InferenceSession(class_model_path, sess_options=so)
# class_input_name = class_session.get_inputs()[0].name

# # Preprocessing parameters (MUST match training)
# MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32)
# STD = np.array([0.229, 0.224, 0.225], dtype=np.float32)
# INPUT_SIZE = (224, 224)

# # Artist labels (VERIFY ORDER MATCHES TRAINING)
# artist_names = ['Claude Monet', 'Da Vinci', 'Picasso', 'Van Gogh']

# # Webcam setup
# cap = cv2.VideoCapture(0)
# cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
# cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

# def preprocess_image(image):
#     # Convert BGR to RGB
#     image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
#     # Resize and normalize
#     image = cv2.resize(image, INPUT_SIZE)
#     image = image.astype(np.float32) / 255.0
#     image = (image - MEAN) / STD
#     # Change from HWC to CHW format
#     image = image.transpose(2, 0, 1)
#     return np.expand_dims(image, axis=0)  # Add batch dimension

# frame_skip = 4  # Process every 5th frame for speed
# frame_count = 0

# print("Press Q to quit")
# while True:
#     success, frame = cap.read()
#     if not success:
#         break

#     key = cv2.waitKey(1) & 0xFF
#     if key in [ord('q'), ord('Q')]:
#         break

#     frame_count += 1
#     if frame_count % frame_skip != 0:
#         continue

#     # Detection
#     start_time = time.time()  # Add timing for reference
#     results = paper_model(frame, conf=0.4, iou=0.5)[0]
#     final_frame = results.plot()

#     # Fallback to phone detection
#     if len(results.boxes) == 0 or results.boxes.conf.max() < 0.3:
#         phone_results = phone_model(frame, conf=0.5, iou=0.5)[0]
#         if len(phone_results.boxes) > 0:
#             final_frame = phone_results.plot()
#             results = phone_results

#     # Classification
#     detected_artists = []
#     if len(results.boxes) > 0:
#         box = results.boxes[0]  # Take the first detection (simplifies to match original)
#         x1, y1, x2, y2 = map(int, box.xyxy[0].tolist())
#         crop = frame[y1:y2, x1:x2]
#         if crop.size == 0:
#             continue

#         input_data = preprocess_image(crop)
#         outputs = class_session.run(None, {class_input_name: input_data.astype(np.float32)})
#         logits = np.squeeze(outputs[0])
#         exp_scores = np.exp(logits - np.max(logits))
#         probabilities = exp_scores / exp_scores.sum()
#         class_id = np.argmax(probabilities)
#         confidence = probabilities[class_id]

#         if confidence > 0.5:
#             artist = artist_names[class_id]
#             label = f"{artist} ({confidence * 100:.0f}%)"
#             detected_artists.append(label)
#             display_artist_on_lcd(artist, confidence * 100)
#         else:
#             detected_artists.append("Unknown artist")
#             display_unknown_on_lcd()

#     # Clear LCD when no detections
#     elif len(results.boxes) == 0:
#         lcd_string("No artwork", 1)
#         lcd_string("detected :(", 2)

#     # Show artist names at the bottom
#     if detected_artists:
#         text = ", ".join(detected_artists)
#         cv2.putText(final_frame, text, (10, final_frame.shape[0] - 10),
#                     cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)

#     print(f"Inference time: {time.time() - start_time:.2f}s, FPS: {1/(time.time() - start_time):.1f}")
#     cv2.imshow('Art Recognizer', final_frame)

# cap.release()
# cv2.destroyAllWindows()
# print("Program ended.")











































import cv2
from ultralytics import YOLO
import time
import numpy as np
import onnxruntime as ort
import smbus

class LCDDisplay:
    def __init__(self, address=0x27, width=16):
        self.bus = smbus.SMBus(1)
        self.LCD_ADDRESS = address
        self.LCD_WIDTH = width
        self._initialize()

    def _lcd_byte(self, bits, mode):
        bits_high = mode | (bits & 0xF0) | 0x08
        bits_low = mode | ((bits << 4) & 0xF0) | 0x08
        self.bus.write_byte(self.LCD_ADDRESS, bits_high)
        self.bus.write_byte(self.LCD_ADDRESS, bits_high | 0x04)
        self.bus.write_byte(self.LCD_ADDRESS, bits_high & ~0x04)
        self.bus.write_byte(self.LCD_ADDRESS, bits_low)
        self.bus.write_byte(self.LCD_ADDRESS, bits_low | 0x04)
        self.bus.write_byte(self.LCD_ADDRESS, bits_low & ~0x04)

    def _initialize(self):
        self._lcd_byte(0x33, 0)
        self._lcd_byte(0x32, 0)
        self._lcd_byte(0x06, 0)
        self._lcd_byte(0x0C, 0)
        self._lcd_byte(0x28, 0)
        self._lcd_byte(0x01, 0)
        time.sleep(0.005)

    def _lcd_string(self, message, line):
        line_address = 0xC0 if line == 2 else 0x80
        safe_msg = ''.join(c if 32 <= ord(c) <= 126 else ' ' for c in message)
        padded_msg = safe_msg.ljust(self.LCD_WIDTH)
        self._lcd_byte(line_address, 0)
        for char in padded_msg:
            self._lcd_byte(ord(char), 1)

    def clear_line(self, line):
        self._lcd_string(" " * self.LCD_WIDTH, line)

    def display_artist(self, artist_name, confidence):
        self.clear_line(1)
        self.clear_line(2)
        self._lcd_string(f"Artist: {artist_name[:self.LCD_WIDTH]}", 1)
        self._lcd_string(f"Conf: {int(confidence):02d} %", 2)

    def display_unknown(self):
        self.clear_line(1)
        self.clear_line(2)
        self._lcd_string("Unknown artist", 1)
        self._lcd_string("Conf: -- %", 2)

    def display_no_artwork(self):
        self.clear_line(1)
        self.clear_line(2)
        self._lcd_string("No artwork", 1)
        self._lcd_string("detected :(", 2)

    def display_out_of_model(self):
        self.clear_line(1)
        self.clear_line(2)
        self._lcd_string("Unknown Artist", 1)
        self._lcd_string("for model :(", 2)

# Initialize LCD
lcd = LCDDisplay()
lcd.clear_line(1)
lcd.clear_line(2)
lcd._lcd_string("Art Recognizer", 1)
lcd._lcd_string("Initializing...", 2)
time.sleep(2)

# Load YOLO models
paper_model_path = 'Models/best.onnx'
phone_model_path = 'Models/Phonebest.onnx'
paper_model = YOLO(paper_model_path)
phone_model = YOLO(phone_model_path)

# Load ResNet18 ONNX classification model with multi-threading
class_model_path = 'Models/artist_classifier_resnet18.onnx'
so = ort.SessionOptions()
so.intra_op_num_threads = 4  # Use all Pi 5 cores
class_session = ort.InferenceSession(class_model_path, sess_options=so)
class_input_name = class_session.get_inputs()[0].name

# Preprocessing parameters (MUST match training)
MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32)
STD = np.array([0.229, 0.224, 0.225], dtype=np.float32)
INPUT_SIZE = (224, 224)

# Artist labels (VERIFY ORDER MATCHES TRAINING)
artist_names = ['Claude Monet', 'Da Vinci', 'Picasso', 'Van Gogh']

# Webcam setup
cap = cv2.VideoCapture(0)
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

def preprocess_image(image):
    # Convert BGR to RGB
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    # Resize and normalize
    image = cv2.resize(image, INPUT_SIZE)
    image = image.astype(np.float32) / 255.0
    image = (image - MEAN) / STD
    # Change from HWC to CHW format
    image = image.transpose(2, 0, 1)
    return np.expand_dims(image, axis=0)  # Add batch dimension

frame_skip = 4  # Process every 5th frame for speed
frame_count = 0
lcd_update_count = 0  # Counter for LCD update frequency
lcd_update_interval = 4  # Update LCD every 4th processed frame

print("Press Q to quit")
while True:
    success, frame = cap.read()
    if not success:
        break

    key = cv2.waitKey(1) & 0xFF
    if key in [ord('q'), ord('Q')]:
        break

    frame_count += 1
    if frame_count % frame_skip != 0:
        continue

    # Detection
    start_time = time.time()  # Add timing for reference
    results = paper_model(frame, conf=0.6, iou=0.5)[0]  # Updated to 0.6
    final_frame = results.plot()

    # Fallback to phone detection
    if len(results.boxes) == 0 or results.boxes.conf.max() < 0.6:  # Updated to 0.6
        phone_results = phone_model(frame, conf=0.5, iou=0.5)[0]  # Updated to 0.5
        if len(phone_results.boxes) > 0:
            final_frame = phone_results.plot()
            results = phone_results

    # Classification
    detected_artists = []
    if len(results.boxes) > 0:
        box = results.boxes[0]  # Take the first detection
        x1, y1, x2, y2 = map(int, box.xyxy[0].tolist())
        crop = frame[y1:y2, x1:x2]
        if crop.size == 0:
            continue

        input_data = preprocess_image(crop)
        outputs = class_session.run(None, {class_input_name: input_data.astype(np.float32)})
        logits = np.squeeze(outputs[0])
        exp_scores = np.exp(logits - np.max(logits))
        probabilities = exp_scores / exp_scores.sum()
        class_id = np.argmax(probabilities)
        confidence = probabilities[class_id]

        if confidence > 0.5:
            if class_id < len(artist_names):  # Check if class_id is within known artists
                artist = artist_names[class_id]
                label = f"{artist} ({confidence * 100:.0f}%)"
                detected_artists.append(label)
                if lcd_update_count % lcd_update_interval == 0:  # Update LCD less frequently
                    lcd.display_artist(artist, confidence * 100)
            else:
                detected_artists.append("Unknown Artist for the model :(")
                if lcd_update_count % lcd_update_interval == 0:  # Update LCD less frequently
                    lcd.display_out_of_model()
                    cv2.putText(final_frame, "Unknown Artist for the model :(",
                                (10, final_frame.shape[0] - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.7,
                                (0, 0, 255), 2)  # Red text
        else:
            detected_artists.append("Unknown artist")
            if lcd_update_count % lcd_update_interval == 0:  # Update LCD less frequently
                lcd.display_unknown()

    # Clear LCD when no detections
    elif len(results.boxes) == 0:
        if lcd_update_count % lcd_update_interval == 0:  # Update LCD less frequently
            lcd.display_no_artwork()

    lcd_update_count += 1  # Increment update counter

    # Show artist names at the bottom
    if detected_artists:
        text = ", ".join(detected_artists)
        cv2.putText(final_frame, text, (10, final_frame.shape[0] - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)

    print(f"Inference time: {time.time() - start_time:.2f}s, FPS: {1/(time.time() - start_time):.1f}")
    cv2.imshow('Art Recognizer', final_frame)

cap.release()
cv2.destroyAllWindows()
print("Program ended.")
























































# cd ~/2024-2025-projectone-ctai-KattanHamzzah-1/RPi
# source ../.venv/bin/activate
# python detection_local.py