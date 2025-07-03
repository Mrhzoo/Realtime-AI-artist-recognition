# # import threading
# # import queue
# # import time

# # # Bluez gatt uart service (SERVER)
# # from ble_utils.bluetooth_uart_server import ble_gatt_uart_loop, get_ble_mac

# # # TODO: 1) print BLE device name on LCD on app start
# # # TODO: 2) print incoming bluetooth messages on LCD



# # def main():
# #     print("[app] Adapter MAC:", get_ble_mac())

# #     rx_q = queue.Queue()
# #     tx_q = queue.Queue()
# #     device_name = "HzPi-Server" # TODO: replace with your own (unique) device name
# #     evt_q = queue.Queue()          # New queue which provides the connection state of our ble server

# #     threading.Thread(target=ble_gatt_uart_loop, args=(rx_q, tx_q, device_name, evt_q), daemon=True).start()
# #     try:         
# #         while True:
# #             try:
# #                 incoming = rx_q.get_nowait()
# #                 print("In main loop: {}".format(incoming))
# #             except queue.Empty:
# #                 pass # nothing in Q 
# #             time.sleep(0.01)
# #     except KeyboardInterrupt:
# #         pass
# #     finally:
# #         # TODO: maybe cleanup if needed or code get's extended
# #         pass
        
# # if __name__ == '__main__':
# #     main()







# import threading
# import queue
# import time
# from RPLCD.i2c import CharLCD
# from ble_utils.bluetooth_uart_server import ble_gatt_uart_loop, get_ble_mac

# print("[app] Starting...")

# try:
#     lcd = CharLCD('PCF8574', 0x27, cols=16, rows=2)
#     print("[app] LCD initialized.")
# except Exception as e:
#     print("[ERROR] LCD init failed:", e)

# def display_lcd(line1="", line2=""):
#     try:
#         lcd.clear()
#         lcd.write_string(line1[:16])
#         lcd.crlf()
#         lcd.write_string(line2[:16])
#     except Exception as e:
#         print("[ERROR] LCD write failed:", e)

# def main():
#     print("[app] Inside main()")

#     try:
#         ble_mac = get_ble_mac()
#         print("[app] Adapter MAC:", ble_mac)
#     except Exception as e:
#         print("[ERROR] get_ble_mac failed:", e)
#         return

#     display_lcd("HzPi-Server", ble_mac)

#     rx_q = queue.Queue()
#     tx_q = queue.Queue()
#     device_name = "HzPi-Server"
#     evt_q = queue.Queue()

#     print("[app] Starting BLE thread...")
#     try:
#         threading.Thread(target=ble_gatt_uart_loop, args=(rx_q, tx_q, device_name, evt_q), daemon=True).start()
#     except Exception as e:
#         print("[ERROR] BLE thread start failed:", e)
#         return

#     print("[app] Entering main loop...")
#     try:
#         while True:
#             try:
#                 incoming = rx_q.get_nowait()
#                 print("In main loop: {}".format(incoming))
#                 display_lcd("Msg received:", incoming)
#             except queue.Empty:
#                 pass
#             time.sleep(0.01)
#     except KeyboardInterrupt:
#         lcd.clear()
#         print("App stopped")
#     finally:
#         lcd.clear()
#         print("[app] Clean exit")

# if __name__ == '__main__':
#     main()







# import cv2
# from ultralytics import YOLO

# def main():
#     # Load the YOLO model. Adjust the path if needed.
#     model_path = "Models/best.pt"
#     print("Loading the model...")
#     try:
#         model = YOLO(model_path)
#         print("Model loaded successfully!")
#     except Exception as e:
#         print(f"Error loading model: {e}")
#         return

#     # Open the webcam (0 is the default device index)
#     cap = cv2.VideoCapture(0)
#     if not cap.isOpened():
#         print("Error: Could not open the webcam.")
#         return

#     print("Starting video stream... Press 'q' to exit.")
#     while True:
#         ret, frame = cap.read()
#         if not ret:
#             print("Error: Failed to grab a frame.")
#             break

#         # Run the model inference on the current frame
#         results = model(frame, conf=0.25)  # You can adjust the confidence threshold if needed

#         # Get the annotated frame using YOLOv8's API (assumes results is a list of one result)
#         annotated_frame = results[0].plot() if results else frame

#         # Show the annotated frame in a window
#         cv2.imshow("Detection", annotated_frame)



#         # Exit if "q" is pressed
#         if cv2.waitKey(1) & 0xFF == ord('q'):
#             break

#     # Clean up resources
#     cap.release()
#     cv2.destroyAllWindows()

# if __name__ == '__main__':
#     main()







# from flask import Flask, Response
# import cv2

# app = Flask(__name__)

# def gen_frames():
#     cap = cv2.VideoCapture(0)  # open the default webcam
#     if not cap.isOpened():
#         print("Error: Webcam not accessible.")
#         return
#     while True:
#         success, frame = cap.read()
#         if not success:
#             break
#         # Encode the frame in JPEG format
#         ret, buffer = cv2.imencode('.jpg', frame)
#         frame = buffer.tobytes()
#         # Yield the output frame in byte format as a part of a multipart HTTP response
#         yield (b'--frame\r\n'
#                b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')

# @app.route('/video_feed')
# def video_feed():
#     # Return a multipart response containing the frames
#     return Response(gen_frames(), mimetype='multipart/x-mixed-replace; boundary=frame')

# @app.route('/')
# def index():
#     # A very simple page that embeds the video stream
#     return ("<html><head><title>Camera Stream</title></head>"
#             "<body><h1>Raspberry Pi Camera Stream</h1>"
#             "<img src='/video_feed' style='width: 80%;' />"
#             "</body></html>")

# if __name__ == '__main__':
#     # Bind to 0.0.0.0 so it can be accessed from any device on your network
#     app.run(host='0.0.0.0', port=5000)













# from flask import Flask, Response
# import cv2
# from ultralytics import YOLO

# app = Flask(__name__)

# # Swap this model path with your Roboflow model path or configuration
# model_path = "RPi/Models/best.pt"
# detection_model = YOLO(model_path)

# def gen_frames():
#     cap = cv2.VideoCapture(0)
#     if not cap.isOpened():
#         print("Error: Webcam not accessible.")
#         return

#     frame_count = 0
#     last_annotated = None
#     while True:
#         success, frame = cap.read()
#         if not success:
#             break

#         frame_count += 1
#         # Process every 5th frame for detection
#         if frame_count % 5 == 0:
#             # Resize frame to a lower resolution for faster inference
#             resized_frame = cv2.resize(frame, (320, 240))
#             results = detection_model(resized_frame, conf=0.6)
#             # Here you could apply filtering of results if necessary
#             # For now using the built-in plot function:
#             annotated_resized = results[0].plot() if results else resized_frame
#             # Optionally, upscale the annotated frame for display (smoothing up to the original dimensions)
#             annotated_frame = cv2.resize(annotated_resized, (frame.shape[1], frame.shape[0]))
#             last_annotated = annotated_frame
#         else:
#             last_annotated = last_annotated if last_annotated is not None else frame

#         ret, buffer = cv2.imencode('.jpg', last_annotated)
#         frame_bytes = buffer.tobytes()
#         yield (
#             b'--frame\r\n'
#             b'Content-Type: image/jpeg\r\n\r\n' + frame_bytes + b'\r\n'
#         )
#     cap.release()

# @app.route('/video_feed')
# def video_feed():
#     return Response(gen_frames(), mimetype='multipart/x-mixed-replace; boundary=frame')

# @app.route('/')
# def index():
#     return ("<html><head><title>Camera Stream - Detection</title></head>"
#             "<body><h1>Raspberry Pi Camera Stream (Detection)</h1>"
#             "<img src='/video_feed' style='width: 80%;' />"
#             "</body></html>")

# if __name__ == '__main__':
#     app.run(host='0.0.0.0', port=5000)



#note new 
















































from flask import Flask, Response, render_template
import cv2
from ultralytics import YOLO
import os
import time

app = Flask(__name__)

# Load the YOLOv8-OBB model
model_path = 'Models/best.pt'  # Adjusted for RPi/Models
model = YOLO(model_path)

def gen_frames():
    cap = cv2.VideoCapture(0)  # Open the default webcam
    if not cap.isOpened():
        print("Error: Webcam not accessible.")
        return
    # Set lower resolution
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
    frame_count = 0
    skip_frames = 80  # Process every 2nd frame

    while True:
        success, frame = cap.read()
        if not success:
            break
        frame_count += 1
        if frame_count % (skip_frames + 1) != 0:  # Skip frames
            continue

        # Run detection
        start_time = time.time()
        results = model(frame, conf=0.1, iou=0.5)  # Lower confidence threshold
        annotated_frame = results[0].plot()  # Add oriented bounding boxes
        inference_time = time.time() - start_time
        print(f"Inference time: {inference_time:.2f}s")

        # Encode the annotated frame
        ret, buffer = cv2.imencode('.jpg', annotated_frame, [int(cv2.IMWRITE_JPEG_QUALITY), 70])  # Lower JPEG quality
        frame = buffer.tobytes()
        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')

@app.route('/video_feed')
def video_feed():
    return Response(gen_frames(), mimetype='multipart/x-mixed-replace; boundary=frame')

@app.route('/')
def index():
    return render_template('index.html')  # Use a template for better control

if __name__ == '__main__':
    os.makedirs('templates', exist_ok=True)
    with open('templates/index.html', 'w') as f:
        f.write("""
        <html><head><title>Camera Stream</title></head>
        <body><h1>Raspberry Pi Camera Stream</h1>
        <img src="/video_feed" style="width: 80%;" />
        </body></html>
        """)
    app.run(host='0.0.0.0', port=5000, threaded=True)