
# 🖼️ AI-Powered Artist Recognition with Raspberry Pi

This project enables real-time artist recognition from paintings using a Raspberry Pi, a USB webcam, and a 16×2 I²C LCD display. It leverages two custom AI models (YOLOv8‑based detector and ResNet‑18 classifier) to identify iconic artists like Monet, Van Gogh, Picasso, and Da Vinci. The system is housed in a 17×27×11 cm wooden box.

## 🔧 Features
- Real-time identification of painting artist names
- Bounding box overlay with confidence scores on webcam feed
- Artist name display on LCD
- Dual-model vision pipeline optimized for Raspberry Pi

## 🪵 Hardware Requirements
- **Raspberry Pi** (Pi 4 or Pi 5 recommended)  
- **USB webcam** (max resolution ~320×240 for speed)
- **16×2 I²C LCD display** (default address `0x27`)
- **Wooden enclosure** (17×27×11 cm with camera and cable cut-outs)
- Supporting materials: USB cables, power supply, I²C/SPI wires

## 💾 Software Dependencies

- Python 3
- `opencv-python`
- `ultralytics` (YOLOv8)
- `onnxruntime`
- `smbus2`

> Install all with:
```bash
pip install opencv-python ultralytics onnxruntime smbus2
```

## 🛠️ Setup & Installation

1. **Flash Raspberry Pi OS** using Raspberry Pi Imager.
2. **Enable I²C** (`sudo raspi-config` → Interface Options → I²C).
3. **Connect hardware**:
   - USB webcam to Pi
   - I²C LCD (wired to SDA/SCL)
4. **Test devices**:
```bash
python -c "import cv2; print(cv2.VideoCapture(0).read())"
i2cdetect -y 1  # ensure LCD shows at address 0x27
```

## 🧠 Model Training & Preparation

- **Dataset**: Photos of paintings by Monet, Van Gogh, Picasso, Da Vinci (printed, webcam‑shot, cropped)  
- **Detection**: Trained YOLOv8‑n to identify paintings within camera feed  
- **Classification**: ResNet‑18 (converted to ONNX) for artist recognition within detected patches  
- **Training flow**: Data augmentation, iterative tuning — ResNet‑18 chosen over ResNet‑34 for being lightweight and Pi‑friendly

## 🧩 Core Files & Structure
```
.
├── models/
│   ├── yolov8n_detector.onnx
│   └── resnet18_classifier.onnx
├── src/
│   ├── detect_and_classify.py  # main pipeline script
│   ├── lcd_display.py          # LCD helper functions
│   └── utils.py                # auxiliary utilities
├── enclosure_design/           # laser-cut & box build files
├── BOM_bill-of-materials.pdf   # cost breakdown (~€220–€300)
└── README.md                   # this file
```

## 🚀 How to Run

1. Launch script:
   ```bash
   python src/detect_and_classify.py
   ```
2. **Process Flow**:
   - Grab frame from webcam at low-res for speed
   - Detect painting area with YOLOv8‑n
   - Crop each detection and classify with ResNet‑18
   - Overlay bounding box + artist confidence on live video
   - Display artist name on LCD

## 🎨 Live Preview & Optimization

- Overlaid feed viewable via SSH + VNC
- Use confidence threshold (`0.5` default)
- To reduce lag: consider frame skipping or lowering resolution further

## 🧪 Testing & Tuning

- Test with varied style prints under different lighting
- Calibrate threshold for single vs. multiple detections
- Monitor LCD for successful artist read-outs

## 🛠️ Future Enhancements

- Expand artist dataset to include lesser-known painters
- Improve hardware enclosure aesthetics or size
- Add GUI dashboard, higher-res camera, or graphical UI
- Potential museum-grade curator tools or art authentication usage

## 📋 Credits & License

- **Author**: Hamzzah Kattan (Ct&AI student at Howest)  
- **Instructables source**: [AI-Powered Artist Recognition With Raspberry Pi](https://www.instructables.com/AI-Powered-Artist-Recognition-With-Raspberry-Pi/)
- Please cite original tutorial and this repo when reusing code/design.
