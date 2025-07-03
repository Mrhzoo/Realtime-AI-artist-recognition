import cv2

cap = cv2.VideoCapture(0)

print("Press 'q' to quit.")

while True:
    ret, frame = cap.read()
    if not ret:
        print("Failed to grab frame.")
        break

    # Try different color conversions below (ONE AT A TIME):
    
    # Uncomment one at a time and see which looks correct:

    # Option A - try raw (may look green/pink)
    # output = frame

    # Option B - YUV to BGR
    # output = cv2.cvtColor(frame, cv2.COLOR_YUV2BGR)

    # Option C - BGR to RGB
    # output = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    # Option D - YUV to RGB
    output = cv2.cvtColor(frame, cv2.COLOR_YUV2RGB)

    cv2.imshow("Camera Test", output)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
