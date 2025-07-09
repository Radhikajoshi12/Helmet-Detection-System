from ultralytics import YOLO
import cv2
import os
import winsound
from datetime import datetime

# Load your trained model
model = YOLO('helmet_detection/helmet_model/weights/best.pt')

# Set detection confidence threshold
model.conf = 0.3

# Create a folder to save violations
os.makedirs('violations', exist_ok=True)
os.makedirs('screenshots', exist_ok=True)


# Start webcam
cap = cv2.VideoCapture(0)

while True:
    ret, frame = cap.read()
    if not ret:
        break

    # Run detection
    results = model(frame, stream=True)

    # Check detections and alert if no helmet
    for r in results:
        boxes = r.boxes
        for box in boxes:
            cls = int(box.cls[0])  # class index
            label = model.names[cls]
            print(f"Detected: {label}")  # For debugging

            if label == 'no_helmet':
                
                winsound.PlaySound(r'C:\Users\HP\Desktop\Radha\ANN\beep.wav', winsound.SND_FILENAME | winsound.SND_ASYNC)


                # Save the frame
                timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
                cv2.imwrite(f'violations/violation_{timestamp}.jpg', frame)

        annotated_frame = r.plot()

    # Show the frame
        # Show the frame
    cv2.imshow('Helmet Detection', annotated_frame)

    key = cv2.waitKey(1) & 0xFF

    # Exit on 'q'
    if key == ord('q'):
        break

    # Save screenshot on 's'
    if key == ord('s'):
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        cv2.imwrite(f'screenshots/screenshot_{timestamp}.jpg', annotated_frame)
        print("Screenshot saved!")

cap.release()
cv2.destroyAllWindows()
