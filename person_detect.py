from ultralytics import YOLO
import cv2

from pathlib import Path
video = Path(__file__).parent/"video.mp4"

model = YOLO("yolov8s.pt")
num_class = 0
cap = cv2.VideoCapture(str(video))


while True:
    ret, image_np = cap.read()
    if not ret: break
    
    # Detecção
    results = model(image_np)
    
    # Bbox
    for r in results:
        for bbox, cls, conf in zip(r.boxes.xyxy, r.boxes.cls, r.boxes.conf):
            if int(cls) == num_class and conf > 0.7:
                x1, y1, x2, y2 = map(int,bbox)
                cv2.rectangle(image_np, (x1, y1), (x2, y2), (0,255,130),2)
                cv2.putText(image_np,f"Person {conf:.2f}",(x1,y1-10),cv2.FONT_HERSHEY_SIMPLEX,0.6,(0,255,130),2)
    
    # exibição
    cv2.imshow("Person Detection", image_np)
    if cv2.waitKey(1) & 0xFF in [ord("q"), 27]:
        break

cap.release()
cv2.destroyAllWindows()
