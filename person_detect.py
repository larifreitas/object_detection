from ultralytics import YOLO
import cv2

from pathlib import Path
video = Path(__file__).parent/"video.mp4"

model = YOLO("yolov8s.pt")
num_class = 0
cap = cv2.VideoCapture(str(video))


while True:
    ret, frame = cap.read()
    if not ret: break

    # pré processamento e melhoria para visão noturna ou ambiente escuro para detecção
    frame = cv2.resize(640,640)
    frame = cv2.cvtColor(frame,cv2.COLOR_BGR2LAB)
    l,a,b = cv2.split(frame)
    l_eq = cv2.equalizeHist(l)
    frame = cv2.merge(l_eq,a,b)
    frame = cv2.cvtColor(frame,cv2.COLOR_LAB2BGR)
    
    # Detecção
    results = model(frame)
    
    # Bbox
    for r in results:
        for bbox, cls, conf in zip(r.boxes.xyxy, r.boxes.cls, r.boxes.conf):
            if int(cls) == num_class and conf > 0.7:
                x1, y1, x2, y2 = map(int,bbox)
                cv2.rectangle(frame, (x1, y1), (x2, y2), (0,255,130),2)
                cv2.putText(frame,f"Person {conf:.2f}",(x1,y1-10),cv2.FONT_HERSHEY_SIMPLEX,0.6,(0,255,130),2)
    
    # exibição
    cv2.imshow("Person Detection", frame)
    if cv2.waitKey(1) & 0xFF in [ord("q"), 27]:
        break

cap.release()
cv2.destroyAllWindows()
