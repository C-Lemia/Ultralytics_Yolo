import cv2
from ultralytics import YOLO

def contar_carros(video_path):
    model = YOLO("yolov8m.pt")  # Use yolov8n.pt se quiser mais leve
    cap = cv2.VideoCapture(video_path)

    CAR_CLASS_ID = 2  # Classe "car" no COCO
    AREA_MINIMA = 3000  # Evita detectar carros muito pequenos/distantes

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        results = model(frame, conf=0.2)[0]  # Confiança mínima reduzida
        car_count = 0

        for box in results.boxes:
            cls = int(box.cls[0].item())
            if cls == CAR_CLASS_ID:
                x1, y1, x2, y2 = map(int, box.xyxy[0])
                area = (x2 - x1) * (y2 - y1)
                if area < AREA_MINIMA:
                    continue  # Ignora objetos pequenos (longe demais)

                car_count += 1
                cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
                cv2.putText(frame, "Carro", (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)

        # Exibe total de carros no topo da imagem
        cv2.putText(frame, f"Carros detectados: {car_count}", (30, 40),
                    cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0, 0, 0), 3)

        _, buffer = cv2.imencode('.jpg', frame)
        frame_bytes = buffer.tobytes()
        yield frame_bytes

    cap.release()
