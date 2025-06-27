import cv2 
from ultralytics import YOLO

def contar_carros(caminho_video):
    #------------ Carrega o modelo YOLO pré-treinado para detecção de objetos
    modelo = YOLO("yolov8m.pt")  
    video = cv2.VideoCapture(caminho_video)

    ID_CARRO = 2  #------------ Classe de carro na base COCO
    AREA_MINIMA = 3000  #------------ Ignora carros muito pequenos ou distantes

    while video.isOpened():
        leitura_ok, quadro = video.read()
        if not leitura_ok:
            break

        #------------ Detecta objetos no quadro com uma confiança mínima
        resultados = modelo(quadro, conf=0.2)[0]  
        total_carros = 0

        for caixa in resultados.boxes:
            classe_detectada = int(caixa.cls[0].item())
            if classe_detectada == ID_CARRO:
                x1, y1, x2, y2 = map(int, caixa.xyxy[0])
                area = (x2 - x1) * (y2 - y1)

                if area < AREA_MINIMA:
                    continue  #------------ Ignora objetos pequenos (provavelmente muito distantes)

                total_carros += 1
                #------------ Desenha a caixa ao redor do carro detectado
                cv2.rectangle(quadro, (x1, y1), (x2, y2), (0, 255, 0), 2)
                cv2.putText(quadro, "Carro", (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)

        #------------ Mostra o total de carros detectados no canto superior esquerdo
        cv2.putText(quadro, f"Carros detectados: {total_carros}", (30, 40),
                    cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0, 0, 0), 3)

        #------------ Codifica a imagem em bytes para exibição no Streamlit
        _, imagem_codificada = cv2.imencode('.jpg', quadro)
        imagem_bytes = imagem_codificada.tobytes()
        yield imagem_bytes

    video.release()
