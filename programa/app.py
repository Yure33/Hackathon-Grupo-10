import cv2
from ultralytics import YOLO


# 1. Função de Alerta Logístico
def acionar_alerta_campus(pombo_id, centro_x, centro_y):
    """
    Função engatilhada automaticamente quando um pombo é detectado na área interna (à esquerda).
    Imprime os logs contendo o ID e as coordenadas cartesianas do centro do objeto.
    """
    print(
        f"🚨 [ALERTA LOGÍSTICO] Pombo ID #{pombo_id} detectado na ZONA INTERNA! Coordenadas Centro: X={centro_x}, Y={centro_y}"
    )


# 2. Inicialização do modelo treinado (Pesos do YOLO)
model = YOLO('best-30-30.pt')

# 3. Configuração do fluxo de captura de vídeo (Webcam: 0 ou arquivo de vídeo)
video_path = 'gemini-pigeon-video-2.mp4'
cap = cv2.VideoCapture(video_path)

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        print("Finalização da transmissão de vídeo ou falha na leitura da câmera.")
        break

    # Dimensões geométricas do frame
    altura, largura, _ = frame.shape

    # Cálculo da linha vertical no centro exato da tela (Eixo X)
    X_LINHA_CENTRAL = largura // 2

    # Renderização da linha delimitadora vertical (Cor: Amarela | Espessura: 2px)
    cv2.line(frame, (X_LINHA_CENTRAL, 0), (X_LINHA_CENTRAL, altura), (0, 255, 255), 2)

    # Rótulos das zonas no topo da tela
    cv2.putText(frame, "ZONA INTERNA (DENTRO)", (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
    cv2.putText(frame, "ZONA EXTERNA (FORA)", (X_LINHA_CENTRAL + 10, 30),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)

    # Execução da inferência computacional
    results = model(frame, conf=0.4)[0]

    pombos_dentro = 0
    pombos_fora = 0

    # Iteração sobre cada objeto detectado pelo modelo
    for idx, box in enumerate(results.boxes, start=1):
        # Mapeamento das coordenadas da caixa delimitadora (Bounding Box)
        x1, y1, x2, y2 = map(int, box.xyxy[0])

        # Cálculo das Coordenadas do Centroide do Retângulo/Pombo
        cx = int((x1 + x2) / 2)
        cy = int((y1 + y2) / 2)

        # LÓGICA DE GEOLOCALIZAÇÃO:
        # Ponto à esquerda da linha central (cx < X_LINHA_CENTRAL) -> DENTRO (Vermelho)
        # Ponto à direita da linha central (cx >= X_LINHA_CENTRAL) -> FORA (Verde)
        if cx < X_LINHA_CENTRAL:
            pombos_dentro += 1
            status_pombo = "DENTRO"
            cor_box = (0, 0, 255)  # Vermelho (BGR)
            acionar_alerta_campus(pombo_id=idx, centro_x=cx, centro_y=cy)
        else:
            pombos_fora += 1
            status_pombo = "FORA"
            cor_box = (0, 255, 0)  # Verde (BGR)

        # 1. Desenha a Bounding Box e o Ponto Central Geométrico
        cv2.rectangle(frame, (x1, y1), (x2, y2), cor_box, 2)
        cv2.circle(frame, (cx, cy), 5, (255, 255, 255), -1)

        # 2. Exibição do Status e Coordenadas no topo da Bounding Box
        label_texto = f"{status_pombo} ({cx}, {cy})"

        # Fundo do texto para melhor visibilidade
        (w_text, h_text), _ = cv2.getTextSize(label_texto, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
        cv2.rectangle(frame, (x1, y1 - h_text - 6), (x1 + w_text, y1), cor_box, -1)
        cv2.putText(frame, label_texto, (x1, y1 - 4),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)

    # Painel do cabeçalho com a contagem geral
    status_geral = f"Pombos Dentro: {pombos_dentro} | Pombos Fora: {pombos_fora}"
    cv2.putText(frame, status_geral, (20, altura - 20),
                cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)

    # Exibição do frame processado em tempo real
    cv2.imshow("Sistema de Monitoramento de Pragas - SECOMP 2026", frame)

    # Interrupção manual do loop de execução via tecla 'q'
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()