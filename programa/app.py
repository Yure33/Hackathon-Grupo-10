import cv2
from ultralytics import YOLO


# 1. Função customizada que será disparada ao cruzar a linha
def acionar_alerta_campus(pombo_id, posicao):
    """
    Função chamada automaticamente quando um pombo cruza a linha demarcada.
    Aqui você pode integrar envios de email, logs ou acionamentos sonoros.
    """
    print(f"🚨 [ALERTA LOGÍSTICO] Pombo detectado cruzando a zona restrita! Posição Y: {posicao}")


# 2. Carrega o modelo com os pesos treinados no Google Colab
# Baixe o arquivo 'best.pt' do Colab e coloque na mesma pasta deste script
model = YOLO('best-30-30.pt')

# 3. Carrega o vídeo de teste ou a webcam (use 0 para webcam)
video_path = 'pigeon-video-1.mp4'
cap = cv2.VideoCapture(video_path)

# Definindo a posição da linha virtual (coordenada Y na tela)
Y_LINHA_RESTRIÇÃO = 350

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    # Pega as dimensões do frame para desenhar a linha de ponta a ponta
    altura, largura, _ = frame.shape

    # Desenha a linha virtual de limite na tela (Cor Vermelha, Espessura 3)
    cv2.line(frame, (0, Y_LINHA_RESTRIÇÃO), (largura, Y_LINHA_RESTRIÇÃO), (0, 0, 255), 3)
    cv2.putText(frame, "LINHA DE LIMITE / ZONA RESTRITA", (10, Y_LINHA_RESTRIÇÃO - 10),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)

    # Executa a inferência do modelo no frame atual
    results = model(frame, conf=0.4)[0]

    pombos_na_zona_critica = 0

    # Percorre cada detecção encontrada pela YOLO
    for box in results.boxes:
        # Coordenadas da caixa delimitadora
        x1, y1, x2, y2 = map(int, box.xyxy[0])

        # Calcula o ponto central inferior do pombo (onde ele está pisando/andando)
        cx = int((x1 + x2) / 2)
        cy = int(y2)

        # Desenha a caixa delimitadora do pombo e o ponto de referência
        cv2.rectangle(frame, (x1, y1), (x2, y2), (255, 105, 180), 2)
        cv2.circle(frame, (cx, cy), 5, (0, 255, 255), -1)

        # VERIFICAÇÃO: O pombo cruzou a linha Y?
        if cy > Y_LINHA_RESTRIÇÃO:
            pombos_na_zona_critica += 1

            # Muda a cor da caixa para vermelho se ele violou o limite
            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 0, 255), 3)

            # DISPARADOR DA SUA FUNÇÃO
            acionar_alerta_campus(pombo_id=1, posicao=cy)

    # Exibe o painel de status na tela
    status_texto = f"Pombos alem do limite: {pombos_na_zona_critica}"
    cv2.putText(frame, status_texto, (20, 50),
                cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0) if pombos_na_zona_critica == 0 else (0, 0, 255), 2)

    # Exibe a imagem processada em tempo real
    cv2.imshow("Sistema de Monitoramento de Pragas - SECOMP 2026", frame)

    # Pressione 'q' para fechar a aplicação
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()