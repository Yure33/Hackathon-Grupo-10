# 🐦 Sistema Inteligente de Monitoramento e Controle Logístico de Aves (RU - UNIFEI)

> **Projeto desenvolvido para o Hackathon da SECOMP 2026 (UNIFEI)**  
> *Tema:* Visão Computacional na Logística Universitária.

---

## 💡 Sobre o Problema

A presença excessiva de pombos em áreas de convivência e praças de alimentação abertas, como o **Restaurante Universitário (RU) da UNIFEI**, gera desafios recorrentes relacionados à higiene, riscos sanitários e à conservação do patrimônio público. 

Contudo, monitorar o fluxo contínuo de aves em um grande salão exige uma infraestrutura complexa e de alto custo.

---

## 🚀 A Solução Proposta

Nossa solução consiste em um **Sistema de Contagem Baseada em Fluxo de Passagem**, posicionado estrategicamente nas portas de acesso do RU. 

Em vez de cobrir todo o salão com câmeras, instalamos sensores visuais (voltados para o chão) no topo das portas de entrada e saída. O software utiliza **Visão Computacional e Deep Learning** para:
1. Rastrear o movimento individual de cada ave (`ByteTrack`).
2. Identificar se o pombo está cruzando o limite da porta para **Dentro** (Incremento `+1`) ou para **Fora** (Decremento `-1`).
3. Manter a contagem acumulada em tempo real do número de pombos presentes na área interna do RU, mesmo após eles saírem do campo de visão imediato da câmera.

---

## 🛠️ Tecnologias Utilizadas

O projeto foi construído utilizando uma stack tecnológica moderna, leve e de alta performance:
* **Python 3.10+** (Linguagem principal)
* **Ultralytics YOLO (v8 / v11)** (Fine-tuning de pesos para detecção de pombos)
* **OpenCV (`cv2`)** (Processamento de vídeo em tempo real e renderização geométrica)
* **Roboflow Universe** (Plataforma de obtenção e anotação do dataset de pombos)

---

## ⚙️ Arquitetura do Sistema e Lógica de Funcionamento

[ Câmera Vertical na Porta ] ──► [ YOLO + Tracking Individual ] ──► [ Linha Virtual (Porta) ] ──► [ Contador Global do RU ]

---

## 📦 Como Executar o Projeto Localmente

### 1. Pré-requisitos
Certifique-se de ter o Python instalado em sua máquina. Clone este repositório e acesse a pasta do projeto.

### 2. Instalação de Dependências
Execute o comando abaixo no terminal para instalar as bibliotecas necessárias:
```bash
pip install ultralytics opencv-python roboflow
```

### 3. Configuração dos Arquivos
Certifique-se de colocar o arquivo de pesos do modelo treinado (best-30-30.pt) na raiz do projeto.

Insira o vídeo de teste (ex: pigeon-video-1.mp4) na mesma pasta, ou configure o código para utilizar a webcam (cv2.VideoCapture(0)).

### 4. Executando a Aplicação
```bash
python main.py
```
(Pressione a tecla q para encerrar a execução da janela de vídeo).

## 👥 Equipe / Autores
Equipe 10 - Integrado:
- João Lucas Cândido Carmo
- Vinícius Kadlubiski Teixeira
- Yuri Samuel Barbosa França 

## 📜 Licença e Agradecimentos
Projeto desenvolvido para fins educacionais e competitivos no âmbito acadêmico da Universidade Federal de Itajubá (UNIFEI).
Datasets obtidos via comunidade aberta do Roboflow Universe.