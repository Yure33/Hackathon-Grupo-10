"""
Sistema de Monitoramento e Contagem de Pombos (Entrada/Saída) - SECOMP 2026 UNIFEI
Arquitetura com Inicialização de Estado no Primeiro Frame e Atualização em Tempo Real.

Regra de Negócio:
    - Frame 0: Pombos à esquerda da linha já iniciam computados no saldo acumulado (+1 cada).
    - Frames N: Transição DIREITA -> ESQUERDA (A -> B): ENTRADA (+1)
                Transição ESQUERDA -> DIREITA (B -> A): SAÍDA (-1)
"""
from __future__ import annotations

import argparse
import csv
import math
from collections import deque
from dataclasses import dataclass, field

import cv2
import numpy as np


@dataclass
class Config:
    line_x_ratio: float = 0.5         # Posição relativa da linha (0.0 a 1.0)
    max_lost_seconds: float = 1.2     # Tolerância temporal para perda de ID
    reid_max_dist_ratio: float = 0.12 # Distância máxima para re-identificação
    min_hits: int = 2                 # Mínimo de frames para validar o rastreio


@dataclass
class PigeonTrack:
    lid: int                           # ID lógico estável
    first_frame: int
    last_frame: int
    last_center: tuple[float, float]
    current_side: str                  # 'A' (Direita/Externo) ou 'B' (Esquerda/Interno)
    first_xs: list[float] = field(default_factory=list)
    last_xs: deque = field(default_factory=deque)
    hits: int = 0

    def add(self, frame_idx: int, cx: float, cy: float, side: str, n: int = 5) -> None:
        if len(self.first_xs) < n:
            self.first_xs.append(cx)
        self.last_xs.append(cx)
        self.last_center = (cx, cy)
        self.last_frame = frame_idx
        self.current_side = side
        self.hits += 1


class RealTimeEntryExitCounter:
    def __init__(self, frame_width: int, fps: float, cfg: Config | None = None):
        self.cfg = cfg or Config()
        self.line_x = frame_width * self.cfg.line_x_ratio
        self.max_lost = max(1, int(self.cfg.max_lost_seconds * fps))
        self.reid_dist = frame_width * self.cfg.reid_max_dist_ratio
        self.fps = fps

        self.count = 0
        self.active: dict[int, PigeonTrack] = {}
        self.tid2lid: dict[int, int] = {}
        self.events: list[dict] = []
        self._next_lid = 1
        self.reid_merges = 0

    def get_side(self, x: float) -> str:
        """A = EXTERNO (Direita | x > line_x) | B = INTERNO (Esquerda | x <= line_x)"""
        return "A" if x > self.line_x else "B"

    def update(self, frame_idx: int, dets: list[tuple[int, float, float]]) -> tuple[dict[int, int], list[str]]:
        n = 5
        assigned: dict[int, int] = {}
        new_dets: list[tuple[int, float, float]] = []
        frame_logs: list[str] = []

        # 1) Associação de IDs conhecidos
        for tid, cx, cy in dets:
            lid = self.tid2lid.get(tid)
            if lid is not None and lid in self.active:
                trk = self.active[lid]
                if trk.last_frame != frame_idx:
                    side_atual = self.get_side(cx)

                    # TRANSIÇÃO EM TEMPO REAL (Apenas após a inicialização)
                    if trk.current_side != side_atual and trk.hits >= self.cfg.min_hits:
                        if trk.current_side == "A" and side_atual == "B":
                            self.count += 1
                            evento = "ENTROU"
                            msg = f"🚨 [ENTRADA] Pombo Lógico #{trk.lid} cruzou A -> B. Total no RU: {self.count}"
                        elif trk.current_side == "B" and side_atual == "A":
                            self.count = max(0, self.count - 1)
                            evento = "SAIU"
                            msg = f"🚨 [SAÍDA] Pombo Lógico #{trk.lid} cruzou B -> A. Total no RU: {self.count}"
                        else:
                            evento = "INDEFINIDO"
                            msg = ""

                        if msg:
                            print(msg, flush=True)
                            frame_logs.append(msg)
                            self.events.append({
                                "lid": trk.lid,
                                "yolo_id": tid,
                                "evento": evento,
                                "frame": frame_idx,
                                "tempo_seg": round(frame_idx / self.fps, 2),
                                "posicao_x": round(cx, 1),
                                "posicao_y": round(cy, 1),
                                "contador_resultante": self.count
                            })

                    trk.add(frame_idx, cx, cy, side_atual, n)
                assigned[tid] = lid
            else:
                new_dets.append((tid, cx, cy))

        # 2) Re-associação de IDs perdidos (Recuperação de Glitches de Rastreio)
        lost = [t for t in self.active.values() if t.last_frame < frame_idx]
        pairs = []
        for i, (_, cx, cy) in enumerate(new_dets):
            for t in lost:
                d = math.hypot(cx - t.last_center[0], cy - t.last_center[1])
                if d <= self.reid_dist:
                    pairs.append((d, i, t.lid))
        pairs.sort()

        used_det: set[int] = set()
        used_lid: set[int] = set()
        for _, i, lid in pairs:
            if i in used_det or lid in used_lid:
                continue
            used_det.add(i)
            used_lid.add(lid)
            tid, cx, cy = new_dets[i]
            self.tid2lid[tid] = lid
            side_atual = self.get_side(cx)
            self.active[lid].add(frame_idx, cx, cy, side_atual, n)
            assigned[tid] = lid
            self.reid_merges += 1

        # 3) Registro de Novos Pombos
        for i, (tid, cx, cy) in enumerate(new_dets):
            if i in used_det:
                continue
            lid = self._next_lid
            self._next_lid += 1
            side_inicial = self.get_side(cx)

            trk = PigeonTrack(
                lid=lid, first_frame=frame_idx, last_frame=frame_idx,
                last_center=(cx, cy), current_side=side_inicial,
                last_xs=deque(maxlen=n),
            )
            trk.add(frame_idx, cx, cy, side_inicial, n)
            self.active[lid] = trk
            self.tid2lid[tid] = lid
            assigned[tid] = lid

            # REGRA DE INICIALIZAÇÃO NO NOVO FRAME (FRAME 0 OU PRIMEIRO RASTREIO NO INTERIOR)
            if frame_idx == 0 and side_inicial == "B":
                self.count += 1
                msg_init = f"✨ [INICIALIZAÇÃO - JÁ DENTRO] Pombo #{lid} detectado no interior no Frame 0. Contador inicial: {self.count}"
                print(msg_init, flush=True)
                self.events.append({
                    "lid": lid,
                    "yolo_id": tid,
                    "evento": "INICIAL_INTERIOR",
                    "frame": frame_idx,
                    "tempo_seg": 0.0,
                    "posicao_x": round(cx, 1),
                    "posicao_y": round(cy, 1),
                    "contador_resultante": self.count
                })
            else:
                log_novo = f"ℹ️ [NOVO RASTREIO] Pombo #{lid} detectado no Lado {side_inicial} (Frame {frame_idx})"
                print(log_novo, flush=True)

        # 4) Limpeza de instâncias inativas
        for lid in [l for l, t in self.active.items() if frame_idx - t.last_frame > self.max_lost]:
            self.active.pop(lid)
            self.tid2lid = {k: v for k, v in self.tid2lid.items() if v != lid}

        return assigned, frame_logs


def run(args) -> None:
    from ultralytics import YOLO

    cap = cv2.VideoCapture(args.video)
    if not cap.isOpened():
        raise SystemExit(f"ERRO: Não foi possível carregar o arquivo de vídeo '{args.video}'.")

    W = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    H = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    cap.release()

    print("\n" + "="*80)
    print("      SISTEMA DE MONITORAMENTO DE PRAGAS E CONTROLE LOGÍSTICO - RU UNIFEI")
    print("="*80)
    print(f"Especificações: {W}x{H}px | {fps:.1f} FPS | Total de Frames: {total_frames}")
    print(f"Modelo YOLO: '{args.model}' | Limiar de Confiança: {args.conf}")
    print("="*80 + "\n")

    cfg = Config(line_x_ratio=args.line, max_lost_seconds=args.max_lost)
    counter = RealTimeEntryExitCounter(W, fps, cfg)

    writer = None
    if args.output:
        writer = cv2.VideoWriter(args.output, cv2.VideoWriter_fourcc(*"mp4v"), fps, (W, H))

    model = YOLO(args.model)
    results = model.track(
        source=args.video, stream=True, persist=True,
        tracker=args.tracker, conf=args.conf, verbose=False
    )

    palette = [(0, 255, 0), (0, 165, 255), (255, 0, 255), (255, 255, 0),
               (0, 0, 255), (255, 128, 0), (128, 255, 128), (200, 100, 255)]

    frame_idx = -1
    for frame_idx, r in enumerate(results):
        frame = r.orig_img.copy()
        dets, boxes = [], {}

        if r.boxes is not None and r.boxes.id is not None:
            ids = r.boxes.id.int().cpu().tolist()
            xyxy = r.boxes.xyxy.cpu().numpy()
            for tid, (x1, y1, x2, y2) in zip(ids, xyxy):
                dets.append((tid, (x1 + x2) / 2, (y1 + y2) / 2))
                boxes[tid] = (int(x1), int(y1), int(x2), int(y2))

        assigned, _ = counter.update(frame_idx, dets)

        # RENDERIZAÇÃO DA INTERFACE GRÁFICA (OpenCV)
        lx = int(counter.line_x)
        cv2.line(frame, (lx, 0), (lx, H), (0, 255, 255), 2)
        cv2.putText(frame, "INTERNO (B)", (lx - 150, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
        cv2.putText(frame, "EXTERNO (A)", (lx + 10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)

        for tid, lid in assigned.items():
            if tid in boxes:
                x1, y1, x2, y2 = boxes[tid]
                color = palette[lid % len(palette)]
                trk = counter.active.get(lid)
                side_str = trk.current_side if trk else "?"

                cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
                cv2.putText(frame, f"P{lid} [{side_str}]", (x1, max(15, y1 - 6)),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)

        # Painel Visual de Status
        cv2.rectangle(frame, (10, H - 60), (380, H - 10), (0, 0, 0), -1)
        cv2.putText(frame, f"POMBOS NO RU: {counter.count}", (20, H - 25),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 255), 2)

        if writer:
            writer.write(frame)
        if args.show:
            cv2.imshow("Monitoramento de Pragas - RU UNIFEI", frame)
            if cv2.waitKey(1) & 0xFF == ord("q"):
                break

    if writer:
        writer.release()
    cv2.destroyAllWindows()

    # RELATÓRIO FINAL DE EXECUÇÃO
    print("\n" + "="*80)
    print("                     RELATÓRIO DE AUDITORIA E LOGÍSTICA")
    print("="*80)
    print(f"Total de Frames Processados: {frame_idx + 1}")
    print(f"Correções de Glitch (Reassociações de ID): {counter.reid_merges}")
    print("-" * 80)

    iniciais = sum(1 for e in counter.events if e["evento"] == "INICIAL_INTERIOR")
    entrou = sum(1 for e in counter.events if e["evento"] == "ENTROU")
    saiu = sum(1 for e in counter.events if e["evento"] == "SAIU")

    print(f"Eventos Confirmados:")
    print(f"  • Pombos Existentes Inicialmente (Lado B): {iniciais}")
    print(f"  • Entradas Registradas (A -> B):          {entrou}")
    print(f"  • Saídas Registradas (B -> A):            {saiu}")
    print(f"  • SALDO ACUMULADO FINAL NO RU:            {counter.count}")
    print("="*80)

    if args.csv and counter.events:
        with open(args.csv, "w", newline="", encoding="utf-8") as f:
            w = csv.DictWriter(f, fieldnames=counter.events[0].keys())
            w.writeheader()
            w.writerows(counter.events)
        print(f"📊 Relatório exportado com sucesso para o arquivo: '{args.csv}'\n")


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--model", default="best-pt/best-30-30.pt", help="Caminho do arquivo .pt treinado")
    p.add_argument("--video", default="videos/claude-pigeon-video-3.mp4", help="Arquivo de vídeo de entrada")
    p.add_argument("--output", default=None, help="Caminho para salvar o vídeo anotado")
    p.add_argument("--csv", default="relatorio_pombos.csv", help="Caminho para salvar o relatório CSV")
    p.add_argument("--line", type=float, default=0.5, help="Posição proporcional da linha (0.0 a 1.0)")
    p.add_argument("--conf", type=float, default=0.3, help="Limiar de confiança da YOLO")
    p.add_argument("--max-lost", type=float, default=1.2, help="Segundos de tolerância para perda de ID")
    p.add_argument("--tracker", default="bytetrack.yaml", help="Configuração do algoritmo de tracking")
    p.add_argument("--show", action=argparse.BooleanOptionalAction, default=True, help="Exibe a janela gráfica")
    run(p.parse_args())