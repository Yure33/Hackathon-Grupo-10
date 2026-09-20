"""
Contador de pombos (entrada/saída) com YOLO + tracking.

Convenção (linha vertical imaginária em x = LINE_X):
    Lado A = EXTERNO  = à DIREITA da linha  (x >  LINE_X)
    Lado B = INTERNO  = à ESQUERDA da linha (x <= LINE_X)

Regra do contador (avaliada quando o rastreamento TERMINA):
    começou em B e terminou em A -> SAIU   -> contador -= 1
    começou em A e terminou em B -> ENTROU -> contador += 1
    começou e terminou no mesmo lado -> nada acontece

Robustez a "glitches" de tracking (duas camadas):
    1) Um pombo só é "encerrado" depois de ficar MAX_LOST_SECONDS sem ser visto.
       Falhas curtas não encerram o rastreamento.
    2) Se o tracker do YOLO perder o objeto e devolver um ID NOVO, o ID novo é
       reassociado ao pombo perdido mais próximo (dentro de um raio máximo),
       herdando a posição inicial original.

Uso:
    python contador_pombos.py --model pombos.pt --video entrada.mp4 \
        --output saida.mp4 --csv eventos.csv     (janela ao vivo; 'q' fecha)
"""
from __future__ import annotations

import argparse
import csv
import math
from collections import deque
from dataclasses import dataclass, field
from statistics import median

import cv2
import numpy as np


# --------------------------------------------------------------------------- #
# Configuração
# --------------------------------------------------------------------------- #
@dataclass
class Config:
    line_x_ratio: float = 0.5         # posição da linha (0..1 da largura do frame)
    max_lost_seconds: float = 1.5     # tempo sem detecção até encerrar o rastreio
    reid_max_dist_ratio: float = 0.10 # raio de reassociação (fração da largura)
    edge_samples: int = 5             # nº de amostras p/ estimar início e fim
    min_hits: int = 5                 # mín. de detecções p/ o rastreio valer


# --------------------------------------------------------------------------- #
# Estrutura de um pombo rastreado ("track lógico", independente do ID do YOLO)
# --------------------------------------------------------------------------- #
@dataclass
class PigeonTrack:
    lid: int                           # ID lógico (estável mesmo se o YOLO trocar o ID)
    first_frame: int
    last_frame: int
    last_center: tuple[float, float]
    first_xs: list[float] = field(default_factory=list)  # primeiras posições x
    last_xs: deque = field(default_factory=deque)         # últimas posições x
    hits: int = 0

    def add(self, frame_idx: int, cx: float, cy: float, n: int) -> None:
        if len(self.first_xs) < n:
            self.first_xs.append(cx)
        self.last_xs.append(cx)
        self.last_center = (cx, cy)
        self.last_frame = frame_idx
        self.hits += 1

    # Usa a MEDIANA de algumas amostras (e não um único frame) para que um
    # ponto ruidoso no primeiro/último frame não defina o lado errado.
    @property
    def start_x(self) -> float:
        return median(self.first_xs)

    @property
    def end_x(self) -> float:
        return median(self.last_xs)


# --------------------------------------------------------------------------- #
# Contador
# --------------------------------------------------------------------------- #
class EntryExitCounter:
    def __init__(self, frame_width: int, fps: float, cfg: Config | None = None):
        self.cfg = cfg or Config()
        self.line_x = frame_width * self.cfg.line_x_ratio
        self.max_lost = max(1, int(self.cfg.max_lost_seconds * fps))
        self.reid_dist = frame_width * self.cfg.reid_max_dist_ratio
        self.fps = fps

        self.count = 0
        self.active: dict[int, PigeonTrack] = {}   # lid -> track
        self.tid2lid: dict[int, int] = {}          # id do YOLO -> lid
        self.events: list[dict] = []
        self._next_lid = 1
        self.reid_merges = 0

    # ---- lado ------------------------------------------------------------- #
    def side(self, x: float) -> str:
        """A = externo (direita da linha) | B = interno (esquerda da linha)."""
        return "A" if x > self.line_x else "B"

    def start_side(self, lid: int) -> str | None:
        t = self.active.get(lid)
        return self.side(t.start_x) if t and t.first_xs else None

    # ---- atualização por frame ------------------------------------------- #
    def update(self, frame_idx: int, dets: list[tuple[int, float, float]]) -> dict[int, int]:
        """
        dets: lista de (tracker_id, cx, cy) do frame atual.
        Retorna {tracker_id: lid} para desenho/depuração.
        """
        n = self.cfg.edge_samples
        assigned: dict[int, int] = {}
        new_dets: list[tuple[int, float, float]] = []

        # 1) IDs do YOLO que já conhecemos
        for tid, cx, cy in dets:
            lid = self.tid2lid.get(tid)
            if lid is not None and lid in self.active:
                trk = self.active[lid]
                if trk.last_frame != frame_idx:      # evita 2 IDs no mesmo track/frame
                    trk.add(frame_idx, cx, cy, n)
                assigned[tid] = lid
            else:
                new_dets.append((tid, cx, cy))

        # 2) IDs novos: tenta reassociar a pombos "perdidos" (glitch do tracker)
        lost = [t for t in self.active.values() if t.last_frame < frame_idx]
        pairs = []
        for i, (_, cx, cy) in enumerate(new_dets):
            for t in lost:
                d = math.hypot(cx - t.last_center[0], cy - t.last_center[1])
                if d <= self.reid_dist:
                    pairs.append((d, i, t.lid))
        pairs.sort()  # menores distâncias primeiro (atribuição gulosa)

        used_det: set[int] = set()
        used_lid: set[int] = set()
        for _, i, lid in pairs:
            if i in used_det or lid in used_lid:
                continue
            used_det.add(i)
            used_lid.add(lid)
            tid, cx, cy = new_dets[i]
            self.tid2lid[tid] = lid
            self.active[lid].add(frame_idx, cx, cy, n)
            assigned[tid] = lid
            self.reid_merges += 1

        # 3) O que sobrou é um pombo realmente novo
        for i, (tid, cx, cy) in enumerate(new_dets):
            if i in used_det:
                continue
            lid = self._next_lid
            self._next_lid += 1
            trk = PigeonTrack(
                lid=lid, first_frame=frame_idx, last_frame=frame_idx,
                last_center=(cx, cy), last_xs=deque(maxlen=n),
            )
            trk.add(frame_idx, cx, cy, n)
            self.active[lid] = trk
            self.tid2lid[tid] = lid
            assigned[tid] = lid

        # 4) Encerra quem ficou tempo demais sem ser visto
        for lid in [l for l, t in self.active.items()
                    if frame_idx - t.last_frame > self.max_lost]:
            self._finalize(lid)

        return assigned

    def finalize_all(self) -> None:
        """Chame ao fim do vídeo: encerra todos os rastreios ainda ativos."""
        for lid in list(self.active):
            self._finalize(lid)

    # ---- encerramento de um rastreio ------------------------------------- #
    def _finalize(self, lid: int) -> None:
        t = self.active.pop(lid)
        self.tid2lid = {k: v for k, v in self.tid2lid.items() if v != lid}

        if t.hits < self.cfg.min_hits:   # provável falso positivo
            return

        start, end = self.side(t.start_x), self.side(t.end_x)
        if start == "B" and end == "A":
            self.count -= 1
            event = "SAIU"
        elif start == "A" and end == "B":
            self.count += 1
            event = "ENTROU"
        else:
            event = "SEM_MUDANCA"

        self.events.append({
            "lid": lid,
            "evento": event,
            "lado_inicio": start,
            "lado_fim": end,
            "x_inicio": round(t.start_x, 1),
            "x_fim": round(t.end_x, 1),
            "frame_inicio": t.first_frame,
            "frame_fim": t.last_frame,
            "deteccoes": t.hits,
            "contador_apos": self.count,
        })


# --------------------------------------------------------------------------- #
# Vídeo + YOLO
# --------------------------------------------------------------------------- #
def run(args) -> None:
    from ultralytics import YOLO  # import aqui p/ permitir testar a lógica sem YOLO

    cap = cv2.VideoCapture(args.video)
    if not cap.isOpened():
        raise SystemExit(f"ERRO: não consegui abrir o vídeo '{args.video}'. "
                         "Confira o nome, a extensão e a pasta.")
    W = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    H = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    cap.release()
    print(f"Vídeo: {args.video} | {W}x{H} | {fps:.1f} fps | {total} frames")
    print(f"Carregando modelo '{args.model}' e processando... (pode demorar)")

    cfg = Config(line_x_ratio=args.line, max_lost_seconds=args.max_lost)
    counter = EntryExitCounter(W, fps, cfg)

    writer = None
    if args.output:
        writer = cv2.VideoWriter(args.output, cv2.VideoWriter_fourcc(*"mp4v"), fps, (W, H))

    model = YOLO(args.model)
    results = model.track(
        source=args.video, stream=True, persist=True,
        tracker=args.tracker, conf=args.conf, verbose=False,
    )

    frame_idx = -1
    trails: dict[int, deque] = {}   # lid -> últimas posições (para desenhar a trilha)
    palette = [(0, 255, 0), (0, 165, 255), (255, 0, 255), (255, 255, 0),
               (0, 0, 255), (255, 128, 0), (128, 255, 128), (200, 100, 255)]

    for frame_idx, r in enumerate(results):
        frame = r.orig_img.copy()
        dets, boxes = [], {}

        if r.boxes is not None and r.boxes.id is not None:
            ids = r.boxes.id.int().cpu().tolist()
            xyxy = r.boxes.xyxy.cpu().numpy()
            for tid, (x1, y1, x2, y2) in zip(ids, xyxy):
                dets.append((tid, (x1 + x2) / 2, (y1 + y2) / 2))
                boxes[tid] = (int(x1), int(y1), int(x2), int(y2))

        assigned = counter.update(frame_idx, dets)

        # ---- desenho ---- #
        lx = int(counter.line_x)
        cv2.line(frame, (lx, 0), (lx, H), (0, 255, 255), 2)
        cv2.putText(frame, "B (interno)", (lx - 170, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 200, 0), 2)
        cv2.putText(frame, "A (externo)", (lx + 10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 200, 255), 2)

        for tid, lid in assigned.items():
            x1, y1, x2, y2 = boxes[tid]
            color = palette[lid % len(palette)]
            s = counter.start_side(lid) or "?"
            trails.setdefault(lid, deque(maxlen=90)).append((int((x1 + x2) / 2), int((y1 + y2) / 2)))
            cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
            cv2.putText(frame, f"P{lid} inicio:{s}", (x1, max(15, y1 - 6)),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)

        # trilhas: mantém também as de pombos temporariamente perdidos (glitch)
        for lid in [l for l in trails if l not in counter.active]:
            del trails[lid]
        for lid, pts in trails.items():
            color = palette[lid % len(palette)]
            if len(pts) > 1:
                cv2.polylines(frame, [np.array(pts, dtype="int32")],
                              False, color, 2)

        cv2.putText(frame, f"Contador: {counter.count}", (10, H - 15),
                    cv2.FONT_HERSHEY_SIMPLEX, 1.0, (255, 255, 255), 3)

        if frame_idx % 100 == 0:
            print(f"  frame {frame_idx}/{total} | contador: {counter.count}", flush=True)

        if writer:
            writer.write(frame)
        if args.show:
            cv2.imshow("Pombos", frame)
            if cv2.waitKey(1) & 0xFF == ord("q"):
                break

    counter.finalize_all()
    if writer:
        writer.release()
    cv2.destroyAllWindows()

    # ---- relatório ---- #
    print(f"\nFrames processados: {frame_idx + 1}")
    print(f"Reassociações de ID (glitches corrigidos): {counter.reid_merges}")
    for e in counter.events:
        print(f"  P{e['lid']}: {e['lado_inicio']} -> {e['lado_fim']}  {e['evento']}")
    entrou = sum(e["evento"] == "ENTROU" for e in counter.events)
    saiu = sum(e["evento"] == "SAIU" for e in counter.events)
    print(f"Entraram: {entrou} | Saíram: {saiu} | Contador final: {counter.count}")

    if args.csv and counter.events:
        with open(args.csv, "w", newline="", encoding="utf-8") as f:
            w = csv.DictWriter(f, fieldnames=counter.events[0].keys())
            w.writeheader()
            w.writerows(counter.events)


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--model", default="best-30-30.pt", help="caminho do .pt treinado")
    p.add_argument("--video", default="gemini-pigeon-video-1.mp4")
    p.add_argument("--output", default=None, help="vídeo anotado de saída")
    p.add_argument("--csv", default=None, help="CSV com os eventos")
    p.add_argument("--line", type=float, default=0.5, help="posição da linha (0..1)")
    p.add_argument("--conf", type=float, default=0.3)
    p.add_argument("--max-lost", type=float, default=1.5, help="segundos sem ver antes de encerrar")
    p.add_argument("--tracker", default="bytetrack.yaml")
    p.add_argument("--show", action=argparse.BooleanOptionalAction, default=True,
                   help="mostra o vídeo ao vivo (use --no-show para desligar)")
    run(p.parse_args())
