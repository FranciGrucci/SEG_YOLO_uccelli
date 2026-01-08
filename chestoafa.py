#!/usr/bin/env python3
import os
import glob
import json
from pathlib import Path
import numpy as np
from scipy.optimize import linear_sum_assignment
from shapely.geometry import Polygon
from PIL import Image, ImageDraw
from collections import defaultdict
import math
import random
import re

# =============================
# PATHS
# =============================
PREDICTIONS_DIR = Path("/mnt/Atlante/SEG_YOLO_uccelli/test_results_seg_sliced/predictions")
TEST_IMAGES_DIR = Path("/mnt/Atlante/SEG_YOLO_uccelli/dataset/images/test")
OUTPUT_DIR = Path("/mnt/Atlante/SEG_YOLO_uccelli/test_results_tracking_masksort_newparams")
VIS_DIR = OUTPUT_DIR / "visualizations"
TRAJ_DIR = OUTPUT_DIR / "trajectories_2d"  # Cartella per le traiettorie 2D separate

OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
VIS_DIR.mkdir(parents=True, exist_ok=True)
TRAJ_DIR.mkdir(parents=True, exist_ok=True)
(OUTPUT_DIR / "json").mkdir(parents=True, exist_ok=True)

# =============================
# CONFIG TRACKER
# =============================
IOU_THRESHOLD = 0.1
DIST_THRESHOLD = 0.3
WEIGHT_MASK = 0.4
WEIGHT_DIST = 0.6
MAX_AGE = 20
MIN_HITS = 2
DT = 1.0
PROCESS_STD = 1.0
MEASUREMENT_STD = 2.0

# =============================
# HELPERS
# =============================
def polygon_from_coco(seg):
    if not seg:
        return None
    flat = seg[0] if isinstance(seg[0], list) else seg
    if len(flat) < 6:
        return None
    pts = [(flat[i], flat[i+1]) for i in range(0, len(flat), 2)]
    try:
        p = Polygon(pts)
        if (not p.is_valid) or p.area == 0:
            p = p.buffer(0)
        if p.is_empty or p.area == 0:
            return None
        return p
    except Exception:
        return None

def mask_iou(a, b):
    if a is None or b is None:
        return 0.0
    A = polygon_from_coco(a)
    B = polygon_from_coco(b)
    if A is None or B is None:
        return 0.0
    inter = A.intersection(B).area
    union = A.union(B).area
    return inter / union if union > 0 else 0.0

def centroid_from_seg(seg):
    if not seg:
        return None
    flat = seg[0] if isinstance(seg[0], list) else seg
    if len(flat) < 6:
        return None
    xs = flat[0::2]
    ys = flat[1::2]
    return sum(xs) / len(xs), sum(ys) / len(ys)

def centroid_from_bbox(b):
    x1, y1, x2, y2 = b
    return ((x1 + x2) / 2, (y1 + y2) / 2)

def color_for_id(tid):
    random.seed(abs(hash(str(tid))) % (2**32))
    return tuple(random.randint(0, 255) for _ in range(3))

# =============================
# KALMAN 2D
# =============================
class Kalman2D:
    def __init__(self, x, y):
        self.x = np.array([x, y, 0., 0.])
        self.P = np.eye(4) * 500
        self.F = np.array([
            [1, 0, DT, 0],
            [0, 1, 0, DT],
            [0, 0, 1, 0],
            [0, 0, 0, 1]
        ])
        self.H = np.array([
            [1, 0, 0, 0],
            [0, 1, 0, 0]
        ])
        self.Q = np.eye(4) * PROCESS_STD**2
        self.R = np.eye(2) * MEASUREMENT_STD**2

    def predict(self):
        self.x = self.F @ self.x
        self.P = self.F @ self.P @ self.F.T + self.Q

    def update(self, x, y):
        z = np.array([x, y])
        S = self.H @ self.P @ self.H.T + self.R
        K = self.P @ self.H.T @ np.linalg.inv(S)
        self.x += K @ (z - self.H @ self.x)
        self.P = (np.eye(4) - K @ self.H) @ self.P

    def get_state(self):
        return self.x

# =============================
# TRACK CLASS
# =============================
class Track:
    def __init__(self, det, tid, frame_num, diag):
        self.id = tid
        self.bbox = det["bbox"]
        self.seg = det.get("segmentation", [])
        c = centroid_from_seg(self.seg) or centroid_from_bbox(self.bbox)
        self.kf = Kalman2D(*c)
        self.hits = 1
        self.time_since_update = 0
        self.frame_num = frame_num
        self.diag = diag

    def predict(self):
        self.kf.predict()
        self.time_since_update += 1

    def update(self, det, frame_num):
        self.bbox = det["bbox"]
        self.seg = det.get("segmentation", [])
        c = centroid_from_seg(self.seg) or centroid_from_bbox(self.bbox)
        self.kf.update(*c)
        self.hits += 1
        self.time_since_update = 0
        self.frame_num = frame_num

    def get_state(self):
        return self.kf.get_state()[:2]

# =============================
# MASKSORT
# =============================
class MaskSORT:
    def __init__(self, img_size):
        self.tracks = []
        self.next_id = 1
        self.diag = math.hypot(*img_size)

    def update(self, dets, frame_num):
        for tr in self.tracks:
            tr.predict()

        N = len(self.tracks)
        M = len(dets)
        cost = np.ones((N, M)) * 1e5

        for i, tr in enumerate(self.tracks):
            tcx, tcy = tr.get_state()
            for j, det in enumerate(dets):
                iou = mask_iou(tr.seg, det.get("segmentation", []))
                dc = centroid_from_seg(det.get("segmentation", [])) or centroid_from_bbox(det["bbox"])
                dist = math.hypot(tcx - dc[0], tcy - dc[1]) / self.diag
                cost[i, j] = WEIGHT_MASK * (1 - iou) + WEIGHT_DIST * dist

        assigned = []
        if N > 0 and M > 0:
            row, col = linear_sum_assignment(cost)
            for r, c in zip(row, col):
                if cost[r, c] < 1.0:
                    dets[c]["track_id"] = self.tracks[r].id
                    self.tracks[r].update(dets[c], frame_num)
                    assigned.append(c)

        for j, det in enumerate(dets):
            if j not in assigned:
                det["track_id"] = self.next_id
                self.tracks.append(Track(det, self.next_id, frame_num, self.diag))
                self.next_id += 1

        self.tracks = [t for t in self.tracks if t.time_since_update <= MAX_AGE]
        return dets

# =============================
# LOAD PREDICTIONS (FIXED)
# =============================
def load_all_predictions(pred_dir):
    """Carica SOLO i file merged (senza _x*_y*), ignorando le slice"""
    frames_by_cam = defaultdict(list)
    
    json_files = sorted(pred_dir.glob("*.json"))
    print(f"📁 File JSON totali: {len(json_files)}")
    
    # FILTRA: prendi solo file senza coordinate slice
    merged_files = [jf for jf in json_files if not re.search(r"_x\d+_y\d+", jf.stem)]
    print(f"📄 File merged (senza slice): {len(merged_files)}")
    print(f"🗑️  File slice ignorati: {len(json_files) - len(merged_files)}")
    
    if len(merged_files) == 0:
        print(f"⚠️  ATTENZIONE: Nessun file merged trovato!")
        print(f"   I file hanno tutti coordinate _x*_y*")
        print(f"   Esempio: {json_files[0].name}")
        return frames_by_cam
    
    print(f"\n📄 Esempi di file merged:")
    for jf in merged_files[:5]:
        print(f"   - {jf.name}")
    
    for jf in merged_files:
        name = jf.stem
        
        # Estrai camera e frame number
        m = re.search(r"(CAM\d+)_(\d+)", name)
        if m:
            cam = m.group(1)
            fnum = int(m.group(2))
        else:
            print(f"⚠️  Pattern non riconosciuto: {name}")
            continue
        
        # Carica il JSON
        try:
            with open(jf, 'r') as f:
                data = json.load(f)
        except Exception as e:
            print(f"⚠️  Errore nel caricare {jf.name}: {e}")
            continue
        
        # data dovrebbe essere una lista
        if not isinstance(data, list):
            print(f"⚠️  Formato inaspettato in {jf.name}: {type(data)}")
            continue
        
        frames_by_cam[cam].append({
            "frame_num": fnum,
            "image_name": name + ".jpg",
            "objects": data
        })
    
    # Stampa statistiche per camera
    print(f"\n📊 Frame per camera:")
    for cam in sorted(frames_by_cam.keys()):
        frames = frames_by_cam[cam]
        total_objects = sum(len(f["objects"]) for f in frames)
        avg_objects = total_objects / len(frames) if frames else 0
        print(f"   {cam}: {len(frames)} frame, {total_objects} oggetti totali (avg: {avg_objects:.1f}/frame)")
    
    # Ordina per frame number
    for cam in frames_by_cam:
        frames_by_cam[cam].sort(key=lambda x: x["frame_num"])
    
    return frames_by_cam

# =============================
# MAIN
# =============================
def run_tracking():
    print("="*70)
    print("🚀 TRACKING 2D PER OGNI CAMERA (SEPARATO)")
    print("="*70)
    
    frames_by_camera = load_all_predictions(PREDICTIONS_DIR)
    
    if not frames_by_camera:
        print("\n❌ ERRORE: Nessun frame caricato!")
        return

    # Processa ogni camera SEPARATAMENTE
    for cam, frames in sorted(frames_by_camera.items()):
        print(f"\n{'='*70}")
        print(f"🎥 Processing {cam}")
        print(f"{'='*70}")
        print(f"Frame: {len(frames)} (da {frames[0]['frame_num']} a {frames[-1]['frame_num']})")
        
        if not frames:
            continue

        # Tracker SEPARATO per questa camera
        camera_trajectories = defaultdict(list)

        # Trova dimensioni immagine
        sample_img = None
        for frame in frames[:10]:
            img_path = TEST_IMAGES_DIR / frame["image_name"]
            if img_path.exists():
                sample_img = Image.open(img_path)
                break
        
        if sample_img is None:
            print(f"⚠️  Nessuna immagine trovata, uso dimensioni default")
            tracker = MaskSORT((3840, 2400))  # Dimensioni da diagnostica
        else:
            tracker = MaskSORT(sample_img.size)
            print(f"✓ Dimensioni immagine: {sample_img.size}")

        cam_tracks_created = 0
        total_detections = 0
        
        for idx, frame in enumerate(frames):
            if idx % 100 == 0:
                print(f"  Processing frame {idx}/{len(frames)}...")
            
            total_detections += len(frame["objects"])
            dets = tracker.update(frame["objects"], frame["frame_num"])

            # Salva JSON con track_id (opzionale)
            out_j = OUTPUT_DIR / "json" / cam / f"{Path(frame['image_name']).stem}_tracked.json"
            out_j.parent.mkdir(parents=True, exist_ok=True)
            with open(out_j, "w") as f:
                json.dump(dets, f, indent=2)

            # Aggiungi alle traiettorie DELLA CAMERA CORRENTE
            for d in dets:
                cx, cy = centroid_from_seg(d.get("segmentation", [])) or centroid_from_bbox(d["bbox"])
                tid = d["track_id"]
                
                # Conta nuove tracce
                if len(camera_trajectories[tid]) == 0:
                    cam_tracks_created += 1
                
                camera_trajectories[tid].append({
                    "frame": frame["image_name"],
                    "frame_num": frame["frame_num"],
                    "x": float(cx),
                    "y": float(cy),
                    "bbox": d["bbox"],
                    "score": d.get("score", 0.0)
                })

            # Visualizzazione (opzionale, solo ogni 10 frame)
            if idx % 10 == 0:
                img_path = TEST_IMAGES_DIR / frame["image_name"]
                if img_path.exists():
                    try:
                        vis = Image.open(img_path).convert("RGB")
                        draw = ImageDraw.Draw(vis)

                        for d in dets:
                            cx, cy = centroid_from_seg(d.get("segmentation", [])) or centroid_from_bbox(d["bbox"])
                            tid = d["track_id"]
                            col = color_for_id(tid)
                            
                            # Disegna
                            r = 10
                            draw.ellipse([cx-r, cy-r, cx+r, cy+r], fill=col, outline="white", width=3)
                            draw.text((cx+15, cy-10), str(tid), fill=col)

                        vis_cam_dir = VIS_DIR / cam
                        vis_cam_dir.mkdir(parents=True, exist_ok=True)
                        vis.save(vis_cam_dir / frame["image_name"])
                    except Exception as e:
                        pass

        # SALVA TRAIETTORIE DELLA CAMERA CORRENTE
        traj_file = TRAJ_DIR / f"trajectories_{cam}.json"
        traj_dict = dict(camera_trajectories)
        
        with open(traj_file, "w") as f:
            json.dump(traj_dict, f, indent=2)
        
        print(f"\n✓ {cam} completato:")
        print(f"  - Tracce create: {cam_tracks_created}")
        print(f"  - Detections totali: {total_detections}")
        print(f"  - Avg detections/frame: {total_detections/len(frames):.1f}")
        print(f"  💾 Traiettorie salvate in: {traj_file}")
        
        # Statistiche per questa camera
        if traj_dict:
            lengths = [len(v) for v in traj_dict.values()]
            print(f"  📊 Statistiche traiettorie:")
            print(f"     - Lunghezza media: {sum(lengths)/len(lengths):.1f} frame")
            print(f"     - Più lunga: {max(lengths)} frame (ID: {max(traj_dict.keys(), key=lambda k: len(traj_dict[k]))})")
            print(f"     - Più corta: {min(lengths)} frame")
            
            # Filtra traiettorie corte
            long_traj = {k: v for k, v in traj_dict.items() if len(v) >= 10}
            print(f"     - Traiettorie lunghe (≥10 frame): {len(long_traj)}")
    
    print(f"\n{'='*70}")
    print(f"🏁 TRACKING 2D COMPLETATO PER TUTTE LE CAMERE!")
    print(f"{'='*70}")
    print(f"\n📁 File generati:")
    for cam in sorted(frames_by_camera.keys()):
        traj_file = TRAJ_DIR / f"trajectories_{cam}.json"
        if traj_file.exists():
            print(f"  ✓ {traj_file}")
    print(f"\n💡 Prossimo step: Usa questi file per la ricostruzione 3D")
    print(f"   con i parametri di calibrazione delle camere")

if __name__ == "__main__":
    run_tracking()