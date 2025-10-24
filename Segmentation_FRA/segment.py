#!/usr/bin/env python3
"""
Convertitore multi-camera da file .txt di segmentazione a formato YOLOv8 Segmentazione.
Versione con debug visivo e gestione streaming memoria.
Autore: ChatGPT (2025)
"""

import os
import json
import random
import numpy as np
import cv2
from tqdm import tqdm
from scipy.spatial import ConvexHull

# =====================================================
# PARAMETRI GLOBALI
# =====================================================

IMG_WIDTH = 3840
IMG_HEIGHT = 2400
MAX_POINTS = 20
DEBUG_SAMPLES = 10  # Numero di frame casuali per cui salvare immagini debug
SIMPLIFY_POLYGON = True

# =====================================================
# FUNZIONI DI SUPPORTO
# =====================================================

def create_bbox_polygon(points, img_width, img_height):
    """Crea un rettangolo se ci sono meno di 3 punti."""
    x_coords = [p[0] for p in points]
    y_coords = [p[1] for p in points]
    x_min, x_max = min(x_coords), max(x_coords)
    y_min, y_max = min(y_coords), max(y_coords)
    bbox = [[x_min, y_min], [x_max, y_min], [x_max, y_max], [x_min, y_max]]
    return normalize_polygon(bbox, img_width, img_height)

def normalize_polygon(points, img_width, img_height):
    """Normalizza coordinate in [0,1]."""
    normalized = []
    for x, y in points:
        x_norm = max(0.001, min(0.999, x / img_width))
        y_norm = max(0.001, min(0.999, y / img_height))
        normalized.extend([x_norm, y_norm])
    return normalized

def point_line_distance(point, line_start, line_end):
    if np.array_equal(line_start, line_end):
        return np.linalg.norm(point - line_start)
    line_vec = line_end - line_start
    point_vec = point - line_start
    proj = np.dot(point_vec, line_vec) / np.dot(line_vec, line_vec)
    proj_point = line_start + proj * line_vec
    return np.linalg.norm(point - proj_point)

def douglas_peucker(points, epsilon):
    """Algoritmo DP per semplificare una polilinea."""
    if len(points) < 3:
        return points
    start, end = points[0], points[-1]
    max_dist, max_index = 0, 0
    for i in range(1, len(points) - 1):
        dist = point_line_distance(points[i], start, end)
        if dist > max_dist:
            max_dist, max_index = dist, i
    if max_dist > epsilon:
        left = douglas_peucker(points[:max_index + 1], epsilon)
        right = douglas_peucker(points[max_index:], epsilon)
        return np.vstack([left[:-1], right])
    else:
        return np.array([start, end])

def process_polygon_points(points, img_width, img_height, simplify=True, max_points=50):
    """Semplifica e normalizza un poligono."""
    if len(points) < 3:
        return create_bbox_polygon(points, img_width, img_height)

    points_array = np.array(points, dtype=np.float32)

    if simplify:
        # Convex Hull per ridurre rumore
        if len(points_array) > 10:
            try:
                hull = ConvexHull(points_array)
                points_array = points_array[hull.vertices]
            except Exception:
                pass

        # Douglas–Peucker (chiuso)
        if len(points_array) > max_points:
            if not np.array_equal(points_array[0], points_array[-1]):
                closed = np.vstack([points_array, points_array[0]])
            else:
                closed = points_array
            bbox_diag = np.hypot(closed[:,0].ptp(), closed[:,1].ptp())
            epsilon = max(1.0, bbox_diag * 0.01)
            simplified = douglas_peucker(closed, epsilon)
            if np.array_equal(simplified[0], simplified[-1]):
                simplified = simplified[:-1]
            if len(simplified) >= 3:
                points_array = simplified

    # Campiona punti se ancora troppi
    if len(points_array) > max_points:
        idx = np.linspace(0, len(points_array)-1, max_points, dtype=int)
        points_array = points_array[idx]

    return normalize_polygon(points_array, img_width, img_height)

# =====================================================
# CONVERSIONE STREAMING
# =====================================================

def convert_file(input_path, output_dir, img_width, img_height,
                 simplify=True, max_points=20, debug_dir=None):
    os.makedirs(output_dir, exist_ok=True)
    stats = {"frames": 0, "objects": 0, "samples": []}
    frame_objects = []
    current_key = None

    if not os.path.exists(input_path):
        print(f"❌ File non trovato: {input_path}")
        return stats

    # Legge tutte le righe
    with open(input_path, 'r') as f:
        for line in tqdm(f, desc=f"Elaboro {os.path.basename(input_path)}"):
            parts = line.strip().split()
            if len(parts) < 4:
                continue

            try:
                camera_id = int(parts[0])
                frame_id = int(parts[1])
                num_pixel = int(parts[3])
                coords = parts[4:]
                if len(coords) != num_pixel * 2:
                    continue

                points = [[int(coords[i]), int(coords[i+1])] for i in range(0, len(coords), 2)]
                polygon = process_polygon_points(points, img_width, img_height, simplify, max_points)
                if not polygon or len(polygon) < 6:
                    continue

                key = (camera_id, frame_id)
                if current_key is None:
                    current_key = key

                if key != current_key:
                    flush_frame(current_key, frame_objects, output_dir)
                    frame_objects = []
                    stats["frames"] += 1
                    current_key = key

                frame_objects.append({'label': 0, 'polygon': polygon})
                stats["objects"] += 1

            except Exception:
                continue

    # flush finale
    if current_key and frame_objects:
        flush_frame(current_key, frame_objects, output_dir)
        stats["frames"] += 1

    # genera immagini debug
    if debug_dir:
        os.makedirs(debug_dir, exist_ok=True)
        generate_debug_images(output_dir, debug_dir, img_width, img_height, DEBUG_SAMPLES)

    print(f"\n📊 Cam {camera_id} -> {stats['frames']} frame | {stats['objects']} oggetti\n")
    return stats

def flush_frame(key, objects, output_dir):
    """Scrive il file YOLO per un singolo frame."""
    camera_id, frame_id = key
    filename = f"CAM{camera_id}_Img_{frame_id:06d}.txt"
    with open(os.path.join(output_dir, filename), 'w') as f:
        for obj in objects:
            coords = ' '.join([f"{c:.6f}" for c in obj['polygon']])
            f.write(f"{obj['label']} {coords}\n")

# =====================================================
# DEBUG VISIVO
# =====================================================

def generate_debug_images(label_dir, debug_dir, img_width, img_height, num_samples=10):
    """Crea immagini di debug con poligoni disegnati."""
    files = [f for f in os.listdir(label_dir) if f.endswith('.txt')]
    if not files:
        print("⚠️ Nessun file label trovato per il debug.")
        return

    samples = random.sample(files, min(num_samples, len(files)))
    for txt_file in samples:
        img = np.zeros((img_height//4, img_width//4, 3), dtype=np.uint8)  # immagine nera ridotta
        path = os.path.join(label_dir, txt_file)
        with open(path) as f:
            for line in f:
                parts = line.strip().split()
                if len(parts) < 7:
                    continue
                coords = list(map(float, parts[1:]))
                poly = np.array([[coords[i]*img_width//4, coords[i+1]*img_height//4]
                                for i in range(0, len(coords), 2)], np.int32)
                cv2.polylines(img, [poly], True, (0,255,0), 1)
        out_path = os.path.join(debug_dir, txt_file.replace(".txt", "_debug.jpg"))
        cv2.imwrite(out_path, img)
    print(f"🖼️ Debug: salvate {len(samples)} immagini in {debug_dir}/")

# =====================================================
# MAIN
# =====================================================

def main():
    base_input_dir = "/mnt/Atlante/PROGgETTO_MATTEO/Acq_uccelli/Segmentation/20221219_palmax_acq_002"
    base_output_dir = "20221219_acq002_labels_yolo_seg_multi"
    debug_dir = os.path.join(base_output_dir, "_debug")

    os.makedirs(base_output_dir, exist_ok=True)

    cams = [1, 2, 3]
    global_stats = {}

    for cam_id in cams:
        input_file = os.path.join(base_input_dir, f"cam{cam_id}_countorns.txt")
        output_dir = os.path.join(base_output_dir, f"cam{cam_id}")
        stats = convert_file(
            input_file,
            output_dir,
            IMG_WIDTH,
            IMG_HEIGHT,
            simplify=SIMPLIFY_POLYGON,
            max_points=MAX_POINTS,
            debug_dir=debug_dir if cam_id == 1 else None  # debug solo su cam1 per velocità
        )
        global_stats[f"cam{cam_id}"] = stats

    with open(os.path.join(base_output_dir, "conversion_report.json"), 'w') as jf:
        json.dump(global_stats, jf, indent=4)

    print("\n✅ Conversione completata per tutte le camere.")
    print(f"📂 Output: {base_output_dir}/")
    print(f"📑 Report: {os.path.join(base_output_dir, 'conversion_report.json')}")

# =====================================================

if __name__ == "__main__":
    main()
