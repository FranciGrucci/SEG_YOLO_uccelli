#!/usr/bin/env python3
"""
generate_yolo_seg_labels_final.py
See comments above in chat for usage.
"""
import os
import math
import argparse
import tempfile
import shutil
import time
from collections import defaultdict

# ---------- CONFIG ----------
BASE_SEG_DIR = "/home/artswarms/Desktop/Segmentation_FRA/20250327_ACQ002/SEGMENTATION"
IMAGES_DIR = "/mnt/Atlante/SEG_YOLO_uccelli/images"
OUTPUT_LABELS_DIR = "/mnt/Atlante/SEG_YOLO_uccelli/labels"

IMG_W = 3840
IMG_H = 2400
CLASS_ID = 0

CAM_COUNTORNS = {
    1: os.path.join(BASE_SEG_DIR, "cam1_countorns.dat"),
    2: os.path.join(BASE_SEG_DIR, "cam2_countorns.dat"),
    3: os.path.join(BASE_SEG_DIR, "cam3_countorns.dat"),
}
SEGMENTATION_DAT = {
    1: os.path.join(BASE_SEG_DIR, "Cam1", "segmentation.dat"),
    2: os.path.join(BASE_SEG_DIR, "Cam2", "segmentation.dat"),
    3: os.path.join(BASE_SEG_DIR, "Cam3", "segmentation.dat"),
}

# ---------- optional libs ----------
USE_ALPHASHAPE = False
USE_SHAPELY = False
try:
    import alphashape  # type: ignore
    from shapely.geometry import Polygon, MultiPoint  # type: ignore
    USE_ALPHASHAPE = True
    USE_SHAPELY = True
    # successful import
except Exception:
    try:
        from shapely.geometry import MultiPoint, Polygon  # type: ignore
        USE_SHAPELY = True
    except Exception:
        USE_SHAPELY = False

# ---------- utilities ----------
def ensure_dirs():
    os.makedirs(OUTPUT_LABELS_DIR, exist_ok=True)

def parse_countorns_all(cam_countorns_map):
    mapping = {}
    for cam, path in cam_countorns_map.items():
        if not os.path.exists(path):
            print(f"⚠️ countorns missing: {path}")
            continue
        with open(path, "r") as f:
            for raw in f:
                line = raw.strip()
                if not line:
                    continue
                parts = line.split()
                if len(parts) < 6:
                    continue
                try:
                    cam_in_file = int(parts[0])
                    contour_id = int(parts[2])
                except Exception:
                    continue
                coords = parts[4:]
                pts = []
                for i in range(0, len(coords), 2):
                    try:
                        x = float(coords[i]); y = float(coords[i+1])
                        pts.append((x,y))
                    except Exception:
                        pass
                mapping[(cam_in_file, contour_id)] = pts
    return mapping

def monotone_chain_convex_hull(points):
    pts = sorted(set(points))
    if len(pts) <= 2:
        return pts
    def cross(o,a,b): return (a[0]-o[0])*(b[1]-o[1]) - (a[1]-o[1])*(b[0]-o[0])
    lower=[]
    for p in pts:
        while len(lower)>=2 and cross(lower[-2], lower[-1], p) <= 0:
            lower.pop()
        lower.append(p)
    upper=[]
    for p in reversed(pts):
        while len(upper)>=2 and cross(upper[-2], upper[-1], p) <= 0:
            upper.pop()
        upper.append(p)
    hull = lower[:-1] + upper[:-1]
    return hull

def build_polygon_from_points(pts):
    if not pts or len(pts) < 3:
        return None
    # try alphashape
    if USE_ALPHASHAPE:
        try:
            try:
                alpha = alphashape.optimizealpha(pts)
            except Exception:
                alpha = None
            if alpha is None or alpha <= 0:
                xs = [p[0] for p in pts]; ys = [p[1] for p in pts]
                bbox_diag = math.hypot(max(xs)-min(xs), max(ys)-min(ys))
                alpha = max(1.0, bbox_diag * 0.02)
            shape = alphashape.alphashape(pts, alpha)
            if shape is None:
                raise RuntimeError("alpha->None")
            if shape.geom_type == "MultiPolygon":
                shape = max(shape, key=lambda p: p.area)
            if shape.geom_type != "Polygon":
                raise RuntimeError("alpha->not polygon")
            exterior = list(shape.exterior.coords)[:-1]
            if len(exterior) >= 3:
                return [(float(x), float(y)) for x,y in exterior]
        except Exception:
            # fallback
            pass
    if USE_SHAPELY:
        try:
            mp = MultiPoint(pts)
            hull = mp.convex_hull
            if hull.geom_type == "Polygon":
                exterior = list(hull.exterior.coords)[:-1]
                if len(exterior) >= 3:
                    return [(float(x), float(y)) for x,y in exterior]
        except Exception:
            pass
    return monotone_chain_convex_hull(pts)

def normalize_and_flatten(polygon):
    flattened=[]
    for x,y in polygon:
        xn = max(0.0, min(1.0, float(x)/IMG_W))
        yn = max(0.0, min(1.0, float(y)/IMG_H))
        flattened.append(f"{xn:.6f} {yn:.6f}")
    return flattened

def safe_write(label_path, content):
    # write to tmp then move
    dirn = os.path.dirname(label_path)
    fd, tmp = tempfile.mkstemp(dir=dirn, prefix=".tmp_label_")
    try:
        with os.fdopen(fd, "w") as f:
            f.write(content)
            f.flush()
            os.fsync(f.fileno())
        # atomic move
        shutil.move(tmp, label_path)
    except Exception:
        try:
            if os.path.exists(tmp):
                os.remove(tmp)
        except Exception:
            pass
        raise

def print_progress_line(c1, c2, c3, total1, total2, total3):
    # minimal in-place 3-cam status
    line = f"CAM1: [{c1}/{total1}]  CAM2: [{c2}/{total2}]  CAM3: [{c3}/{total3}]"
    print("\r" + line.ljust(120), end="", flush=True)

# ---------- main generation ----------
def generate_labels(dry_run=True, test_n=10, progress_pct=1):
    ensure_dirs()
    contours_map = parse_countorns_all(CAM_COUNTORNS)
    if not contours_map:
        print("❌ No contours parsed.")
        return

    # precompute image frame lists per cam
    image_frames = {}
    for cam in (1,2,3):
        prefix = f"CAM{cam}_"
        imgs = [n for n in os.listdir(IMAGES_DIR) if n.startswith(prefix) and n.lower().endswith(".jpg")]
        frames = sorted({int(n.split("_")[1].split(".")[0]) for n in imgs})
        image_frames[cam] = frames

    total1 = len(image_frames[1]); total2 = len(image_frames[2]); total3 = len(image_frames[3])
    if total1==0 and total2==0 and total3==0:
        print("⚠️ No images found in IMAGES_DIR.")
        return

    # parse segmentation.dat per cam
    all_per_frame = {1: defaultdict(list), 2: defaultdict(list), 3: defaultdict(list)}
    for cam in (1,2,3):
        seg_path = SEGMENTATION_DAT[cam]
        if not os.path.exists(seg_path):
            print(f"⚠️ segmentation.dat missing for CAM{cam}: {seg_path}; skipping cam.")
            continue
        with open(seg_path,"r") as f:
            for raw in f:
                line = raw.strip()
                if not line:
                    continue
                parts = line.split()
                if len(parts) < 6:
                    continue
                try:
                    cam_id = int(parts[0])
                    frame_id = int(parts[1])
                    contour_id = int(parts[2])
                except Exception:
                    continue
                pts = contours_map.get((cam_id, contour_id))
                if not pts:
                    # warn once per missing contour id (light)
                    # print(f"⚠️ CAM{cam_id} frame {frame_id:06d} contour {contour_id} not found")
                    continue
                poly = build_polygon_from_points(pts)
                if poly and len(poly) >= 3:
                    all_per_frame[cam_id][frame_id].append(poly)

    # now iterate per cam and per image, create files
    c1=c2=c3=0
    files_created = 0
    non_empty = 0
    missing_images = []
    # set progress step: every ~1%
    step1 = max(1, int(total1 * progress_pct / 100)) if total1>0 else 1
    step2 = max(1, int(total2 * progress_pct / 100)) if total2>0 else 1
    step3 = max(1, int(total3 * progress_pct / 100)) if total3>0 else 1

    # helper to process one cam
    def process_cam(cam, frames, step, counter_ref):

        nonlocal files_created, non_empty
        cur_count = 0
        if cam == 3:
        # riparti dal frame 366 (incluso)
            frames = [f for f in frames if f >= 366]
        
        for idx, frame_id in enumerate(frames):
            # skip if image missing
            img_name = f"CAM{cam}_{frame_id:06d}.jpg"
            img_path = os.path.join(IMAGES_DIR, img_name)
            if not os.path.exists(img_path):
                missing_images.append(img_name)
                # do not create label if image missing (you confirmed)
                cur_count += 1
                if cur_count % step == 0:
                    print_progress_line(c1 if cam!=1 else cur_count, c2 if cam!=2 else cur_count, c3 if cam!=3 else cur_count, total1, total2, total3)
                continue

            label_name = f"CAM{cam}_{frame_id:06d}.txt"
            label_path = os.path.join(OUTPUT_LABELS_DIR, label_name)

            # build content
            polys = all_per_frame[cam].get(frame_id, [])
            content_lines = []
            for poly in polys:
                flat = normalize_and_flatten(poly)
                if len(flat) < 3: continue
                content_lines.append(f"{CLASS_ID} " + " ".join(flat))
            content = "\n".join(content_lines) + ("\n" if content_lines else "")

            # overwrite always (A)
            try:
                safe_write(label_path, content)
            except Exception as e:
                print(f"\n❌ I/O error while writing {label_path}: {e}")
                raise

            files_created += 1
            if content_lines:
                non_empty += 1

            cur_count += 1
            # update global counters for printing
            if cam == 1:
                nonlocal_vars_update(1, cur_count)
            elif cam == 2:
                nonlocal_vars_update(2, cur_count)
            else:
                nonlocal_vars_update(3, cur_count)

            # remove empty if configured A (we remove empty after write)
            if not content_lines:
                try:
                    # remove the empty file
                    if os.path.exists(label_path) and os.path.getsize(label_path) == 0:
                        os.remove(label_path)
                except Exception:
                    pass

            # print progress at steps
            if cur_count % step == 0:
                print_progress_line(c1, c2, c3, total1, total2, total3)

        # final print for cam done
        if cam == 1:
            nonlocal_vars_update(1, cur_count)
        elif cam == 2:
            nonlocal_vars_update(2, cur_count)
        else:
            nonlocal_vars_update(3, cur_count)
        print_progress_line(c1, c2, c3, total1, total2, total3)
        print("")  # newline to end in-place line

    # allow nested updates through closure: update c1,c2,c3
    def nonlocal_vars_update(cam_id, value):
        nonlocal c1, c2, c3
        if cam_id == 1:
            c1 = value
        elif cam_id == 2:
            c2 = value
        elif cam_id == 3:
            c3 = value

    # process cams sequentially (CAM1 then CAM2 then CAM3)
    try:
        process_cam(1, image_frames[1], step1, c1)
        process_cam(2, image_frames[2], step2, c2)
        process_cam(3, image_frames[3], step3, c3)
    except Exception as e:
        print(f"\n❌ Fatal error during processing: {e}")
        return

    # final consistency check
    total_images = len(image_frames[1]) + len(image_frames[2]) + len(image_frames[3])
    total_labels_present = len([n for n in os.listdir(OUTPUT_LABELS_DIR) if n.endswith(".txt")])
    print("\n=== FINAL REPORT ===")
    print(f"Images total (all cams): {total_images}")
    print(f"Label files present: {total_labels_present}")
    print(f"Label files written (non-empty): {non_empty}")
    if missing_images:
        print(f"Missing images skipped: {len(set(missing_images))}")
        for im in sorted(set(missing_images))[:50]:
            print(" -", im)
    print("✅ Done. Labels in:", OUTPUT_LABELS_DIR)

# ---------- CLI ----------
IMG_W = 3840
IMG_H = 2400
def cli():
    global IMG_W, IMG_H
    p = argparse.ArgumentParser(description="Generate YOLO segmentation labels (final).")
    p.add_argument("--dry-run", action="store_true", default=True, help="Test run (default True).")
    p.add_argument("--no-dry-run", dest="dry_run", action="store_false", help="Process all images.")
    p.add_argument("--test-n", type=int, default=10, help="Frames per cam in dry-run.")
    p.add_argument("--img-w", type=int, default=IMG_W)
    p.add_argument("--img-h", type=int, default=IMG_H)
    args = p.parse_args()

    
    IMG_W = args.img_w
    IMG_H = args.img_h

    generate_labels(dry_run=args.dry_run, test_n=args.test_n, progress_pct=1)

if __name__ == "__main__":
    cli()
