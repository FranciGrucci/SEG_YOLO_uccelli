#!/usr/bin/env python3
import os
import argparse
import tempfile
import shutil
from collections import defaultdict

# CONFIG
BASE_SEG_DIR = "/home/artswarms/Desktop/Segmentation_FRA/20250327_ACQ002/SEGMENTATION"
IMAGES_DIR = "/mnt/Atlante/SEG_YOLO_uccelli/images"
OUTPUT_LABELS_DIR = "/mnt/Atlante/SEG_YOLO_uccelli/labels"

IMG_W = 3840
IMG_H = 2400
CLASS_ID = 0

CAM_COUNTORNS = {
    3: os.path.join(BASE_SEG_DIR, "cam3_countorns.dat")
}
SEGMENTATION_DAT = {
    3: os.path.join(BASE_SEG_DIR, "Cam3", "segmentation.dat")
}

def ensure_dirs():
    os.makedirs(OUTPUT_LABELS_DIR, exist_ok=True)

def parse_countorns(cam=3):
    mapping = {}
    path = CAM_COUNTORNS[cam]
    if not os.path.exists(path):
        raise FileNotFoundError(f"countorns missing: {path}")
    with open(path, "r") as f:
        for raw in f:
            parts = raw.strip().split()
            if len(parts) < 6:
                continue
            camid = int(parts[0])
            contour = int(parts[2])
            coords = parts[4:]
            pts=[]
            for i in range(0,len(coords),2):
                try:
                    x=float(coords[i]); y=float(coords[i+1])
                    pts.append((x,y))
                except: pass
            mapping[(camid,contour)] = pts
    return mapping

def convex_hull(points):
    pts = sorted(set(points))
    if len(pts) <= 2:
        return pts
    def cross(o,a,b):
        return (a[0]-o[0])*(b[1]-o[1]) - (a[1]-o[1])*(b[0]-o[0])
    lower=[]
    for p in pts:
        while len(lower)>=2 and cross(lower[-2], lower[-1], p) <=0: lower.pop()
        lower.append(p)
    upper=[]
    for p in reversed(pts):
        while len(upper)>=2 and cross(upper[-2], upper[-1], p) <=0: upper.pop()
        upper.append(p)
    return lower[:-1] + upper[:-1]

def safe_write(label_path, content):
    fd, tmp = tempfile.mkstemp(dir=os.path.dirname(label_path), prefix=".tmp_")
    with os.fdopen(fd,"w") as f:
        f.write(content)
        f.flush()
        os.fsync(f.fileno())
    shutil.move(tmp, label_path)

def generate(cam=3, start_frame=366, dry_run=True, max_frames=None):
    ensure_dirs()
    contours = parse_countorns(cam)
    segpath = SEGMENTATION_DAT[cam]

    per_frame = defaultdict(list)
    written = 0
    processed = 0

    with open(segpath,"r") as f:
        for raw in f:
            parts = raw.strip().split()
            if len(parts)<6: 
                continue
            camid = int(parts[0])
            frame = int(parts[1])
            contour = int(parts[2])
            if camid!=cam or frame < start_frame:
                continue

            pts = contours.get((camid,contour))
            if not pts: 
                continue

            hull = convex_hull(pts)
            if len(hull)>=3:
                per_frame[frame].append(hull)

            # quando cambiamo frame → subito scrivi
            if max_frames and len(per_frame)>=max_frames:
                break

    for frame, hulls in sorted(per_frame.items()):
        img = f"CAM{cam}_{frame:06d}.jpg"
        imgp = os.path.join(IMAGES_DIR,img)
        if not os.path.exists(imgp):
            continue

        lines=[]
        for poly in hulls:
            flat = []
            for x,y in poly:
                flat.append(f"{max(0,min(1,x/IMG_W)):.6f} {max(0,min(1,y/IMG_H)):.6f}")
            if len(flat)>=3:
                lines.append(f"{CLASS_ID} " + " ".join(flat))

        if not dry_run and lines:
            out = os.path.join(OUTPUT_LABELS_DIR, f"CAM{cam}_{frame:06d}.txt")
            safe_write(out, "\n".join(lines)+"\n")
            written+=1

        processed+=1
        print(f"\rProcessed CAM{cam} frame {frame} | written: {written}", end="", flush=True)

    print("\nDone.")

if __name__=="__main__":
    a = argparse.ArgumentParser()
    a.add_argument("--start-frame", type=int, default=366)
    a.add_argument("--no-dry-run", action="store_false", dest="dry_run")
    a.add_argument("--max-frames", type=int, default=None)
    args = a.parse_args()

    generate(start_frame=args.start_frame, dry_run=args.dry_run, max_frames=args.max_frames)
