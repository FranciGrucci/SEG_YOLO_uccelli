#!/usr/bin/env python3
import math

FILE = "/home/artswarms/Desktop/Segmentation_FRA/20250327_ACQ002/SEGMENTATION/cam3_countorns.dat"

with open(FILE, "r") as f:
    line = next(l for l in f.readlines() if l.strip())

parts = line.split()
coords = list(map(float, parts[4:]))

pts = [(coords[i], coords[i+1]) for i in range(0, len(coords), 2)]

def dist(p,q):
    return math.hypot(p[0]-q[0], p[1]-q[1])

# distanza media consecutiva (p_i, p_{i+1})
cons = [dist(pts[i], pts[(i+1) % len(pts)]) for i in range(len(pts))]
# distanza media “random” (salti di 5 posizioni)
rand = [dist(pts[i], pts[(i+5) % len(pts)]) for i in range(len(pts))]

avg_cons = sum(cons)/len(cons)
avg_rand = sum(rand)/len(rand)

print("Avg consecutive:", avg_cons)
print("Avg random    :", avg_rand)

if avg_cons < avg_rand * 0.25:
    print("\n✅ Punti ORDINATI → Il file contiene poligoni validi (Formato A)\n")
else:
    print("\n⚠️  Punti NON ordinati → Serve riordinamento (Formato B)\n")
