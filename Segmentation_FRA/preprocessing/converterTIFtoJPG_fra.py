import os
from PIL import Image

INPUT_DIR = "/mnt/Atlante/PROGgETTO_MATTEO/Segmentation/20221207_acq004/CAM1"
OUTPUT_DIR = "/home/artswarms/Desktop/Segmentation_FRA/seg_images_jpg"

os.makedirs(OUTPUT_DIR, exist_ok=True)

count = 0

for root, dirs, files in os.walk(INPUT_DIR):
    for filename in files:
        if filename.lower().endswith((".tif", ".tiff")):
            count += 1
            tif_path = os.path.join(root, filename)
            # mantengo nome file originale ma con .jpg
            jpg_filename = os.path.splitext(filename)[0] + ".jpg"
            jpg_path = os.path.join(OUTPUT_DIR, jpg_filename)

            with Image.open(tif_path) as img:
                img = img.convert("RGB")  # YOLO vuole 3 canali
                img.save(jpg_path, "JPEG", quality=95)

if count == 0:
    print("⚠️ Nessun .tif trovato nelle sottocartelle!")
else:
    print(f"✅ Conversione completata! {count} immagini convertite.")
