import pandas as pd
import matplotlib.pyplot as plt
import os

exp_dir = "/mnt/Atlante/SEG_YOLO_uccelli/dataset/uccello_exp3"
csv_file = os.path.join(exp_dir, "results.csv")
plots_dir = os.path.join(exp_dir, "plots_manual")

os.makedirs(plots_dir, exist_ok=True)

df = pd.read_csv(csv_file)

metrics = ["metrics/precision", "metrics/recall", "metrics/mAP50", "metrics/mAP50-95",
           "train/box_loss", "train/cls_loss", "train/dfl_loss"]

for m in metrics:
    if m in df.columns:
        plt.figure()
        plt.plot(df[m])
        plt.title(m)
        plt.xlabel("Epoch")
        plt.ylabel(m)
        plt.savefig(os.path.join(plots_dir, f"{m.replace('/', '_')}.png"))
        plt.close()

print(f"✅ Grafici salvati in: {plots_dir}")
