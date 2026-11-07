import os
from collections import Counter
from PIL import Image
import matplotlib.pyplot as plt

ROOT = "data/raw"
CLASSES = ["glioma", "meningioma", "notumor", "pituitary"]

counts = {}
for split in ["Training", "Testing"]:
    counts[split] = {}
    for c in CLASSES:
        folder = os.path.join(ROOT, split, c)
        counts[split][c] = len([f for f in os.listdir(folder)
                                if f.lower().endswith((".jpg", ".jpeg", ".png"))])
    print(split, counts[split], "total =", sum(counts[split].values()))

# dimensions + modes (echantillon de 300 images)
sizes, modes = Counter(), Counter()
for c in CLASSES:
    folder = os.path.join(ROOT, "Training", c)
    for f in os.listdir(folder)[:75]:
        with Image.open(os.path.join(folder, f)) as im:
            sizes[im.size] += 1
            modes[im.mode] += 1
print("Dimensions les plus frequentes:", sizes.most_common(5))
print("Modes:", dict(modes))

# graphique
fig, axes = plt.subplots(1, 2, figsize=(10, 4))
for ax, split in zip(axes, ["Training", "Testing"]):
    ax.bar(counts[split].keys(), counts[split].values())
    ax.set_title(split)
    ax.tick_params(axis="x", rotation=30)
plt.tight_layout()
os.makedirs("report", exist_ok=True)
plt.savefig("report/class_distribution.png", dpi=150)
print("Graphique sauvegarde: report/class_distribution.png")