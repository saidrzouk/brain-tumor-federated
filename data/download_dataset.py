import argparse
import os
import shutil
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
RAW = os.path.join(HERE, "raw")
SLUG = "masoudnickparvar/brain-tumor-mri-dataset"
CLASSES = ["glioma", "meningioma", "no_tumor", "pituitary"]
# noms possibles du dossier source pour chaque classe
SOURCE_NAMES = {
    "glioma": ["glioma"],
    "meningioma": ["meningioma"],
    "no_tumor": ["notumor", "no_tumor"],
    "pituitary": ["pituitary"],
}
EXTS = (".jpg", ".jpeg", ".png")


def count_images(raw=RAW):
    counts = {}
    for c in CLASSES:
        folder = os.path.join(raw, c)
        counts[c] = len([f for f in os.listdir(folder) if f.lower().endswith(EXTS)]) \
            if os.path.isdir(folder) else 0
    return counts


def is_ready(raw=RAW):
    return all(n > 0 for n in count_images(raw).values())


def find_source_root(path):
    """Cherche le dossier qui contient Training/ et Testing/ (ou directement les classes)."""
    for root, dirs, _ in os.walk(path):
        low = {d.lower(): d for d in dirs}
        if "training" in low and "testing" in low:
            return root
    return path


def organize(src_root, raw=RAW):
    """Copie toutes les images dans raw/<classe>/ (Training et Testing regroupes)."""
    splits = [d for d in os.listdir(src_root)
              if d.lower() in ("training", "testing")]
    if not splits:
        splits = [None]  # les classes sont directement dans src_root
    for c in CLASSES:
        dst = os.path.join(raw, c)
        os.makedirs(dst, exist_ok=True)
        for split in splits:
            base = os.path.join(src_root, split) if split else src_root
            src = next((os.path.join(base, n) for n in SOURCE_NAMES[c]
                        if os.path.isdir(os.path.join(base, n))), None)
            if src is None:
                continue
            prefix = (split.lower() + "_") if split else ""
            for f in sorted(os.listdir(src)):
                if f.lower().endswith(EXTS):
                    shutil.copy2(os.path.join(src, f), os.path.join(dst, prefix + f))


def download(source=None, raw=RAW):
    os.makedirs(raw, exist_ok=True)
    if is_ready(raw):
        print("Dataset deja present:", count_images(raw))
        return
    if source is None:
        try:
            import kagglehub
            source = kagglehub.dataset_download(SLUG)
        except Exception as e:
            print("Echec du telechargement Kaggle:", e)
            print(f"Telechargez https://www.kaggle.com/datasets/{SLUG} a la main,")
            print("extrayez-le, puis lancez: python data/download_dataset.py --source <dossier>")
            sys.exit(1)
    organize(find_source_root(source), raw)
    counts = count_images(raw)
    print("Images par classe:", counts, "| total =", sum(counts.values()))
    if not is_ready(raw):
        print("Structure attendue introuvable.")
        sys.exit(1)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--source", default=None,
                    help="dossier local contenant Training/ et Testing/ (evite le telechargement)")
    download(ap.parse_args().source)