
from collections import Counter
from pathlib import Path
from random import Random
import csv

import cv2
import numpy as np

# Direktori proyek
ROOT = Path(__file__).resolve().parents[1]
DATASET = ROOT / "data" / "raw" / "cnrpark"
OUTPUT = ROOT / "outputs"

# Kelas klasifikasi
LABELS = {
    "free": 0,
    "busy": 1
}

# 1. Membaca daftar gambar
records = []

for file in sorted(DATASET.rglob("*.jpg")):
    class_name = file.parent.name.lower()

    if class_name not in LABELS:
        continue

    camera = file.parent.parent.name
    capture_date = file.stem.split("_")[0]

    records.append((
        file,
        LABELS[class_name],
        camera,
        capture_date
    ))

if not records:
    raise SystemExit(
        "Dataset tidak ditemukan. Periksa data/raw/cnrpark."
    )

# 2. Menyimpan metadata dataset
OUTPUT.mkdir(parents=True, exist_ok=True)

manifest = OUTPUT / "manifest_cnrpark.csv"

with manifest.open("w", newline="", encoding="utf-8") as f:
    writer = csv.writer(f)
    writer.writerow(["path", "label", "camera", "date"])

    for file, label, camera, date in records:
        writer.writerow([
            file.relative_to(ROOT).as_posix(),
            label,
            camera,
            date
        ])

# 3. Menampilkan statistik dataset
print("Total gambar:", len(records))

print("Jumlah tiap kelas:",
      dict(Counter(r[1] for r in records)))

print("Jumlah tiap kamera:",
      dict(Counter(r[2] for r in records)))

print("Jumlah tiap tanggal:",
      dict(Counter(r[3] for r in records)))

# 4. Membuat preview 6 gambar acak
rng = Random(42)
panels = []

for label in [0, 1]:
    candidates = [r for r in records if r[1] == label]

    if len(candidates) < 3:
        raise SystemExit("Sampel tiap kelas kurang dari 3.")

    for file, _, _, _ in rng.sample(candidates, 3):
        image = cv2.imread(str(file))

        if image is None:
            raise SystemExit(f"Gagal membaca: {file}")

        image = cv2.resize(image, (180, 180))

        footer = np.full(
            (30, 180, 3), 255, dtype=np.uint8
        )

        status = "KOSONG" if label == 0 else "TERISI"

        cv2.putText(
            footer, status, (10, 21),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.55, (0, 0, 0), 1, cv2.LINE_AA
        )

        panels.append(np.vstack((image, footer)))

preview = np.vstack((
    np.hstack(panels[:3]),
    np.hstack(panels[3:])
))

preview_path = OUTPUT / "preview_cnrpark.jpg"
cv2.imwrite(str(preview_path), preview)

print("Manifest:", manifest)
print("Preview:", preview_path)
