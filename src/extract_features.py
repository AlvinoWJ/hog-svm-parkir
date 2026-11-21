
from pathlib import Path
import csv

import cv2
import numpy as np

from hog_features import preprocess_image, extract_hog


ROOT = Path(__file__).resolve().parents[1]

MANIFEST = ROOT / "outputs" / "manifest_cnrpark.csv"
PROCESSED = ROOT / "data" / "processed"
OUTPUT = ROOT / "outputs"

PROCESSED.mkdir(parents=True, exist_ok=True)
OUTPUT.mkdir(parents=True, exist_ok=True)

if not MANIFEST.exists():
    raise SystemExit(
        "Manifest tidak ditemukan. Jalankan Tahap 1 dahulu."
    )

# 1. Membaca manifest dari Tahap 1
with MANIFEST.open(
    "r", newline="", encoding="utf-8"
) as file:
    records = list(csv.DictReader(file))

print(f"Jumlah data pada manifest: {len(records)}")

# 2. Menyiapkan penyimpanan hasil
X = []
y = []
paths = []
cameras = []
dates = []

examples = {}
failed = 0

# 3. Memproses setiap gambar
for index, record in enumerate(records, start=1):
    image_path = ROOT / record["path"]

    try:
        label = int(record["label"])
    except (ValueError, KeyError):
        failed += 1
        continue

    if label not in (0, 1):
        failed += 1
        continue

    image = cv2.imread(str(image_path))

    if image is None:
        print(f"Gagal membaca: {image_path}")
        failed += 1
        continue

    # Prapemrosesan
    gray = preprocess_image(image)

    # Ekstraksi HOG dan contoh visualisasi
    if label not in examples:
        features, hog_image = extract_hog(
            gray, visualize=True
        )
        examples[label] = (gray, hog_image)
    else:
        features = extract_hog(gray)

    X.append(features)
    y.append(label)
    paths.append(record["path"])
    cameras.append(record["camera"])
    dates.append(record["date"])

    if index % 1000 == 0:
        print(f"Proses: {index}/{len(records)} gambar")

if not X:
    raise SystemExit("Tidak ada gambar yang berhasil diproses.")

# 4. Mengonversi hasil ke NumPy array
X = np.stack(X).astype(np.float32)
y = np.asarray(y, dtype=np.int8)

print("\n=== HASIL EKSTRAKSI ===")
print("Bentuk fitur X:", X.shape)
print("Bentuk label y:", y.shape)
print("Jumlah fitur per gambar:", X.shape[1])
print("Gambar gagal:", failed)
print("Label 0 (kosong):", np.sum(y == 0))
print("Label 1 (terisi):", np.sum(y == 1))

# 5. Menyimpan fitur dan metadata
save_path = PROCESSED / "cnrpark_hog.npz"

np.savez_compressed(
    save_path,
    X=X,
    y=y,
    paths=np.asarray(paths),
    cameras=np.asarray(cameras),
    dates=np.asarray(dates)
)

print("Fitur disimpan:", save_path)

# 6. Membuat visualisasi HOG
panels = []

for label in (0, 1):
    if label not in examples:
        continue

    gray, hog_image = examples[label]

    gray_view = cv2.resize(
        gray, (256, 256),
        interpolation=cv2.INTER_NEAREST
    )

    hog_view = cv2.normalize(
        hog_image,
        None,
        0, 255,
        cv2.NORM_MINMAX
    ).astype(np.uint8)

    hog_view = cv2.resize(
        hog_view, (256, 256),
        interpolation=cv2.INTER_NEAREST
    )

    panel = np.hstack((gray_view, hog_view))
    panel = cv2.cvtColor(
        panel, cv2.COLOR_GRAY2BGR
    )

    status = "KOSONG" if label == 0 else "TERISI"

    cv2.putText(
        panel, f"{status} - CITRA",
        (10, 25),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.6, (0, 255, 0), 2
    )

    cv2.putText(
        panel, "VISUALISASI HOG",
        (270, 25),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.55, (0, 255, 0), 2
    )

    panels.append(panel)

if panels:
    preview = np.vstack(panels)
    preview_path = OUTPUT / "hog_preview.jpg"
    cv2.imwrite(str(preview_path), preview)
    print("Visualisasi disimpan:", preview_path)

print("\nTahap 2 selesai.")
