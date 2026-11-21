
from pathlib import Path
import re
import joblib
import numpy as np
import matplotlib.pyplot as plt

from sklearn.model_selection import GroupShuffleSplit
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import LinearSVC
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    f1_score,
    classification_report,
    confusion_matrix,
    ConfusionMatrixDisplay
)

# ==========================================
# 1. KONFIGURASI DIREKTORI
# ==========================================

ROOT = Path(__file__).resolve().parents[1]

DATA_PATH = ROOT / "data/processed/cnrpark_hog.npz"
MODEL_DIR = ROOT / "models"
OUTPUT_DIR = ROOT / "outputs"

MODEL_DIR.mkdir(parents=True, exist_ok=True)
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# ==========================================
# 2. MEMBACA FITUR HOG
# ==========================================

with np.load(DATA_PATH) as data:
    X = data["X"]
    y = data["y"]
    paths = data["paths"]
    cameras = data["cameras"]

print("=== DATASET HOG ===")
print("Jumlah sampel:", len(X))
print("Jumlah fitur:", X.shape[1])
print("Label unik:", np.unique(y))

# ==========================================
# 3. MEMBUAT KELOMPOK SLOT PARKIR
# ==========================================

groups = []

for path, camera in zip(paths, cameras):
    filename = Path(
        str(path).replace("\\", "/")
    ).stem

    match = re.fullmatch(
        r"\d{8}_\d{4}_(\d+)",
        filename
    )

    if match is None:
        raise ValueError(
            f"Format nama file tidak sesuai: {path}"
        )

    slot_id = match.group(1)
    groups.append(f"{camera}_{slot_id}")

groups = np.asarray(groups)

print("Jumlah kelompok slot:",
      len(np.unique(groups)))

# ==========================================
# 4. PEMBAGIAN DATASET
# ==========================================

# Training 80%, sementara 20%
split1 = GroupShuffleSplit(
    n_splits=1,
    test_size=0.20,
    random_state=42
)

train_idx, temp_idx = next(
    split1.split(X, y, groups=groups)
)

# Membagi bagian sementara menjadi dua
# Validation dan testing
split2 = GroupShuffleSplit(
    n_splits=1,
    test_size=0.50,
    random_state=43
)

val_rel, test_rel = next(
    split2.split(
        X[temp_idx],
        y[temp_idx],
        groups=groups[temp_idx]
    )
)

val_idx = temp_idx[val_rel]
test_idx = temp_idx[test_rel]

# Memastikan kelompok tidak tumpang tindih
assert set(groups[train_idx]).isdisjoint(
    groups[val_idx]
)
assert set(groups[train_idx]).isdisjoint(
    groups[test_idx]
)
assert set(groups[val_idx]).isdisjoint(
    groups[test_idx]
)

print("\n=== PEMBAGIAN DATASET ===")

for name, idx in [
    ("Training", train_idx),
    ("Validation", val_idx),
    ("Testing", test_idx)
]:
    labels, counts = np.unique(
        y[idx], return_counts=True
    )

    if set(labels.tolist()) != {0, 1}:
        raise ValueError(
            f"{name} tidak memiliki kedua kelas"
        )

    print(f"\n{name}: {len(idx)} gambar")
    print("Jumlah kelompok:",
          len(np.unique(groups[idx])))
    print("Distribusi label:",
          dict(zip(labels.tolist(), counts.tolist())))

# Simpan pembagian dataset
np.savez_compressed(
    OUTPUT_DIR / "cnrpark_split.npz",
    train_idx=train_idx,
    val_idx=val_idx,
    test_idx=test_idx
)

# ==========================================
# 5. MEMBANGUN MODEL SVM
# ==========================================

model = Pipeline([
    ("scaler", StandardScaler()),
    ("svm", LinearSVC(
        C=1.0,
        class_weight="balanced",
        dual=False,
        max_iter=10000,
        random_state=42
    ))
])

# ==========================================
# 6. MELATIH MODEL
# ==========================================

print("\n=== PELATIHAN SVM ===")
print("Memulai pelatihan...")

model.fit(
    X[train_idx],
    y[train_idx]
)

print("Pelatihan selesai.")

# ==========================================
# 7. EVALUASI PADA VALIDATION
# ==========================================

y_true = y[val_idx]
y_pred = model.predict(X[val_idx])

accuracy = accuracy_score(y_true, y_pred)
balanced_acc = balanced_accuracy_score(
    y_true, y_pred
)
macro_f1 = f1_score(
    y_true, y_pred, average="macro"
)

report = classification_report(
    y_true,
    y_pred,
    labels=[0, 1],
    target_names=["Kosong", "Terisi"],
    zero_division=0
)

cm = confusion_matrix(
    y_true, y_pred, labels=[0, 1]
)

print("\n=== HASIL VALIDATION ===")
print(f"Accuracy         : {accuracy:.4f}")
print(f"Balanced Accuracy: {balanced_acc:.4f}")
print(f"Macro F1-Score   : {macro_f1:.4f}")

print("\nClassification Report:")
print(report)

print("Confusion Matrix:")
print(cm)

# ==========================================
# 8. MENYIMPAN HASIL EVALUASI
# ==========================================

report_path = OUTPUT_DIR / "validation_report.txt"

with report_path.open("w", encoding="utf-8") as f:
    f.write("HASIL VALIDATION HOG-SVM\n\n")
    f.write(f"Accuracy: {accuracy:.4f}\n")
    f.write(f"Balanced Accuracy: {balanced_acc:.4f}\n")
    f.write(f"Macro F1: {macro_f1:.4f}\n\n")
    f.write(report)
    f.write("\nConfusion Matrix:\n")
    f.write(str(cm))

disp = ConfusionMatrixDisplay(
    confusion_matrix=cm,
    display_labels=["Kosong (0)", "Terisi (1)"]
)

fig, ax = plt.subplots(figsize=(6, 5))

disp.plot(
    ax=ax,
    cmap="Blues",
    colorbar=False
)

plt.title("Confusion Matrix - Validation")
plt.tight_layout()

plt.savefig(
    OUTPUT_DIR / "confusion_matrix_validation.png",
    dpi=150
)

plt.close(fig)

# ==========================================
# 9. MENYIMPAN MODEL
# ==========================================

bundle = {
    "model": model,
    "image_size": (64, 64),
    "hog_config": {
        "orientations": 9,
        "pixels_per_cell": (8, 8),
        "cells_per_block": (2, 2),
        "block_norm": "L2-Hys",
        "transform_sqrt": False
    },
    "labels": {
        0: "Kosong",
        1: "Terisi"
    }
}

model_path = (
    MODEL_DIR / "hog_svm_baseline.joblib"
)

joblib.dump(bundle, model_path)

print("\n=== PENYIMPANAN ===")
print("Model:", model_path)
print("Laporan:", report_path)
print("Confusion matrix berhasil disimpan.")

print("\nTahap 3 - Pelatihan awal selesai.")
print("Data testing belum digunakan.")
