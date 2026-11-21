"""
Kontrol:
    Klik kiri x4 : empat sudut slot (searah jarum jam atau sebaliknya)
    Enter        : konfirmasi slot setelah empat titik
    Z            : hapus titik terakhir yang belum dikonfirmasi
    C            : batalkan empat titik yang sedang dibuat
    U            : hapus slot terakhir yang sudah dikonfirmasi
    S            : simpan JSON, overview, dan gambar crop untuk inspeksi
    Q            : keluar (perubahan yang belum disimpan akan hilang)
"""

import argparse
import json
from pathlib import Path

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_VIDEO = ROOT / "data" / "raw" / "giri-loka" / "parkir_hp.mp4"
CONFIG_FILE = ROOT / "data" / "annotations" / "giri_loka_rois.json"
OVERVIEW_FILE = ROOT / "outputs" / "roi_overview.jpg"
CROP_DIR = ROOT / "outputs" / "roi_crops"
WINDOW = "Anotasi ROI - Giri Loka"


def video_identifier(path: Path) -> str:
    resolved = path.resolve()
    try:
        return resolved.relative_to(ROOT.resolve()).as_posix()
    except ValueError:
        return resolved.as_posix()


def is_valid_quad(points: list[list[int]]) -> bool:
    if len(points) != 4:
        return False
    contour = np.asarray(points, dtype=np.int32).reshape(-1, 1, 2)
    return cv2.contourArea(contour) >= 100 and bool(cv2.isContourConvex(contour))


def save_annotations(frame, slots, video_path, frame_index, second):
    if not slots:
        print("Belum ada slot untuk disimpan.")
        return False

    for folder in (CONFIG_FILE.parent, OVERVIEW_FILE.parent, CROP_DIR):
        folder.mkdir(parents=True, exist_ok=True)

    height, width = frame.shape[:2]
    config = {
        "video": video_identifier(video_path),
        "frame_index": frame_index,
        "requested_second": second,
        "frame_width": width,
        "frame_height": height,
        "slots": slots,
    }
    CONFIG_FILE.write_text(
        json.dumps(config, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )

    overview = frame.copy()
    # Kuning = penanda lokasi ROI, BUKAN label terisi/kosong.
    for slot in slots:
        poly = np.asarray(slot["points"], dtype=np.int32)
        cv2.polylines(overview, [poly], True, (0, 220, 255), 3)
        center = poly.mean(axis=0).astype(int)
        cv2.putText(
            overview, slot["id"], tuple(center),
            cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 220, 255), 2, cv2.LINE_AA,
        )
    if not cv2.imwrite(str(OVERVIEW_FILE), overview):
        raise OSError(f"Gagal menyimpan {OVERVIEW_FILE}")

    # Hasil crop untuk INSPEKSI saja; belum menjadi input HOG-SVM.
    for old in CROP_DIR.glob("slot_*.jpg"):
        old.unlink()
    for slot in slots:
        poly = np.asarray(slot["points"], dtype=np.int32)
        mask = np.zeros((height, width), dtype=np.uint8)
        cv2.fillPoly(mask, [poly], 255)
        x, y, w, h = cv2.boundingRect(poly)
        crop = cv2.bitwise_and(frame, frame, mask=mask)[y:y + h, x:x + w]
        output_path = CROP_DIR / f"{slot['id']}.jpg"
        if not cv2.imwrite(str(output_path), crop):
            raise OSError(f"Gagal menyimpan {output_path}")

    print(f"Tersimpan: {CONFIG_FILE}")
    print(f"Gambar ROI: {OVERVIEW_FILE}")
    print(f"Crop inspeksi: {CROP_DIR} ({len(slots)} gambar)")
    return True


def main():
    parser = argparse.ArgumentParser(description="Anotasi ROI poligon slot parkir")
    parser.add_argument("--video", type=Path, default=DEFAULT_VIDEO)
    parser.add_argument("--second", type=float, default=0.0,
                        help="Perkiraan detik frame yang dipakai untuk anotasi")
    parser.add_argument("--reset", action="store_true",
                        help="Mulai anotasi baru, abaikan ROI tersimpan sebelumnya")
    args = parser.parse_args()

    video_path = args.video.resolve()
    if not video_path.is_file():
        raise SystemExit(f"Video tidak ditemukan: {video_path}")
    if args.second < 0:
        raise SystemExit("--second harus >= 0")

    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise SystemExit(f"Video tidak bisa dibuka: {video_path}")
    cap.set(cv2.CAP_PROP_POS_MSEC, args.second * 1000)
    ok, frame = cap.read()
    if not ok:
        cap.release()
        raise SystemExit("Gagal membaca frame. Coba --second 0")
    frame_index = max(0, int(cap.get(cv2.CAP_PROP_POS_FRAMES)) - 1)
    cap.release()

    frame_h, frame_w = frame.shape[:2]
    display_scale = min(1.0, 1100 / frame_w, 640 / frame_h)
    view_w = max(1, round(frame_w * display_scale))
    view_h = max(1, round(frame_h * display_scale))
    view = cv2.resize(frame, (view_w, view_h), interpolation=cv2.INTER_AREA)

    slots = []
    current_points = []
    dirty = False

    if CONFIG_FILE.exists() and not args.reset:
        previous = json.loads(CONFIG_FILE.read_text(encoding="utf-8"))
        same_video = previous.get("video") == video_identifier(video_path)
        same_size = (previous.get("frame_width"), previous.get("frame_height")) == (frame_w, frame_h)
        if not (same_video and same_size):
            raise SystemExit(
                "Konfigurasi lama berasal dari video/resolusi lain. "
                "Jika ingin menggantinya, jalankan ulang dengan --reset."
            )
        slots = previous.get("slots", [])
        print(f"Memuat {len(slots)} slot dari {CONFIG_FILE}")

    def on_mouse(event, x, y, flags, userdata):
        nonlocal dirty
        if event != cv2.EVENT_LBUTTONDOWN:
            return
        if not (0 <= x < view_w and 0 <= y < view_h):
            return
        if len(current_points) >= 4:
            print("Empat titik sudah dipilih. Tekan Enter, Z, atau C.")
            return
        orig_x = min(frame_w - 1, round(x * frame_w / view_w))
        orig_y = min(frame_h - 1, round(y * frame_h / view_h))
        current_points.append([orig_x, orig_y])
        dirty = True

    def to_view_points(points):
        arr = np.asarray(points, dtype=np.float32).reshape(-1, 2)
        arr[:, 0] *= view_w / frame_w
        arr[:, 1] *= view_h / frame_h
        return np.rint(arr).astype(np.int32)

    cv2.namedWindow(WINDOW, cv2.WINDOW_AUTOSIZE)
    cv2.setMouseCallback(WINDOW, on_mouse)
    print("Klik empat sudut setiap slot secara berurutan (searah jarum jam / sebaliknya).")
    print("Enter: konfirmasi | Z: undo titik | C: batal slot | U: undo slot")
    print("S: simpan | Q: keluar")

    while True:
        screen = view.copy()
        for slot in slots:
            points = to_view_points(slot["points"])
            cv2.polylines(screen, [points], True, (0, 220, 255), 2)
            center = points.mean(axis=0).astype(int)
            cv2.putText(screen, slot["id"], tuple(center),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 220, 255),
                        1, cv2.LINE_AA)

        if current_points:
            points = to_view_points(current_points)
            for x, y in points:
                cv2.circle(screen, (int(x), int(y)), 4, (255, 180, 0), -1)
            if len(points) >= 2:
                cv2.polylines(screen, [points], False, (255, 180, 0), 2)
            if len(points) == 4:
                cv2.line(screen, tuple(points[-1]), tuple(points[0]),
                         (255, 180, 0), 1)

        panel = np.zeros((72, view_w, 3), dtype=np.uint8)
        cv2.putText(panel, f"Slot tersimpan: {len(slots)} | Titik aktif: {len(current_points)}/4",
                    (10, 24), cv2.FONT_HERSHEY_SIMPLEX, 0.56,
                    (255, 255, 255), 1, cv2.LINE_AA)
        cv2.putText(panel, "Klik x4 + Enter | Z: titik | C: batal | U: slot | S: simpan | Q: keluar",
                    (10, 52), cv2.FONT_HERSHEY_SIMPLEX, 0.46,
                    (215, 215, 215), 1, cv2.LINE_AA)
        cv2.imshow(WINDOW, np.vstack([screen, panel]))
        key = cv2.waitKey(30) & 0xFF

        if key in (13, 10):  # Enter
            if len(current_points) != 4:
                print("Pilih tepat empat titik dulu, lalu tekan Enter.")
            elif not is_valid_quad(current_points):
                print("Poligon tidak valid. Pastikan tidak bersilangan dan luasnya cukup.")
            else:
                slot_id = f"slot_{len(slots) + 1:03d}"
                slots.append({"id": slot_id,
                              "points": [p.copy() for p in current_points]})
                current_points.clear()
                dirty = True
                print(f"Ditambahkan: {slot_id}")
        elif key == ord("z") and current_points:
            current_points.pop()
            dirty = True
        elif key == ord("c"):
            current_points.clear()
        elif key == ord("u") and slots:
            removed = slots.pop()
            dirty = True
            print(f"Dihapus: {removed['id']}")
        elif key == ord("s"):
            if current_points:
                print("Ada titik yang belum dikonfirmasi; Enter atau C dahulu.")
            elif save_annotations(frame, slots, video_path, frame_index, args.second):
                dirty = False
        elif key == ord("q"):
            if dirty:
                print("Peringatan: perubahan setelah penyimpanan terakhir tidak tersimpan.")
            break

    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
