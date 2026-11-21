# Klasifikasi Slot Parkir Menggunakan HOG-SVM

Implementasi metode Histogram of Oriented Gradients (HOG) dan Support Vector Machine (SVM) untuk klasifikasi status keterisian slot parkir mobil di area parkir Giri Loka UPN "Veteran" Jawa Timur.

## Metode

- **HOG:** Ekstraksi fitur citra slot parkir.
- **SVM:** Klasifikasi dua kelas, yaitu kosong (0) dan terisi (1).
- **ROI:** Penentuan wilayah slot parkir menggunakan koordinat poligon.
- **Input:** Dataset publik CNRPark dan video rekaman kamera HP.

## Tahapan Implementasi

- [x] Persiapan dataset CNRPark.
- [x] Prapemrosesan dan ekstraksi fitur HOG.
- [x] Pelatihan awal model Linear SVM.
- [ ] Anotasi ROI pada video Giri Loka.
- [ ] Integrasi klasifikasi pada video.
- [ ] Evaluasi akhir sistem.

## Hasil Baseline

Evaluasi awal menggunakan data validation:

| Metrik            | Nilai  |
| ----------------- | ------ |
| Accuracy          | 85,79% |
| Balanced Accuracy | 86,56% |
| Macro F1-score    | 85,76% |

Data testing belum digunakan untuk evaluasi akhir.

## Teknologi

Python, OpenCV, scikit-image, scikit-learn, NumPy, Matplotlib, Pandas, dan joblib.

## Dataset

Dataset publik CNRPark tersedia di https://cnrpark.it/.

Dataset, model hasil pelatihan, dan file video tidak disertakan dalam repository.
