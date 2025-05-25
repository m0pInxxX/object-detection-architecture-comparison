# Deteksi Objek dengan YOLOv8 pada Dataset Pascal VOC

Proyek ini mengimplementasikan deteksi objek menggunakan model YOLOv8 pada dataset Pascal VOC. Proyek ini mencakup preprocessing data, evaluasi model pre-trained, dan analisis hasil.

## Struktur Proyek
```
.
├── data/
│   └── processed/
│       ├── images/         # Gambar-gambar dataset
│       ├── labels/         # Label dalam format YOLO
│       ├── train/         # Symlink untuk data training
│       ├── val/           # Symlink untuk data validasi
│       └── dataset.yaml   # Konfigurasi dataset
├── process_voc.py         # Script preprocessing dataset
├── evaluate_pretrained.py # Script evaluasi model
├── analisis_deteksi_objek.md  # Dokumentasi analisis
└── README.md             # Dokumentasi proyek
```

## Persyaratan
- Python 3.8+
- PyTorch
- Ultralytics (YOLOv8)
- OpenCV
- PIL (Python Imaging Library)
- Matplotlib
- NumPy

Anda dapat menginstal semua dependensi dengan:
```bash
pip install torch ultralytics opencv-python pillow matplotlib numpy
```

## Penggunaan

### 1. Preprocessing Dataset
Untuk memproses dataset Pascal VOC ke format YOLO:
```bash
python process_voc.py
```
Pastikan untuk mengubah `voc_path` di dalam script sesuai dengan lokasi dataset VOC Anda.

### 2. Evaluasi Model
Untuk menjalankan evaluasi menggunakan model pre-trained:
```bash
python evaluate_pretrained.py
```
Hasil deteksi akan disimpan sebagai gambar dengan suffix `_detected.jpg`.

## Kelas Objek
Dataset mencakup 20 kelas objek:
- aeroplane (0)
- bicycle (1)
- bird (2)
- boat (3)
- bottle (4)
- bus (5)
- car (6)
- cat (7)
- chair (8)
- cow (9)
- diningtable (10)
- dog (11)
- horse (12)
- motorbike (13)
- person (14)
- pottedplant (15)
- sheep (16)
- sofa (17)
- train (18)
- tvmonitor (19)

## Analisis
Untuk melihat hasil analisis lengkap, silakan baca [analisis_deteksi_objek.md](analisis_deteksi_objek.md).

## Catatan
- Model yang digunakan adalah YOLOv8n pre-trained
- Format label yang digunakan adalah format YOLO (normalized coordinates)
- Visualisasi mencakup bounding box dan label dengan confidence score 