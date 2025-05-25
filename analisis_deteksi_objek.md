# Analisis Deteksi Objek menggunakan YOLOv8

## Dataset
Dataset yang digunakan adalah Pascal VOC dengan 20 kelas objek:
0. aeroplane
1. bicycle
2. bird
3. boat
4. bottle
5. bus
6. car
7. cat
8. chair
9. cow
10. diningtable
11. dog
12. horse
13. motorbike
14. person
15. pottedplant
16. sheep
17. sofa
18. train
19. tvmonitor

## Preprocessing Data
Data preprocessing dilakukan menggunakan script `process_voc.py` yang melakukan:
1. Konversi format anotasi dari XML (Pascal VOC) ke format YOLO
2. Pembuatan struktur folder yang sesuai
3. Pembagian dataset menjadi train dan validation set
4. Pembuatan file konfigurasi dataset.yaml

## Evaluasi Model
Evaluasi model dilakukan menggunakan script `evaluate_pretrained.py` yang:
1. Menggunakan model YOLOv8n pre-trained
2. Melakukan deteksi objek pada gambar test
3. Memvisualisasikan hasil deteksi dengan bounding box dan label
4. Menyimpan hasil visualisasi

## Hasil dan Analisis
(Bagian ini dapat diisi setelah menjalankan evaluasi pada beberapa gambar test) 