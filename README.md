# model-nyawit

Deteksi pohon kelapa sawit dari citra UAV/drone menggunakan deep learning object detection. Project ini mencakup dua pendekatan model: **RetinaNet** (one-stage) dan **Faster R-CNN** (two-stage), keduanya dengan backbone ResNet50 ImageNet.

## Struktur Folder

```
model-nyawit/
├── main.py                         # Entry point (stub)
├── pyproject.toml                  # Project config & dependencies
├── uv.lock                        # Lock file (uv)
├── .python-version                 # Python 3.11
├── .gitignore
├── README.md                       # Dokumentasi ini
│
├── datasets/
│   ├── annotations/
│   │   ├── instances_train2017.json   # COCO annotations (train)
│   │   └── instances_val2017.json     # COCO annotations (val)
│   ├── train2017/                     # 1.803 gambar training (1024x1024)
│   ├── val2017/                       # 500 gambar validasi (1024x1024)
│
├── notebooks/                      # Jupyter notebooks (source of truth)
│   ├── model_deteksi_sawit_retina-net.ipynb   # RetinaNet pipeline (TF + KerasCV)
│   ├── model_deteksi_sawit_faster-r-cnn.ipynb # Faster R-CNN pipeline (TF + KerasCV)
│
├── models/                         # (kosong, reserved untuk model defs)
│
└── working/                        # Output & cache (tidak di-commit)
    ├── dataset/
    │   ├── records.json               # Cache parsed COCO records
    │   └── preview/                   # Visualisasi sample dengan bbox
    └── runs/
        └── palm_detection_tf/
            └── retinanet_fallback_exp1/  # Checkpoint & logs training
```

## Dataset

Dataset berisi citra udara (UAV/drone) perkebunan kelapa sawit dengan anotasi bounding box format **COCO JSON**.
Source: [UAV Dataset](https://github.com/rs-dl/MOPAD)

### Sumber Data

| Split | Jumlah Gambar | Jumlah Annotasi | Format |
|-------|--------------|-----------------|--------|
| Train | 1.803        | 262.811         | JPG (1024x1024) |
| Val   | 500          | 32.846          | JPG (1024x1024) |

### Kelas (5 kelas)

| Index | Nama      | Deskripsi                    |
|-------|-----------|------------------------------|
| 0     | Dead      | Pohon mati                   |
| 1     | Healthy   | Pohon sehat                  |
| 2     | Grass     | Semak/rumput                 |
| 3     | Small     | Pohon kecil (belum produktif)|
| 4     | Yellow    | Pohon menguning (kurang sehat)|

### Struktur Anotasi (COCO Format)

```json
{
  "images": [{"id": 0, "file_name": "52000_20000_1529_3318.jpg", "height": 1024, "width": 1024}],
  "annotations": [{"image_id": 0, "category_id": 1, "bbox": [x, y, w, h], "iscrowd": 0}],
  "categories": [{"id": 1, "name": "Dead"}, {"id": 2, "name": "Healthy"}, ...]
}
```

Format bbox: `[x, y, width, height]` dalam piksel (origin: kiri-atas).

### Persiapan Dataset

1. Unduh dataset UAV palm tree detection (format COCO).
2. Letakkan folder dan file sesuai struktur:
   ```
   datasets/
   ├── annotations/
   │   ├── instances_train2017.json
   │   └── instances_val2017.json
   ├── train2017/    ← gambar training
   └── val2017/      ← gambar validasi
   ```
3. Dataset **tidak di-commit** (terdaftar di `.gitignore`).

## Tech Stack

| Komponen        | Versi       | Fungsi                          |
|-----------------|-------------|---------------------------------|
| Python          | 3.11        | Runtime                         |
| TensorFlow      | 2.15.0      | Training engine & tf.data       |
| Keras           | 2.15.0      | High-level API                  |
| KerasCV         | 0.8.2       | Object detection model & layers |
| OpenCV          | latest      | Image I/O & visualization       |
| NumPy           | <2          | Numerik                         |
| scikit-learn    | latest      | Split & metrics                 |
| Matplotlib      | latest      | Plotting                        |
| Seaborn         | latest      | Confusion matrix heatmap        |

## Setup

### Prasyarat

- **Python 3.11** (ditentukan di `.python-version`)
- GPU dengan CUDA support (opsional, sangat direkomendasikan untuk training)
- ~10 GB disk space untuk dataset

### Opsi 1: Menggunakan uv (Direkomendasikan)

[uv](https://github.com/astral-sh/uv) adalah package manager Python yang sangat cepat.

```bash
# Install uv (jika belum)
curl -LsSf https://astral.sh/uv/install.sh | sh

# Clone repo
git clone https://github.com/capstone-nyawit/model-nyawit.git
cd model-nyawit

# Sinkronisasi dependencies (otomatis baca pyproject.toml + uv.lock)
uv sync

# Aktifkan virtual environment
source .venv/bin/activate

# Jalankan
python main.py
```

### Opsi 2: Menggunakan pip

```bash
# Clone repo
git clone https://github.com/capstone-nyawit/model-nyawit.git
cd model-nyawit

# Buat virtual environment
python3.11 -m venv .venv
source .venv/bin/activate

# Install dependencies
pip install -e .

# Atau install langsung dari pyproject.toml
pip install numpy<2 tensorflow==2.15.0 keras==2.15.0 keras-cv==0.8.2 \
    opencv-python matplotlib pandas scikit-learn seaborn

# Jalankan
python main.py
```

### Verifikasi Instalasi

```bash
python -c "import tensorflow as tf; import keras_cv; import cv2; print(f'TF={tf.__version__}'); print(f'KerasCV={keras_cv.__version__}'); print(f'OpenCV={cv2.__version__}'); print(f'GPU={tf.config.list_physical_devices(\"GPU\")}')"
```

## Penggunaan

### Training via Notebook

Dua notebook tersedia di `notebooks/`:

1. **RetinaNet** (`notebooks/model_deteksi_sawit_retina-net.ipynb`):
   - One-stage detector, lebih cepat
   - Backbone: ResNet50 ImageNet
   - IMG_SIZE=800, BATCH_SIZE=2
   - LR: Warmup + CosineDecay
   - Augmentasi: RandomFlip, JitteredResize, RandomBrightness, RandomContrast

2. **Faster R-CNN** (`notebooks/model_deteksi_sawit_faster-r-cnn.ipynb`):
   - Two-stage detector, lebih akurat
   - Backbone: ResNet50 ImageNet
   - IMG_SIZE=640, BATCH_SIZE=2
   - LR: PiecewiseConstantDecay
   - Augmentasi: RandomFlip, JitteredResize

```bash
# Jalankan Jupyter
jupyter notebook notebooks/
```

### Pipeline (per notebook)

```
Setup → Data Loading → Split (train/val/test) → tf.data Pipeline
  → Build Model → Compile → Train → Evaluate (COCO mAP)
  → Confusion Matrix → Inference → Export (SavedModel/TFLite/ONNX)
```

### Output Training

Semua artefak tersimpan di `working/runs/`:

```
working/runs/palm_detection_tf/<experiment_name>/
├── best.weights.h5          # Bobot terbaik
├── training_log.csv         # Log training per epoch
├── coco_metrics.json        # COCO mAP metrics
├── loss_curve.png           # Plot loss
├── confusion_matrix.png     # Confusion matrix
├── logs/                    # TensorBoard logs
└── exported/
    ├── saved_model/         # SavedModel format
    ├── best.tflite          # TFLite format
    └── best.onnx            # ONNX format (opsional)
```

### Inference

```python
from pathlib import Path
# Lihat contoh lengkap di notebook cell "Inference: detect_and_count"
counts = detect_and_count("path/to/image.jpg", model, conf=0.5)
# Output: {0: 5, 1: 12, 3: 2} → {Dead: 5, Healthy: 12, Small: 2}
```

## Catatan

- Dataset (`datasets/`) dan output (`working/`) **tidak di-commit** via `.gitignore`.
- `records.json` di `working/dataset/` adalah cache hasil parse COCO JSON. Hapus file ini untuk rebuild dari awal.
- Training membutuhkan GPU dengan minimal ~8 GB VRAM. Tanpa GPU, training akan sangat lambat.
- Untuk environment Kaggle, path otomatis disesuaikan ke `/kaggle/input/` dan `/kaggle/working/`.
