# 🌴 Palm Detection ML Service - RetinaNet

Deteksi pohon kelapa sawit dari citra UAV/drone menggunakan model **RetinaNet** dengan backbone **ResNet50 ImageNet** (menggunakan KerasCV & TensorFlow). Project ini dideploy sebagai layanan REST API berbasis **FastAPI** untuk deteksi real-time.

---

## 📂 Struktur Proyek

```text
model-nyawit/
├── main.py                         # FastAPI ML Service (Inference)
├── pyproject.toml                  # Konfigurasi dependensi Python
├── uv.lock                         # Lockfile dependensi (uv manager)
├── .python-version                 # Versi Python (3.11)
├── .gitignore                      # Git Ignore
├── .dvcignore                      # DVC Ignore
├── README.md                       # Dokumentasi ini
│
├── .dvc/                           # Konfigurasi Data Version Control (DVC)
│   ├── config                      # Konfigurasi remote publik (Google Drive)
│   ├── config.local                # [Gitignored] Kredensial privat Google Drive
│   └── .gitignore                  # Mengabaikan folder cache & config.local
│
├── notebooks/                      # Jupyter notebook untuk training & riset
│   └── model_deteksi_sawit_retina-net.ipynb   # Pipeline training & evaluasi RetinaNet
│
├── working/                        # [Gitignored] Output training & Bobot Model (.h5)
│   └── runs/
│       └── palm_detection/
│           └── retinanet_v2/
│               └── best.weights.h5 # Bobot terbaik model RetinaNet
│
└── working.dvc                     # Pointer DVC melacak direktori working/
```

---

## 🛠️ Cara Instalasi

Ikuti langkah-langkah di bawah ini untuk menyiapkan environment dan dependencies proyek.

### Prasyarat
* **Python 3.11**
* **Package Manager `uv`** (direkomendasikan karena instalasi sangat cepat) atau **`pip`**

### Opsi A: Menggunakan `uv` (Direkomendasikan)
1. **Install uv** (jika belum terpasang):
   ```bash
   curl -LsSf https://astral.sh/uv/install.sh | sh
   ```
2. **Clone Repositori**:
   ```bash
   git clone https://github.com/capstone-nyawit/model-nyawit.git
   cd model-nyawit
   ```
3. **Instal Dependencies & Sinkronisasi Virtual Environment**:
   ```bash
   uv sync
   ```
4. **Aktifkan Virtual Environment**:
   ```bash
   source .venv/bin/activate
   ```

### Opsi B: Menggunakan `pip` standar
1. **Clone Repositori**:
   ```bash
   git clone https://github.com/capstone-nyawit/model-nyawit.git
   cd model-nyawit
   ```
2. **Buat & Aktifkan Virtual Environment**:
   ```bash
   python3 -m venv .venv
   source .venv/bin/activate
   ```
3. **Instal Dependencies**:
   ```bash
   pip install -e .
   ```

---

## 🚀 Cara Menjalankan Sistem

Setelah instalasi selesai, ikuti langkah berikut untuk mengunduh model dan menjalankan layanan API.

### 1. Unduh Bobot Model (DVC Pull)
Bobot model berukuran besar disimpan di remote Google Drive dan dikelola menggunakan DVC. Unduh bobot model ke direktori lokal dengan menjalankan:
```bash
dvc pull
```
*Catatan: Pastikan Anda telah memiliki akses ke remote storage. Jika remote bersifat privat, masukkan kredensial lokal Anda ke `.dvc/config.local` terlebih dahulu.*

### 2. Jalankan API Server (FastAPI)
Jalankan server inferensi menggunakan Python:
```bash
python main.py
```
Server secara default akan berjalan di **`http://localhost:8001`**.

---

## ⚡ API Documentation

Layanan REST API ini digunakan untuk melakukan deteksi objek kelapa sawit secara real-time.

### 1. Health Check
Memeriksa apakah server API berjalan dengan baik.
* **URL**: `/`
* **Method**: `GET`
* **Response (JSON)**:
  ```json
  {
    "status": "ok",
    "message": "Palm Detection ML Service is running"
  }
  ```

### 2. Deteksi Gambar (Inference)
Mendeteksi letak pohon kelapa sawit dari URL gambar yang diberikan.
* **URL**: `/predict`
* **Method**: `POST`
* **Headers**: `Content-Type: application/json`
* **Request Body (JSON)**:
  ```json
  {
    "image_url": "https://res.cloudinary.com/demo/image/upload/v12345678/uav_palm_plantation.jpg"
  }
  ```
* **Response (JSON)**:
  ```json
  {
    "predictions": [
      {
        "box": [0.1523, 0.4412, 0.0821, 0.0894],
        "confidence": 0.8942,
        "class_id": 1
      },
      {
        "box": [0.3415, 0.1205, 0.0754, 0.0762],
        "confidence": 0.9123,
        "class_id": 1
      }
    ],
    "original_shape": [1024, 1024]
  }
  ```

#### Penjelasan Response:
* **`box`**: Koordinat bounding box dalam format `[x, y, w, h]` (koordinat x tengah, y tengah, lebar, tinggi) yang telah dinormalisasi terhadap dimensi gambar (skala 0.0 hingga 1.0).
* **`confidence`**: Tingkat keyakinan model (skala 0.0 sampai 1.0).
* **`class_id`**: Kategori kelapa sawit yang terdeteksi:
  * `0`: **Dead** (Pohon mati)
  * `1`: **Healthy** (Pohon sehat)
  * `2`: **Grass** (Semak/rumput liar)
  * `3`: **Small** (Pohon kecil/belum produktif)
  * `4`: **Yellow** (Pohon menguning/sakit)

---

## 📊 Detail Model & Dataset

* **Model Architecture**: KerasCV RetinaNet
* **Backbone**: ResNet50 (Pretrained on ImageNet)
* **Dataset Format**: COCO JSON (BBox format `[x_min, y_min, width, height]`)
* **Image Input Size**: 800x800 piksel (auto-resized pada API)
