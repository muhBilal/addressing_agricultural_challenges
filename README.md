# Addressing Agricultural Challenges

## Reference :

**Image Preprocess:** Mengacu pada S. Malathy et al. [13] yang mengajukan pendekatan *image processing* untuk deteksi penyakit buah. Mereka menggunakan dataset online dan menerapkan teknik restorasi citra untuk meminimalisir *noise* pada tahap preprocessing.

## Annotated Files : Bounding Box with YOLO

Dataset anotasi menggunakan format YOLO untuk lokalisasi area penyakit.

## Requirements / Tech Stack

* python-opencv
* Scikit Learn
* pandas
* tqdm

# Dragon Fruit Disease Recognition — Pendekatan Paper

Berikut adalah **flow** pengerjaan proyek dari **raw dataset** hingga **model evaluation**, diadaptasi dari metode paper referensi.

---

## 0) Struktur Data (Input)

Pastikan direktori dataset sudah di-setup seperti ini:

* `data/Converted Images/`
Direktori ini berisi *raw images* yang sudah dikonversi (citra berwarna) yang siap diproses. Struktur folder berdasarkan nama kelas (Healthy, Anthracnose, dsb).
* `data/Annotated Files/`
Direktori ini berisi:
* File gambar anotasi (sebagai referensi visual).
* File `.txt` dengan **format YOLO normalized (0–1)** → berisi koordinat *bounding box* area lesi/penyakit (wajib ada untuk *guided segmentation*).



---

## 1) Preprocessing

### Notebook: `preprocess.ipynb`

**Goal:** Mereplikasi teknik preprocessing sesuai paper agar citra siap untuk tahap segmentasi & ekstraksi fitur.

**Main Steps (sesuai paper):**

1. **Load Image** dari direktori `data/Converted Images/`.
2. **Resizing** ke dimensi **300×300** (menggunakan interpolasi `cv2.INTER_LINEAR`).
3. **Noise Reduction:** Implementasi **Gaussian Blur**.
4. **Enhancement:** Penerapan **Gamma Correction**.
5. **Enhancement:** Penerapan **Histogram Equalization** untuk perbaikan kontras.

**Output:**

* Direktori gambar hasil preprocess:
`outputs/preprocessed_index/<ClassName>/*`
* File index (manifest file untuk mapping path):
`outputs/preprocessed_index.csv`

> **Note:** Semua notebook selanjutnya akan me-load `outputs/preprocessed_index.csv` sebagai referensi path utama.

---

## 2) (Opsional) Test Visual Guided Segmentation

### Notebook: `segmentation_test.ipynb`

**Goal:** *Sanity check* / Validasi visual untuk memastikan ROI mask hasil segmentasi sudah presisi menutupi area lesi (objek).

**Main Steps:**

1. Load `outputs/preprocessed_index.csv`.
2. Ambil `orig_path` (citra RGB) → resize 300×300.
3. Parsing file `.txt` YOLO (normalized) → generate `lesion_mask` (ground truth).
4. Eksekusi **Guided KMeans (LAB Color Space)**:
* Lakukan segmentasi KMeans (k=3).
* Filter cluster ROI yang memiliki *IoU (Intersection over Union)* atau overlap terbesar dengan `lesion_mask`.
* **Refine Mask:** Operasi morfologi + seleksi *largest component*.


5. Simpan sampel hasil overlay ROI untuk inspeksi.

**Output:**

* Sampel overlay hasil segmentasi:
`outputs/samples/segmentation_guided/*.png`

> **Note:** Step ini opsional, tapi *highly recommended* buat debugging logika segmentasi sebelum lanjut ke ekstraksi fitur massal.

---

## 3) Feature Extraction (13 Features)

### Notebook: `features_extraction.ipynb`

**Goal:** Generate dataset berisi fitur numerik (ekstraksi ciri) untuk keperluan seleksi fitur & klasifikasi.

**Input:**

* `outputs/preprocessed_index.csv`
* `data/Annotated Files/*.txt` (YOLO normalized)

**Main Steps:**
Iterasi untuk setiap citra:

1. Load `orig_path` (RGB) → resize 300×300 → input untuk **KMeans-LAB guided segmentation**.
2. Load `prep_path` (Grayscale hasil preprocess) → input untuk kalkulasi nilai piksel (**ekstraksi fitur**).
3. **ROI Logic:**
* Jika file `.txt` ada → generate ROI mask via guided segmentation.
* Jika tidak ada (misal kelas Healthy) → *fallback* ROI = *full image*.


4. Ekstrak **13 Fitur** (Statistik + GLCM), meliputi:
* **Intensity Stats:** Mean, Variance, Std Dev, Skewness, Kurtosis, RMS, Entropy, Smoothness.
* **GLCM (Texture):** Contrast, Correlation, Energy, Homogeneity + (IDM).



**Output:**

* Dataset fitur final (CSV):
`outputs/extracted_features.csv`

---

## 4) Feature Selection — ANOVA

### Notebook: `anova_feats_selection.ipynb` (Paper Approach)

**Goal:** Ranking fitur menggunakan metode statistik ANOVA F-test, lalu generate dataset Top-K fitur terbaik.

**Input:**

* `outputs/extracted_features.csv`

**Output:**

* Ranking fitur berdasarkan skor ANOVA:
`outputs/rankings/anova_rank.csv`
* Visualisasi ranking (Bar Plot):
`outputs/rankings/anova_rank_top.png`
* Dataset Top-K ANOVA (siap training):
* `outputs/datasets/data5A.csv` (Top 5)
* `outputs/datasets/data7A.csv` (Top 7)
* `outputs/datasets/data9A.csv` (Top 9)
* `outputs/datasets/data10A.csv` (Top 10)



---

## 5) Feature Selection — LASSO (Repo/Github Style)

### Notebook: `lasso_feats_selection.ipynb` (Paper Approach)

**Goal:** Ranking fitur menggunakan LASSO (seperti implementasi di repo penulis: MinMaxScaler + Lasso).
*Opsional: Komparasi dengan Mutual Information.*

**Input:**

* `outputs/extracted_features.csv`

**Output:**

* Ranking fitur berdasarkan koefisien LASSO:
`outputs/rankings/lasso_rank.csv`
* Ranking Mutual Info (opsional/benchmark):
`outputs/rankings/mutual_info_rank.csv`
* Dataset Top-K LASSO (siap training):
* `outputs/datasets/data5L.csv`
* `outputs/datasets/data7L.csv`
* `outputs/datasets/data9L.csv`
* `outputs/datasets/data10L.csv`



---

## 6) Modeling & Evaluation (ML Classifiers)

### Notebook: `classification_model.ipynb`

**Goal:** Training & Evaluasi performa berbagai algoritma ML menggunakan dataset Top-K hasil ANOVA dan LASSO.

**Tested Models:**

* Logistic Regression
* KNN (K-Nearest Neighbors)
* Decision Tree
* AdaBoost
* Random Forest
* SVM (RBF Kernel)

**Input:**

* Dataset varian ANOVA: `data5A.csv` s/d `data10A.csv`
* Dataset varian LASSO: `data5L.csv` s/d `data10L.csv`

**Output:**

* Plot Komparasi Performa (Metrics):
* `outputs/plots/anova_f1_macro_vs_k.png`
* `outputs/plots/lasso_f1_macro_vs_k.png`
* `outputs/plots/best_f1_macro_anova_vs_lasso.png`


* Confusion Matrix (Best Model):
* `outputs/plots/cm_best_anova_top10.png`
* `outputs/plots/cm_best_lasso_top10.png`



> Notebook ini adalah hasil akhir penelitian: berisi analisis komparasi antar model serta efektivitas Feature Selection (ANOVA vs LASSO).

---

## Rekomendasi Flow Eksekusi

1. `preprocess.ipynb` (Wajib pertama)
2. `segmentation_test.ipynb` (Cek visual ROI)
3. `features_extraction.ipynb` (Generate `extracted_features.csv`)
4. `anova_feats_selection.ipynb`
5. `lasso_feats_selection.ipynb`
6. `classification_model.ipynb` (Final output)

---

## Recap Output Final

Yang harus ada di folder `outputs/` setelah semua *run* selesai:

* **Main Dataset:** `outputs/extracted_features.csv`
* **Feature Rankings:** `outputs/rankings/anova_rank.csv` & `lasso_rank.csv`
* **Training Sets:** `outputs/datasets/data* A.csv` & `data* L.csv`
* **Evaluation Plots:** `outputs/plots/*.png` (Grafik F1-Score & Confusion Matrix)

---
