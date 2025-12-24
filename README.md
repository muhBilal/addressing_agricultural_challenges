# addressing_agricultural_challenges

## Reference :

Image preprocess : S. Malathy et al. [13] introduced an image-processing approach
for detecting fruit disease. They worked with an online dataset and
mentioned an image restoration technique used to minimize image
noise in this image preprocessing.

## Annotated Files : Bounding Box with Yolo

## Requirements 
- python-opencv
- pandas
- tqdm 

# Dragon Fruit Disease Recognition — dengan pendekatan paper

Panduan singkat alur/prosedur project dari **awal (dataset)** sampai **output akhir (evaluasi model)**, sesuai pendekatan paper.

---

## 0) Struktur Data (Input)

Siapkan Dataset ini : 

- `data/Converted Images/`  
  Berisi citra hasil konversi (warna) yang akan diproses (per kelas: Healthy, Anthracnose, dll).

- `data/Annotated Files/`  
  Berisi:
  - file gambar anotasi (ukuran bisa seperti original)
  - file `.txt` **format YOLO normalized (0–1)** → berisi bounding box area lesi/penyakit (dipakai untuk guided segmentation)

---

## 1) Preprocessing

### Notebook: `preprocess.ipynb`

**Tujuan:** Menyamakan preprocessing dengan paper agar citra siap untuk segmentasi & ekstraksi fitur.

**Langkah utama (sesuai paper):**
1. Baca citra dari `data/Converted Images/`
2. Resize ke **300×300** (Bilinear / `cv2.INTER_LINEAR`)
3. Noise reduction: **Gaussian Blur**
4. Enhancement: **Gamma Correction**
5. Enhancement: **Histogram Equalization**

**Output:**
- Folder gambar hasil preprocess:  
  `outputs/preprocessed_index/<ClassName>/*`
- File index (penghubung untuk notebook lain):  
  `outputs/preprocessed_index.csv`

> Semua notebook setelah ini menggunakan `outputs/preprocessed_index.csv` sebagai sumber pathnya.

---

## 2) (Opsional) Test Visual Guided Segmentation

### Notebook: `segmentation_test.ipynb`

**Tujuan:** Validasi visual bahwa ROI mask hasil segmentasi sudah “nempel” ke area lesi (objek).

**Langkah utama:**
1. Load `outputs/preprocessed_index.csv`
2. Ambil `orig_path` (warna) → resize 300×300
3. Ambil `.txt` YOLO (normalized) → buat `lesion_mask`
4. Jalankan **Guided KMeans (LAB)**:
   - segmentasi KMeans (k=3)
   - pilih cluster ROI yang overlap paling besar dengan `lesion_mask`
   - refine mask (morphology + largest component)
5. Simpan contoh hasil overlay ROI

**Output:**
- Sample overlay hasil segmentasi:  
  `outputs/samples/segmentation_guided/*.png`

> Notebook ini opsional, tetapi sangat membantu mengecek apakah segmentasi sudah oke sebelum ekstraksi fitur massal.

---

## 3) Feature Extraction (13 fitur)

### Notebook: `features_extraction.ipynb`

**Tujuan:** Menghasilkan dataset fitur numerik untuk feature selection & klasifikasi.

**Input:**
- `outputs/preprocessed_index.csv`
- `data/Annotated Files/*.txt` (YOLO normalized)

**Langkah utama:**
Untuk setiap citra:
1. Baca `orig_path` (warna) → resize 300×300 → untuk **KMeans-LAB guided segmentation**
2. Baca `prep_path` (grayscale hasil preprocessing 300×300) → untuk **ekstraksi fitur**
3. Jika `.txt` tersedia → buat ROI mask guided  
   Jika tidak (mis. Healthy) → fallback ROI = full image
4. Ekstrak **13 fitur** (statistik + GLCM), contoh:
   - Statistik intensitas: Mean, Variance, Std, Skewness, Kurtosis, RMS, Entropy, Smoothness
   - GLCM: Contrast, Correlation, Energy, Homogeneity + (IDM)

**Output:**
- Dataset fitur final:  
  `outputs/extracted_features.csv`

---

## 4) Feature Selection — ANOVA

### Notebook: `anova_feats_selection.ipynb` (paper-style)

**Tujuan:** Ranking fitur pakai ANOVA F-test, lalu buat dataset Top-K.

**Input:**
- `outputs/extracted_features.csv`

**Output:**
- Ranking fitur ANOVA:  
  `outputs/rankings/anova_rank.csv`
- Plot ranking (bar):  
  `outputs/rankings/anova_rank_top.png`
- Dataset Top-K ANOVA (dipakai modeling):  
  - `outputs/datasets/data5A.csv`
  - `outputs/datasets/data7A.csv`
  - `outputs/datasets/data9A.csv`
  - `outputs/datasets/data10A.csv`

---

## 5) Feature Selection — LASSO (paper/github style)

### Notebook: `lasso_feats_selection.ipynb` (paper-style)

**Tujuan:** Ranking fitur pakai LASSO seperti repo penulis (MinMaxScaler + Lasso).  
Tambahan: Mutual Information untuk pembanding (sesuai repo).

**Input:**
- `outputs/extracted_features.csv`

**Output:**
- Ranking fitur LASSO:  
  `outputs/rankings/lasso_rank.csv`
- Ranking Mutual Info (opsional):  
  `outputs/rankings/mutual_info_rank.csv`
- Dataset Top-K LASSO (dipakai modeling):  
  - `outputs/datasets/data5L.csv`
  - `outputs/datasets/data7L.csv`
  - `outputs/datasets/data9L.csv`
  - `outputs/datasets/data10L.csv`

---

## 6) Modeling & Evaluation (ML Classifiers)

### Notebook: `classification_model.ipynb`

**Tujuan:** Melatih & mengevaluasi model ML berdasarkan dataset Top-K dari ANOVA dan LASSO.

**Model yang diuji:**
- Logistic Regression
- KNN
- Decision Tree
- AdaBoost
- Random Forest
- SVM (RBF)

**Input:**
- `outputs/datasets/data5A.csv`, `data7A.csv`, `data9A.csv`, `data10A.csv`
- `outputs/datasets/data5L.csv`, `data7L.csv`, `data9L.csv`, `data10L.csv`

**Output:**
- Plot perbandingan performa:  
  - `outputs/plots/anova_f1_macro_vs_k.png`
  - `outputs/plots/lasso_f1_macro_vs_k.png`
  - `outputs/plots/best_f1_macro_anova_vs_lasso.png`
- Confusion matrix best setting (contoh):  
  - `outputs/plots/cm_best_anova_top10.png`
  - `outputs/plots/cm_best_lasso_top10.png`

> Notebook ini menjadi output akhir penelitian: perbandingan model + perbandingan feature selection ANOVA vs LASSO untuk Top-K.

---

## Urutan Run yang Disarankan

1. `preprocess.ipynb`
2. `segmentation_test.ipynb` (opsional)
3. `features_extraction.ipynb` 
4. `anova_feats_selection.ipynb`
5. `lasso_feats_selection.ipynb`
6. `classification_model.ipynb` 

---

## Ringkas Output Akhir yang Dicari

- **Dataset fitur:** `outputs/extracted_features.csv`
- **Ranking fitur:** `outputs/rankings/anova_rank.csv` & `outputs/rankings/lasso_rank.csv`
- **Dataset Top-K:** `outputs/datasets/data* A.csv` & `data* L.csv`
- **Grafik & evaluasi model:** `outputs/plots/*.png`

---


