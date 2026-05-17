# Eye Disease Detection

A web application that classifies retinal diseases from **OCT scans** and **Fundus photographs** using deep learning. Built with TensorFlow/Keras and Streamlit, deployable locally or via Docker.

---

## Detected Conditions

### OCT Scan (8 diseases)

| Class | Description |
| --- | --- |
| AMD | Age-related Macular Degeneration |
| CNV | Choroidal Neovascularization |
| CSR | Central Serous Retinopathy |
| DME | Diabetic Macular Edema |
| DR | Diabetic Retinopathy |
| DRUSEN | Drusen deposits under the retina |
| MH | Macular Hole |
| NORMAL | Healthy retina |

### Fundus Photo (5 diseases)

| Class | Description |
| --- | --- |
| Glaucoma | Optic nerve damage from elevated eye pressure |
| Cataract | Clouding of the eye lens |
| Hypertension | Retinal damage from high blood pressure |
| Myopia | Nearsightedness — light focuses in front of the retina |
| Normal | Healthy eye |

---

## Features

- Select imaging type — OCT Scan or Fundus Photo
- Drag & drop image upload
- Prediction with confidence scores for all classes
- Grad-CAM heatmap — highlights the region the model focused on
- Low-confidence warning for uncertain predictions
- Runs locally or in Docker

---

## Project Structure

```text
.
├── app.py                    # Streamlit web app
├── train.py                  # Training script (OCT2017 dataset)
├── predict.py                # CLI prediction script
├── eye_disease_model.keras   # OCT Model 1: CNV, DME, DRUSEN, NORMAL
├── model_oct_c8/             # OCT Model 2: AMD, CSR, MH, DR
│   ├── config.json
│   ├── model.weights.h5
│   └── metadata.json
├── fundus_final/             # Fundus Model: Glaucoma, Cataract, Normal, Hypertension, Myopia
│   ├── config.json
│   ├── model.weights.h5
│   └── metadata.json
├── Dockerfile                # Docker setup
├── requirements.txt          # Python dependencies
└── README.md
```

---

## Datasets

### OCT Model 1 — OCT2017 (Kermany et al., 2018)

- Classes: CNV, DME, DRUSEN, NORMAL
- Source: [kaggle.com/datasets/paultimothymooney/kermany2018](https://www.kaggle.com/datasets/paultimothymooney/kermany2018)

### OCT Model 2 — OCT-C8

- Classes: AMD, CSR, MH, DR
- Source: [kaggle.com/datasets/obulisainaren/retinal-oct-c8](https://www.kaggle.com/datasets/obulisainaren/retinal-oct-c8)

### Fundus Model — Combined Dataset (FUNDUS_DATASET)

Built by merging 4 public Kaggle datasets:

| Source | Classes Used | Images |
| --- | --- | --- |
| [Eye Diseases Classification](https://www.kaggle.com/datasets/gunavenkatdoddi/eye-diseases-classification) | Glaucoma, Cataract, Normal | 704 + 726 + 751 |
| [Hypertension & Hypertensive Retinopathy](https://www.kaggle.com/datasets/harshwardhanfartale/hypertension-and-hypertensive-retinopathy-dataset) | Hypertension | 498 |
| [PALM Pathologic Myopia](https://www.kaggle.com/datasets/fahimaislam1812/myopia) | Myopia | 445 |
| [G1020 Glaucoma](https://www.kaggle.com/datasets/kiamahmed/glaucoma-fundus-imaging-g1020-splitted) | Glaucoma (extra) | 296 |

Final training split:

```text
FUNDUS_DATASET/train/
├── Glaucoma      → 1,000 images
├── Cataract      →   726 images
├── Normal        →   751 images
├── Hypertension  →   498 images
└── Myopia        →   445 images
Total: 3,420 images
```

---

## Running Locally

### Requirements

- Python 3.11
- TensorFlow 2.17.0
- Keras 3.10.0

### Setup

```bash
# Clone the repo
git clone https://github.com/Yaseenjabir/Eye-Diseases-Detection-Model
cd Eye-Diseases-Detection-Model

# Create virtual environment with Python 3.11
py -3.11 -m venv venv

# Activate (choose one)
source venv/Scripts/activate   # Windows Git Bash
venv\Scripts\activate          # Windows CMD
source venv/bin/activate       # Mac/Linux

# Install dependencies
pip install -r requirements.txt

# Run the app
python -m streamlit run app.py
```

---

## Running with Docker

```bash
# Build
docker build -t eye-disease-app .

# Run
docker run -p 8501:8501 eye-disease-app
```

Then open: [http://localhost:8501](http://localhost:8501)

---

## Dependencies

- Python 3.11
- tensorflow==2.17.0
- keras==3.10.0
- streamlit>=1.32.0
- Pillow>=10.0.0
- matplotlib>=3.7.0
- numpy==1.26.4

---

> **Disclaimer:** This tool is for educational purposes only and is not a substitute for professional medical advice.
