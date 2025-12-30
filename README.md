# Applied Machine Learning — Coursework (Python / Jupyter)

This repository contains my coursework for a Machine Learning class, implemented as **Jupyter notebooks** in Python. It covers classical ML (supervised + unsupervised) and introductory deep learning with Keras/TensorFlow.

## Notebooks
- **P1.ipynb — Supervised Learning**
  - **Classification** on the `MeatClassification` dataset.
  - Pipeline includes exploratory analysis, preprocessing (e.g., scaling), model selection and tuning.
  - Models used include **Logistic Regression**, **SVM**, and **Random Forest** (with **GridSearchCV**).
  - Evaluation with **accuracy, precision, recall, F1**, and **confusion matrix**.
  - **Regression** on the `ElectronicTongue` dataset with **Ridge Regression** and **Random Forest Regressor**.
  - Regression metrics include **MAE, MSE/RMSE, MAPE, R², max error**.

- **P2.ipynb — Unsupervised Learning + Pattern Mining**
  - **Clustering** (Old Faithful geyser dataset) using **KMeans** and **DBSCAN**.
  - Model selection via **silhouette score** and qualitative visualization.
  - **Association rule mining** on a transactional dataset using **Apriori** and **FP-Growth** (mlxtend),
    extracting rules and analyzing **support, confidence, lift**, etc.

- **P3.ipynb — Deep Learning + Generative Models (Keras/TensorFlow)**
  - **Image classification** on **Fashion-MNIST** with two deep learning model families (e.g., MLP/CNN),
    including regularization and optimizer experiments (Adam / AdamW).
  - **Generative AI with Autoencoders** on **MNIST**:
    separate **Encoder**, **Decoder**, and **Autoencoder** models for reconstruction and generation.

## Tech stack
- Python 3
- Jupyter (Lab / Notebook)
- numpy, pandas, matplotlib, seaborn
- scikit-learn
- mlxtend (frequent itemsets + association rules)
- tensorflow / keras

## How to run (Google Colab)

### Option A — Open directly in Colab
1. Open the notebook on GitHub.
2. Replace `github.com` with `colab.research.google.com/github` in the URL.

Example:
- GitHub: `https://github.com/<user>/<repo>/blob/main/P1.ipynb`
- Colab:  `https://colab.research.google.com/github/<user>/<repo>/blob/main/P1.ipynb`

### Option B — Upload to Colab
1. Go to Colab → **File → Upload notebook**.
2. Upload `P1.ipynb`, `P2.ipynb`, or `P3.ipynb`.

## Data notes

- Some datasets may be downloaded or loaded directly in the notebooks.
- If you need to upload local datasets to Colab, place them in the Colab session and adjust file paths accordingly.

## Reproducibility

- Where applicable, seeds and consistent validation protocols (train/test split, CV, etc.) are used to make results comparable across runs. Library versions and hardware can still produce small numerical differences.

## Academic integrity

- This repository is shared for educational/portfolio purposes. If you are taking a similar course, do not copy solutions directly—use them only as a learning reference and follow your course policies.
