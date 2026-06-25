# Fake News Detection with DistilBERT

A deep learning project that classifies news articles as **Fake** or **Real** using **DistilBERT** fine-tuned with PyTorch and served via a Flask web application.

> Developed as part of an internship project.

---

## Project Structure

```
fake-news-detector/
├── app.py                    # Flask web server and prediction API
├── download_and_prepare.py   # Dataset download and preprocessing script
├── templates/
│   └── index.html            # Frontend UI
├── static/
│   ├── style.css             # Stylesheet
│   └── js/
│       └── main.js           # Frontend logic
├── models/
│   ├── fake_news_model/      # NOT included — train locally (see below)
│   └── metrics.json          # Saved evaluation metrics
├── confusion_matrix/
│   └── confusion_matrix.png  # Confusion matrix plot
└── data/                     # NOT included — download from Kaggle (see below)
    ├── True.csv
    ├── Fake.csv
    ├── train.csv
    ├── test.csv
    └── processed_data.csv
```

---

## Model Evaluation

### Results Summary

| Metric    | Value  |
|-----------|--------|
| Accuracy  | 99.98% |
| Precision | 99.97% |
| Recall    | 99.97% |
| F1 Score  | 99.97% |

> Evaluated on 4,000 test samples (2,000 Fake, 2,000 Real)

---

### Confusion Matrix

![Confusion Matrix](confusion_matrix/confusion_matrix.png)

|                | Predicted FAKE | Predicted REAL |
|----------------|----------------|----------------|
| **True FAKE**  | 2000 (TN)      | 0 (FP)         |
| **True REAL**  | 1 (FN)         | 1999 (TP)      |

The model achieves near-perfect classification with only **1 misclassification** across 4,000 test samples. All 2,000 fake articles were correctly identified.

---

### Training History

| Epoch | Loss     | Accuracy |
|-------|----------|----------|
| 1     | 0.0621   | 99.90%   |
| 2     | 0.0026   | 99.95%   |
| 3     | 0.0007   | 99.97%   |

- Trained for **3 epochs** using DistilBERT
- Loss dropped sharply from epoch 1 to 3 — model converged quickly
- No overfitting observed across all epochs

---

## Model Architecture

| Component       | Detail                                      |
|-----------------|---------------------------------------------|
| Base Model      | DistilBERT (distilbert-base-uncased)        |
| Input           | Raw article text (title + body, up to 512 tokens) |
| Classifier Head | Linear(768 → 2)                             |
| Output Classes  | FAKE, REAL                                  |
| Optimizer       | AdamW                                       |
| Loss Function   | CrossEntropyLoss                            |
| Device          | CPU                                         |

---

## Web Application

The project includes a fully functional Flask web app with:

- Paste any news article and get an instant **FAKE / REAL verdict**
- **Confidence score** displayed as an animated ring
- **Probability breakdown** bar chart for both classes
- **Keyword highlighting** — key signals in the article are highlighted
- **Model metrics panel** — loads confusion matrix and performance stats on demand

---

## Setup & Usage

### 1. Clone the Repository

```bash
git clone https://github.com/Divyanshu409/fake-news-detector.git
cd fake-news-detector
```

### 2. Install Dependencies

```bash
pip install flask torch transformers numpy scikit-learn pandas
```

### 3. Download and Prepare the Dataset

```bash
python download_and_prepare.py
```

Requires Kaggle API credentials. Downloads the dataset and generates `train.csv` and `test.csv` inside the `data/` folder.

### 4. Train the Model

```bash
python train.py
```

Saves the fine-tuned model to `models/fake_news_model/` and writes `models/metrics.json`.

### 5. Run the Web App

```bash
python app.py
```

Open [http://localhost:5000](http://localhost:5000) in your browser. Paste a news article and click **Analyze**.

---

## Dataset

Uses the [Fake and Real News Dataset](https://www.kaggle.com/datasets/clmentbisaillon/fake-and-real-news-dataset) from Kaggle.

| Split | FAKE   | REAL   | Total  |
|-------|--------|--------|--------|
| Train | 8,000  | 8,000  | 16,000 |
| Test  | 2,000  | 2,000  | 4,000  |

**The dataset is not included** due to size. Download from Kaggle and place `True.csv` and `Fake.csv` inside the `data/` folder, or run `download_and_prepare.py` directly.

---

## Author

**Divyanshu Sharma**  
Intern - InLighnx Global
[GitHub](https://github.com/Divyanshu409)

---

## License

This project is for educational and internship purposes.
