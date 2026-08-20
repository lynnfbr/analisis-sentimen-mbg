# 🍱 Streamlit Application – Sentiment Analysis of MBG

A Streamlit-based application developed as a supporting implementation for the research:

> **Perbandingan Algoritma Naive Bayes dan Support Vector Machine untuk Analisis Sentimen Masyarakat terhadap Program MBG pada Media Sosial X**

This application provides an interactive interface for exploring the research dataset, viewing machine learning model evaluation results, and testing sentiment predictions using pre-trained models.

---

## 📌 Application Features

The application consists of five main pages:

### 1. Beranda
Provides an overview of the research, dataset, research workflow, and model performance comparison.

### 2. Eksplorasi Data
Displays information related to the dataset, including:

- Data filtering results
- Data quality after preprocessing
- Sentiment distribution
- Training and testing data distribution
- Examples of eliminated data
- Positive sentiment wordcloud
- Negative sentiment wordcloud

### 3. Cara Kerja Model
Provides an explanation of the methods used in the research:

- TF-IDF
- Multinomial Naïve Bayes
- Support Vector Machine (SVM)
- Simple TF-IDF calculation simulation

### 4. Evaluasi Model
Displays the evaluation results of the machine learning models, including:

- Hyperparameter optimization
- Accuracy
- Precision
- Recall
- F1-Score
- Confusion Matrix
- Comparison between Naïve Bayes and SVM

### 5. Uji Coba Prediksi
Provides two prediction modes:

- **Input Teks Baru** – Predict sentiment from new text entered by the user.
- **Data Berlabel (Ground Truth)** – Test the model using labeled research data and compare the prediction with the original label.

---

# 💻 Requirements

Before running the application, make sure the following software is installed:

- Python 3.9 or newer
- pip
- Git (optional)

The required Python libraries are listed in:

```text
requirements.txt
```

Main dependencies:

- Streamlit
- Pandas
- NumPy
- Scikit-learn
- Joblib
- Plotly
- Matplotlib
- WordCloud
- NLTK
- Sastrawi

---

# 📁 Project Structure

The repository should have the following structure:

```text
project/
│
├── app.py
├── preprocessing.py
├── requirements.txt
│
├── models/
│   ├── tfidf_vectorizer.pkl
│   ├── naive_bayes_model.pkl
│   └── svm_model.pkl
│
└── data/
    ├── dataset_filtered.csv
    ├── dataset_labeled_binary.csv
    └── dataset_labeled_3class.csv
```

### File and Folder Description

| File / Folder | Description |
|---|---|
| `app.py` | Main Streamlit application |
| `preprocessing.py` | Text preprocessing functions used during prediction |
| `requirements.txt` | List of required Python libraries |
| `models/` | Contains the pre-trained machine learning models and TF-IDF Vectorizer |
| `data/` | Contains the datasets used by the application |

---

# 🚀 Installation

## 1. Clone the Repository

Clone the repository using Git:

```bash
git clone https://github.com/lynnfbr/analisis-sentimen-mbg.git
```

Then navigate to the project directory:

```bash
cd analisis-sentimen-mbg
```

---

## 2. Create a Virtual Environment

It is recommended to use a virtual environment to avoid dependency conflicts.

### Windows

Create the virtual environment:

```bash
python -m venv venv
```

Activate the virtual environment:

```bash
venv\Scripts\activate
```

### macOS / Linux

Create the virtual environment:

```bash
python3 -m venv venv
```

Activate the virtual environment:

```bash
source venv/bin/activate
```

---

## 3. Install Dependencies

After activating the virtual environment, install all required libraries:

```bash
pip install -r requirements.txt
```

Wait until the installation process is completed.

---

# ▶️ Run the Application

After all dependencies have been installed, run the following command:

```bash
streamlit run app.py
```

If the `streamlit` command is not recognized, use:

```bash
python -m streamlit run app.py
```

After the application starts successfully, Streamlit will display a local address similar to:

```text
Local URL: http://localhost:8501
```

Open the displayed URL in your web browser.

---

# 🧭 How to Use the Application

After opening the application in your browser, use the navigation menu to access the available features.

### Step 1 – Beranda

Open **Beranda** to view:

- Research overview
- Dataset summary
- Research workflow
- Model performance comparison

---

### Step 2 – Eksplorasi Data

Open **Eksplorasi Data** to explore:

- Dataset filtering results
- Data distribution
- Sentiment distribution
- Training and testing data
- Eliminated data examples
- Wordcloud visualization

---

### Step 3 – Cara Kerja Model

Open **Cara Kerja Model** to understand the classification process using:

```text
Text
  ↓
Preprocessing
  ↓
TF-IDF
  ↓
┌─────────────────────┐
│                     │
│  Naïve Bayes        │
│  SVM                │
│                     │
└─────────────────────┘
  ↓
Sentiment Prediction
```

---

### Step 4 – Evaluasi Model

Open **Evaluasi Model** to view:

- Hyperparameter optimization results
- Accuracy
- Precision
- Recall
- F1-Score
- Confusion Matrix
- Model comparison

---

### Step 5 – Uji Coba Prediksi

Open **Uji Coba Prediksi** to test the trained models.

## Input Teks Baru

1. Select **Input Teks Baru**.
2. Enter an opinion related to the Program Makan Bergizi Gratis.
3. Enable **Tampilkan tahapan preprocessing** if you want to see the preprocessing process.
4. Click **Prediksi Sentimen**.
5. The application will display the predictions from:
   - Multinomial Naïve Bayes
   - Support Vector Machine (SVM)

Example input:

```text
Program makan bergizi gratis sangat membantu anak-anak sekolah.
```

The application will process the input text and display the predicted sentiment.

---

## Data Berlabel (Ground Truth)

This feature allows users to test the models using data that already has an original sentiment label.

The application displays:

- Original text
- Preprocessed text
- Ground truth label
- Lexicon score
- Naïve Bayes prediction
- SVM prediction
- Prediction correctness

This feature can be used to demonstrate how the trained models perform on labeled research data.

---

# 🤖 Machine Learning Models

The application uses two pre-trained classification models:

### Multinomial Naïve Bayes

Used as one of the classification algorithms for sentiment analysis.

### Support Vector Machine (SVM)

Implemented using `LinearSVC` and used as the second classification algorithm.

The sentiment classes used for prediction are:

```text
0 → Negatif
1 → Positif
```

---

# 🔤 Text Preprocessing

The preprocessing process applied to new text includes:

1. Cleaning
2. Case Folding
3. Remove Character
4. Tokenizing
5. Normalization
6. Stopword Removal
7. Stemming

The preprocessing result is then transformed into numerical features using the trained TF-IDF Vectorizer.

---

# 📐 TF-IDF Configuration

The TF-IDF Vectorizer used in the research was configured with:

```python
max_features = 5000
ngram_range = (1, 2)
min_df = 2
```

The trained vectorizer is stored in:

```text
models/tfidf_vectorizer.pkl
```

---

# 📦 Pre-trained Models

The application requires the following files:

```text
models/
│
├── tfidf_vectorizer.pkl
├── naive_bayes_model.pkl
└── svm_model.pkl
```

These files are loaded automatically by the application.

Make sure the filenames are exactly:

```text
tfidf_vectorizer.pkl
naive_bayes_model.pkl
svm_model.pkl
```

If any of these files are missing, the prediction functionality cannot be used.

---

# 📊 Dataset

The application uses prepared datasets from the research.

The required dataset files are:

```text
data/
│
├── dataset_filtered.csv
├── dataset_labeled_binary.csv
└── dataset_labeled_3class.csv
```

Make sure the `data` folder is located in the same directory as `app.py`.

---

# ⚠️ Important Notes

This repository contains a **supporting application for the research**.

The Streamlit application does not perform the complete research pipeline from the beginning.

The following processes were performed previously during the research:

- Data collection
- Data filtering
- Text preprocessing
- Sentiment labeling
- TF-IDF feature extraction
- Model training
- Hyperparameter optimization
- Model evaluation

The Streamlit application uses the resulting datasets, TF-IDF Vectorizer, and pre-trained machine learning models.

Therefore, users do not need to retrain the models to run the application.

---

# 🔐 Security

Before publishing this repository to GitHub, make sure that no sensitive information is included.

Do **not** upload:

- API keys
- Apify tokens
- Passwords
- Access tokens
- `.env` files containing credentials
- Google account credentials
- Private API credentials

If credentials are required during development, store them locally and add sensitive files to `.gitignore`.

Example:

```text
.env
*.key
*.secret
__pycache__/
venv/
```

---

# 🛠️ Troubleshooting

## `streamlit` is not recognized

If you receive:

```text
'streamlit' is not recognized as an internal or external command
```

try:

```bash
python -m streamlit run app.py
```

If the problem persists, make sure Streamlit has been installed:

```bash
pip install streamlit
```

---

## `ModuleNotFoundError`

Example:

```text
ModuleNotFoundError: No module named 'pandas'
```

Run:

```bash
pip install -r requirements.txt
```

Make sure the virtual environment is active.

---

## Model Not Found

If the application displays a message indicating that the model cannot be found, check the following files:

```text
models/tfidf_vectorizer.pkl
models/naive_bayes_model.pkl
models/svm_model.pkl
```

Make sure the `models` folder is located in the same directory as `app.py`.

---

## Preprocessing Error

If an error related to NLTK or Sastrawi occurs, install the required libraries:

```bash
pip install nltk Sastrawi
```

Then restart the application:

```bash
python -m streamlit run app.py
```

---

# ⏹️ Stop the Application

To stop the Streamlit application, return to the terminal and press:

```text
Ctrl + C
```

---

# 📚 Research Information

**Title:**

> Perbandingan Algoritma Naive Bayes dan Support Vector Machine untuk Analisis Sentimen Masyarakat terhadap program MBG pada media sosial X 

**Author:**  
Linda Febriyanti

**Study Program:**  
Teknik Informatika

**Institution:**  
Universitas Islam Nusantara

---

# 📄 Academic Purpose

This repository is provided as a supporting resource for the academic research and publication associated with the study.

The application is intended for research demonstration, model evaluation, and sentiment prediction based on the prepared research dataset and trained machine learning models.

---

## 👩‍💻 Author

**Linda Febriyanti**  
Teknik Informatika  
Universitas Islam Nusantara