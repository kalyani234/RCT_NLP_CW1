# RCT Classification Using NLP and Machine Learning

## 📌 Project Overview
This repository focuses on **automated identification and classification of Randomised Controlled Trials (RCTs)** from biomedical literature using **Natural Language Processing (NLP)** techniques. The project was developed as part of the MSc **Data Science and Computational Intelligence** programme at **Coventry University**.

Two complementary approaches are explored:
- **Classical Machine Learning (CW1)** using TF-IDF features
- **Deep Learning (CW2)** using word embeddings and neural architectures

The goal is to support **evidence-based healthcare** by improving the efficiency and accuracy of RCT identification.

---
## 📂 Dataset
- **Source**: Bat4RCT Dataset  
- **Size**: 27,063 biomedical articles  
- **Labels**: Binary classification (Intervention vs Control)  
- **Fields**:
  - ID
  - Label
  - Year
  - Title
  - Abstract

Dataset link:  
👉 https://github.com/jennak22/Bat4RCT/blob/main/rct_data.zip

---

## 🧪 Coursework 1: Machine Learning Approach (CW1)

### 🔧 Techniques Used
- Text preprocessing (NLTK)
- TF-IDF vectorisation
- Classical ML models:
  - Support Vector Machine (SVM)
  - Logistic Regression (LR)
  - Gradient Boosting Classifier (GB)

### 📊 Evaluation Metrics
- Accuracy
- Precision
- Recall
- F1 Score
- ROC AUC
- Average Precision

### 🏆 Best Performing Model
| Model | Accuracy | F1 Score | ROC AUC |
|------|---------|---------|--------|
| Gradient Boosting | **94.5%** | **0.858** | **0.968** |



---

## 🤖 Coursework 2: Deep Learning Approach (CW2)

### 🔧 Models Implemented
- Bidirectional LSTM (BiLSTM)
- Convolutional Neural Network (CNN)
- Recurrent Neural Network (RNN)
- Gated Recurrent Unit (GRU)

### 📚 Embeddings & Techniques
- Pre-trained **GloVe embeddings (100D)**
- Padding & tokenisation
- UMAP dimensionality reduction
- KMeans clustering for semantic exploration

### 📊 Performance Summary
| Model | Accuracy | F1 Score | ROC AUC |
|-----|---------|---------|--------|
| **BiLSTM** | **94.37%** | **0.94** | **0.98** |
| CNN | 92.45% | 0.92 | 0.96 |
| GRU | 91.25% | 0.91 | 0.95 |
| RNN | 90.35% | 0.90 | 0.94 |

### 🏆 Best Model
**BiLSTM** demonstrated the strongest generalisation and contextual understanding.


---

## 🔍 Key Features
- End-to-end NLP pipeline
- Robust text preprocessing
- Comparative ML vs DL analysis
- Visualisations: ROC curves, confusion matrices, word clouds
- Strong relevance to **healthcare AI & systematic reviews**

---

## 🚀 Future Improvements
- Transformer-based models (BioBERT, SciBERT)
- Class imbalance handling
- Explainability (SHAP / LIME)
- Ensemble and hybrid architectures

---

## 🛠️ Tech Stack
- Python
- Scikit-learn
- TensorFlow / Keras
- NLTK
- GloVe
- Matplotlib / Seaborn
- UMAP

---

## 📜 Citation
If you use this work, please cite:

> Anantha Navya Kalyani, *Enhancing Evidence-Based Healthcare with NLP: Identification of Randomised Controlled Trials*, Coventry University, 2025.

---

## 📬 Contact

🔗 GitHub: https://github.com/kalyani234  

---

⭐ *If you find this repository useful, please consider giving it a star!*
