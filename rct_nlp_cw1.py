# -*- coding: utf-8 -*-
"""RCT_NLP_CW1.ipynb

Automatically generated by Colab.

Original file is located at
    https://colab.research.google.com/drive/1aRehlVgXYs7OgSTEkSWtRLzUse3DkLz-
"""

import pandas as pd
import numpy as np
import string
import re
import nltk
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from nltk.tokenize import word_tokenize
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix, roc_curve, auc, precision_recall_curve
from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import GradientBoostingClassifier
import matplotlib.pyplot as plt
from tqdm import tqdm
from wordcloud import WordCloud
import seaborn as sns

from google.colab import drive
import os
drive.mount('/content/drive')

dataset_path = '/content/drive/MyDrive/NLP_RCT_Dataset/rct_data.txt'

# Download NLTK data
nltk.download('punkt')
nltk.download('stopwords')
nltk.download('wordnet')

# Function to preprocess text
def preprocess_text(text):
    text = text.lower()  # Convert to lowercase
    text = re.sub(r'[^\w\s]', ' ', text)  # Remove punctuation and typographical symbols, replace with space
    text = re.sub(r'\s+', ' ', text).strip() # Remove extra spaces and strip leading/trailing spaces
    words = word_tokenize(text)  # Tokenize sentences
    words = [word for word in words if word not in stopwords.words('english')]  # Remove stop words
    lemmatizer = WordNetLemmatizer()
    words = [lemmatizer.lemmatize(word) for word in words]  # Lemmatization
    return ' '.join(words)  # Join tokens into a single string

# Read dataset from txt file
df = pd.read_csv(dataset_path, delimiter='\t', header=None, names=['ID', 'Label', 'Year', 'Title', 'Abstract'])

print("No of Rows: {}".format(df.shape[0]))
print("No of Columns: {}".format(df.shape[1]))

print("\nData View :\n")
print(df.head())

#Trim unnecessary spaces for strings
df["Title"] = df["Title"].apply(lambda x: x.strip() if isinstance(x, str) else x)
df["Abstract"] = df["Abstract"].apply(lambda x: x.strip() if isinstance(x, str) else x)

df=df.dropna()
print("No of Rows: {}".format(df.shape[0]))
print("No of Columns: {}".format(df.shape[1]))

print("\nData View :\n")
print(df.head())

# Combine title and abstract into one column
df['text'] = df['Title'] + ' ' + df['Abstract']

# Print before preprocessing
print("Before Preprocessing:")
print(df['text'].head())

# Apply preprocessing
tqdm.pandas()  # Initialize tqdm for progress bar
df['processed_text'] = df['text'].progress_apply(preprocess_text)

# Print after preprocessing
print("\nAfter Preprocessing:")
print(df['processed_text'].head())

# Word Cloud visualization for preprocessed text
all_text = ' '.join(df['processed_text'])
wordcloud = WordCloud(width=800, height=400, background_color='white').generate(all_text)
plt.figure(figsize=(10, 5))
plt.imshow(wordcloud, interpolation='bilinear')
plt.axis('off')
plt.title('Word Cloud of Preprocessed Text')
plt.show()

# Train-test split
train_texts, test_texts, train_labels, test_labels = train_test_split(df['processed_text'], df['Label'], test_size=0.2, random_state=42)

# Print the shape and number of columns
print("\nTraining data shape:", train_texts.shape, "Number of columns:", len(train_texts))
print("Test data shape:", test_texts.shape, "Number of columns:", len(test_texts))

# Vectorize text using TF-IDF
vectorizer = TfidfVectorizer()
X_train_tfidf = vectorizer.fit_transform(train_texts)
X_test_tfidf = vectorizer.transform(test_texts)

# Print the shape of the TF-IDF vectors
print("TF-IDF vectorized training data shape:", X_train_tfidf.shape)
print("TF-IDF vectorized test data shape:", X_test_tfidf.shape)

# Train SVM
svm_clf = SVC(kernel='linear', random_state=42, probability=True)
svm_clf.fit(X_train_tfidf, train_labels)
y_pred_svm = svm_clf.predict(X_test_tfidf)
y_proba_svm = svm_clf.predict_proba(X_test_tfidf)[:, 1]

# Train Logistic Regression
lr_clf = LogisticRegression(random_state=42)
lr_clf.fit(X_train_tfidf, train_labels)
y_pred_lr = lr_clf.predict(X_test_tfidf)
y_proba_lr = lr_clf.predict_proba(X_test_tfidf)[:, 1]

# Train Gradient Boosting
gb_clf = GradientBoostingClassifier(n_estimators=100, random_state=42)
gb_clf.fit(X_train_tfidf, train_labels)
y_pred_gb = gb_clf.predict(X_test_tfidf)
y_proba_gb = gb_clf.predict_proba(X_test_tfidf)[:, 1]

# Evaluation metrics for SVM
svm_accuracy = accuracy_score(test_labels, y_pred_svm)
svm_precision = precision_score(test_labels, y_pred_svm)
svm_recall = recall_score(test_labels, y_pred_svm)
svm_f1 = f1_score(test_labels, y_pred_svm)

# Evaluation metrics for Logistic Regression
lr_accuracy = accuracy_score(test_labels, y_pred_lr)
lr_precision = precision_score(test_labels, y_pred_lr)
lr_recall = recall_score(test_labels, y_pred_lr)
lr_f1 = f1_score(test_labels, y_pred_lr)

# Evaluation metrics for Gradient Boosting
gb_accuracy = accuracy_score(test_labels, y_pred_gb)
gb_precision = precision_score(test_labels, y_pred_gb)
gb_recall = recall_score(test_labels, y_pred_gb)
gb_f1 = f1_score(test_labels, y_pred_gb)

# Print metrics for all models
print("SVM Metrics:")
print(f"Accuracy: {svm_accuracy}")
print(f"Precision: {svm_precision}")
print(f"Recall: {svm_recall}")
print(f"F1 Score: {svm_f1}")

print("\nLogistic Regression Metrics:")
print(f"Accuracy: {lr_accuracy}")
print(f"Precision: {lr_precision}")
print(f"Recall: {lr_recall}")
print(f"F1 Score: {lr_f1}")

print("\nGradient Boosting Metrics:")
print(f"Accuracy: {gb_accuracy}")
print(f"Precision: {gb_precision}")
print(f"Recall: {gb_recall}")
print(f"F1 Score: {gb_f1}")

# Collect metrics
metrics = {
    'Model': ['SVM', 'Logistic Regression', 'Gradient Boosting'],
    'Accuracy': [svm_accuracy, lr_accuracy, gb_accuracy],
    'Precision': [svm_precision, lr_precision, gb_precision],
    'Recall': [svm_recall, lr_recall, gb_recall],
    'F1 Score': [svm_f1, lr_f1, gb_f1]
}

metrics_df = pd.DataFrame(metrics)

# Plot metrics
plt.figure(figsize=(16, 12))

# Create subplots for each metric
plt.subplot(2, 2, 1)
plt.bar(metrics_df['Model'], metrics_df['Accuracy'], color=['blue', 'orange', 'green'])
plt.title('Accuracy')
plt.xticks(rotation=45)

plt.subplot(2, 2, 2)
plt.bar(metrics_df['Model'], metrics_df['Precision'], color=['blue', 'orange', 'green'])
plt.title('Precision')
plt.xticks(rotation=45)

plt.subplot(2, 2, 3)
plt.bar(metrics_df['Model'], metrics_df['Recall'], color=['blue', 'orange', 'green'])
plt.title('Recall')
plt.xticks(rotation=45)

plt.subplot(2, 2, 4)
plt.bar(metrics_df['Model'], metrics_df['F1 Score'], color=['blue', 'orange', 'green'])
plt.title('F1 Score')
plt.xticks(rotation=45)

plt.tight_layout()
plt.show()

# Confusion Matrix for each model
def plot_confusion_matrix(cm, title):
    plt.figure(figsize=(6, 4))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=['Pred 0', 'Pred 1'], yticklabels=['True 0', 'True 1'])
    plt.title(title)
    plt.ylabel('Actual')
    plt.xlabel('Predicted')
    plt.show()

cm_svm = confusion_matrix(test_labels, y_pred_svm)
cm_lr = confusion_matrix(test_labels, y_pred_lr)
cm_gb = confusion_matrix(test_labels, y_pred_gb)

plot_confusion_matrix(cm_svm, 'SVM Confusion Matrix')
plot_confusion_matrix(cm_lr, 'Logistic Regression Confusion Matrix')
plot_confusion_matrix(cm_gb, 'Gradient Boosting Confusion Matrix')

# ROC Curve and AUC for each model
def plot_roc_curve(y_true, y_proba, model_name):
    fpr, tpr, _ = roc_curve(y_true, y_proba)
    roc_auc = auc(fpr, tpr)
    plt.figure(figsize=(8, 6))
    plt.plot(fpr, tpr, color='darkorange', lw=2, label=f'ROC curve (area = {roc_auc:.2f})')
    plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title(f'Receiver Operating Characteristic (ROC) Curve for {model_name}')
    plt.legend(loc='lower right')
    plt.show()

plot_roc_curve(test_labels, y_proba_svm, 'SVM')
plot_roc_curve(test_labels, y_proba_lr, 'Logistic Regression')
plot_roc_curve(test_labels, y_proba_gb, 'Gradient Boosting')

# Precision-Recall Curve for each model
def plot_precision_recall_curve(y_true, y_proba, model_name):
    precision, recall, _ = precision_recall_curve(y_true, y_proba)
    plt.figure(figsize=(8, 6))
    plt.plot(recall, precision, color='b', lw=2, label=f'Precision-Recall curve')
    plt.xlabel('Recall')
    plt.ylabel('Precision')
    plt.title(f'Precision-Recall Curve for {model_name}')
    plt.legend(loc='lower left')
    plt.show()

plot_precision_recall_curve(test_labels, y_proba_svm, 'SVM')
plot_precision_recall_curve(test_labels, y_proba_lr, 'Logistic Regression')
plot_precision_recall_curve(test_labels, y_proba_gb, 'Gradient Boosting')

