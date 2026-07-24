# Fine-Tuning Pre-Trained Language Models for Zomato Sentiment Analysis

## 📌 Project Overview
This project focuses on fine-tuning and evaluating two pre-trained transformer language models—**BERT (`bert-base-uncased`)** and **RoBERTa (`roberta-base`)**—for a 3-class text classification task: **Sentiment Analysis on Zomato Restaurant Reviews** (Negative, Neutral, Positive).

---

## 📊 Dataset Description
- **Source:** Zomato Restaurant Reviews (Kaggle)
- **Task:** Text Classification (Sentiment Analysis)
- **Classes:** 3 Classes (`0: NEGATIVE`, `1: NEUTRAL`, `2: POSITIVE`)
- **Cleaning & Preprocessing:** 
  - Text noise removal (stripped `RATED\n` tags, metadata, tuple brackets, and unwanted characters).
  - Subsampled up to 1,000 samples per class to mitigate extreme class imbalance.
  - **Total Samples:** 2,280 cleaned review texts.
- **Data Splitting (Stratified):**
  - **Train Set (80%):** 1,824 samples
  - **Validation Set (10%):** 228 samples
  - **Test Set (10%):** 228 samples

---

## ⚙️ Experimental Setup & Hyperparameters
Fine-tuning was conducted using PyTorch and the Hugging Face `Trainer` API accelerated on an Apple Silicon GPU (MPS).

| Hyperparameter | Value | Technical Justification |
| :--- | :--- | :--- |
| **Learning Rate** | `2e-5` | Recommended learning rate for transformer fine-tuning to prevent *catastrophic forgetting*. |
| **Effective Batch Size** | `16` (`8 per device` × `2 grad accum`) | Optimizes VRAM allocation while maintaining gradient estimation stability. |
| **Epochs** | `3` | Ensures optimal convergence without overfitting the pre-trained weights. |
| **Optimizer & Regularization** | AdamW with `weight_decay = 0.01` | L2 weight decay regularization to prevent overfitting. |
| **Evaluation Metric** | `Macro F1-Score` | Provides an unbiased evaluation across all sentiment classes. |

---

## 📈 Evaluation Results & Model Comparison

Both fine-tuned models were evaluated on the unseen **Test Set (228 samples)**:

| Model Architecture | Test Accuracy | Test F1 Macro |
| :--- | :---: | :---: |
| **BERT Baseline (`bert-base-uncased`)** | 65.35% | 64.45% |
| **RoBERTa (`roberta-base`)** | **69.30%** | **67.89%** |

### 🔍 Key Findings & Analysis
1. **RoBERTa Outperforms BERT:** **RoBERTa** achieved superior performance over BERT, with an improvement of **+3.95% in Test Accuracy** and **+3.44% in Macro F1-Score**. This performance boost is attributed to RoBERTa's dynamic masking and larger pre-training corpus, making it more robust at interpreting informal review nuances.
2. **Confusion Matrix Insights:**
   - Both models demonstrated high precision in detecting **NEGATIVE** sentiment reviews.
   - The primary source of misclassification occurred between **NEUTRAL** and **POSITIVE** classes due to subtle linguistic boundaries in restaurant feedback (e.g., *"Food was okay"* vs. *"Food was quite good"*).

---

## 📸 Confusion Matrix Visualizations

### BERT Baseline Model
![Confusion Matrix BERT](confusion_matrix_bert.png)

### RoBERTa Fine-Tuned Model
![Confusion Matrix RoBERTa](confusion_matrix_roberta.png)

---

## 🛠️ How to Run
1. Clone this repository:
   ```bash
   https://github.com/NaufalApta/Final_Project_NLP.git

