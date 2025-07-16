# 🧠 Breast Cancer Classification

This project focuses on building a machine learning model to classify breast cancer tumors as **benign** or **malignant** based on diagnostic features. The model leverages data science techniques to assist in early and accurate detection of breast cancer.

## 📊 Dataset

We use the **Breast Cancer Wisconsin Diagnostic Dataset (WBCD)** from the UCI Machine Learning Repository. The dataset contains features computed from digitized images of fine needle aspirate (FNA) of breast masses.

- Number of Instances: 569
- Features: 30 (mean, standard error, and worst for radius, texture, perimeter, etc.)
- Target: `Diagnosis` (M = Malignant, B = Benign)

## 🛠️ Technologies Used

- Python
- Pandas, NumPy
- Scikit-learn
- Matplotlib, Seaborn
- Jupyter Notebook / Google Colab

## 🧪 ML Algorithms Applied

- Logistic Regression
- K-Nearest Neighbors (KNN)
- Support Vector Machine (SVM)
- Decision Tree / Random Forest
- Cross-validation and Hyperparameter Tuning

## 🎯 Evaluation Metrics

- Accuracy
- Precision
- Recall
- F1-score
- Confusion Matrix
- ROC-AUC Curve

## 📈 Sample Results

| Model             | Accuracy |
|------------------|----------|
| LogisticRegression | 96.5%    |
| RandomForest       | 97.3%    |
| SVM                | 98.1%    |

> *Note: Accuracy may vary based on dataset split and parameter tuning.*

## 📌 Project Structure

📂 breast-cancer-classification/
│
├── data/ # Dataset files (if applicable)
├── notebooks/ # Jupyter Notebooks
├── models/ # Saved ML models
├── results/ # Plots, confusion matrix, etc.
├── app.py # (Optional) Streamlit or Flask web app
├── requirements.txt # Python dependencies
└── README.md # Project documentation

bash
Copy
Edit

## 🚀 How to Run

1. Clone the repository:
   ```bash
   git clone https://github.com/yourusername/breast-cancer-classification.git
   cd breast-cancer-classification
Install dependencies:

bash
Copy
Edit
pip install -r requirements.txt
Run the notebook or app:

Open notebooks/BreastCancerClassifier.ipynb in Jupyter/Colab

Or run app.py for the web interface (if available)

📚 References
UCI ML Repository: https://archive.ics.uci.edu/ml/datasets/Breast+Cancer+Wisconsin+(Diagnostic)

Scikit-learn documentation: https://scikit-learn.org/

Kaggle Dataset (optional)
