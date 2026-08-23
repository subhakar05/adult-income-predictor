# Adult Income Prediction Using Machine Learning

## Introduction

This project focuses on predicting whether an individual's annual income is greater than $50K or less than or equal to $50K using machine learning classification techniques. The project uses the **Census Income (Adult) Dataset**, which contains demographic, employment, education, and other socioeconomic attributes that can be used to identify patterns associated with income levels.

To investigate the effectiveness of different classification approaches, five machine learning models are implemented and evaluated: **Decision Tree, Support Vector Machine (SVM), Custom K-Nearest Neighbors (KNN), Naïve Bayes, and Multi-Layer Perceptron (MLP)**. Each model uses a different learning strategy, allowing their predictive capabilities to be compared systematically.

The models are evaluated using multiple performance metrics, including accuracy, precision, recall, F1-score, balanced accuracy, and ROC-AUC. Using multiple metrics provides a more comprehensive evaluation than accuracy alone and helps identify how well each classifier handles the two income classes.

The main objective of this project is to develop a comparative machine learning framework for Adult Income Prediction and determine which classification algorithm provides the most effective overall performance. The experimental results are analyzed to understand the strengths and limitations of each model and to identify the most suitable classifier for the given dataset.

## Objectives

- Predict whether an individual's income is `>50K` or `<=50K`.
- Perform preprocessing and transformation of the Census Income dataset.
- Implement multiple classification algorithms.
- Compare the performance of different machine learning models.
- Evaluate models using multiple classification metrics.
- Analyze the strengths and limitations of each classifier.
- Identify the best-performing model for Adult Income Prediction.

## Dataset

The project uses the **Census Income (Adult) Dataset**, which includes demographic, employment, education, and socioeconomic attributes such as:

- Age, workclass, education, marital status
- Occupation, relationship, race, sex
- Capital gain/loss, hours-per-week, native country
- Target label: income (`>50K` or `<=50K`)

## Project Structure

adult-income-prediction/

├── .gitignore
├── README.md
├── app.py              # Application/demo entry point
├── main.ipynb          # Main notebook: preprocessing, model training, and evaluation
└── requirements.txt

## Methodology

1. **Data Preprocessing**
   - Handling missing values
   - Encoding categorical variables
   - Feature scaling/normalization
   - Train-test split

2. **Model Implementation**
   - Decision Tree
   - Support Vector Machine (SVM)
   - Custom K-Nearest Neighbors (KNN)
   - Naïve Bayes
   - Multi-Layer Perceptron (MLP)

3. **Model Evaluation**
   - Each model is trained and tested on the same data splits for fair comparison.
   - Performance is assessed using multiple metrics rather than accuracy alone.

## Machine Learning Models

The project implements the following classifiers:

1. Decision Tree
2. Support Vector Machine (SVM)
3. Custom K-Nearest Neighbors (KNN)
4. Naïve Bayes
5. Multi-Layer Perceptron (MLP)

## Evaluation Metrics

The models are compared using:

- Accuracy
- Precision
- Recall
- F1-score
- Balanced Accuracy
- ROC-AUC

These metrics provide a comprehensive view of model performance, particularly when the classes are not perfectly balanced.

## Installation

```bash
git clone https://github.com/<your-username>/adult-income-prediction.git
cd adult-income-prediction
pip install -r requirements.txt
```

## Usage

```bash
# Run the full pipeline (preprocessing, training, and evaluation)
# via the notebook
jupyter notebook main.ipynb

# Or run the application
python app.py

## Expected Outcome

The project aims to produce a comparative analysis of the five classifiers across all evaluation metrics, highlighting:

- Which model achieves the highest overall predictive performance.
- Trade-offs between models (e.g., interpretability vs. accuracy, training time vs. performance).
- How each model handles class imbalance in the dataset.
- A recommendation for the most suitable classifier for the Adult Income Prediction task.

## Technologies Used

- Python
- scikit-learn
- NumPy / Pandas
- Matplotlib / Seaborn

## License

This project is intended for academic and research purposes.
