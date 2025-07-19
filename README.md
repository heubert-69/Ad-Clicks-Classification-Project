📈 Click Prediction with Advanced Feature Engineering & Threshold Optimization
This project builds a powerful binary classification model that predicts whether a user will click an ad based on demographic, device, and behavioral features. It uses feature-rich engineering, threshold tuning, and thorough evaluation to achieve high F1 scores and near-perfect recall.

🚀 Features
✅ Data Cleaning & Preprocessing

✅ Advanced Feature Engineering (polynomial, CTR, entropy, lag, rolling, behavioral score, etc.)

✅ Model Training (Random Forest, Logistic Regression)

✅ Threshold Tuning using F1 Score

✅ Evaluation with ROC AUC, F1, Precision, Recall, Confusion Matrix

✅ Interactive Visualizations using Plotly

✅ Production-ready threshold selection and export

📊 Dataset Features
Column	Description
age	User's age (numerical)
gender	Encoded gender
device_type	Encoded device type (e.g., mobile, desktop)
ad_position	The position of the ad on the page
browsing_history	Comma-separated string of visited categories
time_of_day	Time bucket of the session
click	Target variable (1 if clicked, else 0)

🛠 Tech Stack
Python 3.6+

scikit-learn for modeling and metrics

pandas, numpy, scipy for data processing

matplotlib, plotly for visualization

🧠 Feature Engineering Highlights
age_squared, age_cubed, age × ad_position

gender_ctr: click-through-rate by gender

inv_adpos, log_adpos: ad position transforms

bh_entropy: entropy of browsing history

behavior_score: custom weighted user intent

lag, rolling, rank, and cross features

✅ Model Evaluation (Best Threshold ~0.3059)
Metric	Value
Precision	0.754
Recall	0.982 ✅
F1 Score	0.853 ✅
ROC AUC	0.893
Accuracy	77.4%

Confusion Matrix:

```lua
[[ 284  416]
 [  22 1277]]
```
🧪 Model Choice
Although RandomForestClassifier achieved a higher cross-validation score of 86%, I chose to use Logistic Regression for its simplicity, interpretability, and faster inference. Logistic Regression scored 78% in cross-validation but still produced a high F1 score after threshold tuning.

Although `RandomForestClassifier` achieved a higher cross-validation score of **86%**, I chose to use **Logistic Regression** for its simplicity, interpretability, and faster inference. Logistic Regression scored **78%** in cross-validation but still produced a high F1 score after threshold tuning.

## ✅ Model Evaluation (Best Threshold ~0.3058)

| Metric     | Value    |
|------------|----------|
| Precision  | 0.754    |
| Recall     | 0.982 ✅ |
| F1 Score   | 0.853 ✅ |
| ROC AUC    | 0.893    |
| Accuracy   | 77.4%    |


📦 How to Run
Install dependencies

```bash
pip install sklearn pandas numpy matplotlib scipy plotly
```

Train and Evaluate

```python
# In notebook
model.fit(X_train, y_train)
y_proba = model.predict_proba(X_test)[:, 1]
y_pred = (y_proba >= 0.3058).astype(int)
evaluate_model(model, X_test, y_test)
```

📁 Outputs
final_model.pkl: Trained model

best_threshold.pkl: Saved optimal threshold

scored_test_set.csv: Test set with predictions and probabilities

📌 TODO
Deploy as a web API with FastAPI/Flask
