import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from xgboost import XGBClassifier
from sklearn.metrics import accuracy_score
from fairlearn.metrics import MetricFrame, demographic_parity_difference, equalized_odds_difference
from fairlearn.metrics import selection_rate, true_positive_rate, false_positive_rate

# 1. Load the UCI Heart Disease Dataset
url = "https://archive.ics.uci.edu/ml/machine-learning-databases/heart-disease/processed.cleveland.data"
columns = ["age", "sex", "cp", "trestbps", "chol", "fbs", "restecg", 
           "thalach", "exang", "oldpeak", "slope", "ca", "thal", "target"]
df = pd.read_csv(url, names=columns, na_values="?")
df = df.dropna()

# 2. Preprocess the Data
df['target'] = (df['target'] > 0).astype(int)

X = df.drop("target", axis=1)
y = df["target"]
sensitive_features = df["sex"]  # 1.0 = male, 0.0 = female

X_train, X_test, y_train, y_test, A_train, A_test = train_test_split(
    X, y, sensitive_features, test_size=0.3, random_state=42
)

# 3. Define the Models
models = {
    "Logistic Regression": LogisticRegression(max_iter=1000),
    "XGBoost": XGBClassifier(eval_metric='logloss', random_state=42)
}

metrics = {
    'accuracy': accuracy_score,
    'selection_rate': selection_rate,
    'true_positive_rate': true_positive_rate,
    'false_positive_rate': false_positive_rate
}

# 4. Train and Audit Both Models
for model_name, model in models.items():
    print(f"================ {model_name} ================")
    
    # Train and Predict
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    
    print(f"Overall Accuracy: {accuracy_score(y_test, y_pred):.3f}\n")
    
    # Evaluate Fairness
    metric_frame = MetricFrame(
        metrics=metrics,
        y_true=y_test,
        y_pred=y_pred,
        sensitive_features=A_test
    )
    
    print("Metrics by Gender (0.0 = Female, 1.0 = Male):")
    print(metric_frame.by_group)
    print("\nFairness Disparities:")
    print(f"Demographic Parity Difference: {demographic_parity_difference(y_test, y_pred, sensitive_features=A_test):.3f}")
    print(f"Equalized Odds Difference: {equalized_odds_difference(y_test, y_pred, sensitive_features=A_test):.3f}")
    print("\n")