import os
import warnings
warnings.filterwarnings("ignore")

import pandas as pd
import numpy as np
import joblib

from sklearn.model_selection import train_test_split, RandomizedSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.metrics import recall_score, make_scorer

# ===== Load & Preprocess =====
df = pd.read_csv("cardio.csv", sep=";")
df["age"] = (df["age"] / 365).astype(int)
df["bmi"] = df["weight"] / ((df["height"] / 100) ** 2)
df.drop(["id", "height", "weight"], axis=1, inplace=True)

X = df.drop("cardio", axis=1)
y = df["cardio"]

scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

X_train, X_test, y_train, y_test = train_test_split(
    X_scaled, y, test_size=0.2, random_state=42
)

# ===== Model Training: Gradient Boosting =====
print("Training Gradient Boosting...")
recall = make_scorer(recall_score)

gb_search = RandomizedSearchCV(GradientBoostingClassifier(random_state=42), {
    "n_estimators": [100, 150, 200],
    "learning_rate": [0.01, 0.05, 0.1, 0.2],
    "max_depth": [2, 3, 4, 5],
    "min_samples_split": [2, 5]
}, n_iter=5, scoring=recall, cv=2, n_jobs=-1, random_state=42)

gb_search.fit(X_train, y_train)
best_model = gb_search.best_estimator_
print("Gradient Boosting training complete")

# ===== Save Model & Scaler Separately =====
SAVE_DIR = "saved_models"
os.makedirs(SAVE_DIR, exist_ok=True)

joblib.dump(best_model, os.path.join(SAVE_DIR, "model_gradient_boosting.pkl"))
joblib.dump(scaler, os.path.join(SAVE_DIR, "scaler.pkl"))
print("Saved model_gradient_boosting.pkl and scaler.pkl")
