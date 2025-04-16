import joblib
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import GradientBoostingClassifier
import pandas as pd


df = pd.read_csv("cardio.csv", sep=";")
df["age"] = (df["age"] / 365).astype(int)
df["bmi"] = df["weight"] / ((df["height"] / 100) ** 2)
df.drop(["id", "height", "weight"], axis=1, inplace=True)

X = df.drop("cardio", axis=1)
y = df["cardio"]

scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)


model = GradientBoostingClassifier()
model.fit(X_scaled, y)


joblib.dump(model, "saved_models/model_gradient_boosting.pkl")
joblib.dump(scaler, "saved_models/scaler.pkl")
