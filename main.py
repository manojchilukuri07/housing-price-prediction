from pyexpat import features
from sklearn.preprocessing import OneHotEncoder

import pandas as pd

df = pd.read_csv("train.csv")
numerical = df.select_dtypes(include=["number"]).columns

for col in numerical:
    df[col] = df[col].fillna(df[col].median())

categorical = df.select_dtypes(include=["str", "object"]).columns

for col in categorical:
    df[col] = df[col].fillna("Missing")

print(df.isnull().sum().sum())

# Check outliers in GrLivArea
Q1 = df["GrLivArea"].quantile(0.25)
Q3 = df["GrLivArea"].quantile(0.75)

IQR = Q3 - Q1

lower_limit = Q1 - 1.5 * IQR
upper_limit = Q3 + 1.5 * IQR

outliers = df[
    (df["GrLivArea"] < lower_limit) |
    (df["GrLivArea"] > upper_limit)
]

print("Number of GrLivArea outliers:", len(outliers))
df["TotalSF"] = (
    df["TotalBsmtSF"] +
    df["1stFlrSF"] +
    df["2ndFlrSF"]
)
df["HouseAge"] = 2010 - df["YearBuilt"]
df["TotalBathrooms"] = (
    df["FullBath"] +
    0.5 * df["HalfBath"] +
    df["BsmtFullBath"] +
    0.5 * df["BsmtHalfBath"]
)
df["TotalPorchSF"] = (
    df["OpenPorchSF"] +
    df["EnclosedPorch"] +
    df["3SsnPorch"] +
    df["ScreenPorch"] +
    df["WoodDeckSF"]
)
df["TotalOutdoorSF"] = (
    df["TotalPorchSF"] +
    df["PoolArea"]
)
features = [
    "TotalSF",
    "TotalOutdoorSF",
    "TotalBathrooms",
    "OverallQual",
    "YearBuilt",
    "HouseAge",
    "GarageArea",
    "FullBath",
    "BedroomAbvGr",
    "TotRmsAbvGrd",
    "MasVnrArea",
    "Fireplaces",
    "1stFlrSF",
    "2ndFlrSF",
    "Neighborhood",
    "TotalPorchSF"
]
X = pd.get_dummies(df[features], columns=["Neighborhood"])
print("Number of features:", len(features))
print("Features:", features)
y = df["SalePrice"]

from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)
print(X_train.shape)
print(X_test.shape)
from sklearn.linear_model import LinearRegression

model = LinearRegression()

model.fit(X_train, y_train)
print("Model training completed!")
y_pred = model.predict(X_test)

print(y_pred[:5])
from sklearn.metrics import mean_squared_error, r2_score
import numpy as np

rmse = np.sqrt(mean_squared_error(y_test, y_pred))

r2 = r2_score(y_test, y_pred)

print("RMSE:", rmse)
print("R2 Score:", r2)

from sklearn.ensemble import RandomForestRegressor

rf_model = RandomForestRegressor(
    n_estimators=500,
    max_depth=20,
    min_samples_split=2,
    random_state=42
)

rf_model.fit(X_train, y_train)

rf_pred = rf_model.predict(X_test)

rf_rmse = np.sqrt(mean_squared_error(y_test, rf_pred))
rf_r2 = r2_score(y_test, rf_pred)

print("Random Forest RMSE:", rf_rmse)
print("Random Forest R2:", rf_r2)
importance = rf_model.feature_importances_

for feature, value in zip(features, importance):
    print(feature, value)
from xgboost import XGBRegressor

xgb_model = XGBRegressor(
    n_estimators=1000,
    learning_rate=0.03,
    max_depth=3,
    subsample=0.8,
    colsample_bytree=0.8,
    random_state=42
)

xgb_model.fit(X_train, y_train)

xgb_pred = xgb_model.predict(X_test)

xgb_rmse = np.sqrt(mean_squared_error(y_test, xgb_pred))
xgb_r2 = r2_score(y_test, xgb_pred)

print("XGBoost RMSE:", xgb_rmse)
print("XGBoost R2:", xgb_r2)
Q1 = df["GrLivArea"].quantile(0.25)
Q3 = df["GrLivArea"].quantile(0.75)

IQR = Q3 - Q1

lower_limit = Q1 - 1.5 * IQR
upper_limit = Q3 + 1.5 * IQR

outliers = df[
    (df["GrLivArea"] < lower_limit) |
    (df["GrLivArea"] > upper_limit)
]

print("Outliers:", len(outliers))
print(outliers[["GrLivArea", "SalePrice"]].sort_values(
    by="GrLivArea", ascending=False
))
import joblib

joblib.dump(xgb_model, "models/xgb_model.pkl")
joblib.dump(X.columns.tolist(), "models/model_columns.pkl")

print("Model saved successfully!")
import os

print("Model exists:", os.path.exists("models/xgb_model.pkl"))
print("Columns exist:", os.path.exists("models/model_columns.pkl"))