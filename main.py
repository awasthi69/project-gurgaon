
import os
import pandas as pd
import numpy as np
import joblib

from sklearn.model_selection import StratifiedShuffleSplit
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.ensemble import RandomForestRegressor

MODEL_FILE = "model.pkl"
PIPELINE_FILE = "pipeline.pkl"
INPUT_FILE = "input.csv"
OUTPUT_FILE = "output.csv"


def build_pipeline(num_attribs, cat_attribs):
    num_pipeline = Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", StandardScaler())
    ])

    cat_pipeline = Pipeline([
        ("onehot", OneHotEncoder(handle_unknown="ignore"))
    ])

    return ColumnTransformer([
        ("num", num_pipeline, num_attribs),
        ("cat", cat_pipeline, cat_attribs)
    ])


# TRAINING PHASE
if not os.path.exists(MODEL_FILE) or not os.path.exists(PIPELINE_FILE):

    housing = pd.read_csv("housing.csv")

    housing["income_cat"] = pd.cut(
        housing["median_income"],
        bins=[0.0, 1.5, 3.0, 4.5, 6.0, np.inf],
        labels=[1, 2, 3, 4, 5]
    )

    split = StratifiedShuffleSplit(
        n_splits=1, test_size=0.2, random_state=42
    )

    train_index, _ = next(
        split.split(housing, housing["income_cat"])
    )

    housing = housing.iloc[train_index].drop(
        "income_cat", axis=1
    )

    housing_labels = housing["median_house_value"].copy()
    housing_features = housing.drop(
        "median_house_value", axis=1
    )

    num_attribs = housing_features.drop(
        "ocean_proximity", axis=1
    ).columns.tolist()

    cat_attribs = ["ocean_proximity"]

    pipeline = build_pipeline(num_attribs, cat_attribs)

    housing_prepared = pipeline.fit_transform(housing_features)

    model = RandomForestRegressor(random_state=42)
    model.fit(housing_prepared, housing_labels)

    joblib.dump(model, MODEL_FILE)
    joblib.dump(pipeline, PIPELINE_FILE)

    print("Model trained and saved.")

else:
    # LOAD EXISTING MODEL
    model = joblib.load(MODEL_FILE)
    pipeline = joblib.load(PIPELINE_FILE)

    print("Existing model and pipeline loaded.")


# CREATE A SAMPLE INPUT FILE IF IT DOES NOT EXIST
if not os.path.exists(INPUT_FILE):
    sample_data = pd.DataFrame([{
        "longitude": -122.23,
        "latitude": 37.88,
        "housing_median_age": 41.0,
        "total_rooms": 880.0,
        "total_bedrooms": 129.0,
        "population": 322.0,
        "households": 126.0,
        "median_income": 8.3252,
        "ocean_proximity": "NEAR BAY"
    }])

    sample_data.to_csv(INPUT_FILE, index=False)
    print("Sample input.csv created.")


# INFERENCE PHASE
input_data = pd.read_csv(INPUT_FILE)

# Transform input using the trained pipeline
transformed_input = pipeline.transform(input_data)

# Predict house prices
predictions = model.predict(transformed_input)

input_data["median_house_value"] = predictions

# Save output
input_data.to_csv(OUTPUT_FILE, index=False)

print("Inference complete. Results saved to", OUTPUT_FILE)
print(input_data)
