from pathlib import Path
import os
# Reproducibility
SEED = 42
TEST_SIZE = 0.2

# Target
TARGET = "PRICE"

# Default local dataset path (you can override via CLI)
DEFAULT_DATA_PATH = Path("data/raw/Barcelona_2018.csv")


# Features used in your story
FEATURES_LR_BENCH = ["CONSTRUCTEDAREA"]
FEATURES_TREE_2 = ["CONSTRUCTEDAREA", "DISTANCE_TO_CITY_CENTER"]

FEATURES_RF10 = [
    "CONSTRUCTEDAREA",
    "DISTANCE_TO_CITY_CENTER",
    "ROOMNUMBER",
    "LATITUDE",
    "LONGITUDE",
    "CADASTRALQUALITYID",
    "DISTANCE_TO_DIAGONAL",
    "HASPARKINGSPACE",
    "HASDOORMAN",
    "HASGARDEN",
]

# Your final "top 5" (adjust if your final selection differs)
FEATURES_RF5 = [
    "CONSTRUCTEDAREA",
    "LATITUDE",
    "LONGITUDE",
    "DISTANCE_TO_CITY_CENTER",
    "DISTANCE_TO_DIAGONAL",
]

FINAL_MODEL_NAME = "RandomForest (5 vars) + log(y)"

# --- Plot labels/order for model comparison figures ---
MODEL_LABELS = {
    "Benchmark Dummy (median)": "Benchmark\n(Dummy median)",
    "LR (area)": "LR\n(area)",
    "DecisionTree (2 vars, depth=2)": "Decision Tree\n(2 features)",
    "RandomForest (10 vars)": "Random Forest\n(10 features)",
    "RandomForest (5 vars) + log(y)": "Random Forest\n(5 features)",
}


MODEL_ORDER = [
    "Benchmark Dummy (median)",
    "LR (area)",
    "DecisionTree (2 vars, depth=2)",
    "RandomForest (10 vars)",
    "RandomForest (5 vars) + log(y)",
]
