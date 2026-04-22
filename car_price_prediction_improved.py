"""
Used Car Price Prediction

This program reads the cleaned car dataset, trains three models
(Linear Regression, Decision Tree, and Random Forest), compares them,
and then uses the best one to predict the price of a used car.

I also kept a fallback model that depends on the numeric features only.
That part helps when the user enters a manufacturer or model that was
not present in the training data.
"""

from __future__ import annotations

from difflib import get_close_matches
from pathlib import Path
import warnings

import joblib
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer, TransformedTargetRegressor
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor


# Keeping the random state fixed makes the results repeatable.
RANDOM_STATE = 42

# The program expects the dataset to be in the same folder as this file.
DATASET_PATH = Path(__file__).with_name("car dataset cleaned.xlsx")

# The trained models are saved here so the script does not retrain every time.
MODEL_CACHE_PATH = Path(__file__).with_name("car_price_models.joblib")

# These are the columns used in the project.
REQUIRED_COLUMNS = [
    "manufacturer",
    "model",
    "car_age",
    "mileage",
    "accidents_or_damage",
    "one_owner",
    "price",
]

# Numeric features used by the models.
NUMERIC_FEATURES = [
    "car_age",
    "mileage",
    "accidents_or_damage",
    "one_owner",
    "mileage_per_year",
]

# Text features used by the models.
CATEGORICAL_FEATURES = ["manufacturer", "model"]

# The original dataset is very large, so I use a sample to keep training practical.
MAX_ROWS_TO_USE = 300_000


def normalize_text(value: str) -> str:
    """
    Clean text values by removing extra spaces and changing letters to lowercase.
    """
    return " ".join(str(value).strip().lower().split())


def add_engineered_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Add one more useful feature to the dataset.

    mileage_per_year helps the model understand how heavily the car
    was used compared with its age.
    """
    df = df.copy()
    safe_age = df["car_age"].clip(lower=1)
    df["mileage_per_year"] = df["mileage"] / safe_age
    return df


def load_and_clean_data(dataset_path: Path) -> pd.DataFrame:
    """
    Load the dataset and apply the main cleaning steps.
    """
    if not dataset_path.exists():
        raise FileNotFoundError(
            f"Dataset not found: {dataset_path}\n"
            "Place 'car dataset cleaned.xlsx' in the same folder as this file."
        )

    print("\nStep 1: Loading and cleaning the data...")
    df = pd.read_excel(dataset_path)

    missing_columns = [column for column in REQUIRED_COLUMNS if column not in df.columns]
    if missing_columns:
        raise ValueError(f"Missing required columns: {missing_columns}")

    df = df[REQUIRED_COLUMNS].copy()

    # Cleaning the text columns makes similar values consistent.
    df["manufacturer"] = df["manufacturer"].apply(normalize_text)
    df["model"] = df["model"].apply(normalize_text)

    # Converting the numeric columns makes sure calculations work correctly.
    numeric_columns = [
        "car_age",
        "mileage",
        "accidents_or_damage",
        "one_owner",
        "price",
    ]
    for column in numeric_columns:
        df[column] = pd.to_numeric(df[column], errors="coerce")

    # Rows with missing values are removed to keep training cleaner.
    df = df.dropna()

    # These filters remove impossible values.
    df = df[df["car_age"] >= 0]
    df = df[df["mileage"] >= 0]
    df = df[df["price"] > 0]

    # Extreme price values are trimmed so they do not distort the model too much.
    lower_price = df["price"].quantile(0.01)
    upper_price = df["price"].quantile(0.99)
    df = df[(df["price"] >= lower_price) & (df["price"] <= upper_price)]

    df = add_engineered_features(df)

    if len(df) > MAX_ROWS_TO_USE:
        print(
            f"Cleaned rows found: {len(df):,}. "
            f"A sample of {MAX_ROWS_TO_USE:,} rows will be used for training."
        )
        df = df.sample(MAX_ROWS_TO_USE, random_state=RANDOM_STATE).reset_index(drop=True)
    else:
        print(f"Cleaned rows found: {len(df):,}.")

    return df


def create_full_preprocessor() -> ColumnTransformer:
    """
    Prepare the columns for the main models.

    Text columns are one-hot encoded.
    Numeric columns are filled with the median if needed.
    """
    categorical_pipeline = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="most_frequent")),
            ("onehot", OneHotEncoder(handle_unknown="ignore")),
        ]
    )

    numeric_pipeline = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="median")),
        ]
    )

    return ColumnTransformer(
        transformers=[
            ("categorical", categorical_pipeline, CATEGORICAL_FEATURES),
            ("numeric", numeric_pipeline, NUMERIC_FEATURES),
        ]
    )


def create_numeric_only_preprocessor() -> ColumnTransformer:
    """
    Prepare only the numeric columns for the fallback model.
    """
    numeric_pipeline = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="median")),
        ]
    )

    return ColumnTransformer(
        transformers=[
            ("numeric", numeric_pipeline, NUMERIC_FEATURES),
        ]
    )


def build_model(preprocessor: ColumnTransformer, regressor) -> TransformedTargetRegressor:
    """
    Build a complete model pipeline.

    The target price is trained in log form because that usually gives
    more stable results with price data.
    """
    pipeline = Pipeline(
        steps=[
            ("preprocessor", preprocessor),
            ("regressor", regressor),
        ]
    )

    return TransformedTargetRegressor(
        regressor=pipeline,
        func=np.log1p,
        inverse_func=np.expm1,
    )


def get_main_model_candidates() -> dict[str, object]:
    """
    Return the three models used in the comparison.
    """
    return {
        "Linear Regression": LinearRegression(),
        "Decision Tree": DecisionTreeRegressor(
            random_state=RANDOM_STATE,
            max_depth=12,
            min_samples_split=40,
            min_samples_leaf=15,
        ),
        "Random Forest": RandomForestRegressor(
            n_estimators=80,
            random_state=RANDOM_STATE,
            max_depth=18,
            min_samples_split=10,
            min_samples_leaf=4,
            n_jobs=-1,
        ),
    }


def calculate_metrics(y_true: pd.Series, y_pred: np.ndarray) -> dict[str, float]:
    """
    Calculate MAE, RMSE, and R2 for one model.
    """
    y_pred = np.maximum(y_pred, 0)

    mae = mean_absolute_error(y_true, y_pred)
    rmse = np.sqrt(mean_squared_error(y_true, y_pred))
    r2 = r2_score(y_true, y_pred)

    return {"MAE": mae, "RMSE": rmse, "R2": r2}


def train_main_models(df: pd.DataFrame) -> dict:
    """
    Train Linear Regression, Decision Tree, and Random Forest,
    then keep the one with the lowest MAE.
    """
    print("\nStep 2: Training the three main models...")

    X = df.drop(columns="price")
    y = df["price"]

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.20,
        random_state=RANDOM_STATE,
    )

    trained_models = {}
    comparison_rows = []

    for model_name, regressor in get_main_model_candidates().items():
        model = build_model(create_full_preprocessor(), regressor)

        print(f"Training {model_name}...")
        model.fit(X_train, y_train)
        predictions = model.predict(X_test)
        metrics = calculate_metrics(y_test, predictions)

        trained_models[model_name] = model
        comparison_rows.append(
            {
                "Model": model_name,
                "MAE": metrics["MAE"],
                "RMSE": metrics["RMSE"],
                "R2": metrics["R2"],
            }
        )

    results_df = pd.DataFrame(comparison_rows).sort_values("MAE").reset_index(drop=True)
    best_model_name = results_df.loc[0, "Model"]
    best_model = trained_models[best_model_name]

    best_metrics = {
        "MAE": float(results_df.loc[0, "MAE"]),
        "RMSE": float(results_df.loc[0, "RMSE"]),
        "R2": float(results_df.loc[0, "R2"]),
    }

    print("\nModel comparison:")
    print(results_df.to_string(index=False))
    print(f"\nBest model: {best_model_name}")

    return {
        "X_train": X_train,
        "X_test": X_test,
        "y_train": y_train,
        "y_test": y_test,
        "results_df": results_df,
        "trained_models": trained_models,
        "best_model_name": best_model_name,
        "best_model": best_model,
        "best_metrics": best_metrics,
    }


def train_fallback_model(
    X_train: pd.DataFrame,
    X_test: pd.DataFrame,
    y_train: pd.Series,
    y_test: pd.Series,
) -> dict:
    """
    Train a fallback Random Forest using only numeric features.

    This model is used when a completely new manufacturer is entered.
    """
    print("\nStep 3: Training the fallback model...")

    fallback_regressor = RandomForestRegressor(
        n_estimators=80,
        random_state=RANDOM_STATE,
        max_depth=18,
        min_samples_split=10,
        min_samples_leaf=4,
        n_jobs=-1,
    )

    fallback_model = build_model(create_numeric_only_preprocessor(), fallback_regressor)
    fallback_model.fit(X_train[NUMERIC_FEATURES], y_train)

    fallback_predictions = fallback_model.predict(X_test[NUMERIC_FEATURES])
    fallback_metrics = calculate_metrics(y_test, fallback_predictions)

    print(
        "Fallback model metrics: "
        f"MAE = {fallback_metrics['MAE']:.2f}, "
        f"RMSE = {fallback_metrics['RMSE']:.2f}, "
        f"R2 = {fallback_metrics['R2']:.4f}"
    )

    return {
        "model": fallback_model,
        "metrics": fallback_metrics,
    }


def create_lookup_helpers(df: pd.DataFrame) -> tuple[set[str], dict[str, set[str]]]:
    """
    Create helper collections for manufacturer and model lookup.
    """
    known_manufacturers = set(df["manufacturer"].unique())

    known_models_by_make = (
        df.groupby("manufacturer")["model"]
        .apply(lambda values: set(values.unique()))
        .to_dict()
    )

    return known_manufacturers, known_models_by_make


def save_model_bundle(dataset_path: Path, bundle: dict) -> None:
    """
    Save the trained models and helper data.
    """
    bundle["dataset_path"] = str(dataset_path.resolve())
    bundle["dataset_mtime"] = dataset_path.stat().st_mtime
    joblib.dump(bundle, MODEL_CACHE_PATH)
    print(f"\nSaved trained models to: {MODEL_CACHE_PATH}")


def load_cached_model_bundle(dataset_path: Path) -> dict | None:
    """
    Load saved models if the dataset has not changed.
    """
    if not MODEL_CACHE_PATH.exists():
        return None

    try:
        bundle = joblib.load(MODEL_CACHE_PATH)
    except Exception:
        return None

    same_file = bundle.get("dataset_path") == str(dataset_path.resolve())
    same_version = bundle.get("dataset_mtime") == dataset_path.stat().st_mtime

    if same_file and same_version:
        return bundle

    return None


def train_or_load_models(dataset_path: Path) -> dict:
    """
    Load saved models if possible. Otherwise train them from the dataset.
    """
    print("Used Car Price Prediction")
    print("-" * 25)

    cached_bundle = load_cached_model_bundle(dataset_path)
    if cached_bundle is not None:
        print("\nSaved models were found, so retraining is skipped.")
        return cached_bundle

    df = load_and_clean_data(dataset_path)
    known_manufacturers, known_models_by_make = create_lookup_helpers(df)

    main_training = train_main_models(df)
    fallback_training = train_fallback_model(
        main_training["X_train"],
        main_training["X_test"],
        main_training["y_train"],
        main_training["y_test"],
    )

    bundle = {
        "known_manufacturers": known_manufacturers,
        "known_models_by_make": known_models_by_make,
        "results_df": main_training["results_df"],
        "trained_models": main_training["trained_models"],
        "best_model_name": main_training["best_model_name"],
        "best_model": main_training["best_model"],
        "best_metrics": main_training["best_metrics"],
        "fallback_model": fallback_training["model"],
        "fallback_metrics": fallback_training["metrics"],
    }

    save_model_bundle(dataset_path, bundle)
    return bundle


def get_yes_no(prompt: str) -> int:
    """
    Ask the user a yes or no question.
    """
    while True:
        answer = input(prompt).strip().lower()
        if answer in {"yes", "y"}:
            return 1
        if answer in {"no", "n"}:
            return 0
        print("Please enter yes or no.")


def get_non_negative_float(prompt: str) -> float:
    """
    Ask the user for a number that cannot be negative.
    """
    while True:
        try:
            value = float(input(prompt).strip())
            if value < 0:
                print("Please enter a value that is zero or greater.")
                continue
            return value
        except ValueError:
            print("Please enter a numeric value.")


def suggest_similar_value(user_value: str, known_values: set[str]) -> str | None:
    """
    Suggest a close match if the input looks similar to a known value.
    """
    matches = get_close_matches(user_value, sorted(known_values), n=1, cutoff=0.80)
    return matches[0] if matches else None


def resolve_manufacturer(user_value: str, known_manufacturers: set[str]) -> tuple[str, bool]:
    """
    Check whether the entered manufacturer exists in the dataset.
    """
    manufacturer = normalize_text(user_value)

    if manufacturer in known_manufacturers:
        return manufacturer, True

    suggestion = suggest_similar_value(manufacturer, known_manufacturers)
    if suggestion:
        print(f'Did you mean "{suggestion}"?')
        if get_yes_no("Use this manufacturer? (yes/no): "):
            return suggestion, True

    return manufacturer, False


def resolve_model(user_value: str, known_models: set[str]) -> tuple[str, bool]:
    """
    Check whether the entered model exists for the selected manufacturer.
    """
    model = normalize_text(user_value)

    if model in known_models:
        return model, True

    suggestion = suggest_similar_value(model, known_models)
    if suggestion:
        print(f'Did you mean "{suggestion}"?')
        if get_yes_no("Use this model? (yes/no): "):
            return suggestion, True

    return model, False


def collect_user_input(bundle: dict) -> tuple[pd.DataFrame, str, dict[str, float]]:
    """
    Read the user input and decide which prediction path should be used.
    """
    print("\nStep 4: Enter car information for prediction.")

    known_manufacturers = bundle["known_manufacturers"]
    known_models_by_make = bundle["known_models_by_make"]

    manufacturer_input = input("Manufacturer: ")
    manufacturer, manufacturer_is_known = resolve_manufacturer(
        manufacturer_input,
        known_manufacturers,
    )

    model_input = input("Model: ")
    if manufacturer_is_known:
        model, model_is_known = resolve_model(
            model_input,
            known_models_by_make[manufacturer],
        )
    else:
        model = normalize_text(model_input)
        model_is_known = False

    car_age = get_non_negative_float("Car age: ")
    mileage = get_non_negative_float("Mileage: ")
    accidents_or_damage = get_yes_no("Accidents or damage? (yes/no): ")
    one_owner = get_yes_no("One owner? (yes/no): ")

    user_input = pd.DataFrame(
        [
            {
                "manufacturer": manufacturer,
                "model": model,
                "car_age": car_age,
                "mileage": mileage,
                "accidents_or_damage": accidents_or_damage,
                "one_owner": one_owner,
            }
        ]
    )
    user_input = add_engineered_features(user_input)

    if manufacturer_is_known and model_is_known:
        strategy_message = (
            "Prediction mode: best main model\n"
            "Reason: the manufacturer and model were both found in the dataset."
        )
        error_metrics = bundle["best_metrics"]
    elif manufacturer_is_known and not model_is_known:
        strategy_message = (
            "Prediction mode: best main model\n"
            "Reason: the manufacturer is known, and the model is new. "
            "The encoder can still handle this case."
        )
        error_metrics = bundle["best_metrics"]
    else:
        strategy_message = (
            "Prediction mode: fallback model\n"
            "Reason: the manufacturer is new, so the prediction is based on "
            "the numeric features only."
        )
        error_metrics = bundle["fallback_metrics"]

    return user_input, strategy_message, error_metrics


def predict_price(bundle: dict, user_input: pd.DataFrame) -> float:
    """
    Predict the final price using the correct model.
    """
    manufacturer = user_input.loc[0, "manufacturer"]

    if manufacturer in bundle["known_manufacturers"]:
        prediction = bundle["best_model"].predict(user_input)[0]
    else:
        prediction = bundle["fallback_model"].predict(user_input[NUMERIC_FEATURES])[0]

    return max(float(prediction), 0.0)


def show_prediction(
    predicted_price: float,
    strategy_message: str,
    error_metrics: dict[str, float],
    best_model_name: str,
) -> None:
    """
    Show the predicted price and a simple error range.
    """
    estimated_low = max(predicted_price - error_metrics["MAE"], 0)
    estimated_high = predicted_price + error_metrics["MAE"]

    print("\nStep 5: Prediction result")
    print(f"Best training model: {best_model_name}")
    print(strategy_message)
    print(f"\nEstimated car price: ${predicted_price:,.2f}")
    print(
        f"Expected range using MAE: "
        f"${estimated_low:,.2f} to ${estimated_high:,.2f}"
    )


def main() -> None:
    """
    Run the full program from training to final prediction.
    """
    warnings.filterwarnings("ignore", category=UserWarning)

    bundle = train_or_load_models(DATASET_PATH)
    user_input, strategy_message, error_metrics = collect_user_input(bundle)
    predicted_price = predict_price(bundle, user_input)
    show_prediction(
        predicted_price,
        strategy_message,
        error_metrics,
        bundle["best_model_name"],
    )


if __name__ == "__main__":
    main()
