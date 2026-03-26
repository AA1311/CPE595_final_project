import pandas as pd
import numpy as np

from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder
from sklearn.pipeline import Pipeline
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

from sklearn.linear_model import LinearRegression
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor


#  Load and clean dataset
df = pd.read_excel("car dataset cleaned.xlsx")

# Clean categorical columns
df["manufacturer"] = df["manufacturer"].astype(str).str.strip()
df["model"] = df["model"].astype(str).str.strip()

# Clean numeric columns
df["car_age"] = pd.to_numeric(df["car_age"], errors="coerce")
df["mileage"] = pd.to_numeric(df["mileage"], errors="coerce")
df["accidents_or_damage"] = pd.to_numeric(df["accidents_or_damage"], errors="coerce")
df["one_owner"] = pd.to_numeric(df["one_owner"], errors="coerce")
df["price"] = pd.to_numeric(df["price"], errors="coerce")

# Keep only needed columns and drop missing rows
df = df[
    [
        "manufacturer",
        "model",
        "car_age",
        "mileage",
        "accidents_or_damage",
        "one_owner",
        "price",
    ]
].dropna()

# Remove invalid values
df = df[df["car_age"] >= 0]
df = df[df["mileage"] >= 0]
df = df[df["price"] > 0]

# Remove extreme price outliers
lower_price = df["price"].quantile(0.01)
upper_price = df["price"].quantile(0.99)
df = df[(df["price"] >= lower_price) & (df["price"] <= upper_price)]

#  Build lookup helpers
valid_manufacturers = set(df["manufacturer"].unique())

valid_models_by_make = (
    df.groupby("manufacturer")["model"]
    .apply(lambda x: set(x.unique()))
    .to_dict()
)

#  Features and target
X = df.drop("price", axis=1)
y = df["price"]

categorical_cols = ["manufacturer", "model"]

preprocessor = ColumnTransformer(
    transformers=[
        ("cat", OneHotEncoder(handle_unknown="ignore"), categorical_cols)
    ],
    remainder="passthrough"
)


#  Train/test split
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)


#  Define models
models = {
    "Linear Regression": LinearRegression(),
    "Decision Tree": DecisionTreeRegressor(
        random_state=42,
        max_depth=10,
        min_samples_split=50,
        min_samples_leaf=20
    ),
    "Random Forest": RandomForestRegressor(
        n_estimators=50,
        random_state=42,
        max_depth=15,
        min_samples_split=20,
        min_samples_leaf=10,
        n_jobs=-1
    )
}

trained_models = {}
results = []


# Train and evaluate

for model_name, regressor in models.items():
    pipeline = Pipeline([
        ("preprocessor", preprocessor),
        ("regressor", regressor)
    ])

    print(f"\nTraining {model_name}...")
    pipeline.fit(X_train, y_train)

    y_pred = pipeline.predict(X_test)

    mae = mean_absolute_error(y_test, y_pred)
    rmse = np.sqrt(mean_squared_error(y_test, y_pred))
    r2 = r2_score(y_test, y_pred)

    trained_models[model_name] = pipeline
    results.append([model_name, mae, rmse, r2])


#  Print model comparison

results_df = pd.DataFrame(results, columns=["Model", "MAE", "RMSE", "R²"])

print("\nModel Comparison:")
print(results_df.to_string(index=False))


#  Helper for yes/no input
def get_yes_no(prompt):
    while True:
        answer = input(prompt).strip().lower()
        if answer in ["yes", "y"]:
            return 1
        elif answer in ["no", "n"]:
            return 0
        else:
            print("Please enter yes or no.")


#  Get validated user input

print("\nEnter car information to predict price:")

manufacturer = input("Manufacturer: ").strip()
if manufacturer not in valid_manufacturers:
    print("Error: Invalid manufacturer. Please enter a manufacturer from the dataset.")
    raise SystemExit

model_name = input("Model: ").strip()
if model_name not in valid_models_by_make[manufacturer]:
    print(f"Error: Invalid model for {manufacturer}. Please enter a valid model from the dataset.")
    raise SystemExit

try:
    car_age = float(input("Car age: ").strip())
    mileage = float(input("Mileage: ").strip())
except ValueError:
    print("Error: Car age and mileage must be numeric.")
    raise SystemExit

if car_age < 0:
    print("Error: Car age cannot be negative.")
    raise SystemExit

if mileage < 0:
    print("Error: Mileage cannot be negative.")
    raise SystemExit

accidents_or_damage = get_yes_no("Accidents or damage? (yes/no): ")
one_owner = get_yes_no("One owner? (yes/no): ")

user_input = pd.DataFrame([{
    "manufacturer": manufacturer,
    "model": model_name,
    "car_age": car_age,
    "mileage": mileage,
    "accidents_or_damage": accidents_or_damage,
    "one_owner": one_owner
}])


#  Predict with all models

print("\nPredicted Prices:")
for model_name, pipeline in trained_models.items():
    predicted_price = pipeline.predict(user_input)[0]

    # just in case any model predicts a negative price
    predicted_price = max(predicted_price, 0)

    print(f"{model_name}: ${predicted_price:,.2f}")