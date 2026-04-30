import matplotlib.pyplot as plt
from car_price_prediction_improved import DATASET_PATH, train_or_load_models
import numpy as np

# R² chart
models = ['Linear Regression', 'Decision Tree', 'Random Forest']
scores = [0.507, 0.555, 0.650]
colors = ['#4C72B0', '#DD8452', '#55A868']

plt.figure(figsize=(8, 5))
bars = plt.bar(models, scores, color=colors)
plt.title('R² Score by Model')
plt.ylabel('R² Score')
plt.ylim(0, 1)

for bar, score in zip(bars, scores):
    plt.text(bar.get_x() + bar.get_width() / 2,
             score + 0.02,
             str(score),
             ha='center', fontsize=11)

plt.tight_layout()
plt.savefig('model_performance.png')
print('Saved model_performance.png')


# MAE chart
models = ['Linear Regression', 'Decision Tree', 'Random Forest']
mae_scores = [7812, 7352, 6299]
colors = ['#4C72B0', '#DD8452', '#55A868']

plt.figure(figsize=(8, 5))
bars = plt.bar(models, mae_scores, color=colors)
plt.title('MAE by Model (lower is better)')
plt.ylabel('Mean Absolute Error ($)')

for bar, val in zip(bars, mae_scores):
    plt.text(bar.get_x() + bar.get_width() / 2,
             val + 100,
             f'${val:,}',
             ha='center', fontsize=11)

plt.tight_layout()
plt.savefig('model_mae.png')
print('Saved model_mae.png')


# Feature importance chart
bundle = train_or_load_models(DATASET_PATH)

best_model = bundle["best_model"]
pipeline = best_model.regressor_
preprocessor = pipeline.named_steps["preprocessor"]
rf = pipeline.named_steps["regressor"]

cat_features = preprocessor.named_transformers_["categorical"]["onehot"].get_feature_names_out(["manufacturer", "model"])
num_features = ["car_age", "mileage", "accidents_or_damage", "one_owner", "mileage_per_year"]
all_features = list(cat_features) + num_features

importances = rf.feature_importances_

grouped = {"manufacturer": 0, "model": 0}
for name, score in zip(all_features, importances):
    if name.startswith("manufacturer_"):
        grouped["manufacturer"] += score
    elif name.startswith("model_"):
        grouped["model"] += score
    else:
        grouped[name] = score

names = list(grouped.keys())
values = list(grouped.values())
sorted_pairs = sorted(zip(values, names))
names = [p[1] for p in sorted_pairs]
values = [p[0] for p in sorted_pairs]

plt.figure(figsize=(8, 5))
plt.barh(names, values, color='#55A868')
plt.title('Feature Importance')
plt.xlabel('Importance Score')
plt.tight_layout()
plt.savefig('feature_importance.png')
print('Saved feature_importance.png')