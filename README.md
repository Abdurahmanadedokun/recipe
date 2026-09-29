# 🍽️ Recipe Popularity Prediction

Predicting which recipes will drive high user traffic using nutritional and categorical features.

![Python](https://img.shields.io/badge/Python-3.8%2B-blue)
![scikit-learn](https://img.shields.io/badge/scikit--learn-1.0%2B-orange)
![XGBoost](https://img.shields.io/badge/XGBoost-1.5%2B-green)
![License](https://img.shields.io/badge/License-MIT-lightgrey)

---

## 📋 Project Overview

The goal is to build a machine learning system that predicts which recipes on a recipe website are likely to become **popular (high traffic)** with users, based on nutritional content, recipe category, and serving size.

Two classification models — **Logistic Regression** (baseline) and **XGBoost** (final model) — were trained, tuned, and evaluated against a business-defined performance target.

---

## 🎯 Business Objective

The business requirement is to build a model that can identify **at least 80% of popular recipes** (recall ≥ 0.80 on the positive/high-traffic class). This ensures the recommendation system rarely misses a recipe that would have driven engagement, even if that means occasionally surfacing a recipe that turns out to be less popular.

---

## 📊 Dataset

- **Source file:** `recipe_site_traffic_2212.csv`
- **Size:** 947 recipes, 8 columns

| Column | Type | Description | Missing |
|---|---|---|---|
| `recipe` | int64 | Unique recipe ID | 0% |
| `calories` | float64 | Calories per serving | 5.5% (895/947) |
| `carbohydrate` | float64 | Carbohydrate content (g) | 5.5% (895/947) |
| `sugar` | float64 | Sugar content (g) | 5.5% (895/947) |
| `protein` | float64 | Protein content (g) | 5.5% (895/947) |
| `category` | object | Recipe category (11 categories, e.g. Breakfast, Chicken Breast, Dessert, Beverages) | 0% |
| `servings` | object → int | Number of servings (extracted from text) | 0% |
| `high_traffic` (target) | object → binary | `1` = High traffic (popular), `0` = Not high | 39.4% (574/947) |

**Target variable:** `high_traffic` — missing values were treated as `0` (Not High), based on the assumption that the label was only recorded when a recipe was flagged as popular.

**Missing data notes:** Nutritional columns (~5.5% missing) were imputed using the **median**. `servings` was stored as text and parsed into an integer. `high_traffic` had the highest missingness (39.4%) and was encoded as a binary category rather than dropped.

---

## 🔍 Methodology

### Data Validation & Cleaning
- Converted `category` to a categorical dtype and `servings` from text to integer via regex extraction.
- Imputed missing values in `calories`, `carbohydrate`, `sugar`, and `protein` with the **column median**.
- Encoded `high_traffic` as binary (`1` = High, `0` = Not High), treating missing labels as `Not High`.
- Applied **IQR-based outlier detection** on all nutritional features. Outliers (6–9% of records per feature) were identified but **retained** in the dataset, since they represent legitimate high-nutrient recipes (e.g., desserts, meat-heavy dishes) rather than data errors.

### Exploratory Data Analysis (Key Findings)
- **Nutrient distributions** (calories, carbohydrate, sugar, protein) are all **right-skewed**, with a long tail of high-nutrient recipes such as desserts and meat dishes.
- **Category distribution** is fairly balanced (7–11% share each); Breakfast, Chicken Breast, and Beverages are the most common categories, while One Dish Meal and Chicken are the least represented.
- **Calories vs. carbohydrate** show a clear positive relationship — high-carb recipes tend to be high-calorie.
- The **correlation heatmap** confirms strong positive correlations between calories, carbohydrate, and sugar, while protein contributes comparatively little to calorie variation.
- More recipes in the dataset are labeled high-traffic than not, indicating a dataset skewed toward popular recipes.

### Model Development
- **Features used:** `calories`, `carbohydrate`, `sugar`, `protein`, `servings` (5 numeric features).
- **Train/test split:** 80/20, stratified on the target, `random_state=42`.
- **Baseline model — Logistic Regression:** trained on `StandardScaler`-normalized features to establish an interpretable benchmark.
- **Final model — XGBoost (`XGBClassifier`):** tuned via `GridSearchCV` (3-fold CV, `scoring='recall'`) over:
  - `n_estimators`: [200, 300, 400]
  - `learning_rate`: [0.01, 0.05, 0.1]
  - `max_depth`: [4, 6, 8]
  - `subsample`: [0.7, 0.8, 1.0]
  - `colsample_bytree`: [0.7, 0.8, 1.0]
- **Threshold tuning:** the XGBoost decision threshold was swept from 0.10–0.90 to find the lowest threshold that still meets the 80% recall requirement; the optimal threshold was **0.10**.

---

## 📈 Model Evaluation

| Metric | Logistic Regression (baseline) | XGBoost (final) |
|---|---|---|
| **Recall** | 0.957 | 0.922 |
| **Precision** | 0.611 | 0.671 |
| **Accuracy** | 0.605 | 0.679 |
| **AUC** | 0.593 | 0.587 |

Both models exceed the 80% recall requirement. **XGBoost was selected as the final model** because it offers a better precision/recall balance — it catches slightly fewer popular recipes than Logistic Regression, but produces far fewer false positives, making its recommendations more reliable for deployment.

At the tuned threshold of **0.10**, the final XGBoost model achieves a recall of **1.000** and precision of **0.605** on the test set, guaranteeing no popular recipes are missed.

---

## 📉 Key Visualizations

- **Nutrient distribution histograms** (calories, carbohydrate, sugar, protein) — show right-skewed distributions with long tails driven by dessert and meat-heavy categories.
- **Category & high-traffic bar charts** — display recipe counts per category and the class balance of the target variable.
- **Calories vs. carbohydrate scatter plot + calories boxplot** — illustrate the positive relationship between carbs and calories, plus outlier spread.
- **Correlation heatmap** — highlights strong positive correlation among calories, carbohydrate, and sugar.
- **Category composition pie chart** — shows the balanced (7–11%) share of each recipe category.
- **Confusion matrix (XGBoost)** — visualizes true/false positive and negative counts on the test set.
- **Precision–Recall curve (XGBoost)** — shows the recall/precision trade-off across probability thresholds, confirming strong recall at moderate precision.

---

## 💡 Business Recommendations

1. **Start using the model to recommend recipes** — deploy it in the app/website to highlight recipes likely to perform well and boost engagement.
2. **Test the model with real users before full rollout** — run an A/B test comparing the current feed to the model-driven feed, measuring clicks, saves, time-on-page, and cooking attempts.
3. **Collect richer recipe data** — supplement nutrition and category data with ingredients, cooking steps, descriptions, photos, and user ratings to improve future model versions.
4. **Monitor model performance regularly** — build a dashboard tracking recall, accuracy, and engagement trends to catch performance drift early.
5. **Retrain the model on a regular cadence** (e.g., monthly/quarterly) — keep predictions aligned with evolving recipe catalogs and user tastes.
6. **Improve data quality at the source** — encourage consistent, complete data entry for nutrition, category, and servings fields to reduce reliance on imputation.

---

## 🛠️ Installation & Usage

```bash
# Clone the repository
git clone https://github.com/<your-username>/<repo-name>.git
cd <repo-name>

# Create and activate a virtual environment
python -m venv venv
source venv/bin/activate        # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt

# Launch Jupyter and open the notebook
jupyter notebook notebook.ipynb
```

---

## 📁 Project Structure

```
recipe-popularity-prediction/
│
├── notebook.ipynb                    # Full analysis: validation, EDA, modeling, evaluation
├── recipe_site_traffic_2212.csv      # Raw dataset (947 recipes)
├── requirements.txt                  # Python dependencies
├── README.md                         # Project documentation
└── LICENSE                           # MIT License
```

---

## 📦 Requirements

- pandas
- numpy
- matplotlib
- seaborn
- scikit-learn
- xgboost
- jupyter / notebook

See [`requirements.txt`](./requirements.txt) for version constraints.

---

## 🔮 Future Improvements

- Incorporate richer features: ingredients, cooking instructions, images, and user reviews.
- Explore additional models (Random Forest, LightGBM, neural networks) for comparison.
- Address the 39% missingness in `high_traffic` with a more rigorous missing-data strategy or by collecting complete labels.
- Build an automated retraining and monitoring pipeline for production deployment.
- Experiment with feature engineering (e.g., nutrient ratios, category-based encodings) to improve AUC.

---

## 📝 License

This project is licensed under the **MIT License** — see the [LICENSE](./LICENSE) file for details.

---

## 👤 Author

**ADEDOKUN Abdurahman**
