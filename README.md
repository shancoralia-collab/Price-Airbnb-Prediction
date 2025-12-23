# 🏠 Airbnb Price Prediction

A machine learning project to predict Airbnb rental prices based on property characteristics.

##  Project Description

This project aims to predict the **logarithm of rental price** (`log_price`) for Airbnb listings using various features such as property type, amenities, geographic location, and user reviews.

###  Objective

Develop a prediction model capable of accurately estimating Airbnb rental prices from textual and numerical data, by transforming categorical and textual variables into features exploitable by machine learning algorithms.

## Data

### Input Variables

| Variable | Type | Description |
|----------|------|-------------|
| `id` | Numeric | Unique listing identifier |
| `property_type` | Categorical | Property type (apartment, house, etc.) |
| `room_type` | Categorical | Room type (entire home, private room, etc.) |
| `amenities` | Text | List of available amenities |
| `accommodates` | Numeric | Number of guests accommodated |
| `bathrooms` | Numeric | Number of bathrooms |
| `bed_type` | Categorical | Type of bed |
| `cancellation_policy` | Categorical | Cancellation policy |
| `cleaning_fee` | Boolean | Cleaning fee present |
| `city` | Categorical | City of the listing |
| `description` | Text | Property description |
| `first_review` | Date | Date of first review |
| `host_has_profile_pic` | Boolean | Host has profile picture |
| `host_identity_verified` | Boolean | Host identity verified |
| `host_response_rate` | Numeric | Host response rate |
| `host_since` | Date | Host registration date |
| `instant_bookable` | Boolean | Instant booking available |
| `last_review` | Date | Date of last review |
| `latitude` | Numeric | Geographic latitude |
| `longitude` | Numeric | Geographic longitude |
| `name` | Text | Listing name |
| `neighbourhood` | Categorical | Neighborhood |
| `number_of_reviews` | Numeric | Number of reviews |
| `review_scores_rating` | Numeric | Average review score |
| `zipcode` | Categorical | Zip code |
| `bedrooms` | Numeric | Number of bedrooms |
| `beds` | Numeric | Number of beds |

### Target Variable

- **`log_price`** : Natural logarithm of rental price

>  **Important**: The model must predict the **logarithm of the price**, not the price directly. If your model predicts the raw price, you must apply `np.log()` before generating the submission file.

##  Installation

### Prerequisites

- Python 3.8+
- pip

### Install Dependencies

```bash
pip install -r requirements.txt
```

### Main Dependencies

```
pandas
numpy
scikit-learn
matplotlib
seaborn
folium
xgboost / lightgbm
```

##  Usage

### 1. Data Exploration

```python
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Load data
df_train = pd.read_csv('data/airbnb_train.csv')

# Descriptive statistics
print(df_train.describe())

# Correlation matrix
correlation_matrix = df_train.corr()
sns.heatmap(correlation_matrix, annot=True)
```

### 2. Preprocessing

```python
from src.preprocessing import preprocess_airbnb

# Preprocess data
X_train, y_train = preprocess_airbnb(df_train)
X_test = preprocess_airbnb(df_test, is_test=True)
```

### 3. Model Training

```python
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.model_selection import train_test_split

# Split data
X_train_split, X_val, y_train_split, y_val = train_test_split(
    X_train, y_train, test_size=0.2, random_state=42
)

# Training
model = GradientBoostingRegressor(n_estimators=100, random_state=42)
model.fit(X_train_split, y_train_split)

# Evaluation
from sklearn.metrics import mean_squared_error, r2_score
y_pred = model.predict(X_val)
rmse = mean_squared_error(y_val, y_pred, squared=False)
r2 = r2_score(y_val, y_pred)

print(f"RMSE: {rmse:.4f}")
print(f"R²: {r2:.4f}")
```

### 4. Generate Predictions

```python
# Predictions on test set
y_pred_test = model.predict(X_test)

# Create submission file
output = pd.DataFrame({
    "id": df_test["id"],
    "logpred": y_pred_test  # ⚠️ Already in log if model predicts log_price
})

output.to_csv("results/airbnb_predictions1.csv", index=False)
```

##  Results

Based on the analysis performed:

- **Best model**: Gradient Boosting
- **RMSE**: 0.42
- **R²**: ~0.65

### Key Insights

1. **Strong correlations**: 
   - `accommodates`, `bedrooms`, `beds` are strongly correlated with price
   - Amenities play an important role

2. **Geographic variables**:
   - `latitude` and `longitude` highly correlated (0.90) → strong geographic influence
   - Heatmap visualization shows areas with high-price density

3. **Textual features**:
   - Transformation of `amenities` into binary variables
   - Keyword extraction from `description` and `name`

##  Techniques Used

### Feature Engineering

- **Numeric variables**: Normalization, handling missing values
- **Categorical variables**: One-hot encoding, label encoding
- **Textual variables**: 
  - Keyword extraction
  - Amenity counting
  - TF-IDF on descriptions
- **Temporal variables**: 
  - Host tenure
  - Time since last review

### Models Tested

- Random Forest
- Gradient Boosting  (best)
- XGBoost
- LightGBM

##  Submission Format

The prediction file must contain two columns:

```csv
id,logpred
12345,4.532
67890,4.127
...
```

> **Reminder**: The `logpred` column must contain the **logarithm of the price**, not the raw price.

---

**Note**: This project was developed in an educational context to explore machine learning techniques applied to price prediction.
