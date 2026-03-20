import pandas as pd
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, r2_score

df = pd.read_csv('../../../../data/kaggle/House_price.csv')

features = ['Avg. Area Income', 'House Age', 'Number of Rooms',
            'Number of Bedrooms', 'Area Population']

X = df[features]
y = df['Price']

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

model = GradientBoostingRegressor(
    n_estimators=100,
    learning_rate=0.1,
    max_depth=3,
    random_state=42
)

model.fit(X_train, y_train)

y_pred = model.predict(X_test)

print(f"MAE:  {mean_absolute_error(y_test, y_pred):,.0f}")
print(f"R²:   {r2_score(y_test, y_pred):.4f}")

importances = pd.Series(model.feature_importances_, index=features)
print("\nFeature Importance:")
print(importances.sort_values(ascending=False))