import pandas as pd
import xgboost as xgb
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

dtrain = xgb.DMatrix(X_train, label=y_train)
dtest  = xgb.DMatrix(X_test,  label=y_test)

params = {
    'objective':       'reg:squarederror',
    'eval_metric':     'mae',
    'max_depth':        4,
    'eta':              0.1,
    'subsample':        0.8,
    'colsample_bytree': 0.8,
    'seed':             42
}

evals_result = {}
model = xgb.train(
    params,
    dtrain,
    num_boost_round=500,
    evals=[(dtrain, 'train'), (dtest, 'eval')],
    early_stopping_rounds=20,
    evals_result=evals_result,
    verbose_eval=20
)

print(f"\nBest Iteration: {model.best_iteration}")
print(f"Best MAE:   {model.best_score:,.0f}")

y_pred = model.predict(dtest)
print(f"MAE: {mean_absolute_error(y_test, y_pred):,.0f}")
print(f"R²:  {r2_score(y_test, y_pred):.4f}")

importance = model.get_score(importance_type='gain')
importance = pd.Series(importance).sort_values(ascending=False)
print("\nFeature Importance (gain):")
print(importance)