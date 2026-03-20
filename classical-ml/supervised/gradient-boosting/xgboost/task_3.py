import pandas as pd
import xgboost as xgb
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report
from sklearn.preprocessing import LabelEncoder

df = pd.read_csv('../../../../data/kaggle/Iris.csv')

features = ['SepalLengthCm', 'SepalWidthCm', 'PetalLengthCm', 'PetalWidthCm']
X = df[features]
y = df['Species']

le = LabelEncoder()
y = le.fit_transform(y)

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

dtrain = xgb.DMatrix(X_train, label=y_train)
dtest  = xgb.DMatrix(X_test,  label=y_test)

params = {
    'objective':        'multi:softprob',
    'eval_metric':      'mlogloss',
    'num_class':         3,
    'max_depth':         4,
    'eta':               0.1,
    'subsample':         0.8,
    'colsample_bytree':  0.8,
    'seed':              42
}

evals_result = {}
model = xgb.train(
    params,
    dtrain,
    num_boost_round=200,
    evals=[(dtrain, 'train'), (dtest, 'eval')],
    early_stopping_rounds=20,
    evals_result=evals_result,
    verbose_eval=20
)

print(f"\nBest iteration: {model.best_iteration}")

y_pred_proba = model.predict(dtest)       
y_pred       = y_pred_proba.argmax(axis=1) 

print(f"Accuracy: {accuracy_score(y_test, y_pred):.4f}")
print("\nClassification Report:")
print(classification_report(y_test, y_pred, target_names=le.classes_))

importance = model.get_score(importance_type='gain')
importance = pd.Series(importance).sort_values(ascending=False)
print("Feature Importance (gain):")
print(importance)