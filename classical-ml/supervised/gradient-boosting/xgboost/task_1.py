import pandas as pd
import xgboost as xgb
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, roc_auc_score

df = pd.read_csv('../../../../data/kaggle/Titanic-Dataset.csv')

df['Age'] = df['Age'].fillna(df['Age'].median())
df['Sex'] = df['Sex'].map({'male': 0, 'female': 1})
df['Embarked'] = df['Embarked'].fillna('S')
df['Embarked'] = df['Embarked'].map({'S': 0, 'C': 1, 'Q': 2})

features = ['Pclass', 'Sex', 'Age', 'SibSp', 'Parch', 'Fare', 'Embarked']
X = df[features]
y = df['Survived']

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

dtrain = xgb.DMatrix(X_train, label=y_train)
dtest  = xgb.DMatrix(X_test,  label=y_test)

params = {
    'objective':        'binary:logistic',
    'eval_metric':      'auc',
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
    num_boost_round=100,
    evals=[(dtrain, 'train'), (dtest, 'eval')],
    evals_result=evals_result,
    verbose_eval=20
)

y_pred_proba = model.predict(dtest)
y_pred       = (y_pred_proba > 0.5).astype(int)

print(f"\nAccuracy: {accuracy_score(y_test, y_pred):.4f}")
print(f"ROC-AUC:  {roc_auc_score(y_test, y_pred_proba):.4f}")

importance = model.get_score(importance_type='gain')
importance = pd.Series(importance).sort_values(ascending=False)
print("\nFeature Importance (gain):")
print(importance)