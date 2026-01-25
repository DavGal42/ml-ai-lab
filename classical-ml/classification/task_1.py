from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score

X, y = load_iris(return_X_y=True)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

# Without Scaling
model1 = KNeighborsClassifier(n_neighbors=3)
model1.fit(X_train, y_train)
y_pred1 = model1.predict(X_test)
print(f'Accuracy Without Scaling {accuracy_score(y_test, y_pred1)}')

# With Scaling
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

model2 = KNeighborsClassifier(n_neighbors=3)
model2.fit(X_train_scaled, y_train)
y_pred2 = model2.predict(X_test_scaled)
print(f'Accuracy With Scaling {accuracy_score(y_test, y_pred2)}')