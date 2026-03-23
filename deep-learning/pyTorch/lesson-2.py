import torch
import torch.nn as nn
import torch.optim as optim

# создаём случайные данные
x_train = torch.rand(100, 2) * 10

# правильный ответ
y_train = x_train[:,0:1] + x_train[:,1:2]

# for i in range(len(x_train)):
#     print(f"{x_train[i]} -> {y_train[i]}")


# nn.Linear(2,1) - это один нейрон 
# input - 2, output - 1
# Этот нейрон считает --> y_pred = w1*x1 + w2*x2 + b

model = nn.Linear(2,1)
# print(model.weight)
# print(model.bias)


# loss_fn - Это функция ошибки Mean Squared Error
# L = (y - y_pred)²

loss_fn = nn.MSELoss()



# optimizer - обновляет веса модели.
# w_new = w_old - lr * gradient
# lr — learning rate
# gradient — производная dL/dw

optimizer = optim.SGD(model.parameters(), lr=0.01)


for epoch in range(2000):

    y_pred = model(x_train)

    loss = loss_fn(y_pred, y_train)

    optimizer.zero_grad()

    loss.backward()

    optimizer.step()

    if epoch % 200 == 0:
        print("loss:", loss.item())


x_test = torch.tensor([[12.0,7.0]])

with torch.no_grad():
    prediction = model(x_test)

print("Prediction:", prediction)

# print("Weights:", model.weight)
# print("Bias:", model.bias)