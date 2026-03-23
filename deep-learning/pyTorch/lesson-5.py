import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms
import matplotlib.pyplot as plt

# ----------------------------------
# 1 Загрузка датасета MNIST
# ----------------------------------

transform = transforms.ToTensor()

train_dataset = torchvision.datasets.MNIST(
    root="./data",
    train=True,
    download=True,
    transform=transform
)

train_loader = torch.utils.data.DataLoader(
    train_dataset,
    batch_size=64,
    shuffle=True
)

# ----------------------------------
# 2 Создание нейросети
# ----------------------------------

model = nn.Sequential(

    nn.Flatten(),      # 28x28 → 784

    nn.Linear(784,128),
    nn.ReLU(),

    nn.Linear(128,10)
)

# ----------------------------------
# 3 Функция ошибки
# ----------------------------------

loss_fn = nn.CrossEntropyLoss()

# ----------------------------------
# 4 Оптимизатор
# ----------------------------------

optimizer = optim.Adam(model.parameters(), lr=0.001)

# ----------------------------------
# 5 Обучение нейросети
# ----------------------------------

for epoch in range(5):

    for images, labels in train_loader:

        # forward pass
        outputs = model(images)

        # считаем loss
        loss = loss_fn(outputs, labels)

        # очищаем старые градиенты
        optimizer.zero_grad()

        # backpropagation
        loss.backward()

        # обновляем веса
        optimizer.step()

    print("epoch:", epoch, "loss:", loss.item())


# ----------------------------------
# 6 Тест модели
# ----------------------------------

image, label = train_dataset[88]

# показываем картинку
plt.imshow(image.squeeze(), cmap="gray")
plt.title(f"Real digit: {label}")
plt.show()

# prediction
with torch.no_grad():

    prediction = model(image.unsqueeze(0))

predicted_digit = torch.argmax(prediction)

print("Model prediction:", predicted_digit.item())
print("Real digit:", label)