import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms
import matplotlib.pyplot as plt

# преобразование изображения в tensor
# PyTorch работает с Tensor, а не с обычными изображениями
transform = transforms.ToTensor()

# загружаем датасет MNIST (рукописные цифры)
train_dataset = torchvision.datasets.MNIST(
    root="./data",
    train=True,
    transform=transform,
    download=True
)


# DataLoader — разбивает датасет на батчи
train_loader = torch.utils.data.DataLoader(
    train_dataset,
    batch_size=64,
    shuffle=True
)



class MyNeuralNet(nn.Module):
    def __init__(self):
        super(MyNeuralNet, self).__init__()
        
        self.fc1 = nn.Linear(28*28, 128)  # вход: 28*28 пикселей выход: 128 нейронов
        self.fc2 = nn.Linear(128, 64)     # вход: 128 выход: 64 нейронов
        self.fc3 = nn.Linear(64, 10)      # вход: 64 выход: 10 нейронов

    def forward(self, x):
        x = x.reshape(-1, 28*28)    # превращаем изображение 28x28 в вектор из 784 элементов
        x = torch.relu(self.fc1(x)) # первый слой + функция активации ReLU
        x = torch.relu(self.fc2(x)) # второй слой + функция активации ReLU
        x = self.fc3(x)             # выходной слой
        
        return x
    

# создаем экземпляр модели
model = MyNeuralNet()

for name, param in model.named_parameters():
    print(name)
    print(param.shape)
    print(param[:2]) 

# функция потерь
# CrossEntropy используется для задач классификации
criterion = nn.CrossEntropyLoss()

# оптимизатор Adam — обновляет веса сети
optimizer = optim.Adam(model.parameters(), lr=0.001)




for epoch in range(3):

    # берем батчи изображений
    for images, labels in train_loader:
        # forward pass
        # пропускаем изображения через модель
        outputs = model(images)
        # вычисляем ошибку
        loss = criterion(outputs, labels)
        
        # обнуляем градиенты
        optimizer.zero_grad()
        # вычисляем градиенты (backpropagation)
        loss.backward()
        # обновляем веса
        optimizer.step()
    
    print(f"Epoch {epoch+1}, Loss: {loss.item()}")


# берем одно изображение из датасета
image, label = train_dataset[8]

# показываем картинку
plt.imshow(image.squeeze(), cmap="gray")
plt.title(f"Real digit: {label}")
plt.show()


# отключаем вычисление градиентов
# это ускоряет inference
with torch.no_grad():
    # добавляем batch dimension
    prediction = model(image.unsqueeze(0))

# выбираем индекс с максимальной вероятностью
predicted_digit = torch.argmax(prediction)

# выводим результат модели
print("Model prediction:", predicted_digit.item())
# выводим правильный ответ
print("Real digit:", label)
