import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms
import matplotlib.pyplot as plt


transform = transforms.ToTensor()
train_dataset = torchvision.datasets.MNIST(
    root="./data",
    train=True,
    transform=transform,
    download=True
)


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
    

model = MyNeuralNet()
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)




for epoch in range(3):
    for images, labels in train_loader:
        outputs = model(images)
        loss = criterion(outputs, labels)
        
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
    
    print(f"Epoch {epoch+1}, Loss: {loss.item()}")


image, label = train_dataset[15]

plt.imshow(image.squeeze(), cmap="gray")
plt.title(f"Real digit: {label}")
plt.show()


with torch.no_grad():
    prediction = model(image.unsqueeze(0))

predicted_digit = torch.argmax(prediction)

print("Model prediction:", predicted_digit.item())
print("Real digit:", label)



#----------------------------------
# СОХРАНЯЕМ МОДЕЛЬ
# ----------------------------------  

# torch.save(model.state_dict(), "model.pth")

# torch.save(model, "model_full.pth")