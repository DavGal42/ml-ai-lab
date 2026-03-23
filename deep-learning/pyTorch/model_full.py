import torch
import torch.nn as nn  # ДОБАВЬ!
import torchvision
import torchvision.transforms as transforms
import matplotlib.pyplot as plt


class MyNeuralNet(nn.Module):
    def __init__(self):
        super(MyNeuralNet, self).__init__()
        self.fc1 = nn.Linear(28*28, 128)
        self.fc2 = nn.Linear(128, 64)
        self.fc3 = nn.Linear(64, 10)

    def forward(self, x):
        x = x.reshape(-1, 28*28)
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        x = self.fc3(x)
        return x
# ========================================

transform = transforms.ToTensor()

test_dataset = torchvision.datasets.MNIST(
    root="./data",
    train=False,
    transform=transform,
    download=True
)

# Теперь загрузка работает!
model = torch.load("model_full.pth", weights_only=False)
model.eval()

image, label = test_dataset[42]

plt.imshow(image.squeeze(), cmap="gray")
plt.title(f"Real: {label}")
plt.show()

with torch.no_grad():
    prediction = model(image.unsqueeze(0))

predicted = torch.argmax(prediction)

print(f"Предсказано: {predicted.item()}")
print(f"Правильный ответ: {label}")