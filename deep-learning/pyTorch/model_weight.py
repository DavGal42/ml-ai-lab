import torch
import torch.nn as nn
import torchvision
import torchvision.transforms as transforms
import matplotlib.pyplot as plt


transform = transforms.ToTensor()

test_dataset = torchvision.datasets.MNIST(
    root="./data",
    train=False,
    transform=transform,
    download=True
)

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
    


model = MyNeuralNet()
model.load_state_dict(torch.load("model.pth"))
model.eval()  # ОБЯЗАТЕЛЬНО


image, label = test_dataset[42]

plt.imshow(image.squeeze(), cmap="gray")
plt.title(f"Real: {label}")
plt.show()


with torch.no_grad():
    prediction = model(image.unsqueeze(0))

predicted_digit = torch.argmax(prediction)

print("Prediction:", predicted_digit.item())
print("Real:", label)