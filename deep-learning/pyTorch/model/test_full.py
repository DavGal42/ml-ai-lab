import torch
import torchvision
import torchvision.transforms as transforms
import matplotlib.pyplot as plt

# from model import MyNeuralNet


model = torch.load("model_full.pth", weights_only=False)
model.eval()

transform = transforms.ToTensor()

test_dataset = torchvision.datasets.MNIST(
    root="./data",
    train=False,
    transform=transform,
    download=True
)

# берём пример
image, label = test_dataset[17]

plt.imshow(image.squeeze(), cmap="gray")
plt.title(f"Real: {label}")
plt.show()

# предсказание
with torch.no_grad():
    prediction = model(image.unsqueeze(0))

predicted = torch.argmax(prediction)

print("Prediction:", predicted.item())
print("Real:", label)