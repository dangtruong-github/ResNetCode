import torch
import torch.nn as nn
import torch.optim as optim

import matplotlib.pyplot as plt

from utils import GetModel, GetData

device = "cuda" if torch.cuda.is_available() else "cpu"

trainloader, testloader = GetData()

# Prepare model, optimizer
model = GetModel(3, 10, "ResNet18")
criterion = nn.CrossEntropyLoss()
optimizer = optim.SGD(model.parameters(), lr=0.001, momentum=0.9)

# Training loop
epochs = 10
loss_epochs = []
accuracy_epochs = []

for epoch in range(epochs):
    running_loss = 0.0
    correct = 0
    total = 0

    for i, data in enumerate(trainloader):
        inputs, labels = data[0].to(device), data[1].to(device)

        optimizer.zero_grad()

        outputs = model(inputs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        running_loss += loss.item()

        correct += (outputs.argmax(1) == labels).type(torch.float).sum().item()
        total += labels.size(0)

    running_loss /= len(trainloader)
    loss_epochs.append(running_loss)
    accuracy_epochs.append(100*correct/total)
    print(f"Epoch {epoch}/{epochs} | "
          f"Loss: {running_loss:.3f} | "
          f"Accuracy: {accuracy_epochs[-1]:.2f}")

# plot training loss and accuracy

plt.figure(figsize=(10, 5))
plt.subplot(1, 2, 1)
plt.plot(loss_epochs)
plt.title("Training Loss")
plt.xlabel("Epoch")
plt.ylabel("Loss")

plt.subplot(1, 2, 2)
plt.plot(accuracy_epochs)
plt.title("Training Accuracy")
plt.xlabel("Epoch")
plt.ylabel("Accuracy")

plt.savefig("train_loss_and_acc_curve.png")

plt.show()

# Evaluation
correct = 0
total = 0

with torch.no_grad():
    for data in testloader:
        images, labels = data[0].to(device), data[1].to(device)
        outputs = model(images)
        _, predicted = torch.max(outputs.data, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()

print(f"Accuracy of the network on the 10000 test images: "
      f"{100 * correct // total} %")
