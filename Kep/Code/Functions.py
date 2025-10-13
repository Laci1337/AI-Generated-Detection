import os
import torch
import torch.nn.functional as F
import matplotlib.pyplot as plt

border = 0.9

def FindBestBorder(model: torch.nn.Module, device: torch.device, test_loader) -> None:
    xData = []
    yData = []

    border = 0.01

    for i in range(100):
        print("Inspected border: " + str(border))
        xData.append(border)
        yData.append(100.0 * Run_test(model, device, test_loader))
        
        border += 0.01

    plt.plot(xData, yData)
    plt.show()

def Run_test(model: torch.nn.Module, device: torch.device, test_loader) -> float:
    model.eval()
    correct = 0
    total = 0
    with torch.no_grad():
        for images, labels in test_loader:
            images = images.to(device)
            labels = labels.to(device).float().unsqueeze(1)  # (B,) -> (B,1)

            outputs = model(images)
            probs = torch.sigmoid(outputs)
            predicted = (probs > border).float()

            print("Labels: " + str(labels))
            print("Outputs: " + str(probs) + "\n")

            total += labels.size(0)
            correct += (predicted == labels).sum().item()

    print(f"Test Accuracy: {100 * correct / total:.2f}%")

    model.train()
    return (correct / total)


def Run_partial_test(model: torch.nn.Module, device: torch.device, test_loader) -> None:
    model.eval()
    correct = 0
    total = 0
    with torch.no_grad():
        for images, labels in test_loader:
            images = images.to(device)
            labels = labels.to(device).float().unsqueeze(1)

            outputs = model(images)
            probs = torch.sigmoid(outputs)
            predicted = (probs > border).float()

            total += labels.size(0)
            correct += (predicted == labels).sum().item()

            if total > 10:
                break

    print(f"Partial Test Accuracy: {100 * correct / total:.2f}%")
    model.train()
    return (correct / total)