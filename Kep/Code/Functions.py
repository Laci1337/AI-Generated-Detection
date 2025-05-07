import os
import torch
import torch.nn.functional as F

def Run_test(model, device, test_loader):
    model.eval()
    correct = 0
    total = 0
    with torch.no_grad():
        for images, labels in test_loader:
            images = images.to(device)
            labels = labels.to(device)

            outputs = model(images)

            outputs = F.softmax(outputs, dim=1)
            
            #_, predicted = torch.max(outputs, 1)
            predicted = Make_prediction(outputs, device)

            total += labels.size(0)
            correct += (predicted == labels).sum().item()

    print(f"Test Accuracy: {100 * correct / total:.2f}%")

    model.train()

    return (correct / total)


def Run_partial_test(model, device, test_loader):
    model.eval()
    correct = 0
    total = 0
    with torch.no_grad():
        for images, labels in test_loader:
            images = images.to(device)
            labels = labels.to(device)

            outputs = model(images)

            #_, predicted = torch.max(outputs, 1)
            predicted = Make_prediction(outputs, device)

            total += labels.size(0)
            correct += (predicted == labels).sum().item()

            if (total > 10):
                break

    print(f"Partial Test Accuracy: {100 * correct / total:.2f}%")

    model.train()

    return (correct / total)

def Make_prediction(output, device):
    if (output[0][0].item() > 0.9):
        return torch.tensor([0]).to(device)
    else:
        return torch.tensor([1]).to(device)

