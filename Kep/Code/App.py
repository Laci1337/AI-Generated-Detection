import torch
import torch.nn as nn
import torch.optim as optim
import torchvision.transforms as transforms
import torchvision.datasets as datasets
import torch.utils.data as data
import matplotlib.pyplot as plt
import os

import ClassificationNetwork
import Functions

image_size = 240

device = torch.device("cpu")

if (torch.cuda.is_available):
    device = torch.device("cuda")

#data transform, data to tensor
transform = transforms.Compose([
    transforms.Resize(image_size),
    transforms.CenterCrop(image_size),
    transforms.ToTensor(),
    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
])


#load data

#print(os.getcwd())

#train data
train_dataset = datasets.ImageFolder(root='data/train', transform=transform)
train_loader = data.DataLoader(train_dataset, batch_size=64, shuffle=True)

#test data
test_dataset = datasets.ImageFolder(root='data/test', transform=transform)
test_loader = data.DataLoader(test_dataset, batch_size=64, shuffle=False)

#sample data
sample_dataset = datasets.ImageFolder(root='data/sample', transform=transform) 
sample_loader = data.DataLoader(sample_dataset, batch_size=1, shuffle=True)

#model, loss and optim

model = ClassificationNetwork.ClassificationNetwork().to(device)

criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)


#learn and test


num_epochs = 1

#graph data
xData = []
yLossData = []
yTestData = []
n = 1 #for graph 


print("1 - Learn\n2 - Use")

a = (int)(input())

if (a == 1):
    for epoch in range(num_epochs):
        #learn
        model.train()
        running_loss = 0.0

        for images, labels in train_loader:
            images = images.to(device)
            labels = labels.to(device)

            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)

            if (n % 10 == 0):
                xData.append(n / 10)
                yLossData.append(loss.item())
                yTestData.append(Functions.Run_partial_test(model, device, test_loader))      
                #yTestData.append(Functions.Run_partial_test(model, device, sample_loader))        

            n += 1

            print("Loss: " + str(round(loss.item(), 2)))

            loss.backward()
            optimizer.step()
            running_loss += loss.item()

        avg_loss = running_loss / len(train_loader)
        print(f"Epoch {epoch+1}/{num_epochs}, Loss: {avg_loss:.4f}")


        #test
        Functions.Run_test(model, device, test_loader)

    plt.plot(xData, yLossData)
    plt.plot(xData, yTestData)
    plt.show()

    model.save()

elif (a == 2):
    model.load()
    model.eval()

    with torch.no_grad():
        for images, labels in sample_loader:
            images = images.to(device)
            labels = labels.to(device)

            outputs = model(images)
            _, predicted = torch.max(outputs, 1)

            img = images[0].to(torch.device("cpu"))

            #makes the image compatible with plt
            img_np = img.permute(1, 2, 0).numpy()

            #inverse normalisation
            img_np = (img_np * 0.5) + 0.5

            print("Predicted: " + str(predicted.item()) + ", Actual: " + str(labels.item()))

            plt.imshow(img_np)
            plt.axis('off')
            plt.show()