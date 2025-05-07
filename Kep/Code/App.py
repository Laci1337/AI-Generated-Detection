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
import AugmentedDataset

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

batch_size = 64
num_epochs = 3

#load sample data

#sample data
sample_dataset = datasets.ImageFolder(root='data/sample', transform=transform) 
#sample_dataset = test_dataset
sample_loader = data.DataLoader(sample_dataset, batch_size=1, shuffle=True)

#model, loss and optim

model = ClassificationNetwork.ClassificationNetwork().to(device)

criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)


#graph data
xData = []
yLossData = []
yTestData = []
n = 1 #for graph 


print("1 - Learn\n2 - Use")

a = (int)(input())

if (a == 1):
    #load data

    #train_locations = ['train1', 'train2', 'train4']
    train_locations = ['train1', 'train4']

    #test_locations = ['test1', 'test2', 'test4']
    test_locations = ['test1', 'test4']

    #train_locations = ['train1']
    #test_locations = ['test1']

    train_augmentation_n = 10235
    test_augmentation_n = 2264

    #train data
    train_dataset_list = []

    for i in train_locations:
        d = datasets.ImageFolder(root='data/' + str(i), transform=transform)
        train_dataset_list.append(d)
        #print("Class mapping:", d.class_to_idx)

    train_dataset = torch.utils.data.ConcatDataset(train_dataset_list)
    train_dataset = AugmentedDataset.AugmentedDataset(train_dataset, train_augmentation_n)
    train_loader = data.DataLoader(train_dataset, batch_size=batch_size, shuffle=True)

    #test data
    test_dataset_list = []

    for i in test_locations:
        test_dataset_list.append(datasets.ImageFolder(root='data/' + str(i), transform=transform))

    test_dataset = torch.utils.data.ConcatDataset(test_dataset_list)
    test_dataset = AugmentedDataset.AugmentedDataset(test_dataset, test_augmentation_n)
    test_loader = data.DataLoader(test_dataset, batch_size=batch_size, shuffle=True)

    #learn and test


    #print("Class mapping:", train_dataset.class_to_idx)
    print("Device: " + str(device))
    print("Train dataset len: " + str(len(train_dataset)))
    print("Test dataset len: " + str(len(test_dataset)))

    #learning

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

            if (n % 100 == 0):
                xData.append(n / 100)
                yLossData.append(loss.item())
                #yTestData.append(Functions.Run_partial_test(model, device, test_loader))      
                yTestData.append(Functions.Run_test(model, device, sample_loader))        

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

    Functions.Run_test(model, device, sample_loader)

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

            if (predicted.item() == 0):
                predicted_string = "ai generated"
            else:
                predicted_string = "real"

            if (labels.item() == 0):
                labels_string = "ai generated"
            else:
                labels_string = "real"

            print("Predicted: " + predicted_string + ", Actual: " + labels_string)

            plt.imshow(img_np)
            plt.axis('off')
            plt.show()