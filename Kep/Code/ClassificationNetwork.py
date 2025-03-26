import torch
import torch.nn as nn
import torch.optim as optim
import torchvision.transforms as transforms
import torchvision.datasets as datasets
import torch.utils.data as data
import os

class ClassificationNetwork(nn.Module):
    def __init__(self):
        super(ClassificationNetwork, self).__init__()

        #first conv layer: input 3 channels (rgb), output 32 channels (32 filter matrix)
        self.conv1 = nn.Conv2d(in_channels=4, out_channels=32, kernel_size=3, padding=1)
        self.relu = nn.ReLU()
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)

        #second conv layer
        self.conv2 = nn.Conv2d(in_channels=32, out_channels=64, kernel_size=3, padding=1)

        self.fc1 = nn.Linear(64 * 60 * 60, 128)

        self.fc2 = nn.Linear(128, 2)

    def forward(self, x):
        #fourier eleje

        # x shape: (B, 3, H, W)
        B, C, H, W = x.size()

        fft = torch.fft.fft2(x)                    # complex tensor
        fft_mag = torch.abs(torch.fft.fftshift(fft))  # shape (B, 3, H, W)
        spectrum = torch.mean(fft_mag, dim=1, keepdim=True)  # (B, 1, H, W)

        spectrum = (spectrum - spectrum.min()) / (spectrum.max() - spectrum.min() + 1e-6)

        # Concatenate the spectrum to the RGB image
        x = torch.cat([x, spectrum], dim=1)  # shape (B, 4, H, W)

        #fourier vege

        x = self.conv1(x)
        x = self.relu(x)
        x = self.pool(x)

        x = self.conv2(x)
        x = self.relu(x)
        x = self.pool(x)

        #flatten the data
        x = x.view(x.size(0), -1)
        x = self.fc1(x)
        x = self.relu(x)
        x = self.fc2(x)

        #print(x)

        return x
    

    def save(self, file_name = 'model.pth'):
        model_folder_path = 'model'

        if not os.path.exists(model_folder_path):
            os.makedirs(model_folder_path)

        file_name = os.path.join(model_folder_path, file_name)
        torch.save(self.state_dict(), file_name)

        print("File saved")


    def load(self, file_name='model.pth'):
        model_folder_path = 'model'
        file_name = os.path.join(model_folder_path, file_name)

        if os.path.exists(file_name):
            self.load_state_dict(torch.load(file_name))
            print(f"Model successfully loaded from {file_name}")
            return True  #load successful
        else:
            print(f"File not found: {file_name}. Loading skipped.")
            return False  #load fail