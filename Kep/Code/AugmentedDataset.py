import torch
import torchvision.transforms as transforms

class AugmentedDataset(torch.utils.data.Dataset):
    def __init__(self, dataset, n_augmented):
        self.original_dataset = dataset
        self.n_augmented = n_augmented
        self.augmented_indices = []

        count = 0
        for idx in range(len(dataset)):
            _, label = dataset[idx]
            if label == 1:
                self.augmented_indices.append(idx)
                count += 1
                if count >= n_augmented:
                    break

        self.total_len = len(dataset) + len(self.augmented_indices)

    def __len__(self):
        return self.total_len

    def __getitem__(self, idx):
        if idx < len(self.original_dataset):
            return self.original_dataset[idx]
        else:
            aug_idx = self.augmented_indices[idx - len(self.original_dataset)]
            image, label = self.original_dataset[aug_idx]
            image = transforms.functional.rotate(image, 180)
            return image, label
