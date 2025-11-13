from typing import Any, Tuple
import torch
import torchvision.transforms as transforms

class AugmentedDataset(torch.utils.data.Dataset):
    def __init__(self, dataset: torch.utils.data.Dataset) -> None:
        self.original_dataset = dataset
        self.augmented_indices = []

        for idx in range(len(dataset)):
            _, label = dataset[idx]
            if label == 0:
                self.augmented_indices.append(idx)

        self.total_len = len(dataset) + 3 * len(self.augmented_indices)

    def __len__(self) -> int:
        return self.total_len

    def __getitem__(self, idx) -> Tuple[torch.Tensor, int]:
        if idx < len(self.original_dataset):
            return self.original_dataset[idx]
        else:
            aug_idx = (idx - len(self.original_dataset)) // 3
            rotation_variant = (idx - len(self.original_dataset)) % 3

            image, label = self.original_dataset[self.augmented_indices[aug_idx]]

            angle = [90, 180, 270][rotation_variant]
            image = transforms.functional.rotate(image, angle)

            return image, label
