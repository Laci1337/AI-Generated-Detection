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
                    pass
                    #break

        # Minden kiválasztott képhez 3 másolat készül (90, 180, 270)
        self.total_len = len(dataset) + 3 * len(self.augmented_indices)

    def __len__(self):
        return self.total_len

    def __getitem__(self, idx):
        if idx < len(self.original_dataset):
            return self.original_dataset[idx]
        else:
            # Hányadik augmentált kép vagyunk?
            aug_idx = (idx - len(self.original_dataset)) // 3
            rotation_variant = (idx - len(self.original_dataset)) % 3

            image, label = self.original_dataset[self.augmented_indices[aug_idx]]

            # Forgatás 90, 180 vagy 270 fokkal
            angle = [90, 180, 270][rotation_variant]
            image = transforms.functional.rotate(image, angle)

            return image, label
