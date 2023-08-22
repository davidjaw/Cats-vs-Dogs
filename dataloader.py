import os
import torch
from torch.utils.data import Dataset, DataLoader, random_split, Subset
from torchvision import transforms
from PIL import Image
import numpy as np


class RandomPixelDrop(object):
    def __init__(self, drop_percentage=0.1):
        self.drop_percentage = drop_percentage
        self.modes = ['random', 'corner', 'center', 'horizontal', 'vertical']

    def drop_random(self, img):
        np_img = np.array(img)
        mask = np.random.rand(*np_img.shape[:2]) < self.drop_percentage + .1
        for c in range(np_img.shape[2]):
            np_img[mask, c] = 0
        return Image.fromarray(np_img)

    def drop_mask(self, img: Image, mask: np.ndarray):
        mask = mask == 1
        np_img = np.array(img)
        for c in range(np_img.shape[2]):
            np_img[mask, c] = 0
        return Image.fromarray(np_img)

    def drop_corner(self, img):
        w, h = img.size
        mask = np.ones((h, w), dtype=np.int32)
        mask[:int(h*self.drop_percentage), :] = 0
        mask[:, :int(w*self.drop_percentage)] = 0
        return self.drop_mask(img, mask)

    def drop_center(self, img):
        w, h = img.size
        mask = np.zeros((h, w))
        ch, cw = int(h*self.drop_percentage), int(w*self.drop_percentage)
        mask[h//2 - ch//2: h//2 + ch//2, w//2 - cw//2: w//2 + cw//2] = 1
        return self.drop_mask(img, mask)

    def drop_horizontal(self, img):
        w, h = img.size
        num_drops = int(w * self.drop_percentage)
        drop_indices = np.random.choice(w, num_drops, replace=False)

        mask = np.zeros((h, w))
        mask[:, drop_indices] = 1

        return self.drop_mask(img, mask)

    def drop_vertical(self, img):
        w, h = img.size
        num_drops = int(h * self.drop_percentage)
        drop_indices = np.random.choice(h, num_drops, replace=False)

        mask = np.zeros((h, w))
        mask[drop_indices, :] = 1

        return self.drop_mask(img, mask)

    def __call__(self, img):
        mode = np.random.choice(self.modes)
        if mode == 'random':
            return self.drop_random(img)
        elif mode == 'corner':
            return self.drop_corner(img)
        elif mode == 'center':
            return self.drop_center(img)
        elif mode == 'horizontal':
            return self.drop_horizontal(img)
        elif mode == 'vertical':
            return self.drop_vertical(img)
        else:
            return img


class SubsetTransform(Subset):
    def __init__(self, dataset: Dataset, indices: list[int], transform: callable = None):
        super(SubsetTransform, self).__init__(dataset, indices)
        self.transform = transform

    def __getitem__(self, index):
        x, y = self.dataset[self.indices[index]]
        if self.transform:
            x = self.transform(x)
        return x, y


class CatsDogsDataset(Dataset):
    def __init__(self, root_dir, transform=None):
        self.root_dir = root_dir
        self.transform = transform
        self.image_files = [f for f in os.listdir(root_dir) if f.endswith('.jpg')]

    def __len__(self):
        return len(self.image_files)

    def __getitem__(self, idx):
        img_name = os.path.join(self.root_dir, self.image_files[idx])
        image = Image.open(img_name)
        label = 0 if 'cat' in self.image_files[idx] else 1
        label = torch.tensor([label], dtype=torch.float32)

        if self.transform:
            image = self.transform(image)

        return image, label


def denormalize(img_tensor, device, mean=None, std=None):
    if std is None:
        std = [0.229, 0.224, 0.225]
        std = torch.tensor(std).reshape(1, -1, 1, 1)
    if mean is None:
        mean = [0.485, 0.456, 0.406]
        mean = torch.tensor(mean).reshape(1, -1, 1, 1)
    return img_tensor * std.to(device) + mean.to(device)


def get_dataloaders(root_dir, batch_size=32, img_size=224, use_norm=True, use_drop=True):
    img_size = (img_size, img_size)
    # Transformations for the training set
    identical = transforms.Lambda(lambda x: x)
    norm = transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]) if use_norm else identical
    drop = RandomPixelDrop(drop_percentage=0.25) if use_drop else identical
    train_transforms = transforms.Compose([
        transforms.RandomHorizontalFlip(),
        transforms.RandomRotation(10),
        transforms.RandomAutocontrast(),
        transforms.Resize(img_size),
        drop,
        transforms.ToTensor(),
        norm
    ])

    # Transformations for the validation and test sets
    val_test_transforms = transforms.Compose([
        transforms.Resize(img_size),
        transforms.ToTensor(),
        norm
    ])

    # Splitting the dataset into train, val, and test
    dataset = CatsDogsDataset(root_dir=root_dir)
    total = len(dataset)
    train_len, val_len, test_len = int(0.55 * total), int(0.15 * total), int(0.3 * total)
    # set random seed such that the split is the same every time
    seed = 9527
    torch.manual_seed(seed)
    indices = torch.randperm(total).tolist()

    torch.seed()

    train_dataset = SubsetTransform(dataset, indices[:train_len], transform=train_transforms)
    val_dataset = SubsetTransform(dataset, indices[train_len:train_len + val_len], transform=val_test_transforms)
    test_dataset = SubsetTransform(dataset, indices[train_len + val_len:], transform=val_test_transforms)

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

    return train_loader, val_loader, test_loader


if __name__ == '__main__':
    import cv2
    from torchvision.utils import make_grid
    # Example usage:
    root_dir = "G:\\dataset\\dogs-vs-cats\\train"
    train_loader, val_loader, test_loader = get_dataloaders(root_dir, batch_size=32, use_norm=True, use_drop=True)
    iter_train, iter_val, iter_test = iter(train_loader), iter(val_loader), iter(test_loader)

    for i in range(10):
        img_train, label_train = next(iter_train)
        img_val, label_val = next(iter_val)
        img_test, label_test = next(iter_test)

        img_concat = img_train
        grid_img = make_grid(img_concat, nrow=15, padding=2, pad_value=1)
        grid_img = grid_img.numpy().transpose((1, 2, 0))
        grid_img = cv2.cvtColor(grid_img, cv2.COLOR_RGB2BGR)
        cv2.imshow('grid_img', grid_img)
        print(label_train)
        cv2.waitKey(0)
        cv2.destroyAllWindows()

