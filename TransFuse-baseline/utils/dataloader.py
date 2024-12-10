import os
from PIL import Image
import torch
import torch.utils.data as data
import torchvision.transforms as transforms
import numpy as np
import matplotlib.pyplot as plt
import albumentations as A
import cv2


class MIS_dataset(data.Dataset):
    """
    dataloader for skin lesion segmentation tasks
    """
    def __init__(self, image_root, gt_root, num_classes=5):
        self.images = np.load(image_root)
        self.gts = np.load(gt_root)
        self.size = len(self.images)
        self.num_classes = num_classes

        self.img_transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406],
                                 [0.229, 0.224, 0.225])
        ])
        self.gt_transform = transforms.Compose([
            transforms.ToTensor()])

        self.transform = A.Compose(
            [
                A.ShiftScaleRotate(shift_limit=0.15, scale_limit=0.15, rotate_limit=25, p=0.5, border_mode=0),
                A.HorizontalFlip(),
                A.VerticalFlip()
            ]
        )

    def one_hot_encode(self, mask):
        """
        Converts a single-channel mask to one-hot encoding with num_classes channels.
        """
        h, w = mask.shape
        one_hot = np.zeros((self.num_classes, h, w), dtype=np.uint8)
        for i in range(self.num_classes):
          one_hot[i] = (mask == i).astype(np.uint8)
        return one_hot

    def __getitem__(self, index):
        image = self.images[index]
        gt = self.gts[index].astype(np.uint8)  # Ensure mask is integer

        # Apply augmentations
        transformed = self.transform(image=image, mask=gt)
        image = transformed['image']
        gt = transformed['mask']

        # Normalize image to [0, 1] if not already
        if image.max() > 1:
            image = (image / 255.0).astype(np.float32)

        # Convert to tensor
        image = torch.from_numpy(image).permute(2, 0, 1).float()  # Channels-first format
        gt = torch.from_numpy(gt.astype(np.uint8))  # Ensure mask is integer

        # Dynamically one-hot encode the mask
        gt_one_hot = self.one_hot_encode(gt.numpy())  # One-hot encode
        gt_one_hot = torch.from_numpy(gt_one_hot)  # Convert to tensor

        return image, gt_one_hot


    def __len__(self):
        return self.size


def get_loader(image_root, gt_root, batchsize, shuffle=True, num_workers=4, pin_memory=True, num_classes=2):

    dataset = MIS_dataset(image_root, gt_root, num_classes=num_classes)
    data_loader = data.DataLoader(dataset=dataset,
                                  batch_size=batchsize,
                                  shuffle=shuffle,
                                  num_workers=num_workers,
                                  pin_memory=pin_memory)
    return data_loader


class test_dataset:
    def __init__(self, image_root, gt_root, num_classes=2, height=192, width=256):
        self.images = np.load(image_root)  # Flattened images
        self.gts = np.load(gt_root)        # Flattened masks
        self.image_shape = (height, width, 3)  # Shape: (H, W, C) for images
        self.mask_shape = (height, width)     # Shape: (H, W) for masks
        self.num_classes = num_classes
        self.size = len(self.images)
        self.index = 0

        # Transformation for images
        self.transform = transforms.Compose([
            transforms.ToTensor(),  # Converts (H, W, C) -> (C, H, W)
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])  # ImageNet mean/std
        ])

    def load_data(self):
        # Load the current image and mask
        image = self.images[self.index].reshape(self.image_shape)  # Reshape to (192, 256, 3)
        gt = self.gts[self.index].reshape(self.mask_shape).astype(np.uint8)         # Reshape to (192, 256)

        # Normalize image to [0, 1] if not already
        if image.max() > 1:
            image = (image / 255.0).astype(np.float32)

        # Convert to tensor
        image = torch.from_numpy(image).permute(2, 0, 1).float()  # Channels-first format
        image = image.unsqueeze(0) # 1, 3, 192, 256


        # Process the mask
        if self.num_classes > 1:
            # Multi-class: one-hot encode
            gt = torch.tensor(gt, dtype=torch.long)  # Class indices as integers
            gt = torch.nn.functional.one_hot(gt, num_classes=self.num_classes)  # (H, W, num_classes)
            gt = gt.permute(2, 0, 1).float()  # Rearrange to (num_classes, H, W)
            gt = gt.unsqueeze(0)  # (B, num_classes, H, W)
        else:
            # Binary: Normalize to [0, 1]
            gt = torch.tensor(gt, dtype=torch.float32) / 255.0  # (H, W)

        # Increment index for the next image
        self.index += 1

        return image, gt

    def __len__(self):
        return self.size


if __name__ == '__main__':
    path = 'data/'
    tt = MIS_dataset(path+'data_train.npy', path+'mask_train.npy')

    for i in range(50):
        img, gt = tt.__getitem__(i)

        img = torch.transpose(img, 0, 1)
        img = torch.transpose(img, 1, 2)
        img = img.numpy()
        gt = gt.numpy()

        plt.imshow(img)
        plt.savefig('vis/'+str(i)+".jpg")
 
        plt.imshow(gt[0])
        plt.savefig('vis/'+str(i)+'_gt.jpg')
