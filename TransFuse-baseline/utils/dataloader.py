import os
from PIL import Image
import torch
import torch.utils.data as data
import torchvision.transforms as transforms
import numpy as np
import matplotlib.pyplot as plt
import albumentations as A
import cv2


class SkinDataset(data.Dataset):
    """
    dataloader for skin lesion segmentation tasks
    """
    def __init__(self, image_root, gt_root, num_classes=1):
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
                A.ColorJitter(),
                A.HorizontalFlip(),
                A.VerticalFlip()
            ]
        )

    def one_hot_encode(self, mask):
        """
        Converts a single-channel mask to one-hot encoding with num_classes channels.
        """
        if self.num_classes == 1:
            # No one-hot encoding needed; return the mask as is
            return mask.astype(np.float32)
        
        h, w = mask.shape
        one_hot = np.zeros((self.num_classes, h, w), dtype=np.float32)
        one_hot[0] = (mask == 0).astype(np.float32)  # Background
        for i in range(1, self.num_classes):
            one_hot[i] = (mask == i).astype(np.float32)  # Foreground or other classes
        return one_hot

    def __getitem__(self, index):
        
        image = self.images[index]
        gt = self.gts[index]
        
        # Handle grayscale images (convert to 3 channels if necessary)
        if len(image.shape) == 2:  # Grayscale image (height, width)
            image = np.expand_dims(image, axis=-1)  # Convert to (height, width, 1)
            image = np.repeat(image, 3, axis=-1)  # Convert to (height, width, 3) by repeating

        # Apply augmentations
        transformed = self.transform(image=image, mask=gt)
        image = self.img_transform(transformed['image'])
        gt = transformed['mask']

        # Dynamically one-hot encode the mask if necessary
        if self.num_classes > 1:
            gt_encoded = self.one_hot_encode(gt)
            gt_tensor = torch.from_numpy(gt_encoded)  # Convert to tensor
        else:
            # When num_classes=1, just normalize the mask and add a channel dimension
            gt_tensor = torch.from_numpy(gt.astype(np.float32)).unsqueeze(0)

        return image, gt_tensor

    def __len__(self):
        return self.size


def get_loader(image_root, gt_root, batchsize, shuffle=True, num_workers=4, pin_memory=True, num_classes=1):

    dataset = SkinDataset(image_root, gt_root, num_classes=num_classes)
    data_loader = data.DataLoader(dataset=dataset,
                                  batch_size=batchsize,
                                  shuffle=shuffle,
                                  num_workers=num_workers,
                                  pin_memory=pin_memory)
    return data_loader



class test_dataset:
    def __init__(self, image_root, gt_root, num_classes=1):
        self.images = np.load(image_root)
        self.gts = np.load(gt_root)

        self.transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406],
                                 [0.229, 0.224, 0.225])
        ])
        self.num_classes = num_classes
        self.size = len(self.images)
        self.index = 0

    def load_data(self):
        # Load the current image and mask
        image = self.images[self.index]
        gt = self.gts[self.index]

        # Convert grayscale to RGB if needed
        if len(image.shape) == 2:  # Grayscale image
            image = np.stack([image] * 3, axis=-1)  # Convert to RGB
        
        # Apply image transformations
        image = self.transform(image).unsqueeze(0)  # Add batch dimension

        # Handle multi-class masks if required
        if self.num_classes > 1:
            gt = torch.tensor(gt, dtype=torch.long)  # Convert to long type
            gt = torch.nn.functional.one_hot(gt, num_classes=self.num_classes).permute(2, 0, 1).float()

        self.index += 1
        return image, gt

    def __len__(self):
        return self.size



if __name__ == '__main__':
    path = 'data/'
    tt = SkinDataset(path+'data_train.npy', path+'mask_train.npy')

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
