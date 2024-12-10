import torch
from torch.autograd import Variable
import argparse
from datetime import datetime
from lib.TransFuse import TransFuse_S, TransFuse_L, TransFuse_L_384
from utils.dataloader import get_loader, test_dataset
from utils.utils import AvgMeter
import torch.nn.functional as F
import numpy as np
import matplotlib.pyplot as plt
from test_isic import mean_dice_np, mean_iou_np
import os


def structure_loss(pred, mask, num_classes=1):
    if num_classes > 1:
        # Multi-class handling
        if mask.shape[1] == num_classes:
            # Already one-hot encoded
            mask_onehot = mask.float()
        else:
            # Convert to one-hot encoding
            mask = mask.long()
            mask_onehot = F.one_hot(mask, num_classes=num_classes).permute(0, 3, 1, 2).float()

        pred_probs = F.softmax(pred, dim=1)  # Apply softmax to logits for multi-class probabilities

        # Calculate class weights based on mask distribution
        class_counts = mask_onehot.sum(dim=(0, 2, 3))  # Count pixels for each class
        total_pixels = class_counts.sum()
        class_weights = total_pixels / (num_classes * class_counts + 1e-6)  # Inverse frequency weighting
        class_weights /= class_weights.min()  # Normalize to make the smallest weight 1
        class_weights[0] *= 0.05  # Reduce background weight

        # Compute weighted cross-entropy
        weit = 1 + 5 * torch.abs(F.avg_pool2d(mask_onehot, kernel_size=31, stride=1, padding=15) - mask_onehot)
        wce = F.cross_entropy(pred, mask.argmax(dim=1), weight=class_weights, reduction='none')
        wce = (weit * wce.unsqueeze(1)).sum(dim=(2, 3)) / weit.sum(dim=(2, 3))

        # Compute weighted IoU
        inter = ((pred_probs * mask_onehot) * weit).sum(dim=(2, 3))  # Intersection
        union = ((pred_probs + mask_onehot) * weit).sum(dim=(2, 3))  # Union
        wiou = 1 - (inter + 1) / (union - inter + 1)  # Weighted IoU loss

        return (wce + wiou).mean()

def visualize_image(image):
    """
    Visualize an image tensor or NumPy array.

    Parameters:
        image (torch.Tensor or np.ndarray): The input image.
                                            Shape: [Batch, Channel, Height, Width] or [Channel, Height, Width].
    """
    print("image bef: ", image.shape)

    if isinstance(image, torch.Tensor):
        image = image.detach().cpu().numpy()  # Convert to NumPy array

    if len(image.shape) == 4:
      image = np.squeeze(image, axis=0)

    print("image aft: ", image.shape)

    if image.shape[0] == 1:  # Single channel (grayscale)
        plt.imshow(image[0], cmap='gray')
    elif image.shape[0] == 3:  # RGB
        plt.imshow(np.transpose(image, (1, 2, 0)))
    else:
        raise ValueError("Image should have 1 or 3 channels for visualization.")

    plt.axis('off')
    plt.show()

def visualize_mask(mask):
    """
    Visualize a segmentation mask tensor or NumPy array.

    Parameters:
        mask (torch.Tensor or np.ndarray): The input mask.
                                           Shape: [Batch, num_classes, Height, Width] or [num_classes, Height, Width].
    """
    print("mask bef: ", mask.shape)

    if isinstance(mask, torch.Tensor):
        mask = mask.detach().cpu().numpy()  # Convert to NumPy array

    if len(mask.shape) == 4:
      mask = np.squeeze(mask, axis=0)

    print("mask aft: ", mask.shape)

    # Assume the mask is one-hot encoded; convert to a single-channel class map
    class_map = np.argmax(mask, axis=0)  # Shape: [Height, Width]

    plt.imshow(class_map, cmap='tab10')  # Use a categorical colormap
    plt.colorbar()
    plt.axis('off')
    plt.show()

def visualize_data(image, gt, res):
    """
    Visualize an image along with its ground truth and result masks.

    Parameters:
        image (torch.Tensor or np.ndarray): The input image. Shape: [Batch, Channel, Height, Width] or [Channel, Height, Width].
        gt (torch.Tensor or np.ndarray): The ground truth mask. Shape: [Batch, num_classes, Height, Width] or [num_classes, Height, Width].
        res (torch.Tensor or np.ndarray): The result (predicted) mask. Shape: [Batch, num_classes, Height, Width] or [num_classes, Height, Width].
    """
    # Convert inputs to NumPy arrays and handle batch dimensions
    if isinstance(image, torch.Tensor):
        image = image.detach().cpu().numpy()
    if len(image.shape) == 4:
        image = np.squeeze(image, axis=0)

    if isinstance(gt, torch.Tensor):
        gt = gt.detach().cpu().numpy()
    if len(gt.shape) == 4:
        gt = np.squeeze(gt, axis=0)

    if isinstance(res, torch.Tensor):
        res = res.detach().cpu().numpy()
    if len(res.shape) == 4:
        res = np.squeeze(res, axis=0)

    # Prepare the masks for visualization
    gt_class_map = np.argmax(gt, axis=0)  # Convert one-hot to class map
    res_class_map = np.argmax(res, axis=0)  # Convert one-hot to class map

    # Visualize the image and masks
    plt.figure(figsize=(15, 5))

    # Image
    plt.subplot(1, 3, 1)
    if image.shape[0] == 1:  # Grayscale
        plt.imshow(image[0], cmap='gray')
    elif image.shape[0] == 3:  # RGB
        plt.imshow(np.transpose(image, (1, 2, 0)))
    else:
        raise ValueError("Image should have 1 or 3 channels for visualization.")
    plt.title("Image")
    plt.axis('off')

    # Ground truth
    plt.subplot(1, 3, 2)
    plt.imshow(gt_class_map, cmap='tab10')  # Use a categorical colormap
    plt.title("Ground Truth")
    plt.axis('off')

    # Result
    plt.subplot(1, 3, 3)
    plt.imshow(res_class_map, cmap='tab10')  # Use a categorical colormap
    plt.title("Result")
    plt.axis('off')

    plt.tight_layout()
    plt.show()

def test(model, path, num_classes):
    model.eval()  # Set model to evaluation mode
    mean_loss = []
    mean_dice = []
    mean_iou = []
    mean_acc = []
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    class_weights = torch.tensor([0.1, 1.0, 1.0]).to(device)

    for s in ['val', 'test']:
        image_root = f'{path}/data_{s}.npy'
        gt_root = f'{path}/mask_{s}.npy'
        test_loader = test_dataset(image_root, gt_root, num_classes=num_classes)

        dice_bank = []
        iou_bank = []
        loss_bank = []
        acc_bank = []

        for i in range(test_loader.size):
            image, gt = test_loader.load_data()
            image = image.to(device)
            gt = gt.to(device)

            if gt.ndim == 3:  # Check if gt is missing the batch dimension
                gt = gt.unsqueeze(0)  # Add batch dimension to match res shape

            with torch.no_grad():
                _, _, res = model(image)

            # Handle multi-class vs binary segmentation
            if num_classes > 1:
                res = res.softmax(dim=1)  # Apply softmax for probabilities
                loss = structure_loss(res, gt.long(), num_classes)  # Compute loss
            else:
                res = res.sigmoid()  # Binary sigmoid
                loss = structure_loss(res, gt.float(), num_classes)  # Compute loss

            # Convert tensors to NumPy arrays
            image = image.cpu().numpy().squeeze()
            res = res.cpu().numpy().squeeze()
            gt = gt.cpu().numpy().squeeze()

            if i == 1:
                visualize_data(image, gt, res)

            # Compute metrics
            dice = mean_dice_np(gt, res, num_classes=num_classes)
            iou = mean_iou_np(gt, res, num_classes=num_classes)
            acc = np.mean(res == gt)  # Compute accuracy

            loss_bank.append(loss.item())
            dice_bank.append(dice)
            iou_bank.append(iou)
            acc_bank.append(acc)

        print(f'{s} Loss: {np.mean(loss_bank):.4f}, Dice: {np.mean(dice_bank):.4f}, IoU: {np.mean(iou_bank):.4f}, Acc: {np.mean(acc_bank):.4f}')
        mean_loss.append(np.mean(loss_bank))
        mean_dice.append(np.mean(dice_bank))
        mean_iou.append(np.mean(iou_bank))
        mean_acc.append(np.mean(acc_bank))

    return sum(mean_loss)/len(mean_loss), sum(mean_dice)/len(mean_dice), sum(mean_iou)/len(mean_iou), sum(mean_acc)/len(mean_acc)

def train(train_loader, model, optimizer, grad_norm, model_save_path, train_save, test_path, max_epoch, current_epoch, best_loss, no_improve_epoch, num_classes):
    model.train()  # Set model to training mode
    loss_record2, loss_record3, loss_record4 = AvgMeter(), AvgMeter(), AvgMeter()
    accum = 0
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    for i, pack in enumerate(train_loader, start=1):
        # ---- Data Preparation ----
        images, gts = pack
        images = images.to(device)
        gts = gts.to(device)

        # ---- Forward Pass ----
        lateral_map_4, lateral_map_3, lateral_map_2 = model(images)

        # ---- Loss Function ----
        loss4 = structure_loss(lateral_map_4, gts, num_classes)
        loss3 = structure_loss(lateral_map_3, gts, num_classes)
        loss2 = structure_loss(lateral_map_2, gts, num_classes)

        # Combined loss with weights
        loss = 0.6 * loss2 + 0.2 * loss3 + 0.2 * loss4

        # ---- Backward and Optimize ----
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), grad_norm)
        optimizer.step()
        optimizer.zero_grad()

        # ---- Record Loss ----
        batch_size = images.size(0)
        loss_record2.update(loss2.item(), batch_size)
        loss_record3.update(loss3.item(), batch_size)
        loss_record4.update(loss4.item(), batch_size)

        # ---- Train Visualization ----
        total_step = len(train_loader)
        if i % 20 == 0 or i == total_step:
            print(f'{datetime.now()} Epoch [{current_epoch:03d}/{max_epoch:03d}], '
                  f'Step [{i:04d}/{total_step:04d}], '
                  f'[lateral-2: {loss_record2.show():.4f}, '
                  f'lateral-3: {loss_record3.show():.4f}, '
                  f'lateral-4: {loss_record4.show():.4f}]')

    # ---- Save Model if Performance Improves ----
    save_path = f'{model_save_path}/{train_save}/'
    os.makedirs(save_path, exist_ok=True)

    mean_loss, mean_dice, mean_iou, mean_acc = test(model, test_path, num_classes)
    if mean_loss < best_loss:
        print('New best loss:', mean_loss)
        best_loss = mean_loss
        torch.save(model.state_dict(), f'{save_path}TransFuse-S_test_best.pth')
        print('[Saving Snapshot:]', f'{save_path}TransFuse-S_test_best.pth')
        no_improve_epoch = 0
    else:
        no_improve_epoch += 1

    return mean_loss, mean_dice, mean_iou, mean_acc, no_improve_epoch



if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--max_epoch', type=int, default=100, help='max epoch')
    parser.add_argument('--stop_epoch', type=int, default=10, help='stop epoch')
    parser.add_argument('--lr', type=float, default=7e-5, help='learning rate')
    parser.add_argument('--batchsize', type=int, default=16, help='training batch size')
    parser.add_argument('--grad_norm', type=float, default=2.0, help='gradient clipping norm')
    parser.add_argument('--train_path', type=str,
                        default='data/', help='path to train dataset')
    parser.add_argument('--test_path', type=str,
                        default='data/', help='path to test dataset')
    parser.add_argument('--model_save_path', type=str, default='snapshots', help='path to save model')
    parser.add_argument('--train_save', type=str, default='TransFuse_S')
    parser.add_argument('--beta1', type=float, default=0.5, help='beta1 of adam optimizer')
    parser.add_argument('--beta2', type=float, default=0.999, help='beta2 of adam optimizer')
    parser.add_argument('--num_classes', type=float, default=0.999, help='number of classes')

    opt = parser.parse_args()

    # ---- build models ----
    model = TransFuse_S(pretrained=True).cuda()
    params = model.parameters()
    optimizer = torch.optim.Adam(params, opt.lr, betas=(opt.beta1, opt.beta2))
     
    image_root = '{}/data_train.npy'.format(opt.train_path)
    gt_root = '{}/mask_train.npy'.format(opt.train_path)

    train_loader = get_loader(image_root, gt_root, batchsize=opt.batchsize)
    total_step = len(train_loader)

    print("#"*20, "Start Training", "#"*20)

    best_loss = 1e5
    no_improve_epoch = 0    # number of times epoch did not improve
    current_epoch = 1       # current epoch

    while current_epoch < opt.max_epoch + 1 and no_improve_epoch < opt.stop_epoch:
        best_loss, mean_dice, mean_iou, mean_acc, no_improve_epoch = train(train_loader, model, optimizer, current_epoch, best_loss, no_improve_epoch, opt.num_classes)
        current_epoch += 1
    print("Training completed!")