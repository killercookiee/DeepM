import torch
import torch.nn.functional as F
import numpy as np
import os, argparse
from lib.TransFuse import TransFuse_S, TransFuse_L, TransFuse_L_384
from utils.dataloader import test_dataset
import imageio


def mean_dice_np(gt, pred, num_classes):
    dice_scores = []
    for c in range(num_classes):
        gt_c = (gt == c)
        pred_c = (pred == c)
        intersection = np.sum(gt_c & pred_c)
        union = np.sum(gt_c) + np.sum(pred_c)
        dice_scores.append(2 * intersection / union if union > 0 else 1.0)
    return np.mean(dice_scores)

def mean_iou_np(gt, pred, num_classes):
    iou_scores = []
    for c in range(num_classes):
        gt_c = (gt == c)
        pred_c = (pred == c)
        intersection = np.sum(gt_c & pred_c)
        union = np.sum(gt_c | pred_c)
        iou_scores.append(intersection / union if union > 0 else 1.0)
    return np.mean(iou_scores)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--ckpt_path', type=str, default='snapshots/TransFuse-19_best.pth')
    parser.add_argument('--test_path', type=str,
                        default='data/', help='path to test dataset')
    parser.add_argument('--save_path', type=str, default=None, help='path to save inference segmentation')

    opt = parser.parse_args()

    # Check if CUDA is available
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    model = TransFuse_S().to(device)
    model.load_state_dict(torch.load(opt.ckpt_path, map_location=device))
    model.eval()

    if opt.save_path is not None:
        os.makedirs(opt.save_path, exist_ok=True)

    print('Evaluating model:', opt.ckpt_path)

    image_root = f'{opt.test_path}/data_test.npy'
    gt_root = f'{opt.test_path}/mask_test.npy'
    test_loader = test_dataset(image_root, gt_root)

    dice_bank = []
    iou_bank = []
    acc_bank = []

    for i in range(test_loader.size):
        image, gt = test_loader.load_data()
        gt = 1 * (gt > 0.5)
        image = image.to(device)

        with torch.no_grad():
            _, _, res = model(image)

        res = res.sigmoid().data.cpu().numpy().squeeze()
        res = 1 * (res > 0.5)

        if opt.save_path is not None:
            imageio.imwrite(f'{opt.save_path}/{i}_pred.jpg', res)
            imageio.imwrite(f'{opt.save_path}/{i}_gt.jpg', gt)

        dice = mean_dice_np(gt, res)
        iou = mean_iou_np(gt, res)
        acc = np.sum(res == gt) / (res.shape[0] * res.shape[1])

        acc_bank.append(acc)
        dice_bank.append(dice)
        iou_bank.append(iou)

    print('Dice: {:.4f}, IoU: {:.4f}, Acc: {:.4f}'.
          format(np.mean(dice_bank), np.mean(iou_bank), np.mean(acc_bank)))