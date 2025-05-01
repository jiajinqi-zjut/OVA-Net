import argparse
import os

import cupy as cp
import numpy as np
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

from dataset import BasicDataset
from utils.metrics import Evaluator


def parse_args():
    parser = argparse.ArgumentParser(description='Train a FPN Semantic Segmentation network')
    parser.add_argument('--gpu_id', dest='gpu_id',
                        help='use which gpu', default=0, type=int)

    parser.add_argument('--cp_dir', dest='cp_dir',
                        help='directory to save models',
                        default=None)
    parser.add_argument('--save_dir', dest='save_dir',
                        help='directory to save data',
                        default=None)
    parser.add_argument('--start_epoch', dest='start_epoch',
                        help='load which checkpoint',
                        default=1, type=int)
    parser.add_argument('--end_epoch', dest='end_epoch',
                        help='load which checkpoint',
                        default=1, type=int)
    parser.add_argument('--dataset', dest='dataset',
                        help='use which dataset',
                        default='segB', type=str)
    return parser.parse_args()


def collect_own(label):
    label0 = np.argmax(label == 1, axis=0)
    label1 = np.argmax(label == 2, axis=0) - 1
    label2 = (label.shape[0] - 1) - np.argmax((label[::-1, :]) == 2, axis=0)
    return label0, label1, label2


def HD_AMPD_RASMPD_cupy(a, b):
    with cp.cuda.Device(0):
        a = cp.asarray(a)
        b = cp.array(b)

        min_a = cp.sqrt(
            cp.square(a[:, cp.newaxis] - b) + cp.square(cp.arange(len(a))[:, cp.newaxis] - cp.arange(len(b))))
        min_a = cp.min(min_a, axis=1)
        min_a2 = cp.square(min_a)

        min_b = cp.sqrt(
            cp.square(b[:, cp.newaxis] - a) + cp.square(cp.arange(len(b))[:, cp.newaxis] - cp.arange(len(a))))
        min_b = cp.min(min_b, axis=1)
        min_b2 = cp.square(min_b)

        max_value = cp.max(cp.concatenate([min_a, min_b]))
        avg_value = (cp.sum(min_a) + cp.sum(min_b)) / (2 * len(a))
        std_value = cp.sqrt((cp.sum(min_a2) + cp.sum(min_b2)) / (2 * len(a)))

        return cp.asnumpy(max_value), cp.asnumpy(avg_value), cp.asnumpy(std_value)


def compute_miou(label, out, use_all=False):
    if not use_all:
        label = label[:, 200:1600]
        out = out[:, 200:1600]

    and_sum_1 = np.sum(np.logical_and(np.where(label == 1, True, False), np.where(out == 1, True, False)))
    or_sum_1 = np.sum(np.logical_or(np.where(label == 1, True, False), np.where(out == 1, True, False)))
    miou1 = and_sum_1 / or_sum_1
    and_sum_2 = np.sum(np.logical_and(np.where(label == 2, True, False), np.where(out == 2, True, False)))
    or_sum_2 = np.sum(np.logical_or(np.where(label == 2, True, False), np.where(out == 2, True, False)))
    miou2 = and_sum_2 / or_sum_2
    return miou1, miou2


def test_mae(args, device):
    for epoch in range(args.start_epoch, args.end_epoch + 1):
        checkpoint = f'unet{epoch}.pth'
        unet = torch.load(f'{model_path}/{args.cp_dir}/{checkpoint}')
        unet = unet.to(device)
        unet.eval()
        evaluator.reset()
        with tqdm(initial=0, total=int(len(val_loader)), desc=f"checkpoint {checkpoint}") as pbar:
            for iteration, batch in enumerate(val_loader):
                index, images, target = int(batch['idx'][0]), batch['image'], batch['label']
                target = target.squeeze(1)
                image = images / 255.0
                image = image.to(device)
                target = target.to(device)
                with torch.no_grad():
                    out = unet(image, False)
                out = out.data.cpu().numpy()
                label = target.cpu().numpy()
                label = np.squeeze(label)

                out = np.argmax(out, axis=1)
                out = out.squeeze(0)
                label_pre1, label_pre2, label_pre3 = collect_own(out)
                label1, label2, label3 = collect_own(label)

                mae1 = torch.mean(torch.abs(torch.tensor(label_pre1, dtype=torch.float32) - torch.tensor(label1, dtype=torch.float32)))
                mae2 = torch.mean(torch.abs(torch.tensor(label_pre2, dtype=torch.float32) - torch.tensor(label2, dtype=torch.float32)))
                mae3 = torch.mean(torch.abs(torch.tensor(label_pre3, dtype=torch.float32) - torch.tensor(label3, dtype=torch.float32)))

                layer1_MSE = np.sum(np.power([value1 - value2 for value1, value2 in zip(label1, label_pre1)], 2)) / 1792
                layer2_MSE = np.sum(np.power([value1 - value2 for value1, value2 in zip(label2, label_pre2)], 2)) / 1792
                layer3_MSE = np.sum(np.power([value1 - value2 for value1, value2 in zip(label3, label_pre3)], 2)) / 1792

                HDS1, AMPDS1, RASMPDS1 = HD_AMPD_RASMPD_cupy(label1, label_pre1)
                HDS2, AMPDS2, RASMPDS2 = HD_AMPD_RASMPD_cupy(label2, label_pre2)
                HDS3, AMPDS3, RASMPDS3 = HD_AMPD_RASMPD_cupy(label3, label_pre3)

                miou1, miou2 = compute_miou(label, out)

                dir_path = os.path.join(f'{results_path}/{args.save_dir}')
                if not os.path.exists(dir_path):
                    os.mkdir(dir_path)
                csv_file = f'{results_path}/{args.save_dir}/{args.cp_dir}-{checkpoint[:-4]}.csv'
                if not os.path.exists(csv_file):
                    with open(csv_file, 'w') as file:
                        file.write(
                            'train,checkpoint,image,miou1,miou2,'
                            'layer1_MAE,layer1_MSE,HDS1,AMPDS1,RASMPDS1,'
                            'layer2_MAE,layer2_MSE,HDS2,AMPDS2,RASMPDS2,'
                            'layer3_MAE,layer3_MSE,HDS3,AMPDS3,RASMPDS3\n')
                with open(csv_file, 'a') as file:
                    print(
                        f'{args.cp_dir},{checkpoint},{index},{miou1},{miou2},{mae1},{layer1_MSE},{HDS1},{AMPDS1},{RASMPDS1},'
                        f'{mae2},{layer2_MSE},{HDS2},{AMPDS2},{RASMPDS2},{mae3},{layer3_MSE},{HDS3},{AMPDS3},{RASMPDS3}',
                        file=file)
                pbar.update(1)


if __name__ == '__main__':
    NUM_CLASS = 4
    args = parse_args()
    device = torch.device(f"cuda:{args.gpu_id}" if torch.cuda.is_available() else "cpu")
    kwargs = {'num_workers': 0, 'pin_memory': True}

    # TODO change the image_path and seg_label_path
    image_path = ""
    seg_label_path = ""
    val_dataset = BasicDataset(image_path, seg_label_path, train=False)
    val_loader = DataLoader(val_dataset, batch_size=1, shuffle=False, num_workers=0, pin_memory=False)
    model_path = './models'
    results_path = './results'
    evaluator = Evaluator(NUM_CLASS)
    test_mae(args, device)
