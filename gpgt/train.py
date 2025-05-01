import argparse
import os

import torch
from torch.autograd import Variable
from torch.optim.lr_scheduler import _LRScheduler
from torch.utils.data import DataLoader
from tqdm import tqdm

from dataset import BasicDataset, TrainBasicDataset
from unet import UNet
from utils.loss import SegmentationLosses
from utils.metrics import Evaluator


def parse_args():
    """
    Parse input arguments
    """
    parser = argparse.ArgumentParser(description='Train a FPN Semantic Segmentation network')
    parser.add_argument('--resume', dest='resume',
                        help='resume or not',
                        default=False, type=bool)
    parser.add_argument('--start_epoch', dest='start_epoch',
                        help='starting epoch',
                        default=1, type=int)
    parser.add_argument('--end_epoch', dest='end_epoch',
                        help='end epoch',
                        default=1, type=int)
    parser.add_argument('--epochs', dest='epochs',
                        help='number of iterations to train',
                        default=100, type=int)

    parser.add_argument('--save_dir', dest='save_dir',
                        help='directory to save models',
                        default=None, type=str)
    parser.add_argument('--dataset', dest='dataset',
                        help='use which dataset',
                        default='segB', type=str)

    parser.add_argument('--cuda', dest='cuda',
                        help='whether use CUDA', default=True, type=bool)
    parser.add_argument('--gpu_id', dest='gpu_id',
                        help='use which gpu', default=0, type=int)

    parser.add_argument('--batch_size', dest='batch_size',
                        help='batch_size',
                        default=4, type=int)

    # config optimization
    parser.add_argument('--o', dest='optimizer',
                        help='training optimizer',
                        default='sgd', type=str)
    parser.add_argument('--lr', dest='lr',
                        help='starting learning rate',
                        default=0.0001, type=float)
    parser.add_argument('--weight_decay', dest='weight_decay',
                        help='weight_decay',
                        default=0, type=float)
    parser.add_argument('--lr_decay_step', dest='lr_decay_step',
                        help='step to do learning rate decay, uint is epoch',
                        default=50, type=int)
    parser.add_argument('--lr_decay_gamma', dest='lr_decay_gamma',
                        help='learning rate decay ratio',
                        default=0.1, type=float)

    args = parser.parse_args()
    return args


class PolyLR(_LRScheduler):
    """Set the learning rate of each parameter group to the initial lr decayed
    by gamma every epoch. When last_epoch=-1, sets initial lr as lr.

    Args:
        optimizer (Optimizer): Wrapped optimizer.
        gamma (float): Multiplicative factor of learning rate decay.
        last_epoch (int): The index of last epoch. Default: -1.
    """

    def __init__(self, optimizer, max_iter, power, last_epoch=-1):
        self.max_iter = max_iter
        self.power = power
        super(PolyLR, self).__init__(optimizer, last_epoch)

    def get_lr(self):
        return [base_lr * (1 - self.last_epoch / self.max_iter) ** self.power
                for base_lr in self.base_lrs]


NUM_CLASS = 3
args = parse_args()
device = torch.device(f"cuda:{args.gpu_id}" if torch.cuda.is_available() else "cpu")
print('device:{}'.format(device))
kwargs = {'num_workers': 0, 'pin_memory': True}

# TODO change the image_path and seg_label_path
image_path = ""
seg_label_path = ""
train_dataset = BasicDataset(image_path, seg_label_path, train=True)
train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, num_workers=0, pin_memory=False)

unet = UNet(n_channels=1, n_classes=3, bilinear=False)
unet = unet.to(device)

weight = None
criterion = SegmentationLosses(weight=weight, cuda=args.cuda).build_loss(mode='ce')
optimizer = torch.optim.Adam(unet.parameters(), lr=args.lr, weight_decay=args.weight_decay)

optimizer_lr_scheduler = PolyLR(optimizer, max_iter=args.epochs, power=0.9)
evaluator = Evaluator(NUM_CLASS)

def train(epoch, optimizer, train_loader):
    unet.train()
    with tqdm(initial=0, total=int(len(train_loader)), desc=f"epoch {epoch}") as pbar:
        for iteration, batch in enumerate(train_loader):
            image, target = batch['image'], batch['label']
            image = image / 255.0
            target = target.squeeze(1)
            inputs = image.to(device)
            labels = target.to(device)
            unet.zero_grad()

            inputs = Variable(inputs)
            labels = Variable(labels)
            out = unet(inputs, True)
            loss_ss = criterion(out, labels.long())
            loss_ss.backward(torch.ones_like(loss_ss))
            optimizer.step()

            pbar.update(1)
            pbar.set_postfix_str("Loss:{:.4f}".format(loss_ss.data))
    torch.save(unet, f'{model_path}/{args.save_dir}/unet' + str(epoch) + '.pth')


if __name__ == '__main__':
    model_path = './models'
    if not os.path.exists(f"{model_path}/{args.save_dir}"):
        os.makedirs(f"{model_path}/{args.save_dir}")

    start_epoch = 1
    if args.resume:
        start_epoch = args.start_epoch + 1
        unet = torch.load(f'{model_path}/{args.save_dir}/unet' + str(args.start_epoch) + '.pth')
    assert start_epoch < args.end_epoch + 1, "error, start_epoch >= epochs!"

    best_pred = 0.0
    for epoch in range(start_epoch, args.end_epoch + 1):
        train(epoch, optimizer, train_loader)
