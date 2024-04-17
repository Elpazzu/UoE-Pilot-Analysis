import os
import argparse
import csv
import numpy as np
import torch
from torch import nn, optim
from torch.utils.data import DataLoader, random_split
from torch.optim.lr_scheduler import ReduceLROnPlateau
from net.att_unet import AttUNet
from utils.dataloader import ISLES2015SISSDataset
from utils.loss import BinaryDiceLoss, MetricsTracker

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

parser = argparse.ArgumentParser()
parser.add_argument("--tag",            type=str,   help="Tag of this training session",                default="att")  # noatt
parser.add_argument("--backbone",       type=str,   help="type of backbone net",                        default="ResNet50")
parser.add_argument("--batch_size",     type=int,   help="batch size",                                  default=16)
parser.add_argument("--data_root",      type=str,   help="root of train set",                           default="./dataset")
parser.add_argument("--model_root",     type=str,   help="root of train set",                           default="./out_models")
parser.add_argument("--log_root",       type=str,   help="root of train set",                           default="./logs")
parser.add_argument("--pretrained",     type=str,   help="(Optional) Path of pretrained model",         default=None)
parser.add_argument("--lr",             type=float, help="initial learning rate",                       default=1e-3)
parser.add_argument("--attention",      type=int,   help="Use attention gate or not",                   default=1)

args, _ = parser.parse_known_args()
train_tag = args.tag
net_type = args.backbone
batch_size = args.batch_size
data_root = args.data_root
model_root = os.path.join(args.model_root, train_tag)
log_root = os.path.join(args.log_root, train_tag)
model_path = args.pretrained
learning_rate = args.lr
attention = args.attention

if not os.path.exists(model_root):
    os.makedirs(model_root)
if not os.path.exists(log_root):
    os.makedirs(log_root)

data_root = "/dbfs/tcdh-isles/All_Images/Augmented_v2"  # specify DBFS path of input training dataset
dataset = ISLES2015SISSDataset(data_root)
train_dataset, val_dataset = torch.utils.data.random_split(dataset, [int(0.8*len(dataset)), len(dataset)-int(0.8*len(dataset))])
H, W = dataset.get_img_size()  # extract dimensions of input training images

# Construct net
#net = AttUNet(net_type, is_train=True, attention=attention).to(device)  # for training with attention
net = AttUNet(net_type, is_train=True, attention=False).to(device)  # for training without attention

print("Traning Session {} \nBrief:\nBackbone_type:{} | batch_size:{} | lr:{} | model_root:{} | Log root:{}".format(train_tag, net_type, batch_size, learning_rate, model_root, log_root))
print("img_size:{}*{} | trainset size:{}".format(W, H, len(dataset)))

print("\nStart training ...")

# Optimization method
opt = optim.Adam(net.parameters(), lr=learning_rate)
scheduler = ReduceLROnPlateau(opt, mode="min", patience=4, verbose=True)  # learning rate adjustment on validation loss

# Loss function
loss_bce = nn.BCELoss()  # BCE loss
loss_dice = BinaryDiceLoss()  # Dice loss

# Performance metrics
metrics = MetricsTracker(train_tag, log_root)  # create instance of imported MetricsTracker class

epochs = 10
k_folds = 5
batch_size = 16
weight_bce = 0.3  # assign weight to BCE loss during training
weight_dice = 0.7  # assign weight to Dice loss during training
min_epoch_loss = np.inf  # to track the minimum loss observed across epochs during training
min_epoch = 0  # to store number of the epoch with the minimum loss observed during training, initialized at 0

for fold in range(k_folds):
    # Initialize data loaders for training and validation sets for current fold
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size)

    # Start training loop for current fold
    for epoch in range(1, epochs + 1):
        metrics.set_epoch(epoch)
        loss_sum = 0
        iter_count = 0
        min_loss = np.inf

        # Training loop
        for i, (img, label) in enumerate(train_loader):
            img, label = img.to(device), label.to(device)
            out_image = net(img)
            out_image = out_image.repeat(1, 3, 1, 1)
            one_hot = (out_image > 0.69).float()

            iter_dice_loss = loss_dice(one_hot, label)
            out_image = out_image.clone().detach().requires_grad_(True)
            iter_bce_loss = loss_bce(out_image, label)
            iter_loss = weight_bce * iter_bce_loss + weight_dice * iter_dice_loss

            opt.zero_grad()
            iter_loss.backward()
            opt.step()

            if i % 10 == 0:
                print("--Fold: {}, Epoch: {}, iter: {}, iter_loss: {}".format(fold+1, epoch, i, iter_loss.item()))

            with torch.no_grad():
                loss_sum += iter_loss.item()
                iter_count += 1
                if iter_loss.item() < min_loss:
                    min_loss = iter_loss.item()
                metrics.update(one_hot, label, iter_dice_loss.item())

        val_loss_sum = 0
        val_iter_count = 0

        # Validation loop
        net.eval()
        with torch.no_grad():
            for i, (img, label) in enumerate(val_loader):
                img, label = img.to(device), label.to(device)
                out_image = net(img)
                out_image = out_image.repeat(1, 3, 1, 1)
                one_hot = (out_image > 0.69).float()

                iter_dice_loss = loss_dice(one_hot, label)
                out_image = out_image.clone().detach().requires_grad_(True)
                iter_bce_loss = loss_bce(out_image, label)
                iter_loss = weight_bce * iter_bce_loss + weight_dice * iter_dice_loss

                val_loss_sum += iter_loss.item()
                val_iter_count += 1

        avg_val_loss = val_loss_sum / val_iter_count
        curr_dice = metrics.get_metrics()

        if ((epoch - 1) % 5 == 0):
            torch.save(net.state_dict(), os.path.join("/dbfs/tcdh-isles/net/", "{}_{}_{}.pth".format(train_tag, fold+1, epoch)))
            metrics.save_logs()

        if avg_val_loss < min_epoch_loss:
            torch.save(net.state_dict(), os.path.join("/dbfs/tcdh-isles/net/", "{}_{}_min.pth".format(train_tag, fold+1)))
            min_epoch_loss = avg_val_loss
            min_epoch = epoch
            metrics.save_logs()

        print("Fold: {} | Epoch {} | val_loss:{} | min_loss:{} | Dice:{} | min epoch:{}".format(fold+1, epoch, avg_val_loss, min_loss, curr_dice, min_epoch))
        
        scheduler.step(avg_val_loss)
        torch.cuda.empty_cache()
