#!/usr/bin/env python
# coding: utf-8

# In[1]:


import os
import time
import copy
import random
import numpy as np
import torch
import torch.optim as optim
from torch.utils.data import DataLoader, ConcatDataset
from torch.cuda.amp import autocast, GradScaler
from utils.ssd_model import (
    make_datapath_list, VOCDataset, Anno_xml2list, DataTransform,
    SSD, MultiBoxLoss, make_loc_conf, od_collate_fn
)


# In[2]:


# 変換（ノイズ入り）: DataLoader worker で pickle 可能なトップレベル定義
from utils.data_augumentation import (
    Compose, ConvertFromInts, ToAbsoluteCoords, PhotometricDistort, Expand,
    RandomSampleCrop, RandomMirror, ToPercentCoords, Resize, SubtractMeans,
    RandomGaussianNoise
)

class DataTransformNoisy:
    """train: DataTransform + GaussianNoise / val: 既存相当"""
    def __init__(self, input_size=300, color_mean=(104,117,123)):
        self.data_transform = {
            "train": Compose([
                ConvertFromInts(),
                ToAbsoluteCoords(),
                PhotometricDistort(),
                Expand(color_mean),
                RandomSampleCrop(),
                RandomMirror(),
                ToPercentCoords(),
                Resize(input_size),
                RandomGaussianNoise(prob=0.5, sigma_range=(3, 18)),
                SubtractMeans(color_mean),
            ]),
            "val": Compose([
                ConvertFromInts(),
                Resize(input_size),
                SubtractMeans(color_mean),
            ]),
        }

    def __call__(self, img, phase, boxes, labels):
        return self.data_transform[phase](img, boxes, labels)


# In[3]:


def set_seed(seed=42):
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


# In[4]:


rootpath         = "./data/"
weight_path      = "./weights/vgg16_reducedfc.pth"
save_weight_path = "./weights/ssd_finetuned_filterv2.pth"
classes          = ["apple", "orange", "banana"]   # XMLの<name>と一致させる


# In[5]:


color_mean       = (104, 117, 123)
input_size       = 300
batch_size       = 16
num_epochs       = 200
set_seed(42)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("[INFO] device:", device)
torch.backends.cudnn.benchmark = True


# In[6]:


# ========== 3. データセットのパスリスト取得 ==========
train_img_list, train_anno_list, val_img_list, val_anno_list = make_datapath_list(rootpath)
print(f"[INFO] train ids: {len(train_img_list)}, val ids: {len(val_img_list)}")


# In[7]:


transform_clean = DataTransform(input_size, color_mean)
transform_noisy = DataTransformNoisy(input_size=input_size, color_mean=color_mean)
transform_anno  = Anno_xml2list(classes)


# In[8]:


train_dataset_clean = VOCDataset(
    train_img_list, train_anno_list,
    phase="train", transform=transform_clean, transform_anno=transform_anno
)
train_dataset_noisy = VOCDataset(
    train_img_list, train_anno_list,
    phase="train", transform=transform_noisy, transform_anno=transform_anno
)
train_dataset_concat = ConcatDataset([train_dataset_clean, train_dataset_noisy])

val_dataset = VOCDataset(
    val_img_list, val_anno_list,
    phase="val", transform=transform_clean, transform_anno=transform_anno
)


# In[9]:


# ノートブック実行では、まずは worker=0 で通すのが安全
# 速度を上げたい場合は num_workers を徐々に増やしてください（例: 2, 4 ...）
num_workers = 0
use_workers = num_workers > 0

train_dataloader = DataLoader(
    train_dataset_concat, batch_size=batch_size, shuffle=True,
    collate_fn=od_collate_fn, drop_last=False,
    num_workers=num_workers,
    pin_memory=True,
    persistent_workers=use_workers,                 # ← 条件付きに
    prefetch_factor=(4 if use_workers else None)    # ← 条件付きに
)
val_dataloader = DataLoader(
    val_dataset, batch_size=batch_size, shuffle=False,
    collate_fn=od_collate_fn, drop_last=False,
    num_workers=(max(1, num_workers // 2) if use_workers else 0),
    pin_memory=True,
    persistent_workers=use_workers,                 # ← 条件付きに
    prefetch_factor=(2 if use_workers else None)    # ← 条件付きに
)
dataloaders_dict = {"train": train_dataloader, "val": val_dataloader}


# In[10]:


for phase in ['train','val']:
    imgs, tars = next(iter(dataloaders_dict[phase]))
    print(f"[CHECK] {phase} batch: imgs={imgs.shape}, targets={[t.shape for t in tars]}")


# In[11]:


for phase in ['train', 'val']:
    try:
        imgs, tars = next(iter(dataloaders_dict[phase]))
        print(f"[CHECK] {phase} batch: imgs={imgs.shape}, targets={[t.shape for t in tars]}")
        break
    except Exception as e:
        print(f"[STOP] dataloader error in {phase} :", e)
        raise


# In[12]:


ssd_cfg = {
    'num_classes': len(classes) + 1,  # 背景含む
    'input_size': 300,
    'bbox_aspect_num': [4, 6, 6, 6, 4, 4],
    'feature_maps': [38, 19, 10, 5, 3, 1],
    'steps': [8, 16, 32, 64, 100, 300],
    'min_sizes': [21, 45, 99, 153, 207, 261],
    'max_sizes': [45, 99, 153, 207, 261, 315],
    'aspect_ratios': [[2], [2, 3], [2, 3], [2, 3], [2], [2]],
}

net = SSD(phase="train", cfg=ssd_cfg)

# VGG初期重みロード（CPUでロード→GPUへ）
print("[INFO] load VGG weights from:", weight_path)
vgg_weights = torch.load(weight_path, map_location="cpu")
net.vgg.load_state_dict(vgg_weights)


# In[13]:


def change_head(net, num_classes):
    loc_layers, conf_layers = make_loc_conf(num_classes, ssd_cfg["bbox_aspect_num"])
    net.loc = loc_layers
    net.conf = conf_layers
    # 再初期化
    for layer in net.conf:
        torch.nn.init.xavier_uniform_(layer.weight)
        if layer.bias is not None:
            torch.nn.init.constant_(layer.bias, 0)
    for layer in net.loc:
        torch.nn.init.xavier_uniform_(layer.weight)
        if layer.bias is not None:
            torch.nn.init.constant_(layer.bias, 0)

change_head(net, ssd_cfg['num_classes'])
net = net.to(device)
print("[INFO] model ready on:", next(net.parameters()).device)


# In[14]:


criterion = MultiBoxLoss(jaccard_thresh=0.5, neg_pos=3, device=device)
optimizer = optim.SGD(net.parameters(), lr=1e-3, momentum=0.9, weight_decay=5e-4)
scaler = GradScaler(enabled=(device.type == 'cuda'))

def get_lr(optim):
    for g in optim.param_groups:
        return g.get("lr", None)


# In[15]:


# ---------- 学習ループ（AMP + 最良モデル保存） ----------
best_loss = float('inf')
best_model_wts = copy.deepcopy(net.state_dict())
print("使用デバイス:", device)
history = {"train_loss": [], "val_loss": []}

for epoch in range(num_epochs):
    epoch_start = time.time()
    print(f"\nEpoch {epoch+1}/{num_epochs}")
    epoch_losses = {"train": None, "val": None}

    for phase in ["train", "val"]:
        net.train(mode=(phase == "train"))
        running_loss = 0.0
        sample_count = 0

        for images, targets in dataloaders_dict[phase]:
            images = images.to(device, non_blocking=True)
            targets = [ann.to(device) for ann in targets]

            optimizer.zero_grad(set_to_none=True)
            with torch.set_grad_enabled(phase == "train"):
                with autocast(enabled=(device.type == 'cuda')):
                    outputs = net(images)
                    loss_l, loss_c = criterion(outputs, targets)
                    loss = loss_l + loss_c
                if phase == "train":
                    scaler.scale(loss).backward()
                    scaler.step(optimizer)
                    scaler.update()

            bsz = images.size(0)
            running_loss += loss.item() * bsz
            sample_count += bsz

        epoch_loss = running_loss / max(sample_count, 1)
        epoch_losses[phase] = epoch_loss
        history[f"{phase}_loss"].append(epoch_loss)

        if phase == "val" and epoch_loss < best_loss:
            best_loss = epoch_loss
            best_model_wts = copy.deepcopy(net.state_dict())
            os.makedirs(os.path.dirname(save_weight_path), exist_ok=True)
            torch.save(best_model_wts, save_weight_path)
            print(f"[BEST] val loss improved to {best_loss:.4f} → saved: {save_weight_path}")

    epoch_time = time.time() - epoch_start
    print(
        f"epoch {epoch+1:>3}/{num_epochs} | {epoch_time:.2f} sec | "
        f"train_Loss: {epoch_losses['train']:.4f} | val_Loss: {epoch_losses['val']:.4f}"
    )

# 最良重みで保存（冪等）
net.load_state_dict(best_model_wts)
torch.save(best_model_wts, save_weight_path)
print("\nTraining complete!")
print(f"Best val loss: {best_loss:.4f} | Saved: {save_weight_path}")


# In[ ]:


get_ipython().system(' jupyter nbconvert --to python "train base cfg_filterv1".ipynb')


# ---------------------------
