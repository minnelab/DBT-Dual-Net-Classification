import cv2
import numpy as np

import matplotlib.pyplot as plt
import torch.nn as nn
import torch
import torchvision
from torchvision import  transforms
from torch.utils.data import Dataset,DataLoader
from torchvision import models
import torch.optim as optim
from PIL import Image
import json
import os
import random
import torch.nn.functional as F

class CustomDataset(Dataset):
    def __init__(self, dataset_path, json_path, transform=None, augment_transform=None, n=2):
        self.dataset_path = dataset_path
        self.transform = transform
        self.augment_transform = augment_transform
        self.n = n

        with open(json_path, 'r') as f:
            self.labels = json.load(f)
        
        # 获取所有PNG图片的文件名
        self.image_files = [f for f in os.listdir(dataset_path) if f.endswith('.png')]
    
    def __len__(self):                              # 数据量扩大n倍
        return len(self.image_files) * self.n
    
    def _extract_prefix(self, filename):
        parts = filename.split('_')
        if len(parts) >= 3:
            prefix = '_'.join(parts[:3])
            view_char = parts[2]
            return prefix,view_char
        return filename  # 如果不足三个 '_'，返回原文件名
    
    def __getitem__(self, idx):
        original_idx = idx % len(self.image_files)

        img_name = self.image_files[original_idx]

        img_path = os.path.join(self.dataset_path, img_name)

        image = cv2.imread(img_path, cv2.IMREAD_COLOR)  # 读取灰度图像

        slice_r = image[:, :, 0]
        slice_g = image[:, :, 1]
        slice_b = image[:, :, 2]  

        def pseudo_rgb(slice):
            normalized_slice = slice.astype(np.float32) / 255.0
            normalized_slice = (normalized_slice * 255).astype(np.uint8)
            return cv2.applyColorMap(normalized_slice, cv2.COLORMAP_JET)

        pseudo_rgb_r = pseudo_rgb(slice_r)
        pseudo_rgb_g = pseudo_rgb(slice_g)
        pseudo_rgb_b = pseudo_rgb(slice_b)

        rgb_images = [
            Image.fromarray(cv2.cvtColor(pseudo_rgb_r, cv2.COLOR_BGR2RGB)),
            Image.fromarray(cv2.cvtColor(pseudo_rgb_g, cv2.COLOR_BGR2RGB)),
            Image.fromarray(cv2.cvtColor(pseudo_rgb_b, cv2.COLOR_BGR2RGB))
        ]

        img_prefix,view_char = self._extract_prefix(img_name)
        if view_char.lower().startswith('r'):
            rgb_images = [transforms.functional.hflip(img) for img in rgb_images]  # 水平翻转图像
        label = -1  # 默认标签
        for key in self.labels:
            key_prefix,_ = self._extract_prefix(key)
            if key_prefix == img_prefix:
                label = self.labels[key]
                break
        if idx >= len(self.image_files) and self.augment_transform:
            # 对数据增强的样本应用增强变换
            rgb_images = [self.augment_transform(img) for img in rgb_images]  # 数据增强
        elif self.transform:
            rgb_images = [self.transform(img) for img in rgb_images]  # 默认变换
        
        return rgb_images, label
    


dataset_path ="/home/maia-user/lyq/Dataset/Train_tumor/train_phase2/rgb_all_png/train_extend_DBT_slice_rgb_patch3/"
json_path = "/home/maia-user/lyq/clear_dataset/codes/codes_benign_malignant/file_benign_dict.json"
# dataset_path =r"F:\Dataset\Train_tumor\train_phase2\rgb_all_png\train_extend_DBT_slice_rgb_patch3"
# json_path = r"F:\Dataset\clear_dataset\codes\codes_benign_malignant\file_benign_dict.json"

default_transform = transforms.Compose([
    transforms.Resize((224, 224)),  # 调整图片大小
    transforms.ToTensor(),          # 转换为Tensor
])

augment_transform = transforms.Compose([
    transforms.Resize((224, 224)),            # 调整图片大小
    #transforms.RandomHorizontalFlip(p=0.5),  # 随机水平翻转
    transforms.RandomRotation(10),           # 随机旋转
    #transforms.ColorJitter(brightness=0.2, contrast=0.2),  # 随机调整亮度和对比度
    transforms.ToTensor(),                   # 转换为Tensor
])

dataset = CustomDataset(
    dataset_path=dataset_path,
    json_path=json_path,
    transform=default_transform,
    augment_transform=augment_transform,
    n=3
)

def custom_collate_fn(batch):
    rgb_images_batch = []
    labels_batch = []
    for rgb_images, label in batch:
        rgb_images_batch.append(rgb_images)  # 每个样本的切片列表
        labels_batch.append(label)
    # 将每个样本的切片列表转换为张量
    rgb_images_batch = [torch.stack(slices) for slices in rgb_images_batch]  # 每个样本的切片堆叠为 [N, C, H, W]
    # 将所有样本的切片列表堆叠为批量数据
    rgb_images_batch = torch.stack(rgb_images_batch)  # [batch_size, N, C, H, W]
    labels_batch = torch.tensor(labels_batch, dtype=torch.float32)  # [batch_size]
    return rgb_images_batch, labels_batch


train_dataloader = DataLoader(dataset, batch_size=2, shuffle=True,collate_fn=custom_collate_fn)

val_dataset  ="/home/maia-user/lyq/Dataset/Val/val_phase2/rgb_all_png/val_extend_DBT_slice_rgb_patch3"
val_json = "/home/maia-user/lyq/Dataset/Val/results_val.json"

# val_dataset =r"F:\Dataset\Val\val_phase2\rgb_all_png\val_extend_DBT_slice_rgb_patch3"
# val_json = r"F:\Dataset\Val\results_val.json"



val_dataset =CustomDataset(
    dataset_path=val_dataset,
    json_path=val_json,
    transform=default_transform,
    augment_transform=augment_transform,
    n=2
)

val_dataloader = DataLoader(val_dataset, batch_size=2, shuffle=False,collate_fn=custom_collate_fn)







import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.metrics import (accuracy_score, precision_score, recall_score, f1_score,
                             roc_auc_score, average_precision_score, matthews_corrcoef,
                             cohen_kappa_score, confusion_matrix, classification_report)
import numpy as np

def train_model(train_dataloader, val_dataloader, model, criterion, optimizer, num_epochs=10):
    best_auc = 0.0  # 用于保存最高的 AUC 值
    log_file = "training_log_mobile.txt"  # 日志文件路径

    # 打开日志文件
    with open(log_file, "w") as f:
        f.write("Epoch\tTrain Loss\tTrain Accuracy\tTrain AUC\tTrain Precision\tTrain Sensitivity\tTrain Specificity\tTrain f1\tVal Accuracy\tVal AUC\tVal Precision\tVal Sensitivity\tVal Specificity\tVal f1\n")

    for epoch in range(num_epochs):
        model.train()
        running_loss = 0.0
        train_preds = []
        train_labels = []

        for rgb_images, labels in train_dataloader:
            labels = labels.to(device)
            labels = labels.float()
            optimizer.zero_grad()

            batch_preds = []
            for slices in rgb_images: 
                slices = [slice.to(device) for slice in slices]  # slices 是一个batch的多个切片的集合
                slice_preds = []
                for slice in slices:
                    outputs = model(slice.unsqueeze(0))
                    slice_preds.append(outputs)
                slice_preds = torch.cat(slice_preds, dim=0) # 将多个切片的预测结果取平均
                final_pred = torch.sigmoid(slice_preds.mean(dim=0))
                batch_preds.append(final_pred)
            batch_preds = torch.stack(batch_preds).squeeze()
            loss = criterion(batch_preds, labels)
            loss.backward()
            optimizer.step()

            running_loss += loss.item()
            train_preds.extend(batch_preds.detach().cpu().numpy())
            train_labels.extend(labels.cpu().numpy())

        train_loss = running_loss / len(train_dataloader)
        train_accuracy = accuracy_score(train_labels, (np.array(train_preds) > 0.5).astype(int))
        train_auc = roc_auc_score(train_labels, train_preds)
        train_precision = precision_score(train_labels, (np.array(train_preds) > 0.5).astype(int))
        train_sensitivity = recall_score(train_labels, (np.array(train_preds) > 0.5).astype(int))
        train_specificity = recall_score(train_labels, (np.array(train_preds) > 0.5).astype(int), pos_label=0)
        train_f1score = f1_score(train_labels, (np.array(train_preds) > 0.5).astype(int))

        model.eval()
        val_preds = []
        val_labels = []
        val_probs = []

        with torch.no_grad():
            for rgb_images, labels in val_dataloader:
                labels = labels.to(device)
                labels = labels.float()
                batch_preds = []
                batch_probs = [] 
                for slices in rgb_images: 
                    slices = [slice.to(device) for slice in slices]  # slices 是一个batch的多个切片的集合
                    slice_preds = []
                    slice_probs_list = []
                    for slice in slices:
                        outputs = model(slice.unsqueeze(0))
                        slice_preds.append(outputs)
                    slice_preds = torch.cat(slice_preds, dim=0) # 将多个切片的预测结果取平均
                    slice_probs = torch.sigmoid(slice_preds.mean(dim=0))
                    slice_votes = (slice_probs > 0.5).float()
                    final_pred = slice_votes.mode(dim=0).values
                    batch_preds.append(final_pred)
                    batch_probs.append(slice_probs.cpu().numpy())
                batch_preds = torch.stack(batch_preds).squeeze()
                val_preds.extend(batch_preds.cpu().numpy())
                val_probs.extend(batch_probs)
                val_labels.extend(labels.cpu().numpy())

        val_accuracy = accuracy_score(val_labels, (np.array(val_preds) > 0.5).astype(int))
        val_auc = roc_auc_score(val_labels, val_probs)
        val_precision = precision_score(val_labels, (np.array(val_preds) > 0.5).astype(int))
        val_sensitivity = recall_score(val_labels, (np.array(val_preds) > 0.5).astype(int))
        val_specificity = recall_score(val_labels, (np.array(val_preds) > 0.5).astype(int), pos_label=0)
        val_f1score = f1_score(val_labels, (np.array(val_preds) > 0.5).astype(int))

        print(f"Epoch [{epoch + 1}/{num_epochs}], "
              f"Train Loss: {train_loss:.4f}, "
              f"Train Accuracy: {train_accuracy:.4f}, "
              f"Train AUC: {train_auc:.4f}, "
              f"Train Precision: {train_precision:.4f}, "
              f"Val Accuracy: {val_accuracy:.4f}, "
              f"Val AUC: {val_auc:.4f}, "
              f"Val Precision: {val_precision:.4f}")

        # 保存指标到日志文件
        with open(log_file, "a") as f:
            f.write(f"{epoch + 1}\t{train_loss:.4f}\t{train_accuracy:.4f}\t{train_auc:.4f}\t{train_precision:.4f}\t{train_sensitivity:.4f}\t{train_specificity:.4f}\t{train_f1score:.4f}\t"
                    f"{val_accuracy:.4f}\t{val_auc:.4f}\t{val_precision:.4f}\t{val_sensitivity:.4f}\t{val_specificity:.4f}\t{val_f1score:.4f}\n")

        if val_auc > best_auc:
            best_auc = val_auc
            torch.save(model.state_dict(), "best_model_mobile.pth")
            print(f"New best model saved with AUC: {best_auc:.4f}")




# class SimpleCNN(nn.Module):
#     def __init__(self):
#         super(SimpleCNN, self).__init__()
#         self.conv1 = nn.Conv2d(3, 16, kernel_size=3, stride=1, padding=1)
#         self.conv2 = nn.Conv2d(16, 32, kernel_size=3, stride=1, padding=1)
#         self.conv3 = nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1)
#         self.pool = nn.MaxPool2d(kernel_size=2, stride=2)
#         self.fc1 = nn.Linear(64 * 28 * 28, 128)  # 假设输入图像大小为 224x224
#         self.fc2 = nn.Linear(128, 1)
#         # self.sigmoid = nn.Sigmoid()

#     def forward(self, x):
#         x = self.pool(F.relu(self.conv1(x)))
#         x = self.pool(F.relu(self.conv2(x)))
#         x = self.pool(F.relu(self.conv3(x)))
#         x = x.view(x.size(0), -1)  # 展平
#         x = F.relu(self.fc1(x))
#         x = self.fc2(x)
#         return x

class BinaryMobileNet(nn.Module):
    def __init__(self):
        super(BinaryMobileNet, self).__init__()
        self.mobilenet = models.mobilenet_v2(pretrained=False)
        num_ftrs = self.mobilenet.last_channel
        self.mobilenet.classifier = nn.Sequential(
            nn.Linear(num_ftrs, 1)
        )
        # self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        x = self.mobilenet(x)
        return x

model = BinaryMobileNet()


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)
criterion = nn.BCELoss()
optimizer = optim.Adam(model.parameters(), lr=0.0001)



train_model(train_dataloader, val_dataloader, model, criterion, optimizer, num_epochs=150)

