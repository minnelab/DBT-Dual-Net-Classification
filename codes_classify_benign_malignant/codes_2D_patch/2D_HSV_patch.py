import cv2
import numpy as np

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


import os
import json
from PIL import Image
import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
import random

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
    
    def __len__(self):
        # 数据量扩大n倍
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

        image = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)  # 读取灰度图像

        img_prefix,view_char = self._extract_prefix(img_name)
        if view_char.lower().startswith('r'):
            image = cv2.flip(image, 1)  # 水平翻转图像

        normalized_image = image.astype(np.float32) / 255.0
        normalized_image = (normalized_image * 255).astype(np.uint8)    
        rgb_image = cv2.applyColorMap(normalized_image, cv2.COLORMAP_JET)

        rgb_image = Image.fromarray(cv2.cvtColor(rgb_image, cv2.COLOR_BGR2RGB))

        label = -1  # 默认标签
        for key in self.labels:
            key_prefix,_ = self._extract_prefix(key)
            if key_prefix == img_prefix:
                label = self.labels[key]
                break
        if idx >= len(self.image_files) and self.augment_transform:
            # 对数据增强的样本应用增强变换
            rgb_image = self.augment_transform(rgb_image)
        elif self.transform:
            # 对原始样本应用默认变换
            rgb_image = self.transform(rgb_image)
        
        return rgb_image, label
    


dataset_path ="/home/maia-user/lyq/Dataset/Train_tumor/train_phase2/rgb_all_png/train_extend_DBT_slice_rgb_patch3/"
json_path = "/home/maia-user/lyq/clear_dataset/codes/codes_benign_malignant/file_benign_dict.json"


default_transform = transforms.Compose([
    transforms.Resize((512, 512)),  # 调整图片大小
    transforms.ToTensor(),          # 转换为Tensor
])

augment_transform = transforms.Compose([
    transforms.Resize((512, 512)),            # 调整图片大小
    transforms.RandomHorizontalFlip(p=0.5),  # 随机水平翻转
    transforms.RandomRotation(10),           # 随机旋转
    #transforms.ColorJitter(brightness=0.2, contrast=0.2),  # 随机调整亮度和对比度
    transforms.ToTensor(),                   # 转换为Tensor
])

dataset = CustomDataset(
    dataset_path=dataset_path,
    json_path=json_path,
    transform=default_transform,
    augment_transform=augment_transform,
    n=2
)

train_dataloader = DataLoader(dataset, batch_size=2, shuffle=True)

val_dataset  ="/home/maia-user/lyq/Dataset/Val/val_phase2/rgb_all_png/val_extend_DBT_slice_rgb_patch3"
val_json = "/home/maia-user/lyq/Dataset/Val/results_val.json"

val_dataset =CustomDataset(
    dataset_path=val_dataset,
    json_path=val_json,
    transform=default_transform,
    augment_transform=augment_transform,
    n=2
)

val_dataloader = DataLoader(val_dataset, batch_size=2, shuffle=False)







import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.metrics import (accuracy_score, precision_score, recall_score, f1_score,
                             roc_auc_score, average_precision_score, matthews_corrcoef,
                             cohen_kappa_score, confusion_matrix, classification_report)
import numpy as np

def train_model(train_dataloader, val_dataloader, model, criterion, optimizer, num_epochs=10):
    best_auc = 0.0  # 用于保存最高的 AUC 值
    log_file = "training_log.txt"  # 日志文件路径

    # 打开日志文件
    with open(log_file, "w") as f:
        f.write("Epoch\tTrain Loss\tTrain Accuracy\tTrain AUC\tTrain Precision\tTrain Sensitivity\tTrain Specificity\tTrain f1\tVal Accuracy\tVal AUC\tVal Precision\tVal Sensitivity\tVal Specificity\tVal f1\n")

    for epoch in range(num_epochs):
        model.train()
        running_loss = 0.0
        train_preds = []
        train_labels = []

        for images, labels in train_dataloader:
            images = images.to(device)
            labels = labels.to(device)
            labels = labels.float()
            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs.squeeze(), labels)
            loss.backward()
            optimizer.step()

            running_loss += loss.item()
            train_preds.extend(outputs.squeeze().detach().cpu().numpy())
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

        with torch.no_grad():
            for images, labels in val_dataloader:
                images = images.to(device)
                labels = labels.to(device)
                labels = labels.float()
                outputs = model(images)
                val_preds.extend(outputs.squeeze().cpu().numpy())
                val_labels.extend(labels.cpu().numpy())

        val_accuracy = accuracy_score(val_labels, (np.array(val_preds) > 0.5).astype(int))
        val_auc = roc_auc_score(val_labels, val_preds)
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
            torch.save(model.state_dict(), "best_model.pth")
            print(f"New best model saved with AUC: {best_auc:.4f}")



# class BinaryResNeXt(nn.Module):
#     def __init__(self):
#         super(BinaryResNeXt, self).__init__()
#         self.resnext = models.resnext50_32x4d(pretrained=False)  # 加载预训练模型
        
#         # 修改最后一层为二分类输出
#         num_ftrs = self.resnext.fc.in_features
#         self.resnext.fc = nn.Linear(num_ftrs, 1)
#         self.sigmoid = nn.Sigmoid()

#     def forward(self, x):
#         x = self.resnext(x)
#         x = self.sigmoid(x)
#         return x


class BinaryResnet(nn.Module):
    def __init__(self):
        super(BinaryResnet, self).__init__()
        #self.resnet = models.resnet50(pretrained=False)
        self.resnet = models.resnet101(pretrained=False)
        num_ftrs = self.resnet.fc.in_features
        self.resnet.fc = nn.Linear(num_ftrs, 1)
        self.sigmoid = nn.Sigmoid()
    
    def forward(self, x):
        x = self.resnet(x)
        x = self.sigmoid(x)
        return x

model = BinaryResnet()

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.0001)



train_model(train_dataloader, val_dataloader, model, criterion, optimizer, num_epochs=50)