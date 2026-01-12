import os
import torch
import torch.utils.data as data
from torchvision import transforms
from PIL import Image
import numpy as np
import torch
import matplotlib.pyplot as plt
import numpy as np
from torchvision import models, transforms
from PIL import Image
import cv2
import os
import numpy as np
from PIL import Image, ImageDraw
import torch
from torchcam.utils import overlay_mask
from scipy.ndimage import label

class TumorDataset(data.Dataset):
    def __init__(self, tumor_dir, non_tumor_dir, transform=None,augment=False):
        self.tumor_images = [os.path.join(tumor_dir, f) for f in os.listdir(tumor_dir)]
        self.non_tumor_images = [os.path.join(non_tumor_dir, f) for f in os.listdir(non_tumor_dir)]
        self.images = self.tumor_images + self.non_tumor_images
        self.labels = [1] * len(self.tumor_images) + [0] * len(self.non_tumor_images)  # 1 for tumor, 0 for non-tumor
        self.transform = transform
        self.augment = augment

    def __len__(self):
        return len(self.images) * 5 if self.augment else len(self.images)

    def __getitem__(self, idx):
        if self.augment:
            idx = idx % len(self.images)
        img_path = self.images[idx]
        label = self.labels[idx]
        image = Image.open(img_path).convert('RGB')
        image = np.array(image)
        green_channel = image[:, :, 1]
        image = Image.fromarray(green_channel)
        if self.transform:
            image = self.transform(image)

        return image, label

transform = transforms.Compose([
    transforms.Resize((2048, 1024)),  # Resize to 2048x1024
    transforms.RandomHorizontalFlip(),  # Random horizontal flip
    transforms.RandomVerticalFlip(),  # Random vertical flip
    transforms.RandomRotation(30),  # Random rotation between -30 and 30 degrees
    transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.2),  # Random color jitter
    transforms.RandomResizedCrop(2048, scale=(0.8, 1.0)),  # Random crop with scaling
    transforms.ToTensor(),  # Convert image to tensor
    transforms.Normalize(mean=[0.485], std=[0.229])  # Normalize image
])

# Load datasets
train_tumor_dir = "F:/Dataset/clear_dataset/2D_whole_image/2D_train/2D_train_tumor"
train_non_tumor_dir = "F:/Dataset/clear_dataset/2D_whole_image/2D_train/2D_train_normal"

val_tumor_dir ="F:/Dataset/clear_dataset/2D_whole_image/2D_val/2D_val_tumor"
val_non_tumor_dir = "F:/Dataset/clear_dataset/2D_whole_image/2D_val/2D_val_normal"
train_dataset = TumorDataset(train_tumor_dir, train_non_tumor_dir, transform=transform, augment=False)
val_dataset = TumorDataset(val_tumor_dir, val_non_tumor_dir, transform=transform, augment=False)

# Create DataLoader
train_loader = data.DataLoader(train_dataset, batch_size=1, shuffle=True)
val_loader = data.DataLoader(val_dataset, batch_size=1, shuffle=True)



import torch.nn as nn
from torchvision import models
import torch.optim as optim



def train(model, dataloader, criterion, optimizer, device, print_every=5):
    model.train()  # Set the model to training mode
    running_loss = 0.0
    corrects = 0
    total = 0
    batch_loss = []
    all_batch_loss = []
    
    for batch_idx, (images, labels) in enumerate(dataloader):
        images, labels = images.to(device), labels.to(device)
        optimizer.zero_grad()  # Zero the parameter gradients
        outputs = model(images)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        running_loss += loss.item() * images.size(0)
        batch_loss.append(loss.item())
        all_batch_loss.append(loss.item())

        _, predicted = torch.max(outputs, 1)
        corrects += (predicted == labels).sum().item()
        total += labels.size(0)

        # Print average loss every `print_every` batches
        if (batch_idx + 1) % print_every == 0:
            avg_loss = sum(batch_loss) / len(batch_loss)
            print(f'Batch {batch_idx + 1}/{len(dataloader)}: Average Loss = {avg_loss:.4f}')
            batch_loss = []  # Reset batch_loss for the next set of batches

        del images, labels, outputs, predicted  # Free memory

    epoch_loss = running_loss / total
    epoch_accuracy = corrects / total

    return epoch_loss, epoch_accuracy,all_batch_loss


def validate(model, dataloader, device):
    model.eval()
    corrects = 0
    total = 0
    with torch.no_grad():
        for images, labels in dataloader:
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            _, predicted = torch.max(outputs, 1)
            corrects += (predicted == labels).sum().item()
            total += labels.size(0)

        epoch_accuracy = corrects / total
        print(f"Average Accuracy: {epoch_accuracy:.4f}")
    return epoch_accuracy


def main_training_loop(model,train_loader,val_loader,criterion,optimizer,num_epochs,lr, device='cuda',folder_name = 'resnext_metric_2D'):
    device = torch.device(device)
    model.to(device)
    metric_folder = folder_name
    os.makedirs(metric_folder, exist_ok=True)
    epoch_accuracies = []
    val_accuracies = []
    val_auc = []
    best_val_accuracy = -1
    for epoch in range(num_epochs):
        print(f'Epoch {epoch + 1}/{num_epochs}')
        epoch_loss, epoch_accuracy,all_batch_loss = train(model, train_loader, criterion, optimizer, device)
        epoch_accuracies.append(epoch_accuracy)
        print(f'Loss: {epoch_loss:.4f}, Accuracy: {epoch_accuracy:.4f}')
        

        val_accuracy = validate(model, val_loader, device)
        val_accuracies.append(val_accuracy)

        with open(os.path.join(metric_folder, 'accuracy.txt'), 'a') as f:
                    f.write(f'Epoch {epoch + 1}: Train Accuracy = {epoch_accuracy:.4f}, Validation Accuracy = {val_accuracy:.4f}\n')

        # Save the best model
        if val_accuracy > best_val_accuracy :#and auc > best_metric:
                best_val_accuracy = val_accuracy
                model_path = os.path.join(metric_folder, f'best_model.pth')
                torch.save(model.state_dict(), model_path)
                print(f'Best model saved to {model_path}')     
        else:
                print('this is not the best model')
    return model


model = models.resnet18(pretrained=False)
model.conv1 = nn.Conv2d(1, 64, kernel_size=(7, 7), stride=(2, 2), padding=(3, 3), bias=False)

model.fc = nn.Sequential(
    nn.Linear(model.fc.in_features, 512),
    nn.ReLU(),
    nn.Dropout(0.5),
    nn.Linear(512, 2)  # Output layer for binary classification (2 classes)
)

# Loss function and optimizer
criterion = nn.CrossEntropyLoss()  # For binary classification (0, 1)
optimizer = optim.Adam(model.parameters(), lr=1e-4,weight_decay=1e-5)
num_epochs = 50
folder_name = 'CAM_resnet18_2D_whole_image'
trained_model = main_training_loop(model, train_loader,val_loader,criterion,optimizer,num_epochs=50, lr=0.001, device='cuda', folder_name=folder_name)




model.load_state_dict(torch.load('CAM_resnet18_2D_whole_image/best_model.pth'), strict=False)
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = model.to(device)
model.eval()

from torchcam.methods import GradCAM, GradCAMpp, SmoothGradCAMpp, ScoreCAM, SSCAM,LayerCAM,XGradCAM,ISCAM
from torchcam.methods import CAM

cam_extractor = CAM(model, target_layer=model.layer4[-1])


from torchvision import transforms
test_transform = transforms.Compose([
    transforms.Resize((2048, 1024)),  # Resize to 2048x1024
    transforms.ToTensor(),  # Convert image to tensor
    transforms.Normalize(mean=[0.485], std=[0.229])  # Normalize image
])




#这段代码用于加载测试集图像并且进行 CAM 可视化

input_folder  ="F:/Dataset/clear_dataset/2D_whole_image/2D_val/2D_val_tumor"
output_folder = "F:/Dataset/clear_dataset/2D_whole_image/2D_val/processed_CAM_output"

if not os.path.exists(output_folder):
    os.makedirs(output_folder)

for filename in os.listdir(input_folder):
    if filename.endswith(".png"):                                   # 只处理PNG文件
        img_path = os.path.join(input_folder, filename)
        image = Image.open(img_path).convert('RGB')
        image = np.array(image)
        green_channel = image[:, :, 1]
        image = Image.fromarray(green_channel)
        input_tensor = test_transform(image).unsqueeze(0).to(device)
        pred_logits = model(input_tensor)
        pred_id = torch.topk(pred_logits, 1)[1].detach().cpu().numpy().squeeze().item()
        activation_map = cam_extractor(pred_id, pred_logits)
        activation_map = activation_map[0][0].detach().cpu().numpy()


        orig_width, orig_height = image.size
        activation_map_rescaled = Image.fromarray(activation_map)
        activation_map_rescaled = activation_map_rescaled.resize((orig_width, orig_height), Image.Resampling.LANCZOS)

        image = Image.open(img_path).convert('RGB')
        result = overlay_mask(image, activation_map_rescaled, alpha=0.7)
        activation_map_rescaled = np.array(activation_map_rescaled)

        threshold = np.percentile(activation_map_rescaled, 99)
        binary_map = activation_map_rescaled > threshold
        # 使用label函数来提取连通区域
        labeled_map, num_features = label(binary_map)
        regions = []
        for region_id in range(1, num_features + 1):  # 从1开始，因为0是背景
            # 提取该区域的坐标
            region_coords = np.argwhere(labeled_map == region_id)
            
            # 计算该区域的边界框（min_x, max_x, min_y, max_y）
            min_y, min_x = region_coords.min(axis=0)
            max_y, max_x = region_coords.max(axis=0)
            
            regions.append((min_x, min_y, max_x, max_y))

        # 创建PIL ImageDraw对象来绘制矩形框
        draw = ImageDraw.Draw(result)
        box_color = (255, 0, 0)  # 红色
        box_width = 5

        # 初始化一个列表来保存每个区域的图像
        extracted_regions = []

        # 为每个高亮区域绘制矩形框并提取区域
        for region in regions:
            min_x, min_y, max_x, max_y = region
            draw.rectangle([min_x, min_y, max_x, max_y], outline=box_color, width=box_width)
            
            # 提取该区域并添加到列表中
            cropped_region = image.crop((min_x, min_y, max_x, max_y))
            extracted_regions.append(np.array(cropped_region))  # 将裁剪的区域转为 NumPy 数组并存储

        # 保存叠加热力图结果到指定文件夹
        output_path = os.path.join(output_folder, f"result_{filename}")
        result.save(output_path)

        print(f"Processed and saved: {output_path}")