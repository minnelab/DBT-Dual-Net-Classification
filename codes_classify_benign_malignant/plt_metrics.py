import pandas as pd
import numpy as np
import re
import os
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import roc_curve, auc, precision_recall_curve, confusion_matrix
import torch
import torch.nn.functional as F
import torch.optim as optim
from sklearn.metrics import roc_auc_score,accuracy_score,f1_score,recall_score

def plot_roc_curve(y_true, y_scores, folder,filename):
    os.makedirs(folder, exist_ok=True)
    save_path = os.path.join(folder, filename)
    fpr, tpr, _ = roc_curve(y_true, y_scores)
    roc_auc = auc(fpr, tpr)

    plt.figure(figsize=(10, 6))
    plt.plot(fpr, tpr, color='blue', lw=2, label=f'ROC curve (area = {roc_auc:.2f})')
    plt.plot([0, 1], [0, 1], color='gray', lw=2, linestyle='--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('Receiver Operating Characteristic')
    plt.legend(loc="lower right")
    plt.grid(True)
    plt.savefig(save_path)
    plt.close()


def plot_confusion_matrix(vote_label, vote_pred, folder,filename):
    os.makedirs(folder, exist_ok=True)
    save_path = os.path.join(folder, filename)

    cm = confusion_matrix(vote_label, vote_pred)

    plt.figure(figsize=(10, 8))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
    plt.title('confusion metrix')
    plt.ylabel('Actual Label')
    plt.xlabel('Predicted Label')

    plt.savefig(save_path)
    plt.close()


def plot_f1_curve(y_true, y_scores, folder,filename):

    os.makedirs(folder, exist_ok=True)
    save_path = os.path.join(folder, filename)
    precision, recall, thresholds = precision_recall_curve(y_true, y_scores)
    f1_scores = 2 * precision[:-1] * recall[:-1] / (precision[:-1] + recall[:-1])
    f1_scores = np.nan_to_num(f1_scores)

    max_f1_index = np.argmax(f1_scores)
    max_f1 = f1_scores[max_f1_index]
    best_threshold = thresholds[max_f1_index]

    plt.figure(figsize=(10, 6))
    plt.plot(thresholds, f1_scores, label='F1 Score')
    plt.scatter(thresholds[max_f1_index], max_f1, color='red', label=f'Best F1 Score: {max_f1:.2f}')
    plt.xlabel('Thresholds')
    plt.ylabel('F1 Score')
    plt.title('F1 Score vs. Thresholds')
    plt.legend()
    plt.grid(True)
    plt.savefig(save_path)
    plt.close()

def save_accuracy_auc_plot(accuracy_history, auc_history, folder,filename):

    os.makedirs(folder, exist_ok=True)
    save_path = os.path.join(folder, filename)

    plt.figure(figsize=(12, 6))

    plt.subplot(1, 2, 1)
    plt.plot(accuracy_history, label='Accuracy')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.title('Accuracy over Epochs')
    plt.legend()

    plt.subplot(1, 2, 2)
    plt.plot(auc_history, label='AUC', color='orange')
    plt.xlabel('Epoch')
    plt.ylabel('AUC')
    plt.title('AUC over Epochs')
    plt.legend()

    plt.tight_layout()
    plt.savefig(save_path)
    plt.close()

def save_train_val_accuracy_plot(epoch_accuracies, val_accuracies,folder,filename):
    # Plot accuracy after training
    os.makedirs(folder, exist_ok=True)
    save_path = os.path.join(folder, filename)

    plt.figure()
    plt.plot(epoch_accuracies, label='Train Accuracy')
    plt.plot(val_accuracies, label='Validation Accuracy')
    plt.xlabel('Epochs')
    plt.ylabel('Accuracy')
    plt.title('Training and Validation Accuracy')
    plt.legend()
    
    plt.savefig(save_path)

    plt.close()  # Close the plot to free up memory

def plot_loss_curve(all_batch_loss, metric_folder, filename):
    # 创建文件夹如果不存在
    os.makedirs(metric_folder, exist_ok=True)
    
    # 绘制损失曲线
    plt.figure(figsize=(10, 6))
    plt.plot(all_batch_loss, label='Batch Loss', color='blue')
    plt.title('Loss Curve')
    plt.xlabel('Batch Number')
    plt.ylabel('Loss')
    plt.legend()
    plt.grid(True)
    
    # 保存图像
    file_path = os.path.join(metric_folder, filename)
    plt.savefig(file_path)
    plt.close()  # 关闭图形以释放内存


def majority_vote(votes):
    num_ones = np.sum(votes)
    num_zeros = len(votes) - num_ones

    return 1 if num_ones > num_zeros else 0



def train(model, dataloader, criterion, optimizer, device, print_every=5):
    model.train()
    running_loss = 0.0
    corrects = 0
    total = 0
    batch_loss = []
    all_batch_loss = []
    
    for batch_idx, (images, labels) in enumerate(dataloader):
        images, labels = images.to(device), labels.to(device)
        
        optimizer.zero_grad()
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

        if (batch_idx + 1) % print_every == 0:
            avg_loss = sum(batch_loss) / len(batch_loss)
            print(f'Batch {batch_idx + 1}/{len(dataloader)}: Average Loss = {avg_loss:.4f}')
            batch_loss = []

    epoch_loss = running_loss / total
    epoch_accuracy = corrects / total

    return epoch_loss, epoch_accuracy, all_batch_loss

def validate(model, dataloader, device):
    model.eval()  # Set the model to evaluation mode

    vote_pred = []
    vote_label = []
    vote_pred_prob = []

    #group_pred = []
    #group_label = []     #三个一组，用于将三个data合并在一起

    with torch.no_grad():
        for images, labels in dataloader:
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            probilities = F.softmax(outputs,dim=1)
            binary_outputs_cpu = torch.argmax(probilities, dim=1).cpu().numpy()

            labels_cpu = labels.cpu().numpy()
            vote_pred.extend(binary_outputs_cpu)
            vote_label.extend(labels_cpu)

            probilities = probilities.cpu().numpy()[:, 1]

            vote_pred_prob.extend(probilities)

        #print(vote_label)
        #print(vote_pred)

        accuracy = accuracy_score(vote_label, vote_pred)
        auc = roc_auc_score(vote_label, vote_pred_prob)
        best_f1_vote = f1_score(vote_label, vote_pred)
        best_recall_vote = recall_score(vote_label, vote_pred)

        print(
                f"Validation results - AUC: {auc:.4f}, "
                f"Average Accuracy: {accuracy:.4f}"
                f"F1: {best_f1_vote:.4f}"
                f"recall: {best_recall_vote:.4f}"                    
            )

    return accuracy, auc, vote_label, vote_pred

def save_loss_to_file(losses, folder, filename):
    os.makedirs(folder, exist_ok=True)
    file_path = os.path.join(folder, filename)
    with open(file_path, 'w') as file:
        for loss in losses:
            file.write(f'{loss}\n')

def main_training_loop(model, train_loader, val_loader, num_epochs=10, lr=0.0001, device='cuda',metric_folder = 'resnext_ACS_metric'):
    device = torch.device(device)
    model.to(device)

    criterion = torch.nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=lr)

    metric_folder = metric_folder
    os.makedirs(metric_folder, exist_ok=True)

    epoch_accuracies = []
    val_accuracies = []
    val_auc = []

    best_val_accuracy = -1
    best_metric = -1

    for epoch in range(num_epochs):
        print(f'Epoch {epoch + 1}/{num_epochs}')
        
        # Train
        epoch_loss, epoch_accuracy, all_batch_loss = train(model, train_loader, criterion, optimizer, device)
        print(f'Train Loss: {epoch_loss:.4f}, Train Accuracy: {epoch_accuracy:.4f}')
        
        # Validate
        val_accuracy,auc,vote_label, vote_pred = validate(model, val_loader, device)
        epoch_accuracies.append(epoch_accuracy)
        val_accuracies.append(val_accuracy)
        val_auc.append(auc)


        with open(os.path.join(metric_folder, 'accuracy.txt'), 'a') as f:
            f.write(f'Epoch {epoch + 1}: Train Accuracy = {epoch_accuracy:.4f}, Validation Accuracy = {val_accuracy:.4f}\n')

        # Save the best model
        if val_accuracy > best_val_accuracy and auc > best_metric:
            best_val_accuracy = val_accuracy
            best_metric = auc

            save_loss_to_file(all_batch_loss, metric_folder, f'loss_epoch_best_model.txt')
            plot_loss_curve(all_batch_loss, metric_folder, f'loss_curve_best_model.png')

            model_path = os.path.join(metric_folder, f'best_model.pth')
            torch.save(model.state_dict(), model_path)
            print(f'Best model saved to {model_path}')
            
            plot_roc_curve(vote_label, vote_pred, metric_folder,f'roc_curve.png')
            plot_confusion_matrix(vote_label, vote_pred, metric_folder,f'confusion_matrix.png')
            plot_f1_curve(vote_label, vote_pred, metric_folder,f'f1_curve.png')

        else:
            print('this is not the best model')
    save_accuracy_auc_plot(val_accuracies, val_auc, metric_folder,f'accuracy_auc_plot.png')
    save_train_val_accuracy_plot(epoch_accuracies, val_accuracies, metric_folder,f'train_val_accuracy_plot.png')
    return model