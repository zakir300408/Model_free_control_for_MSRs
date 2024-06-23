import os
import numpy as np
import torch
import matplotlib.pyplot as plt
from torch import nn, optim
from torch.utils.data import DataLoader, random_split
from sklearn.metrics import precision_score, recall_score, f1_score
from torch.optim.lr_scheduler import ReduceLROnPlateau

# Assuming model.py and dataset.py are properly imported
from model import UNet
from dataset import RobotDataset

import albumentations as A
from albumentations.pytorch import ToTensorV2
import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, random_split
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import precision_score, recall_score, f1_score, confusion_matrix
import seaborn as sns
from albumentations import Compose, HorizontalFlip, VerticalFlip, RandomBrightnessContrast, Blur
from albumentations.pytorch import ToTensorV2
import json
import joblib

class EarlyStopping:
    """Early stops the training if validation loss doesn't improve after a given patience."""
    def __init__(self, patience=7, verbose=False, delta=0):
        self.patience = patience
        self.verbose = verbose
        self.counter = 0
        self.best_score = None
        self.early_stop = False
        self.val_loss_min = np.Inf
        self.delta = delta

    def __call__(self, val_loss, model):
        score = -val_loss
        if self.best_score is None:
            self.best_score = score
        elif score < self.best_score + self.delta:
            self.counter += 1
            print(f'EarlyStopping counter: {self.counter} out of {self.patience}')
            if self.counter >= self.patience:
                self.early_stop = True
        else:
            self.best_score = score
            self.counter = 0


class RobotSegmentation:
    def __init__(self, dataset_dir, n_classes=3, batch_size=8, epochs=250, lr=0.0001):
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.dataset_dir = dataset_dir
        self.n_classes = n_classes
        self.batch_size = batch_size
        self.epochs = epochs

        self.model = UNet(n_classes).to(self.device)
        self.criterion = nn.CrossEntropyLoss()
        self.optimizer = optim.Adam(self.model.parameters(), lr=lr)
        self.scheduler = optim.lr_scheduler.ReduceLROnPlateau(self.optimizer, 'min', patience=20)
        self.train_loader, self.val_loader = self.load_data()

        self.train_losses = []
        self.val_losses = []
        self.iou_scores = []
        self.precisions = []
        self.recalls = []
        self.f1_scores = []

    def load_data(self):
        """Load and preprocess data."""
        transforms = Compose([
            HorizontalFlip(p=0.5),
            VerticalFlip(p=0.5),
            RandomBrightnessContrast(brightness_limit=0.2, contrast_limit=0.2, p=0.5),
            Blur(blur_limit=3, p=0.5),
            ToTensorV2(),
        ])

        annotation_path = os.path.join(self.dataset_dir, 'annotations.json')
        image_dir = os.path.join(self.dataset_dir, 'images')

        full_dataset = RobotDataset(image_dir, annotation_path, img_size=512, augmentations=transforms)
        train_size = int(0.75 * len(full_dataset))
        val_size = len(full_dataset) - train_size
        train_dataset, val_dataset = random_split(full_dataset, [train_size, val_size])

        train_loader = DataLoader(train_dataset, batch_size=self.batch_size, shuffle=True)
        val_loader = DataLoader(val_dataset, batch_size=self.batch_size, shuffle=False)
        return train_loader, val_loader

    def visualize_data(self, loader):
        """Visualize data for debugging."""
        images, masks = next(iter(loader))
        fig, axs = plt.subplots(2, 3, figsize=(8, 5))

        # Define the color map with the first color as black
        colors = plt.cm.jet(np.linspace(0, 1, self.n_classes))
        colors[0] = np.array([0, 0, 0, 1])  # Set the first color to black (R, G, B, A)

        for i in range(3):
            axs[0, i].imshow(images[i].permute(1, 2, 0).numpy())
            axs[0, i].axis('off')
            class_map = np.zeros((masks[i].shape[0], masks[i].shape[1], 3))
            for cls in range(self.n_classes):
                class_map[masks[i].numpy() == cls, :] = colors[cls][:3]
            axs[1, i].imshow(class_map)
            axs[1, i].axis('off')

        plt.tight_layout()
        # Save the high-quality image
        plt.savefig('data_visualization.png', format='png', dpi=1200)
        plt.show()

    def calculate_iou(self, pred, target):
        """Calculate IoU for each class."""
        pred = torch.argmax(pred, dim=1)
        iou_list = []
        for cls in range(self.n_classes):
            pred_inds = (pred == cls)
            target_inds = (target == cls)
            intersection = (pred_inds[target_inds]).float().sum()
            union = pred_inds.float().sum() + target_inds.float().sum() - intersection
            iou = float(intersection) / float(max(union, 1))
            iou_list.append(iou)
        return iou_list

    def calculate_metrics(self, pred, target):
        """Calculate Precision, Recall, F1 score for each class."""
        pred = torch.argmax(pred, dim=1).cpu().numpy()
        target = target.cpu().numpy()
        precision = precision_score(target.flatten(), pred.flatten(), labels=list(range(self.n_classes)), average=None,
                                    zero_division=0)
        recall = recall_score(target.flatten(), pred.flatten(), labels=list(range(self.n_classes)), average=None,
                              zero_division=0)
        f1 = f1_score(target.flatten(), pred.flatten(), labels=list(range(self.n_classes)), average=None,
                      zero_division=0)
        return precision, recall, f1

    def save_model(self, filepath):
        """Save the complete model to a file."""
        torch.save(self.model, filepath)
        print(f"Model saved to {filepath}")

    def plot_training_curves(self):
        """Plot training and validation loss curves."""
        plt.figure(figsize=(10, 5))
        plt.plot(self.train_losses, label='Train Loss')
        plt.plot(self.val_losses, label='Validation Loss')
        plt.xlabel('Epochs')
        plt.ylabel('Loss')
        plt.legend()
        plt.title('Training and Validation Loss')
        plt.savefig('training_curves.png', format='png', dpi=1200)
        plt.show()

    def plot_confusion_matrix(self, y_true, y_pred):
        """Plot confusion matrix for validation set."""
        cm = confusion_matrix(y_true, y_pred, labels=list(range(self.n_classes)))
        plt.figure(figsize=(10, 7))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
        plt.xlabel('Predicted')
        plt.ylabel('True')
        plt.title('Confusion Matrix')
        plt.savefig('confusion_matrix.png', format='png', dpi=1200)
        plt.show()

    def train(self):
        early_stopping = EarlyStopping(patience=20, verbose=True)
        for epoch in range(self.epochs):
            train_loss = self.train_model()
            val_loss, avg_iou, avg_precision, avg_recall, avg_f1, all_preds, all_targets = self.validate_model()
            self.scheduler.step(val_loss)

            # Save metrics for plotting
            self.train_losses.append(train_loss)
            self.val_losses.append(val_loss)
            self.iou_scores.append(avg_iou)
            self.precisions.append(avg_precision)
            self.recalls.append(avg_recall)
            self.f1_scores.append(avg_f1)

            print(f'Epoch {epoch + 1}, Train Loss: {train_loss}, Val Loss: {val_loss}')
            print(f'Class-wise metrics: IoU: {avg_iou}, Precision: {avg_precision}, Recall: {avg_recall}, F1: {avg_f1}')
            # Checkpoint
            if val_loss < early_stopping.val_loss_min:
                print(
                    f'Validation loss decreased ({early_stopping.val_loss_min:.6f} --> {val_loss:.6f}). Saving model...')
                torch.save(self.model.state_dict(), 'checkpoint.pth')
                early_stopping.val_loss_min = val_loss

            # Early stopping
            early_stopping(val_loss, self.model)
            if early_stopping.early_stop:
                print("Early stopping")
                break

            # Visualize the ground truth vs prediction every 5 epochs
            if (epoch + 1) % 5 == 0:
                with torch.no_grad():
                    imgs, masks = next(iter(self.val_loader))
                    imgs, masks = imgs.to(self.device), masks.to(self.device).long()
                    outputs = self.model(imgs)
                    pred = torch.argmax(outputs[0], dim=0).cpu().numpy()
                    gt = masks[0].cpu().numpy()

        # Save confusion matrix after training
        self.plot_confusion_matrix(all_targets, all_preds)
        self.plot_training_curves()

        # Save the complete model after the last epoch
        self.save_model("Multiclass_2_model_5_6_24.pth")

        # Save training data
        self.save_training_data()

    def train_model(self):
        """Single epoch training logic."""
        self.model.train()
        losses = []
        for imgs, masks in self.train_loader:
            imgs, masks = imgs.to(self.device), masks.to(self.device).long()
            outputs = self.model(imgs)
            loss = self.criterion(outputs, masks)
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()
            losses.append(loss.item())
        return np.mean(losses)

    def validate_model(self):
        """Validate the model and calculate metrics."""
        self.model.eval()
        losses, ious, avg_precision, avg_recall, avg_f1 = [], [], [], [], []
        all_preds = []
        all_targets = []
        with torch.no_grad():
            for imgs, masks in self.val_loader:
                imgs, masks = imgs.to(self.device), masks.to(self.device).long()
                outputs = self.model(imgs)
                loss = self.criterion(outputs, masks)
                losses.append(loss.item())
                iou = self.calculate_iou(outputs, masks)
                ious.append(iou)
                precision, recall, f1 = self.calculate_metrics(outputs, masks)
                avg_precision.append(precision)
                avg_recall.append(recall)
                avg_f1.append(f1)

                all_preds.extend(torch.argmax(outputs, dim=1).cpu().numpy().flatten())
                all_targets.extend(masks.cpu().numpy().flatten())

        avg_loss = np.mean(losses)
        avg_iou = np.mean(ious, axis=0)
        avg_precision = np.mean(avg_precision, axis=0)
        avg_recall = np.mean(avg_recall, axis=0)
        avg_f1 = np.mean(avg_f1, axis=0)

        return avg_loss, avg_iou, avg_precision, avg_recall, avg_f1, all_preds, all_targets

    def save_training_data(self):
        """Save training data for future plot generation."""
        training_data = {
            'train_losses': self.train_losses,
            'val_losses': self.val_losses,
            'iou_scores': self.iou_scores,
            'precisions': self.precisions,
            'recalls': self.recalls,
            'f1_scores': self.f1_scores,
        }
        joblib.dump(training_data, 'training_data.pkl')
        print("Training data saved to training_data.pkl")


if __name__ == "__main__":
    # Initialize and run the training
    robot_segmentation = RobotSegmentation(dataset_dir='multilabel_dataset1_2/')
    robot_segmentation.visualize_data(robot_segmentation.train_loader)  # Visualize some training data
    robot_segmentation.train()
