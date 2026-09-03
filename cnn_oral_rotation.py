import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchvision import datasets, transforms
from torch.utils.data import DataLoader, random_split
from collections import Counter
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.utils.class_weight import compute_class_weight


class RotConv2d(nn.Module):
    """Conv2d that shares one kernel across 0/90/180/270° rotations, combined via max-pooling."""
    def __init__(self, in_channels, out_channels, kernel_size=3, padding=1, bias=True):
        super().__init__()
        self.weight = nn.Parameter(torch.empty(out_channels, in_channels, kernel_size, kernel_size))
        nn.init.kaiming_uniform_(self.weight, a=5 ** 0.5)
        self.bias = nn.Parameter(torch.zeros(out_channels)) if bias else None
        self.padding = padding

    def forward(self, x):
        rotated_outputs = []
        for k in range(4):
            w_rot = torch.rot90(self.weight, k, dims=(2, 3))
            out = F.conv2d(x, w_rot, bias=self.bias, padding=self.padding)
            rotated_outputs.append(out)
        out = torch.stack(rotated_outputs, dim=0).max(dim=0)[0]
        return out


class RotationCNN(nn.Module):
    def __init__(self, num_classes, img_size):
        super().__init__()
        self.features = nn.Sequential(
            RotConv2d(3, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.MaxPool2d(2),                      # img_size / 2

            RotConv2d(32, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.MaxPool2d(2),                      # img_size / 4

            RotConv2d(64, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.MaxPool2d(2),                      # img_size / 8

            RotConv2d(128, 256, kernel_size=3, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(),
            nn.MaxPool2d(2),                      # img_size / 16
        )

        reduced_size = img_size // 16
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Dropout(0.5),
            nn.Linear(256 * reduced_size * reduced_size, 256),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(256, num_classes)
        )

    def forward(self, x):
        x = self.features(x)
        x = self.classifier(x)
        return x


def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Using device:", device)

    IMG_SIZE = 128
    BATCH_SIZE = 32
    NUM_EPOCHS = 40
    PATIENCE = 7

    train_transform = transforms.Compose([
        transforms.Resize((IMG_SIZE, IMG_SIZE)),
        transforms.RandomHorizontalFlip(),
        transforms.RandomRotation(10),
        transforms.ColorJitter(brightness=0.1, contrast=0.1),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

    test_transform = transforms.Compose([
        transforms.Resize((IMG_SIZE, IMG_SIZE)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

    TRAIN_DIR = r"./Oral Cancer/Oral Cancer Dataset"
    TEST_DIR  = r"./Oral cancer Dataset 2.0/OC Dataset kaggle new"  # update to exact folder name

    train_full = datasets.ImageFolder(TRAIN_DIR, transform=train_transform)
    test_full  = datasets.ImageFolder(TEST_DIR, transform=test_transform)

    assert train_full.class_to_idx == test_full.class_to_idx, \
        f"Class mismatch: {train_full.class_to_idx} vs {test_full.class_to_idx}"

    class_names = train_full.classes
    num_classes = len(class_names)
    print("Classes:", class_names)
    print("Train (dataset 1) size:", len(train_full))
    print("Test (dataset 2) size:", len(test_full))

    def count_classes(dataset, class_names):
        counts = Counter()
        for _, label in dataset.samples:
            counts[class_names[label]] += 1
        return counts

    print("Train class distribution:", count_classes(train_full, class_names))
    print("Test class distribution:", count_classes(test_full, class_names))

    val_size = int(0.15 * len(train_full))
    train_size = len(train_full) - val_size
    train_dataset, val_dataset = random_split(
        train_full, [train_size, val_size],
        generator=torch.Generator().manual_seed(42)
    )

    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=0)
    val_loader   = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=0)
    test_loader  = DataLoader(test_full, batch_size=BATCH_SIZE, shuffle=False, num_workers=0)

    model = RotationCNN(num_classes=num_classes, img_size=IMG_SIZE).to(device)
    print(model)

    labels_list = [label for _, label in train_full.samples]
    class_weights = compute_class_weight('balanced', classes=np.unique(labels_list), y=labels_list)
    class_weights = torch.tensor(class_weights, dtype=torch.float).to(device)
    print("Class weights:", class_weights)

    criterion = nn.CrossEntropyLoss(weight=class_weights)
    optimizer = optim.Adam(model.parameters(), lr=1e-3, weight_decay=1e-4)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=3)

    def train_one_epoch(model, loader, criterion, optimizer):
        model.train()
        running_loss, correct, total = 0.0, 0, 0
        for images, labels in loader:
            images, labels = images.to(device), labels.to(device)
            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            running_loss += loss.item() * images.size(0)
            correct += (outputs.argmax(1) == labels).sum().item()
            total += labels.size(0)
        return running_loss / total, correct / total

    def evaluate(model, loader, criterion):
        model.eval()
        running_loss, correct, total = 0.0, 0, 0
        with torch.no_grad():
            for images, labels in loader:
                images, labels = images.to(device), labels.to(device)
                outputs = model(images)
                loss = criterion(outputs, labels)
                running_loss += loss.item() * images.size(0)
                correct += (outputs.argmax(1) == labels).sum().item()
                total += labels.size(0)
        return running_loss / total, correct / total

    train_acc_history, val_acc_history, test_acc_history = [], [], []
    train_loss_history, val_loss_history = [], []

    best_val_loss = float('inf')
    patience_counter = 0

    for epoch in range(NUM_EPOCHS):
        train_loss, train_acc = train_one_epoch(model, train_loader, criterion, optimizer)
        val_loss, val_acc = evaluate(model, val_loader, criterion)
        test_loss, test_acc = evaluate(model, test_loader, criterion)  # monitoring only
        scheduler.step(val_loss)

        train_acc_history.append(train_acc)
        val_acc_history.append(val_acc)
        test_acc_history.append(test_acc)
        train_loss_history.append(train_loss)
        val_loss_history.append(val_loss)

        print(f"Epoch {epoch+1}/{NUM_EPOCHS} | "
              f"Train Loss {train_loss:.4f} Acc {train_acc:.4f} | "
              f"Val Loss {val_loss:.4f} Acc {val_acc:.4f} | "
              f"Test Acc {test_acc:.4f}")

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            patience_counter = 0
            torch.save(model.state_dict(), "best_rotation_cnn.pth")
        else:
            patience_counter += 1
            if patience_counter >= PATIENCE:
                print("Early stopping triggered.")
                break

    epochs_range = range(1, len(train_acc_history) + 1)
    plt.figure(figsize=(12, 5))

    plt.subplot(1, 2, 1)
    plt.plot(epochs_range, train_acc_history, label='Train Accuracy', marker='o')
    plt.plot(epochs_range, val_acc_history, label='Val Accuracy', marker='o')
    plt.plot(epochs_range, test_acc_history, label='Test Accuracy (Dataset 2)', marker='o')
    plt.xlabel('Epoch'); plt.ylabel('Accuracy')
    plt.title('Rotation-Equivariant CNN: Train vs Val vs Test Accuracy')
    plt.legend(); plt.grid(True, alpha=0.3)

    plt.subplot(1, 2, 2)
    plt.plot(epochs_range, train_loss_history, label='Train Loss', marker='o')
    plt.plot(epochs_range, val_loss_history, label='Val Loss', marker='o')
    plt.xlabel('Epoch'); plt.ylabel('Loss')
    plt.title('Train vs Val Loss')
    plt.legend(); plt.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig("rotation_cnn_training_curves.png", dpi=150)
    plt.show()

    model.load_state_dict(torch.load("best_rotation_cnn.pth"))
    model.eval()

    y_true, y_pred = [], []
    with torch.no_grad():
        for images, labels in test_loader:
            images = images.to(device)
            outputs = model(images)
            preds = outputs.argmax(1)
            y_pred.extend(preds.cpu().numpy())
            y_true.extend(labels.numpy())

    print("\n=== Cross-dataset test results (Rotation-Equivariant CNN) ===")
    print(classification_report(y_true, y_pred, target_names=class_names))
    print(confusion_matrix(y_true, y_pred))


if __name__ == "__main__":
    main()