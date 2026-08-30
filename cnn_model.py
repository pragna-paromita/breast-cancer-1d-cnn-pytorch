import os
import zipfile
import matplotlib.pyplot as plt
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler

# 1. Unzip Data Automatically
if not os.path.exists("breast-cancer.csv"):
    zip_name = "breast-cancer-dataset.zip"
    if os.path.exists(zip_name):
        with zipfile.ZipFile(zip_name, "r") as zip_ref:
            zip_ref.extractall("./")
        print("Dataset extracted successfully!")

# 2. Load and Preprocess Data
df = pd.read_csv("breast-cancer.csv")

if "id" in df.columns:
    df = df.drop(columns=["id"])

X = df.drop(columns=["diagnosis"]).values
y = df["diagnosis"].values

label_encoder = LabelEncoder()
y = label_encoder.fit_transform(y)

scaler = StandardScaler()
X = scaler.fit_transform(X)

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

X_train_t = torch.tensor(X_train, dtype=torch.float32).unsqueeze(1)
y_train_t = torch.tensor(y_train, dtype=torch.long)
X_test_t = torch.tensor(X_test, dtype=torch.float32).unsqueeze(1)
y_test_t = torch.tensor(y_test, dtype=torch.long)


# 3. Define 1D CNN Architecture
class TabularCNN(nn.Module):

    def __init__(self):
        super(TabularCNN, self).__init__()
        self.conv1 = nn.Conv1d(
            in_channels=1, out_channels=16, kernel_size=3, padding=1
        )
        self.relu = nn.ReLU()
        self.pool = nn.MaxPool1d(kernel_size=2)
        self.conv2 = nn.Conv1d(
            in_channels=16, out_channels=32, kernel_size=3, padding=1
        )

        self.fc1 = nn.Linear(32 * 7, 32)
        self.fc2 = nn.Linear(32, 2)

    def forward(self, x):
        x = self.pool(self.relu(self.conv1(x)))
        x = self.pool(self.relu(self.conv2(x)))
        x = torch.flatten(x, 1)
        x = self.relu(self.fc1(x))
        x = self.fc2(x)
        return x


model = TabularCNN()

# 4. Training Loop with Epoch Metric History
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

epochs = 50
train_acc_history = []
test_acc_history = []
train_loss_history = []
test_loss_history = []

print("Starting Training...")
for epoch in range(epochs):
    # --- Training Phase ---
    model.train()
    optimizer.zero_grad()

    outputs = model(X_train_t)
    loss = criterion(outputs, y_train_t)

    loss.backward()
    optimizer.step()

    # Calculate train accuracy
    _, train_preds = torch.max(outputs, 1)
    train_acc = (train_preds == y_train_t).sum().item() / y_train_t.size(0)

    # --- Evaluation Phase ---
    model.eval()
    with torch.no_grad():
        test_outputs = model(X_test_t)
        test_loss = criterion(test_outputs, y_test_t)

        _, test_preds = torch.max(test_outputs, 1)
        test_acc = (test_preds == y_test_t).sum().item() / y_test_t.size(0)

    # Store metrics for plotting
    train_acc_history.append(train_acc * 100)
    test_acc_history.append(test_acc * 100)
    train_loss_history.append(loss.item())
    test_loss_history.append(test_loss.item())

    if (epoch + 1) % 10 == 0:
        print(
            f"Epoch [{epoch+1}/{epochs}] | "
            f"Train Acc: {train_acc*100:.2f}% | Test Acc: {test_acc*100:.2f}%"
        )

# 5. Plot Accuracy & Loss Curves
plt.figure(figsize=(12, 5))

# Plot 1: Accuracy Curve
plt.subplot(1, 2, 1)
plt.plot(range(1, epochs + 1), train_acc_history, label="Train Accuracy")
plt.plot(range(1, epochs + 1), test_acc_history, label="Test Accuracy")
plt.title("Accuracy vs Epochs")
plt.xlabel("Epoch")
plt.ylabel("Accuracy (%)")
plt.legend()
plt.grid(True)

# Plot 2: Loss Curve
plt.subplot(1, 2, 2)
plt.plot(range(1, epochs + 1), train_loss_history, label="Train Loss")
plt.plot(range(1, epochs + 1), test_loss_history, label="Test Loss")
plt.title("Loss vs Epochs")
plt.xlabel("Epoch")
plt.ylabel("Loss")
plt.legend()
plt.grid(True)

plt.tight_layout()
plt.savefig("metrics_plot.png")  # Saves graph to workspace directory
plt.show()  # Opens plot pop-up window