import os
import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms
import matplotlib.pyplot as plt

# デバイスの設定
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Using device: {device}")

# ハイパーパラメータ
num_epochs = 10
batch_size = 128
learning_rate = 0.001
weight_decay = 1e-4

# MLPの層サイズ
hidden1 = 512
hidden2 = 256

# モデル保存ファイル名
MODEL_PATH = "mlp_cifar10.pth"

# ✅ 修正: RandAugment（num_ops=2, magnitude=14）
transform_train = transforms.Compose([
    transforms.RandAugment(num_ops=2, magnitude=14),
    transforms.ToTensor(),
    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
])

# テスト用：Augmentationなし
transform_test = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
])

# データセットの読み込み
train_dataset = torchvision.datasets.CIFAR10(root='./data', train=True, download=True, transform=transform_train)
test_dataset = torchvision.datasets.CIFAR10(root='./data', train=False, download=True, transform=transform_test)

train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

# Augmentationなしの訓練データで評価
train_eval_dataset = torchvision.datasets.CIFAR10(root='./data', train=True, download=False, transform=transform_test)
train_eval_loader = torch.utils.data.DataLoader(train_eval_dataset, batch_size=batch_size, shuffle=False)

classes = ('plane', 'car', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck')

class MLP(nn.Module):
    def __init__(self, input_size=32*32*3, hidden1=512, hidden2=256, num_classes=10, dropout=0.3):
        super(MLP, self).__init__()
        self.flatten = nn.Flatten()
        self.fc = nn.Sequential(
            nn.Linear(input_size, hidden1),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden1, hidden2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden2, num_classes)
        )

    def forward(self, x):
        x = self.flatten(x)
        return self.fc(x)

# モデルの構築
model = MLP(hidden1=hidden1, hidden2=hidden2).to(device)

# 損失関数と最適化
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=learning_rate, weight_decay=weight_decay)

def evaluate(loader):
    model.eval()
    total_loss = 0.0
    correct = 0
    total = 0
    with torch.no_grad():
        for images, labels in loader:
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            loss = criterion(outputs, labels)
            total_loss += loss.item()
            _, predicted = outputs.max(1)
            total += labels.size(0)
            correct += predicted.eq(labels).sum().item()
    accuracy = 100. * correct / total
    avg_loss = total_loss / len(loader)
    return avg_loss, accuracy

# --- ここから追加 ---
train_losses = []
train_accuracies = []
test_losses = []
test_accuracies = []

if os.path.exists(MODEL_PATH):
    print(f"モデルファイル {MODEL_PATH} が見つかりました。モデルを読み込みます。")
    checkpoint = torch.load(MODEL_PATH, map_location=device)
    model.load_state_dict(checkpoint['model_state_dict'])
    train_losses = checkpoint.get('train_losses', [])
    train_accuracies = checkpoint.get('train_accuracies', [])
    test_losses = checkpoint.get('test_losses', [])
    test_accuracies = checkpoint.get('test_accuracies', [])
else:
    print("モデルファイルが見つかりません。学習を開始します。")
    for epoch in range(num_epochs):
        model.train()
        for images, labels in train_loader:
            images, labels = images.to(device), labels.to(device)
            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

        train_loss, train_acc = evaluate(train_eval_loader)
        test_loss, test_acc = evaluate(test_loader)

        train_losses.append(train_loss)
        train_accuracies.append(train_acc)
        test_losses.append(test_loss)
        test_accuracies.append(test_acc)

        print(f"Epoch [{epoch+1}/{num_epochs}], "
              f"Train Loss: {train_loss:.4f}, Acc: {train_acc:.2f}%, "
              f"Test Loss: {test_loss:.4f}, Acc: {test_acc:.2f}%")

    # --- 学習済みモデルと履歴を保存 ---
    torch.save({
        'model_state_dict': model.state_dict(),
        'train_losses': train_losses,
        'train_accuracies': train_accuracies,
        'test_losses': test_losses,
        'test_accuracies': test_accuracies
    }, MODEL_PATH)
    print(f"モデルを {MODEL_PATH} に保存しました。")
# --- ここまで追加 ---

# 結果のプロット
x_epochs = range(1, len(train_losses) + 1)
plt.figure(figsize=(12, 5))

plt.subplot(1, 2, 1)
plt.plot(x_epochs, train_losses, label='Train Loss')
plt.plot(x_epochs, test_losses, label='Test Loss')
plt.title('Loss Curve')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.legend()

plt.subplot(1, 2, 2)
plt.plot(x_epochs, train_accuracies, label='Train Accuracy')
plt.plot(x_epochs, test_accuracies, label='Test Accuracy')
plt.title('Accuracy Curve')
plt.xlabel('Epoch')
plt.ylabel('Accuracy (%)')
plt.legend()

plt.tight_layout()
plt.show()