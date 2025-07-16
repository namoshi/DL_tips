import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
import torchvision.transforms as T
from torch.utils.data import DataLoader
from tqdm import tqdm
import numpy as np
import matplotlib.pyplot as plt
import os

# ログ出力

def print_both(*args, **kwargs):
    """printの内容をファイル(mmdropout_cifar-10.log)にも保存"""
    print(*args, **kwargs)
    with open("mmdropout_cifar-10.log", "a", encoding="utf-8") as f:
        print(*args, **kwargs, file=f)

class MCResNet18(nn.Module):
    """
    ResNet18にDropoutを追加したモデル
    """
    def __init__(self, dropout_p=0.2):
        super().__init__()
        base = torchvision.models.resnet18(weights=None)
        base.conv1 = nn.Conv2d(3, 64, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)
        base.maxpool = nn.Identity()
        base.fc = nn.Identity()  # 最終層は自分で定義
        self.features = base
        self.dropout = nn.Dropout(p=dropout_p)
        self.fc = nn.Linear(512, 10)

    def forward(self, x):
        x = self.features(x)
        x = torch.flatten(x, 1)
        x = self.dropout(x)
        x = self.fc(x)
        return x

def main():
    # デバイス設定
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print_both(f"Using device: {device}")

    # データセット
    print_both("\nLoading CIFAR-10 dataset...")
    train_transform = T.Compose([
        T.RandomCrop(32, padding=4),
        T.RandomHorizontalFlip(),
        T.ToTensor(),
        T.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
    ])
    test_transform = T.Compose([
        T.ToTensor(),
        T.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
    ])
    train_dataset = torchvision.datasets.CIFAR10(root='./data', train=True, download=True, transform=train_transform)
    test_dataset = torchvision.datasets.CIFAR10(root='./data', train=False, download=True, transform=test_transform)
    train_loader = DataLoader(train_dataset, batch_size=128, shuffle=True, num_workers=2)
    test_loader = DataLoader(test_dataset, batch_size=100, shuffle=False, num_workers=2)
    classes = ('plane', 'car', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck')

    # モデル定義
    model = MCResNet18(dropout_p=0.2).to(device)
    print_both("\n--- Model Structure (MC Dropout ResNet18 for CIFAR-10) ---")
    print_both(model)
    model_path = "mmdropout_resnet18_cifar10.pth"

    # モデルファイルが存在すればロード、なければ学習
    if os.path.exists(model_path):
        print_both(f"\nFound trained model file: {model_path}. Loading model...")
        model.load_state_dict(torch.load(model_path, map_location=device))
        print_both("Model loaded. Skipping training.")
        do_train = False
    else:
        do_train = True

    # 学習
    if do_train:
        criterion = nn.CrossEntropyLoss()
        optimizer = optim.Adam(model.parameters(), lr=0.001)
        num_epochs = 15
        print_both("\nStarting model training...")
        train_loss_list = []
        test_loss_list = []
        train_acc_list = []
        test_acc_list = []
        for epoch in range(num_epochs):
            model.train()
            running_loss = 0.0
            correct_train = 0
            total_train = 0
            for images, labels in tqdm(train_loader, desc=f"Epoch {epoch+1}/{num_epochs}", leave=False):
                images, labels = images.to(device), labels.to(device)
                optimizer.zero_grad()
                outputs = model(images)
                loss = criterion(outputs, labels)
                loss.backward()
                optimizer.step()
                running_loss += loss.item()
                _, predicted = torch.max(outputs.data, 1)
                total_train += labels.size(0)
                correct_train += (predicted == labels).sum().item()
            epoch_loss = running_loss / len(train_loader)
            train_loss_list.append(epoch_loss)
            train_acc = 100 * correct_train / total_train
            train_acc_list.append(train_acc)
            # テスト
            model.eval()
            test_running_loss = 0.0
            test_batches = 0
            correct_test = 0
            total_test = 0
            with torch.no_grad():
                for images, labels in test_loader:
                    images, labels = images.to(device), labels.to(device)
                    outputs = model(images)
                    loss = criterion(outputs, labels)
                    test_running_loss += loss.item()
                    test_batches += 1
                    _, predicted = torch.max(outputs.data, 1)
                    total_test += labels.size(0)
                    correct_test += (predicted == labels).sum().item()
            test_epoch_loss = test_running_loss / test_batches
            test_loss_list.append(test_epoch_loss)
            test_acc = 100 * correct_test / total_test
            test_acc_list.append(test_acc)
            print_both(f"Epoch [{epoch+1}/{num_epochs}], Loss: {epoch_loss:.4f}, Test Loss: {test_epoch_loss:.4f}, Train Acc: {train_acc:.2f}%, Test Acc: {test_acc:.2f}%")
        print_both("Training finished.")
        # グラフ保存
        plt.figure()
        plt.plot(range(1, num_epochs+1), train_loss_list, marker='o', label='Train Loss')
        plt.plot(range(1, num_epochs+1), test_loss_list, marker='s', label='Test Loss')
        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        plt.title('Training & Test Loss Curve')
        plt.legend()
        plt.grid(True)
        plt.tight_layout()
        plt.savefig('mmdropout_training_loss_curve.png')
        plt.figure()
        plt.plot(range(1, num_epochs+1), train_acc_list, marker='o', label='Train Accuracy')
        plt.plot(range(1, num_epochs+1), test_acc_list, marker='s', label='Test Accuracy')
        plt.xlabel('Epoch')
        plt.ylabel('Accuracy (%)')
        plt.title('Training & Test Accuracy Curve')
        plt.legend()
        plt.grid(True)
        plt.tight_layout()
        plt.savefig('mmdropout_training_accuracy_curve.png')
        torch.save(model.state_dict(), model_path)

        print_both(f"Trained model saved to: {model_path}")

    # --------------------------------------------------------------------------
    # 事後確率による認識性能（MC Dropoutなし）
    # --------------------------------------------------------------------------
    print_both("\nEvaluating standard (no MC Dropout) prediction accuracy...")
    model.eval()
    correct_std = 0
    total_std = 0
    softmax = nn.Softmax(dim=1)
    with torch.no_grad():
        for images, labels in tqdm(test_loader, desc="Standard Evaluation"):
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            probs = softmax(outputs)
            _, predicted = torch.max(probs, 1)
            total_std += labels.size(0)
            correct_std += (predicted == labels).sum().item()
    acc_std = 100 * correct_std / total_std
    print_both(f"Standard prediction accuracy (no MC Dropout): {acc_std:.2f} %")

    # --------------------------------------------------------------------------
    # MC Dropoutによる平均事後確率での認識性能
    # --------------------------------------------------------------------------
    print_both("\nEvaluating MC Dropout mean-probability prediction accuracy...")
    model.eval()
    def enable_dropout(m):
        if type(m) == nn.Dropout:
            m.train()
    model.apply(enable_dropout)
    mc_passes = 30
    correct_mc = 0
    total_mc = 0
    with torch.no_grad():
        for images, labels in tqdm(test_loader, desc="MC Dropout Mean Prob Evaluation"):
            images, labels = images.to(device), labels.to(device)
            probs_mc = []
            for _ in range(mc_passes):
                outputs = model(images)
                probs = softmax(outputs)
                probs_mc.append(probs.cpu().numpy())
            probs_mc = np.stack(probs_mc, axis=0)  # (mc_passes, batch, num_classes)
            mean_probs = np.mean(probs_mc, axis=0)  # (batch, num_classes)
            predicted = np.argmax(mean_probs, axis=1)
            correct_mc += (predicted == labels.cpu().numpy()).sum()
            total_mc += labels.size(0)
    acc_mc = 100 * correct_mc / total_mc
    print_both(f"MC Dropout mean-probability prediction accuracy: {acc_mc:.2f} %")



    # モンテカルロDropoutによる不確実性推定
    print_both("\nEvaluating with Monte Carlo Dropout...")
    model.eval()
    def enable_dropout(m):
        if type(m) == nn.Dropout:
            m.train()
    model.apply(enable_dropout)
    mc_passes = 30
    all_entropy = []
    all_mean_probs = []
    all_labels = []
    softmax = nn.Softmax(dim=1)
    with torch.no_grad():
        for images, labels in tqdm(test_loader, desc="MC Dropout Evaluation"):
            images = images.to(device)
            probs_mc = []
            for _ in range(mc_passes):
                outputs = model(images)
                probs = softmax(outputs)
                probs_mc.append(probs.cpu().numpy())
            probs_mc = np.stack(probs_mc, axis=0)  # (mc_passes, batch, num_classes)
            mean_probs = np.mean(probs_mc, axis=0)  # (batch, num_classes)
            entropy = -np.sum(mean_probs * np.log(mean_probs + 1e-8), axis=1)  # (batch,)
            all_entropy.extend(entropy.tolist())
            all_mean_probs.extend(mean_probs.tolist())
            all_labels.extend(labels.numpy().tolist())
    all_entropy = np.array(all_entropy)
    all_labels = np.array(all_labels)
    # 不確実性ヒストグラム
    plt.figure()
    plt.hist(all_entropy, bins=50, color='blue', alpha=0.7)
    plt.xlabel('Predictive Entropy')
    plt.ylabel('Count')
    plt.title('Uncertainty (Predictive Entropy) Histogram')
    plt.tight_layout()
    plt.savefig('mmdropout_uncertainty_hist.png')
    print_both("Uncertainty histogram saved to: mmdropout_uncertainty_hist.png")
    # クラスごとの不確実性平均
    class_entropy = [all_entropy[all_labels == i].mean() for i in range(10)]
    plt.figure()
    plt.bar(classes, class_entropy)
    plt.xlabel('Class')
    plt.ylabel('Mean Predictive Entropy')
    plt.title('Mean Uncertainty per Class')
    plt.tight_layout()
    plt.savefig('mmdropout_uncertainty_per_class.png')
    print_both("Uncertainty per class barplot saved to: mmdropout_uncertainty_per_class.png")
    print_both("\n--- Finished ---")

    # --------------------------------------------------------------------------
    # 不確実性の高い10枚のテストサンプルの詳細を表示（画像も保存＆表示）
    # --------------------------------------------------------------------------
    print_both("\nTop 10 most uncertain test samples:")
    topk_idx = np.argsort(-all_entropy)[:10]
    all_mean_probs = np.array(all_mean_probs)  # shape: (num_samples, num_classes)
    for rank, idx in enumerate(topk_idx, 1):
        true_label = all_labels[idx]
        mean_probs = all_mean_probs[idx]
        pred_label = np.argmax(mean_probs)
        entropy = all_entropy[idx]
        image, label = test_dataset[idx]
        # 画像を表示・保存
        img_np = image.numpy().transpose(1, 2, 0)
        img_np = (img_np * np.array([0.2023, 0.1994, 0.2010])) + np.array([0.4914, 0.4822, 0.4465])
        img_np = np.clip(img_np, 0, 1)
        plt.figure(figsize=(2.5,2.5))
        plt.imshow(img_np)
        plt.axis('off')
        plt.title(f"True: {classes[true_label]}\nPred: {classes[pred_label]}\nEntropy: {entropy:.3f}")
        plt.tight_layout()
        plt.savefig(f"mmdropout_uncertain_sample_{rank}.png")
        plt.close()
        # 画像ファイル名もログに出力
        print_both(f"\nSample {rank}: (image saved: mmdropout_uncertain_sample_{rank}.png)")
        print_both(f"  True label      : {classes[true_label]} ({true_label})")
        print_both(f"  Predicted label : {classes[pred_label]} ({pred_label})")
        print_both(f"  Uncertainty (entropy): {entropy:.4f}")
        print_both(f"  Mean posterior probabilities:")
        for i, prob in enumerate(mean_probs):
            print_both(f"    {classes[i]:>6}: {prob:.4f}")
        # 各クラスごとの不確実性（エントロピー）再計算
        image_ = image.unsqueeze(0).to(device)
        mc_probs = []
        model.apply(enable_dropout)
        with torch.no_grad():
            for _ in range(mc_passes):
                outputs = model(image_)
                probs = softmax(outputs)
                mc_probs.append(probs.cpu().numpy()[0])
        mc_probs = np.stack(mc_probs, axis=0)  # (mc_passes, num_classes)
        class_entropies = -np.sum(mc_probs * np.log(mc_probs + 1e-8), axis=0)
        print_both(f"  Per-class uncertainty (entropy):")
        for i, ce in enumerate(class_entropies):
            print_both(f"    {classes[i]:>6}: {ce:.4f}")

if __name__ == '__main__':
    main()
