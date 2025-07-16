import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
import torchvision.transforms as T
from torch.utils.data import DataLoader
from tqdm import tqdm
import numpy as np

# 学習曲線描画用
import matplotlib.pyplot as plt

# モデル保存・読み込み用
import os

# RandAugment
from torchvision.transforms import RandAugment

def print_both(*args, **kwargs):
    """printの内容をファイル(tta_cifar-10.log)にも保存"""
    print(*args, **kwargs)
    with open("tta_cifar-10.log", "a", encoding="utf-8") as f:
        print(*args, **kwargs, file=f)


def main():
    """
    メイン関数：TTAの全プロセスを実行します。
    """
    # --------------------------------------------------------------------------
    # 1. 初期設定
    # --------------------------------------------------------------------------
    # デバイスの設定 (GPUが利用可能ならGPUを使用)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print_both(f"Using device: {device}")

    # --------------------------------------------------------------------------
    # 2. データセットとデータローダーの準備
    # --------------------------------------------------------------------------
    print_both("\nLoading CIFAR-10 dataset...")
    # 学習用データ拡張（RandAugmentを追加）
    train_transform = T.Compose([
        T.RandomCrop(32, padding=4),
        T.RandomHorizontalFlip(),
        RandAugment(num_ops=2, magnitude=9),  # RandAugmentを追加
        T.ToTensor(),
        T.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
    ])

    # テスト用データ拡張 (TTAなしのベースライン評価用)
    test_transform = T.Compose([
        T.ToTensor(),
        T.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
    ])

    # CIFAR-10データセットの読み込み
    train_dataset = torchvision.datasets.CIFAR10(root='./data', train=True, download=True, transform=train_transform)
    test_dataset_for_baseline = torchvision.datasets.CIFAR10(root='./data', train=False, download=True, transform=test_transform)
    # TTA評価用データセット (変換は後から手動で適用するため、ここではtransform=None)
    test_dataset_for_tta = torchvision.datasets.CIFAR10(root='./data', train=False, download=True, transform=None)


    train_loader = DataLoader(train_dataset, batch_size=128, shuffle=True, num_workers=2)
    test_loader_baseline = DataLoader(test_dataset_for_baseline, batch_size=100, shuffle=False, num_workers=2)

    classes = ('plane', 'car', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck')


    # --------------------------------------------------------------------------
    # 3. CNNモデルの定義 (ResNet18を使用)
    # --------------------------------------------------------------------------
    # torchvisionからResNet18をロードし、CIFAR-10用に調整
    # weights=None で未学習のモデルを初期化
    model = torchvision.models.resnet18(weights=None)
    # CIFAR-10は32x32と画像が小さいため、最初の層を調整
    model.conv1 = nn.Conv2d(3, 64, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)
    model.maxpool = nn.Identity() # MaxPool層を無効化
    model.fc = nn.Linear(model.fc.in_features, 10) # 出力層を10クラスに変更
    model = model.to(device)

    print("\n--- Model Structure (ResNet18 for CIFAR-10) ---")
    print(model)

    model_path = "trained_resnet18_cifar10.pth"


    # モデルファイルが存在すればロード、なければ学習
    if os.path.exists(model_path):
        print(f"\nFound trained model file: {model_path}. Loading model...")
        print("Model loaded. Skipping training.")
        do_train = False
    else:
        do_train = True

    # --------------------------------------------------------------------------
    # 4. モデルの学習
    # --------------------------------------------------------------------------

    if do_train:
        criterion = nn.CrossEntropyLoss()
        optimizer = optim.Adam(model.parameters(), lr=0.001)
        num_epochs = 15 # 学習のエポック数

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

                # 学習データでの認識率
                _, predicted = torch.max(outputs.data, 1)
                total_train += labels.size(0)
                correct_train += (predicted == labels).sum().item()

            epoch_loss = running_loss / len(train_loader)
            train_loss_list.append(epoch_loss)
            train_acc = 100 * correct_train / total_train
            train_acc_list.append(train_acc)

            # --- テストデータでのロス・認識率計算 ---
            model.eval()
            test_running_loss = 0.0
            test_batches = 0
            correct_test = 0
            total_test = 0
            with torch.no_grad():
                for images, labels in test_loader_baseline:
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


        # --- 学習曲線の描画（ロス） ---
        plt.figure()
        plt.plot(range(1, num_epochs+1), train_loss_list, marker='o', label='Train Loss')
        plt.plot(range(1, num_epochs+1), test_loss_list, marker='s', label='Test Loss')
        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        plt.title('Training & Test Loss Curve')
        plt.legend()
        plt.grid(True)
        plt.tight_layout()
        plt.savefig('training_loss_curve.png')


        # --- 学習曲線の描画（認識率） ---
        plt.figure()
        plt.plot(range(1, num_epochs+1), train_acc_list, marker='o', label='Train Accuracy')
        plt.plot(range(1, num_epochs+1), test_acc_list, marker='s', label='Test Accuracy')
        plt.xlabel('Epoch')
        plt.ylabel('Accuracy (%)')
        plt.title('Training & Test Accuracy Curve')
        plt.legend()
        plt.grid(True)
        plt.tight_layout()
        plt.savefig('training_accuracy_curve.png')

        # --- 学習済みモデルの保存 ---
        torch.save(model.state_dict(), model_path)
        print_both(f"Trained model saved to: {model_path}")

    # --------------------------------------------------------------------------
    # 5. TTAなしでの評価 (ベースライン)
    # --------------------------------------------------------------------------
    print_both("\nEvaluating without Test-Time Adaptation (Baseline)...")
    model.eval()
    correct = 0
    total = 0
    with torch.no_grad():
        for images, labels in tqdm(test_loader_baseline, desc="Evaluating (no TTA)"):
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

    accuracy_no_tta = 100 * correct / total
    print_both(f"Accuracy without TTA: {accuracy_no_tta:.2f} %")

    # --------------------------------------------------------------------------
    # 6. TTAありでの評価
    # --------------------------------------------------------------------------
    # Test-Time Augmentationsの定義
    tta_transforms = [
        # 1. 元の画像
        T.Compose([
            T.ToTensor(),
            T.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
        ]),
        # 2. 水平フリップした画像
        T.Compose([
            T.RandomHorizontalFlip(p=1.0),
            T.ToTensor(),
            T.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
        ]),
        # 3. 5ピクセルでパディング後、ランダムクロップ
        T.Compose([
            T.Pad(5),
            T.RandomCrop(32),
            T.ToTensor(),
            T.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
        ])
    ]

    print_both("\nEvaluating with Test-Time Adaptation...")
    model.eval()
    correct_tta = 0
    total_tta = 0
    softmax = nn.Softmax(dim=1)

    with torch.no_grad():
        # PIL画像を1枚ずつ処理
        for image_pil, label in tqdm(test_dataset_for_tta, desc="Evaluating (with TTA)"):
            label_tensor = torch.tensor([label]).to(device)
            tta_predictions = []

            # 定義した各TTA変換を適用して予測
            for transform in tta_transforms:
                # PIL Imageを変換し、バッチ次元を追加してモデルに入力
                augmented_image = transform(image_pil).unsqueeze(0).to(device)
                output = model(augmented_image)
                # Softmaxで確率に変換
                tta_predictions.append(softmax(output))

            # 複数の予測確率の平均を計算
            avg_prediction = torch.stack(tta_predictions).mean(0)
            _, final_prediction = torch.max(avg_prediction, 1)

            total_tta += 1
            correct_tta += (final_prediction == label_tensor).sum().item()

    accuracy_tta = 100 * correct_tta / total_tta
    print_both(f"Accuracy with TTA: {accuracy_tta:.2f} %")

    # --------------------------------------------------------------------------
    # 7. 結果の比較
    # --------------------------------------------------------------------------
    print_both("\n--- Final Results ---")
    print_both(f"Baseline Accuracy (without TTA): {accuracy_no_tta:.2f} %")
    print_both(f"Accuracy with TTA:               {accuracy_tta:.2f} %")
    print_both(f"Improvement:                     {accuracy_tta - accuracy_no_tta:+.2f} %")

if __name__ == '__main__':
    main()
